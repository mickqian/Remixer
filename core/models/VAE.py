from typing import List

import numpy
from diffusers import DPMSolverMultistepScheduler, DDPMScheduler
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import AutoencoderKL, ConsistencyDecoderVAE, VQModel
from diffusers.pipelines import ImagePipelineOutput, DiffusionPipeline
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils.torch_utils import randn_tensor

from core.data.codec import *
from core.utils import *


def build_pipeline(vae, generator, scheduler_path, config):
    pipeline = DiTPipeline(vae=vae, generator=generator, scheduler=scheduler_path, labels_to_id=genre2Id, config=config)
    return pipeline


def get_auto_encoder(config: TrainingConfig, s="vae"):
    if s == "vae":
        return AutoencoderKL(
            in_channels=config.vae_in_channels,
            out_channels=config.vae_out_channels,
            latent_channels=config.latent_channels,
            layers_per_block=config.layers_per_block,
            block_out_channels=config.block_out_channels,
            down_block_types=config.down_block_types,
            up_block_types=config.up_block_types,
            scaling_factor=config.scaling_factor,
            act_fn=config.act_fn,
            norm_num_groups=4,
        ).to(dtype=torch.float32)
    elif s == "consistent_vae":
        return ConsistencyDecoderVAE(
            encoder_in_channels=config.vae_in_channels,
            # if double_z is enabled, latent channels = 2 * encoder channels
            latent_channels=config.latent_channels,
            encoder_layers_per_block=config.layers_per_block,
            decoder_layers_per_block=config.layers_per_block,
            encoder_act_fn=config.act_fn,
            encoder_down_block_types=config.vq_down_block_types,
            encoder_out_channels=config.latent_channels,
            # decoder_block_out_channels=1,
            decoder_up_block_types=config.vq_up_block_types,
            scaling_factor=config.scaling_factor,
        ).to(dtype=torch.float32)
    elif s == "vq_vae":
        return VQModel(
            in_channels=config.vae_in_channels,
            latent_channels=config.vq_latent_channels,
            layers_per_block=config.vq_layers_per_block,
            act_fn=config.act_fn,
            up_block_types=config.up_block_types,
            out_channels=config.out_channels,
            down_block_types=config.down_block_types,
            block_out_channels=config.block_out_channels,
            num_vq_embeddings=config.num_vq_embeddings,
            scaling_factor=config.scaling_factor,
        ).to(dtype=torch.float32)

    else:
        raise Exception(s)


class VQGenerator(ModelMixin, ConfigMixin):
    """A spectrogram sd generator, which composes of:
    1. A VQ-VAE model to encode the audio. Vector-Quantizer helps to capture the repetitive patterns in music, e.g. chord progression
    2. A Transformer with cross attention on
    original audio
    """

    @register_to_config
    def __init__(self, config: TrainingConfig = None) -> None:
        super().__init__()
        if not config:
            config = TrainingConfig()
        from diffusers import Transformer2DModel

        self.transformer: Transformer2DModel = Transformer2DModel(
            in_channels=config.latent_channels,
            out_channels=config.latent_channels,
            num_attention_heads=config.num_attention_heads,
            attention_head_dim=config.attention_head_dim,
            sample_size=config.sample_size[1],
            norm_num_groups=config.norm_nums_groups,
            num_layers=config.generator_num_layers,
        )
        self.guidance_scale = config.guidance_scale
        # self.scheduler = DPMSolverMultistepScheduler(num_train_timesteps=config.num_inference_steps)
        self.scheduler = DDPMScheduler(num_train_timesteps=config.num_inference_steps)

    def __call__(self, x, timesteps, class_ids=None, styled_inputs=None):
        r"""Single inference process, predict the noise of a clean image

        Args:
            x: batch of encoded inputs

            class_ids: the genre_id of the input x

            styled_inputs: the encoded state of the styled input(in remix stage)

        Returns:
            The predicted noise
        """
        guidance_scale = self.guidance_scale

        latent_model_input = x

        bs, height, width, channels = latent_model_input.shape

        if class_ids is not None and styled_inputs is not None:
            raise ValueError("`Class_labels` and `style` can't be provided at the same time.")
        if class_ids is not None:
            # if class_labels is provided
            class_ids = class_ids.clone().detach().reshape(-1)
            # I don't know what the magic number is about
            class_null = torch.tensor([1000] * bs, device=x.device)
            class_labels_input = torch.cat([class_ids, class_null], 0) if guidance_scale > 1 else class_ids
        else:
            class_labels_input = None
        if styled_inputs:
            pass

        if guidance_scale > 1:
            half = latent_model_input[: len(latent_model_input) // 2]
            latent_model_input = torch.cat([half, half], dim=0)

        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = latent_model_input.device.type == "mps"
            if isinstance(timesteps, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=latent_model_input.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(latent_model_input.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(latent_model_input.shape[0])
        # predict noise model_output
        noise_pred = self.transformer(
            latent_model_input, timestep=timesteps, class_labels=class_labels_input, encoder_hidden_states=styled_inputs
        ).sample

        latent_channels = self.transformer.in_channels
        num_channels = noise_pred.shape[1]
        # perform guidance
        if guidance_scale > 1:
            eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
            if latent_channels == num_channels:
                cond_eps = eps
                uncond_eps = torch.empty(cond_eps.shape, device=noise_pred.device)
            else:
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)

            noise_pred = torch.cat([eps, rest], dim=1)

        # learned sigma
        if self.transformer.config.out_channels // 2 == latent_channels:
            model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
        else:
            model_output = noise_pred

        return model_output


class DiTPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation based on a Transformer backbone instead of a UNet.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        generator ([`Transformer2DModel`]):
            A class conditioned `Transformer2DModel` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        scheduler ([`DDIMScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """
    model_cpu_offload_seq = "transformer->vae"

    def __init__(
        self,
        generator: Union[VQGenerator, str],
        vae: Union[AutoencoderKL, str],
        scheduler: Union[KarrasDiffusionSchedulers, str],
        labels_to_id: Optional[Dict[str, int]],
        config: TrainingConfig,
    ):
        super().__init__()
        if isinstance(generator, str):
            generator = load_model(VQGenerator(config), file_path=generator)
        if isinstance(vae, str):
            vae = load_model(get_auto_encoder(config), file_path=vae)
        if isinstance(scheduler, str):
            # scheduler = .from_pretrained(scheduler)
            scheduler = DDPMScheduler.from_pretrained(scheduler)

        self.register_modules(generator=generator, vae=vae, scheduler=scheduler)

        # create a imagenet -> id dictionary for easier use
        self.labels_to_ids = labels_to_id

    def generate_audio(
        self,
        codec,
        genres: List[str] = None,
        desired_width=None,
        config: TrainingConfig = TrainingConfig(),
        count=1,
    ):
        """
        Generate a styled audio from specified genre

        Returns:
            A reconstructed audio
        """
        if not desired_width:
            desired_width = PreprocessingConfig().input_width

        if not genres:
            import random

            genres = random.sample(GENRES, count)

        class_ids = self.get_label_ids(genres)

        def generate(class_id, genre):
            feature = self(
                class_labels=[class_id],
                generator=torch.manual_seed(config.seed),
                num_inference_steps=config.num_inference_steps,
                output_type=None,
                # output_shape=(config.preprocessing.input_height, desired_width)
            )[0]

            feature = numpy.squeeze(feature)
            print(f"{feature.shape}")
            # file_name = utils.get_sample_file_path(genre, config)

            return feature, codec.decode(feature), feature_to_image(feature, config.preprocessing, show=True)

        return [generate(class_id, genre) for class_id, genre in zip(class_ids, genres)]

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        r"""

        Map label strings from ImageNet to corresponding class ids.

        Parameters:
            label (`str` or `dict` of `str`):
                Label strings to be mapped to class ids.

        Returns:
            `list` of `int`:
                Class ids to be processed by pipeline.
        """

        if not isinstance(label, list):
            label = list(label)

        for l in label:
            if l not in self.labels_to_ids:
                raise ValueError(
                    f"{l} does not exist. Please make sure to select one of the following labels: \n {self.labels_to_ids}."
                )

        return [self.labels_to_ids[l] for l in label]

    def prepare_latents(self, shape, dtype, device, generator, latents=None):
        batch_size = shape[0]
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        class_labels: List[int],
        style_input: Optional[torch.Tensor] = None,
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        output_shape: Tuple[int, int] = None,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            class_labels (List[int]):
                List of ImageNet class labels for the images to be generated.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 250):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        scheduler: DPMSolverMultistepScheduler = self.scheduler
        batch_size = len(class_labels)
        if output_shape:
            height, width = output_shape[0], output_shape[1]
        else:
            height, width = self.generator.config.config.sample_size
        device = torch.device("cuda")
        transformer = self.generator.transformer
        latent_channels = transformer.config.in_channels
        latent_model_input = self.prepare_latents(
            shape=(batch_size, latent_channels, height, width),
            generator=generator,
            device=device,
            dtype=transformer.dtype,
        )
        # print(f"{latents=}")

        # latent_model_input = torch.cat([latents] * 2).to(device=device) if guidance_scale > 1 else latents

        # class_labels = torch.tensor(class_labels, device=device).reshape(-1)
        # class_null = torch.tensor([1000] * batch_size, device=device)
        # class_labels_input = torch.cat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels
        class_labels_input = torch.tensor(class_labels, device=device).reshape(-1)

        # set step values
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps

        # encode hidden state for cross_attention
        if style_input:
            style_input = self.vae.encode(style_input).latent_dist.mode()

        # print(f"{latent_model_input}")
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        print(f"{num_warmup_steps=}")
        with self.progress_bar(self.scheduler.timesteps) as progress_bar:
            for i, t in enumerate(timesteps):
                if torch.any(torch.isnan(latent_model_input)):
                    print(f"nan detected at {t}")
                # if guidance_scale > 1:
                #     half = latent_model_input[: len(latent_model_input) // 2]
                #     latent_model_input = torch.cat([half, half], dim=0)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                timesteps = t
                if not torch.is_tensor(timesteps):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
                    is_mps = latent_model_input.device.type == "mps"
                    if isinstance(timesteps, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    timesteps = torch.tensor([timesteps], dtype=dtype, device=latent_model_input.device)
                elif len(timesteps.shape) == 0:
                    timesteps = timesteps[None].to(latent_model_input.device)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timesteps = timesteps.expand(latent_model_input.shape[0])
                # predict noise model_output
                noise_pred = transformer(
                    latent_model_input,
                    timestep=timesteps,
                    class_labels=class_labels_input,
                    encoder_hidden_states=style_input,
                ).sample

                # perform guidance
                # if guidance_scale > 1:
                #     # guidance on the latent channels only, weighted average on halves of latent inputs
                #     eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                #     cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                #
                #     half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                #     eps = torch.cat([half_eps, half_eps], dim=0)
                #
                #     # concat back
                #     noise_pred = torch.cat([eps, rest], dim=1)

                # learned sigma
                if transformer.config.out_channels // 2 == latent_channels:
                    model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
                else:
                    model_output = noise_pred

                # compute previous image: x_t -> x_t-1
                latent_model_input = scheduler.step(model_output, t, latent_model_input).prev_sample

                # displaying intermediate images
                if i % 200 == 0:
                    # decode
                    samples = 1 / self.vae.config.scaling_factor * latent_model_input
                    samples = self.vae.decode(samples).sample
                    samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
                    # print(f"{samples.shape}")
                    sample = samples.squeeze()
                    # sample = self.numpy_to_pil(samples)[0]
                    _image = feature_to_image(sample, show=True, title=f"timestep {i}")

                # progress_bar.update()
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        print(f"{latent_model_input.shape=}")

        # if guidance_scale > 1:
        #     latents, _ = latent_model_input.chunk(2, dim=0)
        # else:
        latents = latent_model_input

        latents = 1 / self.vae.config.scaling_factor * latents
        sample = self.vae.decode(latents).sample.cpu().float().numpy()

        # samples = (samples / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        # sample = sample.permute(0, 2, 3, 1)

        if output_type == "pil":
            sample = self.numpy_to_pil(sample)

        if not return_dict:
            return (sample,)

        return ImagePipelineOutput(images=sample)


PipelineOrPaths = Union[DiTPipeline, Tuple[str, str]]
