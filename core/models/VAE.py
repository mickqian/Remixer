from typing import List

import numpy
import torch.nn as nn
from diffusers import DPMSolverMultistepScheduler
from diffusers.models import AutoencoderKL
from diffusers.pipelines import ImagePipelineOutput, DiffusionPipeline
from diffusers.pipelines.stable_diffusion import *
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils.torch_utils import randn_tensor

from core import utils
from core.utils import *


def build_pipeline(vae_path, generator_path, config):
    pipeline = DiTPipeline(vae=vae_path, generator=generator_path, scheduler=None, labels_to_id=genre2Id, config=config)
    return pipeline


def build_VAE(config: TrainingConfig):
    return AutoencoderKL(in_channels=config.vae_in_channels, out_channels=1, latent_channels=config.latent_channels
                         , block_out_channels=(2,), norm_num_groups=2).to(dtype=torch.float32)


class VQGenerator(nn.Module):
    """A spectrogram sd generator, which composes of:
    1. A VQ-VAE model to encode the audio. Vector-Quantizer helps to capture the repetitive patterns in music, e.g. chord progression
    2. A Transformer with cross attention on
    original audio
    """

    def __init__(self, config: TrainingConfig = None) -> None:
        super().__init__()
        if not config:
            config = TrainingConfig()
        from diffusers import Transformer2DModel, DPMSolverMultistepScheduler
        self.transformer: Transformer2DModel = Transformer2DModel(in_channels=config.latent_channels, out_channels=1,
                                                                  num_attention_heads=config.num_attention_heads,
                                                                  attention_head_dim=config.attention_head_dim,
                                                                  sample_size=config.latent_width,
                                                                  norm_num_groups=config.norm_nums_groups)
        self.guidance_scale = config.guidance_scale
        self.scheduler = DPMSolverMultistepScheduler(num_train_timesteps=config.num_inference_steps)

    def __call__(self, x, timesteps, class_ids=None, style=None):
        r"""Single inference process, predict the noise of a clean image

        Args:
            x: batch of encoded inputs

            class_ids: the genre_id of the input x

            style: the encoded state of the styled input(in remix stage)

        Returns:
            The predicted noise
        """
        guidance_scale = self.guidance_scale

        latent_model_input = x

        bs, height, width, channels = latent_model_input.shape

        if class_ids is not None and style is not None:
            raise ValueError(
                "`Class_labels` and `style` can't be provided at the same time."
            )
        if class_ids is not None:
            # if class_labels is provided
            class_ids = class_ids.clone().detach().reshape(-1)
            # I don't know what the magic number is about
            class_null = torch.tensor([1000] * bs, device=x.device)
            class_labels_input = torch.cat([class_ids, class_null], 0) if guidance_scale > 1 else class_ids
        else:
            class_labels_input = None
        if style:
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
            latent_model_input, timestep=timesteps, class_labels=class_labels_input, encoder_hidden_states=style
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

    def sample(self, shape: Union[Tuple, List] = None):
        batch_size = 1
        num_channels_latents = 3
        desired_height, desired_width = shape
        device = torch.cuda.device
        dtype = torch.float32
        generator = generator
        pipeline: StableDiffusionPipeline = self.pipeline

        latents = pipeline.prepare_latents(batch_size, num_channels_latents, desired_height, desired_width,
                                           dtype=dtype, device=device, generator=generator, )

        return latents

    def generate_audio(self, genre: str, desired_width=None, config: TrainingConfig = TrainingConfig(), vae_path=None):
        """
        Generate a styled audio from specified genre

        Returns:
            A reconstructed audio
        """

        if not desired_width:
            desired_width = PreprocessingConfig().input_width

        pipeline = self.build_generation_pipeline(vae_name=vae_path)
        class_ids = pipeline.get_label_ids([genre])

        feature = pipeline(
            class_labels=class_ids,
            generator=torch.manual_seed(config.seed),
            num_inference_steps=config.num_inference_steps,
            output_type=None,
            output_shape=(config.preprocessing.input_height, desired_width)
        ).images[0]

        file_name = utils.get_sample_file_path(genre, config)
        return utils.feature_to_audio(feature, file_name, config.preprocessing)

    def build_generation_pipeline(self, vae_name=None):
        vae = load_model(build_VAE(TrainingConfig()), file_name=vae_name)
        pipeline = DiTPipeline(generator=self.transformer, vae=vae,
                               scheduler=KarrasDiffusionSchedulers.DPMSolverMultistepScheduler, labels_to_id=genre2Id)
        pipeline.scheduler = DPMSolverMultistepScheduler(num_train_timesteps=TrainingConfig().num_inference_steps)
        return pipeline

    @staticmethod
    def load(artifact_name):
        import os

        if isinstance(artifact_name, str):
            # load from wandb, whatever
            # model_artifact: wandb.Artifact = wandb.use_artifact(artifact_name)
            # model_dir = model_artifact.download()
            # model_config = model_artifact.metadata

            model_dir = "."
            # artifact_name = "model_epoch_4"
            model_config = TrainingConfig()
            vae = VQGenerator()
            model_path = os.path.join(model_dir, f"{artifact_name}.pth")
            vae.load_state_dict(torch.load(model_path))
            # vae = VQGenerator(model_config)

        # a hack to DitPipeline, controls the output image be of shape sample_size * sample_size
        vae.transformer.config.sample_size = PreprocessingConfig().input_width
        return vae


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
            generator: Tuple[VQGenerator, str],
            vae: Tuple[AutoencoderKL, str],
            scheduler: KarrasDiffusionSchedulers,
            labels_to_id: Optional[Dict[str, int]],
            config: TrainingConfig,
    ):
        super().__init__()
        if isinstance(generator, str):
            generator = load_model(VQGenerator(config), file_path=generator)
        if isinstance(vae, str):
            vae = load_model(build_VAE(config), file_path=vae)
        if not scheduler:
            scheduler = DPMSolverMultistepScheduler(config.num_inference_steps)

        self.register_modules(generator=generator, vae=vae, scheduler=scheduler)

        # create a imagenet -> id dictionary for easier use
        self.labels_to_ids = labels_to_id

    def generate_audio(self, genre: str, desired_width=None, config: TrainingConfig = TrainingConfig()):
        """
        Generate a styled audio from specified genre

        Returns:
            A reconstructed audio
        """

        if not desired_width:
            desired_width = PreprocessingConfig().input_width

        class_ids = self.get_label_ids([genre])

        feature = self(
            class_labels=class_ids,
            generator=torch.manual_seed(config.seed),
            num_inference_steps=config.num_inference_steps,
            output_type=None,
            output_shape=(config.preprocessing.input_height, desired_width)
        ).images[0]

        feature = numpy.squeeze(feature)

        file_name = utils.get_sample_file_path(genre, config)
        return utils.feature_to_audio(feature, file_name, config.preprocessing)

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
            output_shape: Tuple[int, int] = None
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

        Examples:

        ```py
        >>> from diffusers import DiTPipeline, DPMSolverMultistepScheduler
        >>> import torch

        >>> pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
        >>> pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        >>> pipe = pipe.to("cuda")

        >>> # pick words from Imagenet class labels
        >>> pipe.labels_to_ids  # to print all available words

        >>> # pick words that exist in ImageNet
        >>> words = ["white shark", "umbrella"]

        >>> class_ids = pipe.get_label_ids(words)

        >>> generator = torch.manual_seed(33)
        >>> output = pipe(class_labels=class_ids, num_inference_steps=25, generator=generator)

        >>> image = output.images[0]  # label 'white shark'
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """

        batch_size = len(class_labels)
        if output_shape:
            height, width = output_shape[0], output_shape[1]
        else:
            height = width = latent_size
        device = torch.device("cuda")
        transformer = self.generator.transformer
        latent_channels = transformer.config.in_channels
        latents = randn_tensor(
            shape=(batch_size, latent_channels, height, width),
            generator=generator,
            device=device,
            dtype=transformer.dtype,
        )
        latent_model_input = torch.cat([latents] * 2).to(device=device) if guidance_scale > 1 else latents

        class_labels = torch.tensor(class_labels, device=device).reshape(-1)
        class_null = torch.tensor([1000] * batch_size, device=device)
        class_labels_input = torch.cat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        # encode hidden state for cross_attention
        if style_input:
            style_input = self.vae.encode(style_input).latent_dist.mode()

        for t in self.progress_bar(self.scheduler.timesteps):
            if guidance_scale > 1:
                half = latent_model_input[: len(latent_model_input) // 2]
                latent_model_input = torch.cat([half, half], dim=0)
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
                latent_model_input, timestep=timesteps, class_labels=class_labels_input,
                encoder_hidden_states=style_input,
            ).sample

            # perform guidance
            if guidance_scale > 1:
                # guidance on the latent channels only, weighted average on halves of latent inputs
                eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

                half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)

                # concat back
                noise_pred = torch.cat([eps, rest], dim=1)

            # learned sigma
            if transformer.config.out_channels // 2 == latent_channels:
                model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
            else:
                model_output = noise_pred

            # compute previous image: x_t -> x_t-1
            latent_model_input = self.scheduler.step(model_output, t, latent_model_input).prev_sample

        if guidance_scale > 1:
            latents, _ = latent_model_input.chunk(2, dim=0)
        else:
            latents = latent_model_input

        latents = 1 / self.vae.config.scaling_factor * latents
        samples = self.vae.decode(latents).sample

        samples = (samples / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            samples = self.numpy_to_pil(samples)

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)


PipelineOrPaths = Union[DiTPipeline, Tuple[str, str]]
