"""training framework."""
import gc

import diffusers
import torch.utils.data.distributed
import wandb
from dataclass_wizard import asdict
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.functional import F
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm

from core.data import *
from core.models.VAE import get_auto_encoder, VQGenerator
from core.utils import *
from test_model import *


def tt():
    import torch
    from encodec import EncodecModel
    from audiocraft.data.audio import audio_read

    bandwidth = 3.0  # 1.5, 3.0, 6.0
    encodec = EncodecModel.encodec_model_24khz()

    wav, sr = audio_read(somepath)
    with torch.no_grad():
        emb = mbd.get_condition(audio, sample_rate=mbd.codec_model.sample_rate)
        size = wav.size()
        out = self.generate(emb, size=size)
        compressed_diffusion = mbd.regenerate(wav, sample_rate=sr)
        compressed_diffusion = mbd.enco(wav, sample_rate=sr)


class Trainer:
    config: TrainingConfig
    vae_path = ""
    generator_path = ""
    accelerate_path = ""

    def __init__(
        self,
        vae_path="",
        generator_path="",
        accelerate_path="",
        config=TrainingConfig(),
        dataset_paths=[],
        vae="vae",
        codec=None,
    ):
        self.config = config
        self.vae_path = vae_path
        self.vae = vae
        self.generator_path = generator_path
        self.accelerate_path = accelerate_path

        self.codec = codec
        [train_dl, _val_dl, _test_dl] = self.prepare_dataloaders(dataset_paths, codec)
        self.train_dl = train_dl

    def train_vae(
        self,
        is_train: bool = True,
        _epoch_callback=None,
    ):
        r"""
        train a VAE
        """
        from core.data.codec import show_spec

        config = self.config

        accelerator = build_accelerator(
            config, accelerate_state=self.accelerate_path, use_wandb=True, project_name="VAE"
        )

        # inception_model = inception_v3(pretrained=True, transform_input=False)
        # inception_model.fc = torch.nn.Identity()  # 用于提取特征，而不是分类

        scaler = get_scaler(config.preprocessing.scale_method)

        vae = load_model(get_auto_encoder(self.config, self.vae), file_path=self.vae_path)

        vae, optimizer, lr_scheduler = self.build_models(vae, lr=self.config.vae_learning_rate)
        name = vae.__class__.__name__
        _bandwidth = 3.0  # 1.5, 3.0, 6.0
        # self.vae = MultiBandDiffusion.get_mbd_24khz(bw=bandwidth)
        # self.vae = load_model(get_auto_encoder(config,self.vae), file_path=self.vae_path)

        # model AutoEncoderKL
        model = accelerator.prepare(vae)
        # model = model.module
        # inner = model.codec_model

        # enabled when FSDP
        optimizer, dataloader, lr_scheduler = accelerator.prepare(optimizer, self.train_dl, lr_scheduler)

        if accelerator.is_main_process:
            run: wandb.sdk.wandb_run.Run = accelerator.get_tracker("wandb", unwrap=True)
            # print(run.__class__.__name__)
            run_name = run.name
            wandb.watch(model, log="all", log_freq=30, log_graph=False)

        print(f"scaler: {scaler}")

        global_step = 0

        history_best_loss = best_loss = math.inf

        if is_train:
            model.train()
        else:
            model.eval()
        # self.test_vae(vae, config)
        for epoch in range(config.num_epochs if is_train else 1):
            progress_bar = tqdm(
                total=len(dataloader), leave=True, position=0, disable=not accelerator.is_local_main_process, ncols=150
            )
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(dataloader):
                with accelerator.accumulate(model):
                    # inner = model
                    inner = model.module
                    scaling_factor = inner.config.scaling_factor

                    clean_images, (_files, _ids) = batch
                    # if accelerator.is_local_main_process and step == 0 and epoch % config.save_image_epochs == 0:
                    #     # feature_to_image(clean_images[0, :].cpu().float().numpy(), PreprocessingConfig(), True)
                    #     feature_to_image(clean_images[0, 0, :].cpu().float().numpy(), PreprocessingConfig(), True)

                    # inner = model.codec_model.to(device=accelerator.device)
                    # inner: diffusers.AutoencoderKL = model.module

                    # Encode
                    x = clean_images * scaling_factor
                    # clean_images = inner.get_condition(clean_images sample_rate=inner.codec_model.sample_rate)
                    if "VQ" in name:
                        z = inner.encode(x).latents
                    else:
                        posterior = inner.encode(x).latent_dist
                        # clean_images, _scale = inner.encode(clean_images)

                        # size = clean_images.size()
                        # posterior = DiagonalGaussianDistribution(clean_images.float())
                        # Sample, Reparameterization
                        z = posterior.sample()

                    # Decode
                    z = 1 / scaling_factor * z
                    # z = z.int()
                    dec = inner.decode(z).sample

                    if config.preprocessing.scale_method == "min_max":
                        dec = dec.clip(min=scaler.feature_range[0], max=scaler.feature_range[1])
                    # dec = torch.clip(dec, -1, 1)
                    # dec = inner.decode(z)
                    # dec = inner.generate(z, size=size)

                    loss = 0
                    # reconstruction loss and regularization loss
                    if "VQ" in name:
                        kl = 0.0
                        pass
                    else:
                        kl_loss_factor = 0.00001
                        kl_loss = kl_loss_factor * posterior.kl().mean()
                        loss += kl_loss
                    # if dec.shape[3] < clean_images.shape[3]:
                    #     clean_images = clean_images[:, :, :, :dec.shape[3]]
                    # clean_images = scaler.inverse_transform(clean_images.cpu())
                    # dec = scaler.inverse_transform(dec)
                    reconstruction_loss = F.mse_loss(clean_images, dec)
                    loss += reconstruction_loss
                    factor = 1
                    loss = loss * factor

                    # if loss > 660:
                    #     image = feature_to_image(clean_images[0, :], config=config.preprocessing)
                    #     images = wandb.Image(image, caption="Spectrogram")
                    #     wandb.log({"spectrogram": images})
                    #     print(files)

                    del dec, x
                    if is_train:
                        if step % 15 == 0 and epoch % config.save_image_epochs == 0:
                            inner.eval()

                            if "VQ" in name:
                                dec = inner.decode(z).sample
                                pass
                            else:
                                # we should take the mode to test the reconstruction ability
                                z = posterior.mode()[0:1, :]
                                z = 1 / scaling_factor * z
                                dec = inner.decode(z).sample
                            reconstructed, clean_image = (
                                dec[0, 0, :].cpu().detach().numpy(),
                                clean_images[0, 0, :].cpu().detach().numpy(),
                            )
                            shape = reconstructed.shape
                            if config.preprocessing.scale_method == "power":
                                reconstructed, clean_image = reconstructed.flatten().reshape(
                                    -1, 1
                                ), clean_image.flatten().reshape(-1, 1)
                            reconstructed, clean_image = scaler.inverse_transform(
                                reconstructed
                            ), scaler.inverse_transform(clean_image)

                            if config.preprocessing.scale_method == "power":
                                reconstructed, clean_image = (
                                    reconstructed.reshape(shape),
                                    clean_image.reshape(shape),
                                )
                            mse = ((clean_image - reconstructed) ** 2).mean()
                            accelerator.log({"mse": mse}, step=global_step)
                            # fid = calculate_fid_of_images(image, clean_image, inception_model)
                            show_spec([clean_image, reconstructed], config.preprocessing, True, title=f"mse: {mse}")
                            mel_to_audio(reconstructed, config.preprocessing)
                            inner.train()

                            artifact_name = f"model_epoch_{epoch}"
                            self.save_model(
                                accelerator=accelerator,
                                models=[accelerator.unwrap_model(model) for model in [inner]],
                                run_name=run_name,
                                name=artifact_name,
                            )
                            del dec, clean_image, reconstructed

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(inner.parameters(), 1.0)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                progress_bar.update(1)
                reconstruction_loss = reconstruction_loss.detach().item()
                l = loss.detach().item()
                logs = {
                    "loss": l,
                    "kl_loss": kl_loss.detach().item(),
                    "reconstruction_loss": reconstruction_loss,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch + (step / len(dataloader)),
                }
                best_loss = min(best_loss, l)
                accelerator.log(logs, step=global_step)
                logs.update({"step": global_step})
                progress_bar.set_postfix(**logs)
                global_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if is_train and accelerator.is_main_process:
                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    if best_loss < history_best_loss:
                        # history_best_loss = best_loss
                        artifact_name = f"model_epoch_{epoch}"
                        self.save_model(
                            accelerator=accelerator,
                            models=[accelerator.unwrap_model(model) for model in [model]],
                            run_name=run_name,
                            name=artifact_name,
                        )
        if not os.path.exists("scaler.joblib"):
            import joblib

            joblib.dump(scaler, f"{config.preprocessing.scale_method}_scaler.joblib")
        accelerator.wait_for_everyone()

        return best_loss

    def test_vae(self, vae, config: TrainingConfig):
        """
        Test the VAE by sampling a randn tensor
        """
        from diffusers.utils.torch_utils import randn_tensor

        device = vae.device
        [height, width] = config.sample_size
        latent_channels = config.latent_channels
        z = randn_tensor((1, latent_channels, height, width), device=device)
        # z = z * scheduler.init_noise_sigma
        z = 1 / vae.config.scaling_factor * z
        z = vae.decode(z).sample

        image = get_scaler(config.preprocessing.scale_method).inverse_transform(z[0, 0, :].cpu().detach().numpy())
        show_spec(
            [image],
            config.preprocessing,
            True,
        )

    def train_diffusion(self, is_train=True, class_inputs=True):
        r"""
        Main training loop for generator

        Args:
             class_inputs: (`bool`):
                Whether class_inputs will be provided to the model. If not, styled audio will be passed to as encoder-hidden-state
        """
        vae = load_model(
            get_auto_encoder(self.config, self.vae),
            file_path=self.vae_path,
        )
        generator = load_model(
            VQGenerator(self.config),
            file_path=self.generator_path,
        )
        generator, optimizer, lr_scheduler = self.build_models(generator, lr=self.config.generator_learning_rate)
        config = self.config

        accelerator = build_accelerator(
            self.config, accelerate_state=self.accelerate_path, use_wandb=True, project_name="Generator"
        )

        # Prepare everything
        model = accelerator.prepare(generator)

        optimizer, dataloader, lr_scheduler, vae = accelerator.prepare(optimizer, self.train_dl, lr_scheduler, vae)
        inner = model.module
        noise_scheduler = inner.scheduler

        if accelerator.is_main_process:
            run: wandb.sdk.wandb_run.Run = accelerator.get_tracker("wandb", unwrap=True)
            # print(run.__class__.__name__)
            run_name = run.name
            wandb.watch(inner, log="all", log_freq=30, log_graph=False)

        losses = []
        global_step = 0

        vae_dtype = torch.float32
        model_dtype = torch.float32
        # vae = vae.to(dtype=vae_dtype)

        history_best_loss = math.inf
        vae: diffusers.AutoEncoderKL = vae.module
        vae.eval()
        if is_train:
            inner.train()
        else:
            inner.eval()

        for epoch in range(config.num_epochs if is_train else 1):
            progress_bar = tqdm(
                total=len(dataloader),
                leave=True,
                position=0,
                disable=not accelerator.is_local_main_process,
                ncols=150,
                colour="green",
            )
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in enumerate(dataloader):
                with accelerator.accumulate(model):
                    if class_inputs:
                        clean_images, (_files, class_ids) = batch
                    else:
                        clean_images, styled_inputs = batch

                    bs = clean_images.shape[0]

                    # 1. Encode the clean images with VAE encoder
                    clean_images = vae.config.scaling_factor * clean_images
                    clean_latents = (
                        vae.encode(clean_images.to(dtype=vae_dtype)).latent_dist.sample().to(dtype=model_dtype)
                    )
                    # clean_latents = encode_output.latent_dist.mode()

                    # 2. Sample a noise to add to all latents
                    noise = torch.randn(clean_latents.shape, device=accelerator.device, dtype=torch.float32)

                    # 3. Sample a random timestep for each latent
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bs,), dtype=torch.int64
                    ).long()

                    # 4. Add noise to the clean images according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(clean_latents, noise, timesteps).to(torch.float32)
                    noisy_latents = noise_scheduler.scale_model_input(noisy_latents, timesteps).to(torch.float32)

                    if not class_inputs:
                        # encode the styled_input
                        styled_inputs = vae.config.scaling_factor * styled_inputs
                        styled_inputs = (
                            vae.encode(styled_inputs.to(dtype=vae_dtype)).latent_dist.sample().to(dtype=model_dtype)
                        )

                    # predict the noise residual
                    if class_inputs:
                        noise_pred = model(noisy_latents, timesteps, class_ids, styled_inputs=None)
                    else:
                        noise_pred = model(noisy_latents, timesteps, class_ids=None, styled_inputs=styled_inputs)

                    # predict the L1-norm loss of noise and predicted_noise
                    noise_loss = 1000 * F.mse_loss(noise, noise_pred)

                    loss = noise_loss

                    if is_train:
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                progress_bar.update(1)
                l = loss.detach().item()
                logs = {"loss": l, "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                losses += [l]
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if is_train and accelerator.is_main_process:
                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    # evaluate(config, epoch, model)
                    best_loss = min(losses, default=math.inf)
                    if best_loss < history_best_loss:
                        # history_best_loss = best_loss
                        artifact_name = f"model_epoch_{epoch}"
                        self.save_model(
                            accelerator=accelerator,
                            models=[accelerator.unwrap_model(model) for model in [model, noise_scheduler]],
                            run_name=run_name,
                            name=artifact_name,
                        )
                if (epoch + 1) % config.save_image_epochs == 0:
                    self.evaluate_model(vae, accelerator.unwrap_model(model), noise_scheduler)

            gc.collect()
            # epoch_callback(epoch, )
        # accelerator.wait_for_everyone()

        import joblib

        joblib.dump(get_scaler(), "scaler.joblib")
        return model

    def save_model(self, accelerator, models, run_name, name):
        base = f"artifacts/{run_name}/"
        os.makedirs(base, exist_ok=True)
        print(f"saving in {base}")

        def save(state_dict, model_name=None):
            config = self.config
            artifact = wandb.Artifact(f"{model_name}", type="model", metadata={"params": asdict(config)})
            base = f"artifacts/{run_name}/{model_name}"
            os.makedirs(base, exist_ok=True)

            # path = os.path.join(base, f'{name}.pth')
            # torch.save(state_dict, path)
            # artifact.add_file(path)

            path = os.path.join(base, f"{name}.pth")
            accelerator.save(state_dict, path)
            artifact.add_file(path)
            wandb.log_artifact(artifact)

        for model in models:
            if isinstance(model, FullyShardedDataParallel):
                with FullyShardedDataParallel.state_dict_type(
                    model,
                    StateDictType.LOCAL_STATE_DICT,  # or any other StateDictType
                ):
                    save(state_dict=model.state_dict(), model_name=str(model.module.__class__.__name__))
            elif isinstance(model, DistributedDataParallel):
                save(state_dict=model.module.state_dict(), model_name=str(model.module.__class__.__name__))
            elif isinstance(model, nn.Module):
                save(state_dict=model.state_dict(), model_name=str(model.__class__.__name__))
                # save(state_dict=model.module.state_dict(), model_name=str(model.module.__class__.__name__))
            elif hasattr(model, "save_pretrained"):
                model.save_pretrained(base)

        # artifact.wait()
        # for file in artifact.files():
        #     print(file.path())

    def evaluate_model(
        self,
        vae,
        generator,
        scheduler,
    ):
        r"""
        Evaluate a model by:
        1. Test on the test dataset
        2. Generate some audios
        """
        pipeline = build_pipeline(vae=vae, generator=generator, scheduler_path=scheduler, config=self.config)

        config = self.config
        results = pipeline.generate_audio(
            codec=self.codec,
            genres=None,
            config=config,
            count=1,
        )

        make_grid([image for _, _, image in results])
        # generate_from_genre()

    def generate(self):
        device = torch.device("cuda")
        vae = load_model(
            self.config,
            file_path=self.vae_path,
        ).to(device=device)
        generator = load_model(
            VQGenerator(self.config),
            file_path=self.generator_path,
        ).to(device=device)
        generator, optimizer, lr_scheduler = self.build_models(generator, self.config.generator_learning_rate)
        _config = self.config
        self.evaluate_model(vae, generator, generator.scheduler)

    def prepare_dataloaders(self, dataset_paths, codec):
        from torch.utils.data import DataLoader, ConcatDataset

        tr_dss = []
        val_dss = []
        test_dss = []
        for dataset_path in dataset_paths:
            files, ids, _genres = read_remixer_dataset(dataset_path, all=True)
            tr_ds, val_ds, test_ds = build_remix_dataset(files=files, ids=ids, ratios=self.config.ratios, codec=codec)
            tr_dss += [tr_ds]
            val_dss += [val_ds]
            test_dss += [test_ds]

        return [
            DataLoader(ConcatDataset(dss), self.config.train_batch_size, shuffle=True)
            for dss in [tr_dss, val_dss, test_dss]
        ]

    def build_models(
        self,
        model_or_type,
        lr,
        step_per_epoch=None,
    ):
        if isinstance(model_or_type, type):
            model = model_or_type(self.config)
        else:
            model = model_or_type

        if step_per_epoch is None:
            step_per_epoch = len(self.train_dl)
        # print(model.transformer)
        # torch.distributed.init_process_group(backend='nccl')
        #
        # model = FullyShardedDataParallel(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=(step_per_epoch * self.config.num_epochs),
        )

        return model, optimizer, lr_scheduler


@dataclass
class dd:
    a: int


if __name__ == "__main__":
    constants.init()
    init()
    codec_config = PreprocessingConfig()

    os.environ["LD_LIBRARY_PATH"] = ""

    vae_config = TrainingConfig()

    # vae_config.train_batch_size = 11
    vae_config.train_batch_size = 7
    vae_config.scaling_factor = 0.18215
    # vae_config.layers_per_block = 4

    # torch.backends.cudnn.enabled = False
    vae_config.num_epochs = 5
    vae_config.vae_learning_rate = 1e-5
    vae_config.save_model_epochs = 2
    vae_config.save_image_epochs = 1
    vae_config.preprocessing.scale_method = "power"

    from core.data.codec import *

    trainer = Trainer(
        # vae_path='/root/autodl-tmp/remixer/training/artifacts/silver-frost-416/VQModel/model_epoch_0.pth',
        # vae_path="/root/autodl-tmp/remixer/training/artifacts/devout-firebrand-505/AutoencoderKL/model_epoch_0.pth",
        # generator_path='/root/autodl-tmp/remixer/training/artifacts/major-waterfall-21/VQGenerator/model_epoch_1.pth',
        config=vae_config,
        dataset_paths=["fma"],
        # codec=build_codec('audio', vae_config.preprocessing)
        codec=build_codec("mel", vae_config.preprocessing),
        vae="vae",
    )

    trainer.train_vae(True, None)
    # trainer.train_diffusion()
    # trainer.genera
