"""training framework."""
import gc

import diffusers
import math
import torch.utils.data.distributed
import wandb
from dataclass_wizard import asdict
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch import nn
from torch.functional import F
from tqdm.auto import tqdm

from core.data import *
from core.models.VAE import VQGenerator, build_VAE
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
    vae_path = ''
    generator_path = ''
    accelerate_path = ''

    def __init__(self, vae_path='', generator_path='', accelerate_path='', config=TrainingConfig(), dataset_paths=[],
                 codec=None):
        self.config = config
        self.vae_path = vae_path
        self.generator_path = generator_path
        self.accelerate_path = accelerate_path

        self.vae = load_model(build_VAE(config), file_name=None, file_path=vae_path, )
        [train_dl, _val_dl, _test_dl] = self.prepare_dataloaders(dataset_paths, codec)
        self.train_dl = train_dl

    def train_vae(self, is_train: bool, _epoch_callback=None, ):
        r"""
        train a VAE
        """
        model, optimizer, lr_scheduler = self.build_models(VQGenerator, len(self.train_dl),
                                                           lr=self.config.vae_learning_rate)

        self.generator = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        config = self.config

        bandwidth = 3.0  # 1.5, 3.0, 6.0
        # self.vae = MultiBandDiffusion.get_mbd_24khz(bw=bandwidth)
        # self.vae = load_model(build_VAE(config), file_path=self.vae_path)

        accelerator = build_accelerator(config, accelerate_state=self.accelerate_path, use_wandb=True,
                                        project_name="VAE")
        self.accelerator = accelerator

        # model AutoEncoderKL
        model = accelerator.prepare(self.vae)
        # model = model.module
        # inner = model.codec_model
        inner = model

        # enabled when FSDP
        optimizer, dataloader, lr_scheduler = accelerator.prepare(
            self.optimizer, self.train_dl, self.lr_scheduler
        )

        accelerator.log({'model_param_count': model_param(model)})

        if accelerator.is_main_process:
            run: wandb.sdk.wandb_run.Run = accelerator.get_tracker('wandb', unwrap=True)
            # print(run.__class__.__name__)
            run_name = run.name

        global_step = 0

        history_best_loss = best_loss = math.inf

        if is_train:
            inner.train()
        else:
            inner.eval()

        for epoch in range(config.num_epochs if is_train else 1):

            progress_bar = tqdm(total=len(dataloader), leave=True, position=0,
                                disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(dataloader):
                clean_images, (files, _ids) = batch
                if accelerator.is_local_main_process and step == 0 and epoch % config.save_image_epochs == 0:
                    # feature_to_image(clean_images[0, :].cpu().float().numpy(), PreprocessingConfig(), True)
                    feature_to_image(clean_images[0, 0, :].cpu().float().numpy(), PreprocessingConfig(), True)

                with accelerator.accumulate(model):
                    # inner = model.codec_model.to(device=accelerator.device)
                    inner: diffusers.AutoencoderKL = model.module
                    from diffusers.models.vae import DiagonalGaussianDistribution
                    x = clean_images
                    scaling_factor = inner.config.scaling_factor

                    # Encode
                    x = x * scaling_factor
                    # x = inner.get_condition(x, sample_rate=inner.codec_model.sample_rate)
                    x = inner.encode(x)
                    # x, _scale = inner.encode(x)

                    # size = x.size()
                    # posterior = DiagonalGaussianDistribution(x.float())
                    posterior: DiagonalGaussianDistribution = x.latent_dist

                    # Sample, Reparameterization
                    z = posterior.sample()

                    # Decode
                    z = 1 / scaling_factor * z
                    # z = z.int()
                    dec = inner.decode(z).sample
                    # dec = torch.clip(dec, -1, 1)
                    # dec = inner.decode(z)
                    # dec = inner.generate(z, size=size)

                    # reconstruction loss and regularization loss
                    kl_loss = 0.0001 * posterior.kl().mean()
                    reconstruction_loss = torch.mean(F.mse_loss(clean_images, dec))

                    loss = reconstruction_loss + kl_loss
                    if loss > 660:
                        image = feature_to_image(clean_images[0, :], config=config.preprocessing)
                        images = wandb.Image(image, caption="Spectrogram")
                        wandb.log({"spectrogram": images})
                        print(files)
                    if is_train:
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(inner.parameters(), 1.0)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                progress_bar.update(1)
                l = loss.detach().item()
                kl = kl_loss.detach().item()
                reconstruction_loss = reconstruction_loss.detach().item()
                logs = {
                    "loss": l,
                    "kl_loss": kl,
                    "reconstruction_loss": reconstruction_loss,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step
                }
                best_loss = min(best_loss, l)
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if is_train and accelerator.is_main_process:
                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    if best_loss < history_best_loss:
                        # history_best_loss = best_loss
                        artifact_name = f'model_epoch_{epoch}'
                        self.save_model([accelerator.unwrap_model(model) for model in [model]], run_name=run_name,
                                        name=artifact_name)

        accelerator.wait_for_everyone()

        return best_loss

    def train_diffusion(self):
        r"""
             Main training loops.

             Args:
                  class_inputs: (`bool`):
                     Whether class_inputs will be provided to the model. If not, styled audio will be passed to as encoder-hidden-state
             """

        vae = self.vae.to(
            device=accelerator.device)

        config = self.config

        accelerator = build_accelerator(self.config, accelerate_state=self.accelerate_path, use_wandb=True,
                                        project_name="Generator")

        # Prepare everything
        model = accelerator.prepare(self.generator)

        # enabled when FSDP
        optimizer, dataloader, lr_scheduler = accelerator.prepare(
            self.optimizer, self.dataloader, self.lr_scheduler
        )
        run_name = "random"
        if accelerator.is_main_process:
            run: wandb.sdk.wandb_run.Run = accelerator.get_tracker('wandb', unwrap=True)
            # print(run.__class__.__name__)
            run_name = run.name

        losses = []
        global_step = 0

        vae_dtype = torch.float32
        model_dtype = torch.float32
        vae = vae.to(dtype=vae_dtype)
        history_best_loss = math.inf
        vae.requires_grad_(False)
        if is_train:
            model.train()
        else:
            model.eval()

        for epoch in range(config.num_epochs if is_train else 1):
            progress_bar = tqdm(total=len(dataloader), leave=True, position=0,
                                disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in enumerate(dataloader):
                with accelerator.accumulate(model):
                    if class_inputs:
                        clean_images, class_ids = batch
                    else:
                        clean_images, styled_inputs = batch

                    bs = clean_images.shape[0]

                    # 1. Encode the clean images with VAE encoder
                    clean_images = vae.bert_config.scaling_factor * clean_images
                    clean_latents = vae.encode(clean_images.to(dtype=vae_dtype)).latent_dist.sample().to(
                        dtype=model_dtype)
                    # clean_latents = encode_output.latent_dist.mode()

                    # 2. Sample a noise to add to all latents
                    noise = torch.randn(clean_latents.shape, device=accelerator.device, dtype=torch.float32)

                    # 3. Sample a random timestep for each latent
                    timesteps = torch.randint(0, noise_scheduler.bert_config.num_train_timesteps, (bs,),
                                              dtype=torch.int64).int()

                    # 4. Add noise to the clean images according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(clean_latents, noise, timesteps).to(torch.float32)
                    noisy_latents = noise_scheduler.scale_model_input(noisy_latents, timesteps).to(torch.float32)

                    if not class_inputs:
                        # encode the styled_input
                        styled_inputs = vae.bert_config.scaling_factor * styled_inputs
                        styled_inputs = vae.encode(styled_inputs.to(dtype=vae_dtype)).latent_dist.sample().to(
                            dtype=model_dtype)

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
                        artifact_name = f'model_epoch_{epoch}'
                        self.save_model([accelerator.unwrap_model(model) for model in [model, noise_scheduler]],
                                        run_name=run_name,
                                        name=artifact_name)
                if (epoch + 1) % config.save_image_epochs == 0:
                    evaluate_model(config, vae, accelerator.unwrap_model(model), noise_scheduler)

            gc.collect()
            # epoch_callback(epoch, )
        # accelerator.wait_for_everyone()
        return model

    def save_model(self, models, run_name, name):
        base = f'artifacts/{run_name}/'
        os.makedirs(base, exist_ok=True)
        print(f"saving in {base}")

        def save(state_dict, model_name=None):
            config = self.config
            artifact = wandb.Artifact(f'{model_name}', type='model', metadata={'params': asdict(config)})
            base = f'artifacts/{run_name}/{model_name}'
            os.makedirs(base, exist_ok=True)

            # path = os.path.join(base, f'{name}.pth')
            # torch.save(state_dict, path)
            # artifact.add_file(path)

            path = os.path.join(base, f'{name}.pth')
            self.accelerator.save(state_dict, path)
            artifact.add_file(path)
            wandb.log_artifact(artifact)

        from torch.distributed.fsdp import FullyShardedDataParallel, StateDictType
        from torch.nn.parallel import DistributedDataParallel

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
            elif hasattr(model, 'save_pretrained'):
                model.save_pretrained(base)

        # artifact.wait()
        # for file in artifact.files():
        #     print(file.path())

    def evaluate_model(self):
        r"""
        Evaluate a model by:
        1. Test on the test dataset
        2. Generate some audios
        """
        pipeline = build_pipeline(vae=self.vae, generator=self.generator, scheduler_path=self.scheduler,
                                  config=self.config)

        config = self.config
        results = pipeline.generate_audio(None, config.latent_width, config, count=1)

        make_grid([image for _, _, image in results])
        # generate_from_genre()

    def prepare_dataloaders(self, dataset_paths, codec):
        from torch.utils.data import DataLoader, ConcatDataset
        tr_dss = []
        val_dss = []
        test_dss = []
        for dataset_path in dataset_paths:
            dataset_datas = read_remixer_dataset(dataset_path, all=True)
            tr_ds, val_ds, test_ds = build_remix_dataset(*dataset_datas, ratios=self.config.ratios,
                                                         codec=codec)
            tr_dss += [tr_ds]
            val_dss += [val_ds]
            test_dss += [test_ds]

        return [DataLoader(ConcatDataset(dss), self.config.train_batch_size, shuffle=True) for dss in
                [tr_dss, val_dss, test_dss]]

    def build_models(self, model_or_type, step_per_epoch, lr):
        if isinstance(model_or_type, type):
            model = model_or_type(self.config)
        else:
            model = model_or_type
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
class dd():
    a: int


if __name__ == "__main__":
    constants.init()
    init()
    codec_config = PreprocessingConfig()

    os.environ['LD_LIBRARY_PATH'] = ''

    vae_config = TrainingConfig()

    # vae_config.train_batch_size = 11
    vae_config.train_batch_size = 58

    # torch.backends.cudnn.enabled = False
    vae_config.num_epochs = 10
    vae_config.vae_learning_rate = 1e-5
    vae_config.save_model_epochs = 2
    vae_config.save_image_epochs = 2

    from core.data.codec import *

    trainer = Trainer(
        # vae_path='/root/autodl-tmp/remixer/training/artifacts/stoic-universe-125/AutoencoderKL/model_epoch_3.pth',
        vae_path='',
        generator_path='',
        config=vae_config,
        dataset_paths=['fma'],
        # codec=build_codec('audio', vae_config.preprocessing)
        codec=build_codec('mel', vae_config.preprocessing)
    )

    accelerate_path = ''
    trainer.train_vae(True, None)
