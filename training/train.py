"""training framework."""
import gc

import diffusers
import math
import torch.utils.data.distributed
import wandb
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.schedulers import DPMSolverMultistepScheduler
from torch import nn
from torch.functional import F
from tqdm.auto import tqdm

from core.data import *
from core.models.VAE import VQGenerator, build_VAE
from core.utils import *
from test_model import *


def train_vae(config: TrainingConfig, is_train: bool, _epoch_callback=None, accelerate_path=None, model_path=None):
    r"""
    train a VAE
    """
    accelerator = build_accelerator(config)

    model = build_VAE(config)

    if model_path:
        model = load_model(model, file_path=model_path)
    [train_dl, _val_dl, _test_dl] = prepare_dataloaders(['gtzan'], config.ratios, config.train_batch_size)

    model, optimizer, lr_scheduler = prepare_models(model, config, len(train_dl))

    # model AutoEncoderKL
    model = accelerator.prepare(model)

    # enabled when FSDP
    optimizer, dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dl, lr_scheduler
    )

    run: wandb.sdk.wandb_run.Run = accelerator.get_tracker('wandb').tracker()
    run_name = run.name
    # run.config_static.

    # logger = wandb.sdk.wandb_run.logger;
    # logger.co

    losses = []
    global_step = 0

    history_best_loss = math.inf

    if is_train:
        model.train()
    else:
        model.eval()

    for epoch in range(config.num_epochs if is_train else 1):

        progress_bar = tqdm(total=len(dataloader), leave=True, position=0,
                            disable=not accelerator.is_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(dataloader):
            clean_images, _ids = batch

            with accelerator.accumulate(model):
                from torch.nn.parallel import DistributedDataParallel
                if isinstance(model, DistributedDataParallel):
                    model = model.module
                from diffusers.models.vae import DiagonalGaussianDistribution
                x = clean_images

                # Encode
                posterior: DiagonalGaussianDistribution = model.encode(x).latent_dist

                # Reparameterization
                z = posterior.sample()

                # Decode
                dec = model.decode(z).sample

                # reconstruction loss and regularization loss
                kl_loss = posterior.kl().mean()
                reconstruction_loss = torch.mean(F.mse_loss(clean_images, dec))
                loss = reconstruction_loss + kl_loss

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
                artifact_name = f'{run_name}_model_epoch_{epoch}'
                best_loss = min(losses, default=history_best_loss)
                if best_loss < history_best_loss:
                    history_best_loss = best_loss
                    save_model(model, accelerator, config, artifact_name)
    accelerator.wait_for_everyone()

    def model_param(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    num_params = model_param(model)
    accelerator.log({'model_param_count': num_params})
    print(f'{num_params=}')
    return losses


def main_loop(config: TrainingConfig, accelerator: Accelerator, dataloader, model: nn.Module, vae,
              noise_scheduler: DPMSolverMultistepScheduler,
              optimizer,
              lr_scheduler,
              is_train: bool, class_inputs=False):
    r"""
    Main training loops.

    Args:
        class_inputs: (`bool`):
            Whether class_inputs will be provided to the model. If not, styled audio will be passed to as encoder-hidden-state
    """

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model = accelerator.prepare([model])

    # enabled when FSDP
    optimizer, dataloader, lr_scheduler = accelerator.prepare(
        optimizer, dataloader, lr_scheduler
    )

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
                clean_latents = vae.encode(clean_images.to(dtype=vae_dtype)).latent_dist.sample().to(
                    dtype=model_dtype)
                # clean_latents = encode_output.latent_dist.mode()
                clean_latents = vae.config.scaling_factor * clean_latents

                # 2. Sample a noise to add to all latents
                noise = torch.randn(clean_latents.shape, device=accelerator.device, dtype=torch.float32)

                # 3. Sample a random timestep for each latent
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), dtype=torch.int64)

                # 4. Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                # noisy_latents = noise_scheduler.add_noise(clean_latents, noise, timesteps).to(torch.float32)

                # # encode, Reparameterization trick
                # encoder_output = vae.encode(inputs).latent_dist
                # inputs = encoder_output.sample()
                if not class_inputs:
                    # encode the styled_input
                    encode_output = vae.encode(styled_inputs).latent_dist
                    styled_inputs = encode_output.sample().to(dtype=torch.float32)
                    styled_inputs = vae.config.scaling_factor * styled_inputs

                # Finally, scale the model input
                # noisy_latents = noise_scheduler.scale_model_input(noisy_latents, timesteps).to(torch.float32)

                # predict the noise residual
                if class_inputs:
                    noise_pred = model(clean_latents, timesteps, class_ids)
                else:
                    noise_pred = model(clean_latents, timesteps, class_ids=None, styled_inputs=styled_inputs)

                # predict the L1-norm loss of noise and predicted_noise
                noise_loss = F.mse_loss(noise, noise_pred)

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
            # unwraped = accelerator.unwrap_model(model)
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                # evaluate(config, epoch, model)
                # vae.to_onnx()
                # pipeline = vae.build_generation_pipeline()
                # pipeline.save_pretrained(config.output_dir)
                best_loss = min(losses, default=math.inf)
                if best_loss < history_best_loss:
                    history_best_loss = best_loss
                    artifact_name = f'model_epoch_{epoch}'
                    save_model(model, accelerator=accelerator, config=config, name=artifact_name)
        gc.collect()
        # epoch_callback(epoch, )
    # accelerator.wait_for_everyone()
    return losses


def train_diffusion(config: TrainingConfig, ):
    [train_dl, _val_dl, _test_dl] = prepare_dataloaders(['gtzan'], config.ratios, config.train_batch_size)

    accelerator = build_accelerator(config)

    model, optimizer, lr_scheduler = prepare_models(VQGenerator, config, len(train_dl))
    noise_scheduler = model.scheduler
    vae = load_model(build_VAE(config),
                     file_path='/root/autodl-tmp/remixer/training/artifacts/AutoencoderKL:v1/model_epoch_0.pth').to(
        device=accelerator.device)

    losses = main_loop(config, accelerator, train_dl, model, vae, noise_scheduler, optimizer, lr_scheduler,
                       is_train=True,
                       class_inputs=True)

    return model


def save_model_if_best(current_loss, best_loss, model, config, accelerator: Accelerator, name=''):
    if current_loss < best_loss:
        save_model(model, accelerator, config, name)
    return best_loss


def save_model(model, accelerator, config, name, version='v1', ):
    # save in the same fashion as wandb
    def save(state_dict, artifact_name=None):
        # if not artifact_name:
        #     artifact_name = str(model.__class__.__name__)
        artifact = wandb.Artifact(artifact_name, type='model', metadata={'params': config})
        base = f'artifacts/{artifact_name}'
        os.makedirs(base, exist_ok=True)

        torch_ckpt = f'{name}.pth'
        path = os.path.join(base, torch_ckpt)
        torch.save(state_dict, path)
        artifact.add_file(path)

        accelerator_ckpt = f'{name}.acc_ckpt'
        path = os.path.join(base, accelerator_ckpt)
        accelerator.save(state_dict, path)
        artifact.add_file(path)
        wandb.log_artifact(artifact)

    from torch.distributed.fsdp import FullyShardedDataParallel, StateDictType

    if isinstance(model, FullyShardedDataParallel):
        with FullyShardedDataParallel.state_dict_type(
                model,
                StateDictType.LOCAL_STATE_DICT,  # or any other StateDictType
        ):
            save(state_dict=model.state_dict(), artifact_name=str(model.module.__class__.__name__))
    else:
        save(state_dict=model.state_dict(), artifact_name=str(model.__class__.__name__))


def evaluate_model(config: TrainingConfig, dataloader, accelerator, model, noise_scheduler, optimizer, lr_scheduler):
    # Evaluate a model by:
    # 1. Test on the test dataset
    # 2. Generate some audios
    _losses = main_loop(config, accelerator=accelerator, dataloader=dataloader, model=model,
                        noise_scheduler=noise_scheduler,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        class_inputs=True, is_train=False)

    # wandb.log({'eval_avg_loss': sum(losses) / len(losses)})

    # generate_from_genre()


def prepare_dataloaders(dataset_paths, ratios, batch_size, ):
    tr_dss = []
    val_dss = []
    test_dss = []
    for dataset in dataset_paths:
        tr_ds, val_ds, test_ds = build_remix_dataset(*read_dataset(dataset), ratios=ratios)
        tr_dss += [tr_ds]
        val_dss += [val_ds]
        test_dss += [test_ds]

    return [DataLoader(ConcatDataset(dss), batch_size, ) for dss in [tr_dss, val_dss, test_dss]]


def prepare_models(model_or_type, config: TrainingConfig, train_sample_count):
    if isinstance(model_or_type, Type):
        model = model_or_type(config)
    else:
        model = model_or_type
    # torch.distributed.init_process_group(backend='nccl')
    #
    # model = FullyShardedDataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(train_sample_count * config.num_epochs),
    )

    return model, optimizer, lr_scheduler


def init(init_cuda=True):
    wandb.login()
    wandb.apis._disable_ssl()
    os.environ['WANDB_HTTP_TIMEOUT'] = "5"
    if init_cuda:
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.synchronize()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    constants.init()
    init()
    diffusers.training_utils.set_seed(SEED)
    train_diffusion(TrainingConfig())
