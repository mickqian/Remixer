"""Experiment-running framework."""
import os

import accelerate

from constants import GENRES, WANDB_PROJECT_NAME
from core.utils import serve_request, TrainingConfig


def make_grid(images, rows):
    cols = (len(images) // rows) + 1
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, S in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def build_accelerator(config: TrainingConfig, accelerate_state=None, use_wandb=True):
    if accelerate_state:
        accelerator = accelerate.Accelerator.load_state(accelerate_state)
    elif use_wandb:
        accelerator = accelerate.Accelerator(
            # split_batches=True,
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="wandb",
            # logging_dir=os.path.join(config.output_dir, "logs")
        )
    else:
        accelerator = accelerate.Accelerator(
            # split_batches=True,
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            # logging_dir=os.path.join(config.output_dir, "logs")
        )
    if accelerator.is_local_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if use_wandb:
            import wandb
            wandb.apis._disable_ssl()
            # wandb.login(relogin=True)
            accelerator.init_trackers(WANDB_PROJECT_NAME, config={'TrainingConfig': config})

    # os.environ["WANDB_DISABLED"] = "true"
    # wandb.init()

    return accelerator


def generate_from_genre(genre=None, vae_path=None, generator_path=None):
    """
    Test a GPT2-decoder model on the PICa test dataset.
    """
    import random
    if not genre:
        genre = random.choice(GENRES)

    # if not model:
    #     # default to latest model
    #     # artifact_name = "vae:latest"
    #     artifact_name = "model_epoch_0"
    #     vae = load_model(build_VAE(TrainingConfig()), file_path=vae_path)
    #     generator = load_model(VQGenerator(), file_path=generator_path)
    #     generator.vae = vae
    #     generator.eval()
    #     generator.to(device=torch.device("cuda"))

    accelerator = build_accelerator(config, False)

    config = TrainingConfig()
    pipeline = build_pipeline(vae_path, generator_path, config)
    generator, vae = accelerator.prepare(pipeline.generator, pipeline.vae)
    pipeline.generator = generator
    pipeline.vae = vae

    output, _audio, img = serve_request(genre=genre, content=None, style=None, pipeline=pipeline)

    return output, img
