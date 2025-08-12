"""Experiment-running framework."""

from dataclasses import asdict

import accelerate
from accelerate.utils import DistributedDataParallelKwargs
from torch.distributed.fsdp._init_utils import *
from torch.distributed.fsdp.wrap import always_wrap_policy

from constants import GENRES, WANDB_PROJECT_NAME, OUTPUT_DIR
from core.models.VAE import build_pipeline
from core.utils import serve_request, TrainingConfig


def make_grid(images, cols=4):
    from PIL import Image

    rows = (len(images) // cols) + 1
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    grid.show()
    return grid


def build_accelerator(config, accelerate_state=None, use_wandb=True, project_name=""):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # plugin = accelerate.FullyShardedDataParallelPlugin(sharding_strategy=ShardingStrategy.FULL_SHARD,
    #                                                    backward_prefetch=BackwardPrefetch.BACKWARD_POST,
    #                                                    state_dict_config=StateDictConfig(offload_to_cpu=True),
    #                                                    cpu_offload=CPUOffload(True),
    #                                                    state_dict_type=StateDictType.SHARDED_STATE_DICT,
    #                                                    use_orig_params=False, auto_wrap_policy=always_wrap_policy)
    plugin = accelerate.FullyShardedDataParallelPlugin()
    if use_wandb:
        accelerator = accelerate.Accelerator(
            # split_batches=True,
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="wandb",
            # log_with="wandb",
            # fsdp_plugin=plugin,
            kwargs_handlers=[kwargs],
            # logging_dir=os.path.join(OUTPUT_DIR, "logs")
        )
    else:
        accelerator = accelerate.Accelerator(
            # split_batches=True,
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            # fsdp_plugin=plugin,
            kwargs_handlers=[kwargs],
            # logging_dir=os.path.join(OUTPUT_DIR, "logs")
        )
    if accelerate_state:
        accelerator.load_state(accelerate_state)
    if accelerator.is_local_main_process:
        if OUTPUT_DIR is not None:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        if use_wandb:
            import wandb

            wandb.apis._disable_ssl()
            # wandb.login(relogin=True)
            accelerator.init_trackers(
                project_name if project_name else WANDB_PROJECT_NAME, config={"TrainingConfig": asdict(config)}
            )
        # os.environ["WANDB_DISABLED"] = "true"
        # wandb.init()
    accelerator.wait_for_everyone()
    return accelerator


def generate_from_genre(config, genre=None, vae_path=None, generator_path=None, scheduler_path=None):
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

    accelerator = build_accelerator(config, accelerate_state=None, use_wandb=False)

    pipeline = build_pipeline(vae_path, generator_path, scheduler_path, config)
    generator, vae = accelerator.prepare(pipeline.generator, pipeline.vae)
    pipeline.generator = generator
    pipeline.vae = vae

    output, _audio, img = serve_request(config, genre=genre, content=None, style=None, pipeline=pipeline)

    return output, img


if __name__ == "__main__":
    # wandb.init()
    config = TrainingConfig()
    # config.num_inference_steps = 399
    config.num_inference_steps = 999

    vae_path = "/root/autodl-tmp/remixer/training/artifacts/feasible-glitter-11/AutoencoderKL/model_epoch_9.pth"
    generator_path = "/root/autodl-tmp/remixer/training/artifacts/likely-fog-6/VQGenerator/model_epoch_1.pth"
    scheduler_path = "/root/autodl-tmp/remixer/training/artifacts/likely-fog-6/scheduler_config.json"

    # vae_path = '/root/autodl-tmp/remixer/training/artifacts/kind-spaceship-380/AutoencoderKL/kind-spaceship-380_models_epoch_9.pth'
    config.num_inference_steps = 999
    datas = [
        generate_from_genre(
            config, genre=None, vae_path=vae_path, generator_path=generator_path, scheduler_path=scheduler_path
        )
        for _ in range(3)
    ]

    make_grid([img for output, img in datas])
