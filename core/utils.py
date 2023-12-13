"""Utility functions for remixer module."""
import collections
import os
from dataclasses import dataclass
from typing import *
from urllib.request import urlretrieve

import torch
from tqdm import tqdm

import constants
from constants import *

counter_dict = collections.defaultdict(int)


def model_param(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_workers():
    import os
    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0))
        except NotImplementedError:
            pass

        # On PyPy and possibly other platforms, os.sched_getaffinity does not exist
        # or raises NotImplementedError, let's try with the psutil if installed.
    try:
        import psutil

        p = psutil.Process()
        if hasattr(p, "cpu_affinity"):
            return len(p.cpu_affinity())
    except  NotImplementedError:
        pass

    return None


def load_model(model_object, file_name="model_epoch_0.pth", file_path=None):
    import os
    name = model_object.__class__.__name__

    # load from wandb, whatever
    if not file_name and file_path is None:
        import wandb
        name = f'{name}:latest'
        wandb.login()
        artifact: wandb.Artifact = wandb.use_artifact(name)
        path = artifact.get_path(file_name)
        model_path = str(path.path)
        if not os.path.exists(model_path):
            path.download()
    elif file_name and file_path is None:
        base = ROOT_DIR / "training" / "artifacts" / f"{name}:v1" / file_name
        model_path = str(base)
    else:
        model_path = file_path

    if model_path:
        print(f'loading {model_path}')
        state_dict = torch.load(model_path)
        # consume_prefix_in_state_dict_if_present(state_dict)
        model_object.load_state_dict(state_dict)
    return model_object


def serve_request(config, genre: str, content: Union[int, np.array], style: Union[int, np.array], progression: str,
                  pipeline):
    if progression:
        chords = progression.split(',')

    if genre and style:
        print("At most one style can be chosen. Currently two. Try removing one")
        return None
    if genre and content:
        print("Both genre and content is chosen. Set only `genre` for generating, both for remixing")
        return None

    if not content:
        # perform random generation from genre
        output, audio, image = pipeline.generate_audio([genre], config=config)[0]
        return output, audio, image

    if style and content:
        # perform style transfer(remix)
        pass
    return None


class TqdmUpTo(tqdm):
    """From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py"""

    def update_to(self, blocks=1, bsize=1, tsize=None):
        """
        Parameters
        ----------
        blocks: int, optional
            Number of blocks transferred so far [default: 1].
        bsize: int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(blocks * bsize - self.n)  # will also set self.n = b * bsize


def download_url(url, filename=None):
    """Download a file from url to filename, with a progress bar."""
    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:
        path, _ = urlretrieve(url, filename, reporthook=t.update_to, data=None)  # noqa: S310
        return path


def get_sample_file_path(genre, config, epoch=None):
    test_dir = os.path.join(OUTPUT_DIR, "samples")
    os.makedirs(test_dir, exist_ok=True)
    if epoch:
        file_name = f"{test_dir}/{genre}_{epoch:04d}"
    else:
        cnt = len(os.listdir(test_dir))
        file_name = f"{test_dir}/{genre}_{cnt}"
    return file_name


# Function to create a hash for a given number
def get_file_name_from(data):
    if isinstance(data, np.ndarray):
        data = data.squeeze()
        if len(data.shape) > 1:
            data = data[:, -1]
        if data.shape[0] > 5:
            data = data[:5]
        data = data.tolist()

    data = data[:5]
    s = str(hash(tuple(data)))

    # Return the hexadecimal digest of the hash
    return s[1:6]


@dataclass
class PreprocessingConfig:
    sr = 22050
    mono = False
    n_fft = 2048
    input_height = 16
    input_width = 600
    n_mels = 128
    hop_length = 32
    # time series count
    T = 10
    clipped_frames = 1290
    clipped_samples = sr * T


@dataclass
class TrainingConfig:
    train_batch_size = 20
    eval_batch_size = 2  # how many images to sample during evaluation
    num_epochs = 1

    ## VAE
    scaling_factor = 0.18215
    vae_in_channels = 1
    # 1
    vae_out_channels = 1
    layers_per_block = 4
    latent_channels = 8
    vae_learning_rate = 1e-4
    down_block_types = [
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    ]
    up_block_types = [
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D"
    ]
    block_out_channels = [
        32,
        64,
        128,
        256,
        512,
    ]

    ## Transformer
    num_attention_heads = 16
    attention_head_dim = 128
    latent_width = 646
    norm_nums_groups = 2
    num_layers = 2
    generator_learning_rate = 1e-4

    accelerator = 'cuda'
    num_workers = get_workers()
    on_gpu = bool(accelerator == "cuda")
    gradient_accumulation_steps = 1
    guidance_scale = 1.0
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    lr_warmup_steps = 500
    num_inference_steps = 999
    save_image_epochs = 10
    save_model_epochs = 3
    mixed_precision = 'no'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = ROOT_DIR / "training" / "out"  # the model namy locally and on the HF Hub
    push_to_hub = True  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = SEED
    ratios = [0.6, 0.2, 0.2]
    device = torch.device(accelerator)

    genres = constants.GENRES
    codec = None
    genre2Id = {genre: id for id, genre in enumerate(set(genres))}
