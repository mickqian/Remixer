"""Utility functions for remixer module."""
import collections
import os
from dataclasses import dataclass, field
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
    except NotImplementedError:
        pass

    return None


def load_model(model_object, file_path=None):
    import os

    name = model_object.__class__.__name__

    # load from wandb, whatever
    if file_path is None:
        import wandb

        name = f"{name}:latest"
        wandb.login()
        artifact: wandb.Artifact = wandb.use_artifact(name)
        path = artifact.get_path(file_name)
        model_path = str(path.path)
        if not os.path.exists(model_path):
            path.download()
    else:
        model_path = file_path

    if model_path:
        print(f"loading {model_path}")
        state_dict = torch.load(model_path)
        # consume_prefix_in_state_dict_if_present(state_dict)
        model_object.load_state_dict(state_dict)

    print(f"{model_object.__class__.__name__} params count : {model_param(model_object)}")
    return model_object


def serve_request(
    config, genre: str, content: Union[int, np.array], style: Union[int, np.array], progression: str, pipeline
):
    if progression:
        _chords = progression.split(",")

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


def get_sample_file_path(genre, _config, epoch=None):
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
    import random

    data = random.sample(data, 5)
    s = str(hash(tuple(data)))

    # Return the hexadecimal digest of the hash
    return s[1:6]


@dataclass
class PreprocessingConfig:
    scale_method: str = "min_max"
    sr: int = 22050
    mono: bool = False
    n_fft: int = 2048
    n_mels: int = 128

    # size of mel-spectrogram
    input_height: int = n_mels
    input_width: int = 416
    hop_length: int = 32
    # time series count
    T: int = 10
    clipped_frames: int = 1290
    clipped_samples: int = sr * T


def get_inception_features(images: np.ndarray, model):
    model.eval()
    with torch.no_grad():
        features = model(torch.tensor(images).repeat(1, 3, 1, 1).to(model.device))
    return features.cpu().numpy()


def calculate_fid(real_features, fake_features):
    from scipy.linalg import sqrtm

    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm((sigma1 @ sigma2).cpu().numpy())

    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def calculate_fid_of_images(original, reconstruct, model):
    # from skimage.color import gray2rgb
    real_features = get_inception_features(original, model)
    reconstruct_features = get_inception_features(reconstruct, model)

    fid = calculate_fid(real_features, reconstruct_features)

    return fid


@dataclass
class TrainingConfig:
    train_batch_size: int = 20
    eval_batch_size: int = 2  # how many images to sample during evaluation
    num_epochs: int = 1
    save_image_epochs: int = 5
    save_model_epochs: int = 1

    ## VAE
    scaling_factor: int = 0.18215
    vae_in_channels: int = 1
    act_fn: str = "silu"
    # "mish"
    # "gelu"
    out_channels: int = 1
    num_vq_embeddings: int = 512
    # 1
    vae_out_channels: int = 1
    layers_per_block: int = 6
    vq_layers_per_block: int = 5
    latent_channels: int = 12
    vq_latent_channels: int = 24
    vae_learning_rate: int = 1e-4
    down_block_types: List[str] = field(
        default_factory=lambda: [
            # "DownEncoderBlock2D",
            # "DownEncoderBlock2D",
            # "AttnDownBlock2D",
            # "DownEncoderBlock2D",
            "AttnDownEncoderBlock2D",
            # "DownEncoderBlock2D",
            # "DownEncoderBlock2D",
            "AttnDownEncoderBlock2D",
            # "DownEncoderBlock2D",
            # "SimpleCrossAttnDownBlock2D",
            # "AttnSkipDownBlock2D",
            # "KCrossAttnDownBlock2D",
            # "CrossAttnDownBlock2D",
        ]
    )
    up_block_types: List[str] = field(
        default_factory=lambda: [
            "AttnUpDecoderBlock2D",
            # "UpDecoderBlock2D",
            # "UpDecoderBlock2D",
            "AttnUpDecoderBlock2D",
            # "UpDecoderBlock2D",
            # "AttnUpBlock2D",
            # "UpDecoderBlock2D",
            # "UpDecoderBlock2D",
            # "SimpleCrossAttnUpBlock2D"
            # "AttnSkipUpBlock2D",
            # "KCrossAttnUpBlock2D",
            # "CrossAttnUpBlock2D",
        ]
    )

    vq_down_block_types: List[str] = field(
        default_factory=lambda: [
            # "DownEncoderBlock2D",
            # "DownEncoderBlock2D",
            # "AttnDownBlock2D",
            # "DownEncoderBlock2D",
            "AttnDownEncoderBlock2D",
            "DownEncoderBlock2D",
            # "DownEncoderBlock2D",
            "AttnDownEncoderBlock2D",
            "DownEncoderBlock2D",
            # "SimpleCrossAttnDownBlock2D",
            # "AttnSkipDownBlock2D",
            # "KCrossAttnDownBlock2D",
            # "CrossAttnDownBlock2D",
        ]
    )
    vq_up_block_types: List[str] = field(
        default_factory=lambda: [
            "UpDecoderBlock2D",
            "AttnUpDecoderBlock2D",
            # "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "AttnUpDecoderBlock2D",
            # "UpDecoderBlock2D",
            # "AttnUpBlock2D",
            # "UpDecoderBlock2D",
            # "UpDecoderBlock2D",
            # "SimpleCrossAttnUpBlock2D"
            # "AttnSkipUpBlock2D",
            # "KCrossAttnUpBlock2D",
            # "CrossAttnUpBlock2D",
        ]
    )
    block_out_channels: List[str] = field(
        default_factory=lambda: [
            # 32,
            # 32,
            32,
            # 32,
            # 128,
            # 128,
            128,
            # 256,
        ]
    )
    vq_block_out_channels: List[str] = field(
        default_factory=lambda: [
            # 32,
            32,
            32,
            # 32,
            # 128,
            128,
            128,
            # 256,
        ]
    )

    ## Transformer
    num_attention_heads: int = 16
    attention_head_dim: int = 128
    norm_nums_groups: int = 2
    generator_num_layers: int = 4
    generator_learning_rate: float = 5e-4
    sample_size: tuple[int] = (1, 26)

    accelerator: str = "cuda"
    num_workers: int = get_workers()
    on_gpu: bool = bool(accelerator == "cuda")
    gradient_accumulation_steps: int = 1
    guidance_scale: float = 1.0
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    lr_warmup_steps: int = 500
    num_inference_steps: int = 999

    mixed_precision: str = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir: str = ROOT_DIR / "training" / "out"  # the model namy locally and on the HF Hub
    push_to_hub: bool = True  # whether to upload the saved model to the HF Hub
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True  # overwrite the old model when re-running the notebook
    seed: int = SEED
    ratios: List[str] = field(default_factory=lambda: [0.6, 0.2, 0.2])
    device: torch.device = torch.device(accelerator)

    genres: List[str] = field(default_factory=lambda: constants.GENRES)
    codec = None
    genre2Id = {genre: id for id, genre in enumerate(set(genres.default_factory()))}
