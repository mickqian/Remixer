"""Utility functions for remixer module."""
import collections
import os
from dataclasses import dataclass
from typing import *
from urllib.request import urlretrieve

import librosa
import torch
from tqdm import tqdm

import constants
from constants import *

counter_dict = collections.defaultdict(int)


@dataclass
class PreprocessingConfig:
    sr: Optional[float] = 22050
    mono = False
    n_fft = 2048
    input_height = 16
    input_width = 600
    n_mels = 128
    hop_length = 32
    # time series count
    T = 30


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


@dataclass
class TrainingConfig:

    train_batch_size = 20
    norm_nums_groups = 1
    eval_batch_size = 2  # how many images to sample during evaluation
    num_epochs = 1
    vae_in_channels = 1
    num_attention_heads = 2
    accelerator = 'cuda'
    num_workers = get_workers()
    on_gpu = bool(accelerator == "cuda")
    gradient_accumulation_steps = 1
    guidance_scale = 1.0
    latent_channels = 4
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    attention_head_dim = 2
    learning_rate = 1e-4

    latent_width = 646
    lr_warmup_steps = 500
    num_inference_steps = 1000
    save_image_epochs = 10
    save_model_epochs = 1
    mixed_precision = 'no'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = ROOT_DIR / "training" / "out"  # the model namy locally and on the HF Hub
    push_to_hub = True  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = SEED
    ratios = [0.6, 0.2, 0.2]
    device = torch.device(accelerator)

    genres = constants.GENRES

    genre2Id = {genre: id for id, genre in enumerate(set(genres))}


def load_model(model_object, file_name="model_epoch_0.pth", file_path=None):
    import os
    name = model_object.__class__.__name__
    # load from wandb, whatever
    if not file_name and not file_path:
        import wandb
        name = f'{name}:latest'
        artifact: wandb.Artifact = wandb.use_artifact(name)
        path = artifact.get_path(file_name)
        model_path = str(path.path)
        if not os.path.exists(model_path):
            path.download()
    elif file_name and not file_path:
        base = ROOT_DIR / "training" / "artifacts" / f"{name}:v1" / file_name
        model_path = str(base)
    else:
        model_path = file_path

    print(f'loading {model_path}')
    model_object.load_state_dict(torch.load(model_path))
    return model_object


def serve_request(genre: str, content: Union[int, np.array], style: Union[int, np.array], pipeline):
    if genre and style:
        print("At most one style can be chosen. Currently two. Try removing one")
        return None
    if genre and content:
        print("Both genre and content is chosen. Set only `genre` for generating, both for remixing")
        return None

    if not content:
        # perform random generation from genre
        output, audio, image = pipeline.generate_audio(genre)
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


def feature_to_image(log_S, config: PreprocessingConfig):
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(log_S, sr=config.sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title('Mel Spectrogram')
    ax.colorbar(format='%+02.0f dB')

    # 将Matplotlib绘制的图像保存到内存中的字节缓冲区
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    # 使用PIL从内存中的字节缓冲区创建Image对象
    img = Image.open(buf)
    buf.close()
    return img


def feature_to_audio(f, name, config: PreprocessingConfig):
    from .data.codes import mel_codec
    audio_reconstructed = mel_codec.decode(f)

    import soundfile as sf
    # Save the reconstructed audio
    path = f'{name}.wav'
    sf.write_wav(path, audio_reconstructed, sr=config.sr)
    print(f"audio file saved to: {path}")
    return audio_reconstructed


def get_sample_file_path(genre, config, epoch=None):
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    # Save the images
    if epoch:
        file_name = f"{test_dir}/{genre}_{epoch:04d}.png"
    else:
        counter_dict[genre] += 1
        cnt = counter_dict[genre]
        file_name = f"{test_dir}/{genre}_{cnt}.png"
    return file_name
