"""Base Dataset class."""

from typing import *

import torch
from PIL import Image
from torch.utils.data import Dataset

SequenceOrTensor = Union[Sequence, torch.Tensor]


def convert_strings_to_labels(strings: Sequence[str], mapping: Dict[str, int], length: int) -> torch.Tensor:
    """
    Convert sequence of N strings to a (N, length) ndarray, with each string wrapped with <S> and <E> tokens,
    and padded with the <P> token.
    """
    labels = torch.ones((len(strings), length), dtype=torch.long) * mapping["<P>"]
    for i, string in enumerate(strings):
        tokens = list(string)
        tokens = ["<S>", *tokens, "<E>"]
        for ii, token in enumerate(tokens):
            labels[i, ii] = mapping[token]
    return labels


def resize_image(image: Image.Image, scale_factor: int) -> Image.Image:
    """Resize image by scale factor."""
    if scale_factor == 1:
        return image
    return image.resize((image.width // scale_factor, image.height // scale_factor), resample=Image.BILINEAR)


def feature_to_audio(f, name, codec):
    audio_reconstructed = codec.decode(f)

    import soundfile as sf
    # Save the reconstructed audio
    path = f'{name}.wav'
    sf.write(path, audio_reconstructed, samplerate=config.sr)
    print(f"audio file saved to: {path}")
    return audio_reconstructed


def split_dataset(base_dataset: Dataset, ratios: List[float]):
    """
    Split input base_dataset into 3  datasets
    """
    count = len(base_dataset)
    split_a_size = int(ratios[0] * count)
    split_b_size = int(ratios[1] * count)
    split_c_size = count - split_a_size - split_b_size
    return torch.utils.data.random_split(  # type: ignore
        base_dataset, [split_a_size, split_b_size, split_c_size]
    )
