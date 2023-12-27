"""Base Dataset class."""

from typing import *

import torch
from PIL import Image
from torch.utils.data import Dataset

SequenceOrTensor = Union[Sequence, torch.Tensor]


def resize_image(image: Image.Image, scale_factor: int) -> Image.Image:
    """Resize image by scale factor."""
    if scale_factor == 1:
        return image
    return image.resize((image.width // scale_factor, image.height // scale_factor), resample=Image.BILINEAR)


def split_dataset(base_dataset: Dataset, ratios: List[float]):
    """
    Split input base_dataset into 3  datasets
    """
    count = len(base_dataset)
    split_a_size = int(ratios[0] * count)
    split_b_size = int(ratios[1] * count)
    split_c_size = count - split_a_size - split_b_size
    return torch.utils.data.random_split(base_dataset, [split_a_size, split_b_size, split_c_size])  # type: ignore
