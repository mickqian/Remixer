"""Base Dataset class."""
from typing import Any, Callable, Dict, Sequence, Tuple, Union
import numpy as np
import sklearn
import torch
from torch.utils.data import Dataset, Subset
from constants import *
from torch.utils.data import Dataset
from PIL import Image
import librosa
import librosa.feature
from sklearn.pipeline import make_pipeline
import torch
from typing import *

import functools
from dataclasses import dataclass
import core.utils
from core import utils
from core.utils import PreprocessingConfig

SequenceOrTensor = Union[Sequence, torch.Tensor]


class RemixerDataset(torch.utils.data.Dataset):
    """Base Dataset class that simply processes data and targets through optional transforms.
    Parameters
    ----------
    data
        commonly these are torch tensors, numpy arrays, or PIL Images
    targets
        commonly these are torch tensors or numpy arrays
    transform
        function that takes a datum and returns the same
    target_transform
        function that takes a target and returns the same
    """

    def __init__(
            self,
            data: SequenceOrTensor,
            targets: SequenceOrTensor,
            transform: Callable = None,
            target_transform: Callable = None,
            config: PreprocessingConfig = PreprocessingConfig(),
    ) -> None:
        super().__init__()

        if len(data) != len(targets):
            raise ValueError("Data and targets must be of equal length")
        self.data = data
        self.transform = transform
        self.target_transform = target_transform
        self.targets = targets
        from core.data import codes
        self.config = config
        self.codec = codes.mel_codec

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Return a datum and its target, after processing by transforms.

        Parameters
        ----------
        index

        Returns
        -------
        (datum, target)
        """
        datum, target = self.data[index], self.targets[index]

        if self.codec is not None:
            datum = self.codec.encode(datum)

        if self.target_transform is not None:
            target = target

        return datum, target


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


def split_dataset(base_dataset: Dataset, ratios: List[float]) -> List[Subset[RemixerDataset]]:
    """
    Split input base_dataset into 3  datasets
    """
    count = len(base_dataset)
    split_a_size = int(ratios[0] * count)
    split_b_size = int((ratios[1] - ratios[0]) * count)
    split_c_size = int(count - split_a_size - split_b_size)
    return torch.utils.data.random_split(  # type: ignore
        base_dataset, [split_a_size, split_b_size, split_c_size], generator=torch.Generator().manual_seed(SEED)
    )


def resize_image(image: Image.Image, scale_factor: int) -> Image.Image:
    """Resize image by scale factor."""
    if scale_factor == 1:
        return image
    return image.resize((image.width // scale_factor, image.height // scale_factor), resample=Image.BILINEAR)
