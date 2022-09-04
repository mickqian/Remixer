"""Base DataModule class."""
import argparse
import sys
import os
import psutil
from pathlib import Path
from typing import Collection, Dict, Optional, Tuple, Union
import torch
from torch.utils.data import ConcatDataset, DataLoader
from core.data.util import RemixerDataset
import psutil
import numpy as np
from .util import split_dataset
from ..utils import TrainingConfig, DATASET_PATH
from constants import *


def load_and_print_info(data_module_class) -> None:
    """Load Dataset and print info."""
    parser = argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser)
    args = parser.parse_args()
    dataset = data_module_class(args)
    dataset.prepare_data()
    dataset.setup()
    print(dataset)


def read_dataset(root=''):
    files = []
    ids = []

    for genre_dir in os.listdir(DATASET_PATH / root):
        genre = str(genre_dir)
        if genre not in genre2Id:
            genre2Id[genre] = len(genre2Id)
        genre_id = genre2Id[genre]
        cnt = 0
        for audio in os.listdir(DATASET_PATH / root / genre):
            files += [str(DATASET_PATH / root / genre / audio)]
            cnt += 1
        ids += [genre_id] * cnt
    return np.array(files), np.array(ids)


def build_remix_dataset(files: np.ndarray, ids: np.ndarray, ratios):
    base_dataset = RemixerDataset(files, ids)
    # Use the indices to create the dataset splits
    return [
        RemixerDataset(dataset.dataset.data, dataset.dataset.targets)
        for dataset in split_dataset(base_dataset, ratios, )
    ]


