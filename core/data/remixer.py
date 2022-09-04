"""Base DataModule class."""
import os

from torch.utils.data import Dataset

from constants import *
from core.utils import PreprocessingConfig
from .codec import build_codec
from .utils import *


class RemixerDataset(Dataset):
    def __init__(
            self,
            data: SequenceOrTensor,
            targets: SequenceOrTensor,
            codec=None
    ) -> None:
        super().__init__()

        if len(data) != len(targets):
            raise ValueError("Data and targets must be of equal length")
        if codec == None:
            raise RuntimeError("empty codec provided")
        self.data = data
        self.targets = targets
        self.codec = codec

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        while True:
            if index >= len(self.data):
                return None
            datum, target = self.data[index], self.targets[index]

            if self.codec is not None:
                datum = self.codec.encode(datum)
                if not datum:
                    index += 1
                    continue
                if isinstance(datum, list):
                    datum = datum[0]

            target = target
            return datum, (self.data[index], target)


class RemixerLoader(torch.utils.data.DataLoader):
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
        self.config = config
        self.codec = build_codec('mel', PreprocessingConfig())


excluded = ["electronic", "experimental", "hip-hop"]
keywords = ["hip"]


def read_remixer_dataset(root='', all=False):
    files = []
    ids = []
    for genre_dir in os.listdir(DATASET_PATH / root):
        if not genre_dir.startswith(".") and os.path.isdir(DATASET_PATH / root / genre_dir):
            genre = str(genre_dir).lower()
            if genre in excluded or any(keyword in genre for keyword in keywords):
                print(f"skipping {genre_dir}")
                continue
            if genre not in genre2Id and not all:
                continue
                # genre2Id[genre] = len(genre2Id)
            genre_id = genre2Id[genre] if genre in genre2Id else 0
            cnt = 0
            for audio in os.listdir(DATASET_PATH / root / genre_dir):
                files += [str(DATASET_PATH / root / genre_dir / audio)]
                cnt += 1
            print(f"{genre}: {cnt} songs")
            ids += [genre_id] * cnt
    return np.array(files), np.array(ids)


def build_remix_dataset(files: np.ndarray, ids: np.ndarray, ratios, codec=None):
    base_dataset = RemixerDataset(files, ids, codec=codec)
    # Use the indices to create the dataset splits
    return [
        RemixerDataset(dataset.dataset.data, dataset.dataset.targets, codec=codec)
        for dataset in split_dataset(base_dataset, ratios, )
    ]
