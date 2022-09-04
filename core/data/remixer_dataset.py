import argparse
import json
import os
from pathlib import Path
import random
import numpy as np

import pathlib

import requests
from io import BytesIO
from typing import Callable, Dict, Optional, Sequence, Tuple
from constants import SEED
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils import data
from core import utils
from core.data.util import RemixerDataset
from constants import *


# gtzan = torchaudio.datasets.GTZAN(DATASET_PATH, folder_in_archive='gtzan', download=True)


class GTZAN(torch.utils.data.Dataset):
    """GTZAN Dataset"""

    def __init__(self, ratios=[]):
        self.ratios = ratios
        # self.data_test = gtzan

    def __getitem__(self, item):
        self.data_test.__getitem__()


