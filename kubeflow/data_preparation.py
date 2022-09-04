import kfp
from kfp import dsl
from kfp.dsl.component_factory import create_component_from_func
from pathlib import Path
import os
import librosa
from constants import *
from core.utils import PreprocessingConfig




# the data should be structured as follows:
# genre
#   - audios
# where 'genre' is str, audios should be of format .wav, .mp3 or .au
@create_component_from_func
def organize_dataset(input_data_path: str, dataset):
    if dataset == 'GTZAN':
        # the gtzan dataset is already organized
        pass


@dsl.pipeline(
    name='Data Preparation Pipeline',
    description='A pipeline that processes data'
)
def data_preparation_gtzan(
        input_data_path: str,
        output_data_path: str,
        dataset: str,
):
    organize_dataset(input_data=input_data_path, dataset=dataset)
