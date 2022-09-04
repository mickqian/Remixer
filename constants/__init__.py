from pathlib import Path

import dotenv
import numpy as np

# import wandb

WANDB_PROJECT_NAME = "Remixer"

# Autodl


SEED = 246

# accelerator = 'cpu'
GENRES = ['blues', 'folk', 'country', 'metal', 'hiphop', 'jazz', 'rock', 'pop', 'classical', 'disco', 'reggae']

genre2Id = {genre: id for id, genre in enumerate(set(GENRES))}


def seed_everything():
    import random
    np.random.seed(SEED)
    random.seed(SEED)


def init():
    # wandb.init(project=WANDB_PROJECT_NAME)
    seed_everything()
    dotenv.load_dotenv(ROOT_DIR / ".env")


ROOT_DIR = Path(__file__).parents[1]

# print(f'{ROOT_DIR}')

MODULE_DIR = 'core'
MODULE_PATH = ROOT_DIR / MODULE_DIR

TEST_PATH = ROOT_DIR / 'test'

DATASET_PATH = ROOT_DIR / "data" / "downloaded"

OUTPUT_DIR = ROOT_DIR / "training" / "out"  # the model namy locally and on the HF Hub

# print(f'{DATASET_PATH}')

env = dotenv.load_dotenv(ROOT_DIR / ".env")
