from pathlib import Path
import dotenv
import numpy as np

# import wandb
import sys
import psutil

WANDB_PROJECT_NAME = "Remixer"


# Autodl
def speedup():
    # pass
    import subprocess
    import os

    result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True,
                            text=True)
    output = result.stdout
    for line in output.splitlines():
        if '=' in line:
            var, value = line.split('=', 1)
            os.environ[var] = value


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
    # speedup()
    dotenv.load_dotenv(ROOT_DIR / ".env")


ROOT_DIR = Path(__file__).parents[1]

# print(f'{ROOT_DIR}')

MODULE_DIR = 'core'
MODULE_PATH = ROOT_DIR / MODULE_DIR

TEST_PATH = ROOT_DIR / 'test'

DATASET_PATH = ROOT_DIR / "data" / "downloaded"

# print(f'{DATASET_PATH}')

env = dotenv.load_dotenv(ROOT_DIR / ".env")
