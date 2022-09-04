import base64
import collections
import contextlib
import hashlib
import logging
import argparse
import importlib
from constants import MODULE_DIR
import librosa
from io import BytesIO
import os
from pathlib import Path
from typing import Union
from urllib.request import urlretrieve
import soundfile as sf
from core.models import VAE_Model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm



def get_genre_embeds(genre: str):
    if genre not in GENRES:
        print(f"unknown genre: {genre}")
    genre_embeds[genre] = encode_genre(genre)
    return genre_embeds[genre]


def setup_callbacks(args):
    log_dir = ROOT_DIR / "training" / "logs"
    _ensure_logging_dir(log_dir)

    goldstar_metric = "validation/loss"
    filename_format = "epoch={epoch:04d}-validation.loss={validation/loss:.3f}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=5,
        filename=filename_format,
        monitor=goldstar_metric,
        mode="min",
        auto_insert_metric_name=False,
        dirpath=experiment_dir,
        every_n_epochs=args.check_val_every_n_epoch,
    )

    # summary_callback = pl.callbacks.ModelSummary(max_depth=2)

    logger = pl.loggers.WandbLogger(log_model="all", save_dir=str(log_dir), job_type="train")
    logger.watch(model, log_freq=max(100, args.log_every_n_steps))
    logger.log_hyperparams(vars(args))
    experiment_dir = logger.experiment.dir

    callbacks.append(cb.ImageToTextLogger())


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid








vae_dict = collections.defaultdict(lambda: VAE)


