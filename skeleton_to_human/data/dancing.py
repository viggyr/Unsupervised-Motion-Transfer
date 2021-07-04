"""
EMNIST dataset. Downloads from NIST website and saves as .npz file if not already present.
"""
from pathlib import Path
from typing import Sequence
import json
import os
import shutil
import zipfile

from torchvision import transforms
import numpy as np
import toml

from .base_data_module import BaseDataModule
from .aligned_pair_dataset import AlignedPairDataset
from .util import split_dataset

TRAIN_FRAC = 0.8


class Dancing(BaseDataModule):
    """
    "The EMNIST dataset is a set of handwritten character digits derived from the NIST Special Database 19
    and converted to a 28x28 pixel image format and dataset structure that directly matches the MNIST dataset."
    From https://www.nist.gov/itl/iad/image-group/emnist-dataset
    The data split we will use is
    EMNIST ByClass: 814,255 characters. 62 unbalanced classes.
    """

    def __init__(self, args=None):
        super().__init__(args)

        self.dims = (1,512, 512)  # Extra dimension is added by ToTensor()
        self.output_dims = (1,)
        self.args=args

    def prepare_data(self, *args, **kwargs) -> None:
        pass

    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            data_trainval = AlignedPairDataset(self.args)
            self.data_train, self.data_val = split_dataset(base_dataset=data_trainval, fraction=TRAIN_FRAC, seed=42)

    def __repr__(self):
        basic = f"Dataset\nNum classes: {len(self.mapping)}\nDims: {self.dims}\n"
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}\n"
            f"Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
        )
        return basic + data