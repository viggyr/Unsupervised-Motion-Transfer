"""Base DataModule class."""
from pathlib import Path
from skeleton_to_human.data.aligned_pair_dataset import AlignedPairDataset
from typing import Dict, Optional, Union
import argparse
from pathlib import Path

from .aligned_pair_dataset import AlignedPairDataset
from .util import split_dataset

TRAIN_FRAC = 0.8

from torch.utils.data import ConcatDataset, DataLoader
import pytorch_lightning as pl

from skeleton_to_human import util


def load_and_print_info(data_module_class) -> None:
    """Load EMNISTLines and print info."""
    parser = argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser)
    args = parser.parse_args()
    dataset = data_module_class(args)
    dataset.prepare_data()
    dataset.setup()
    print(dataset)


def _download_raw_dataset(metadata: Dict, dl_dirname: Path) -> Path:
    dl_dirname.mkdir(parents=True, exist_ok=True)
    filename = dl_dirname / metadata["filename"]
    if filename.exists():
        return filename
    print(f"Downloading raw dataset from {metadata['url']} to {filename}...")
    util.download_url(metadata["url"], filename)
    print("Computing SHA-256...")
    sha256 = util.compute_sha256(filename)
    if sha256 != metadata["sha256"]:
        raise ValueError("Downloaded data file SHA-256 does not match that listed in metadata document.")
    return filename


BATCH_SIZE = 128
NUM_WORKERS = 0


class Dancing(pl.LightningDataModule):
    """
    Base DataModule.
    Learn more at https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.dims = (1,512, 512)  # Extra dimension is added by ToTensor()
        self.output_dims = (1,)

        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)

        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))

        # Make sure to set the variables below in subclasses
        self.data_train: Union[AlignedPairDataset, ConcatDataset]
        self.data_val: Union[AlignedPairDataset, ConcatDataset]
        self.data_test: Union[AlignedPairDataset, ConcatDataset]

    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[3] / "data"

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size", type=int, default=BATCH_SIZE, help="Number of examples to operate on per forward step."
        )
        parser.add_argument(
            "--num_workers", type=int, default=NUM_WORKERS, help="Number of additional processes to load data."
        )
        parser.add_argument('--no_instance', action='store_true', help='if specified, do *not* add instance map as input')        
        parser.add_argument('--dataroot', type=str, default='./data/spiderman') 
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
        return parser


    def prepare_data(self, *args, **kwargs) -> None:
        """
        Use this method to do things that might write to disk or that need to be done only from a single GPU
        in distributed settings (so don't set state `self.x = y`).
        """

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Split into train, val, test, and set dims.
        Should assign `torch Dataset` objects to self.data_train, self.data_val, and optionally self.data_test.
        """

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )
        
    def prepare_data(self, *args, **kwargs) -> None:
        pass

    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            data_trainval = AlignedPairDataset(self.args)
            self.data_train, self.data_val = split_dataset(aligned_dataset=data_trainval, fraction=TRAIN_FRAC, seed=42)

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