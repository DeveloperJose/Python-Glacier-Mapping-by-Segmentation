"""Lightning data module for glacier mapping."""

import pathlib
from typing import List, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from glacier_mapping.data.data import GlacierDataset, BAND_NAMES


class GlacierDataModule(pl.LightningDataModule):
    """Lightning data module for glacier segmentation datasets."""

    def __init__(
        self,
        processed_dir: str,
        batch_size: int = 8,
        use_channels: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        output_classes: List[int] = [0, 1, 2],
        class_names: List[str] = ["BG", "CleanIce", "Debris"],
        normalize: str = "mean-std",
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        """
        Initialize the Glacier data module.

        Args:
            processed_dir: Root of prepared dataset (contains train/val/test subfolders)
            batch_size: Batch size for DataLoaders
            use_channels: Indices into BAND_NAMES
            output_classes: 0=BG, 1=CleanIce, 2=Debris. If len==1 → binary (NOT~cls vs cls)
            class_names: Names for each class
            normalize: "min-max" or "mean-std"
            num_workers: Number of worker processes for DataLoaders
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        super().__init__()
        self.processed_dir = pathlib.Path(processed_dir)
        self.batch_size = batch_size
        self.use_channels = use_channels
        self.output_classes = output_classes
        self.class_names = class_names
        self.normalize = normalize
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Data augmentation transforms for training (disabled for debugging)
        self.train_transform = None

        # No augmentation for validation/test
        self.val_transform = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training, validation, and testing."""
        if stage == "fit" or stage is None:
            self.train_dataset = GlacierDataset(
                self.processed_dir / "train",
                self.use_channels,
                self.output_classes,
                self.normalize,
                transforms=self.train_transform,
            )
            
            self.val_dataset = GlacierDataset(
                self.processed_dir / "val",
                self.use_channels,
                self.output_classes,
                self.normalize,
                transforms=self.val_transform,
            )

        if stage == "test" or stage is None:
            self.test_dataset = GlacierDataset(
                self.processed_dir / "test",
                self.use_channels,
                self.output_classes,
                self.normalize,
                transforms=self.val_transform,
            )

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )