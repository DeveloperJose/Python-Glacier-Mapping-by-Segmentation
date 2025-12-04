"""Lightning data module for glacier mapping."""

import pathlib
from typing import List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from glacier_mapping.data.data import GlacierDataset


class GlacierDataModule(pl.LightningDataModule):
    """Lightning data module for glacier segmentation datasets."""

    def __init__(
        self,
        processed_dir: str,
        batch_size: int = 8,
        landsat_channels=True,
        dem_channels=True,
        spectral_indices_channels=True,
        hsv_channels=True,
        physics_channels=False,
        velocity_channels=True,
        output_classes: List[int] = [0, 1, 2],
        class_names: List[str] = ["BG", "CleanIce", "Debris"],
        normalize: str = "mean-std",
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        """
        Initialize Glacier data module.

        Args:
            processed_dir: Root of prepared dataset (contains train/val subfolders)
            batch_size: Batch size for DataLoaders
            landsat_channels: Landsat band selection (true/false/list)
            dem_channels: DEM feature selection (true/false/list)
            spectral_indices_channels: Spectral indices selection (true/false/list)
            hsv_channels: HSV channel selection (true/false/list)
            physics_channels: Physics feature selection (true/false/list)
            velocity_channels: Velocity data selection (true/false/list)
            output_classes: 0=BG, 1=CleanIce, 2=Debris. If len==1 → binary (NOT~cls vs cls)
            class_names: Names for each class
            normalize: "min-max" or "mean-std"
            num_workers: Number of worker processes for DataLoaders
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        super().__init__()
        self.processed_dir = pathlib.Path(processed_dir)
        self.batch_size = batch_size

        # Store channel group selections
        self.landsat_channels = landsat_channels
        self.dem_channels = dem_channels
        self.spectral_indices_channels = spectral_indices_channels
        self.hsv_channels = hsv_channels
        self.physics_channels = physics_channels
        self.velocity_channels = velocity_channels

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
        """Setup datasets for training and validation."""
        # Import here to avoid circular imports
        from glacier_mapping.data.data import resolve_channel_selection

        # Resolve channel groups to indices
        self.use_channels = resolve_channel_selection(
            self.processed_dir,
            landsat_channels=self.landsat_channels,
            dem_channels=self.dem_channels,
            spectral_indices_channels=self.spectral_indices_channels,
            hsv_channels=self.hsv_channels,
            physics_channels=self.physics_channels,
            velocity_channels=self.velocity_channels,
        )

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
