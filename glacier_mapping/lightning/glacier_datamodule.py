"""Lightning data module for glacier mapping."""

import pathlib
import random
from typing import List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from glacier_mapping.data.data import (
    ChwGeometricAugmentations,
    GlacierDataset,
    load_band_names,
)


class GlacierDataModule(pl.LightningDataModule):
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
        augmentations: Optional[dict] = None,
        output_classes: List[int] = [0, 1, 2],
        class_names: List[str] = ["BG", "CleanIce", "Debris"],
        normalize: str = "mean-std",
        robust_scaling: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: Optional[bool] = None,
        prefetch_factor: Optional[int] = None,
        seed: int = 42,
        augmentation_seed: Optional[int] = None,
        val_shuffle: bool = False,
        shared_loader_generator: bool = False,
    ):
        super().__init__()
        self.processed_dir = pathlib.Path(processed_dir)
        self.batch_size = batch_size
        self.landsat_channels = landsat_channels
        self.dem_channels = dem_channels
        self.spectral_indices_channels = spectral_indices_channels
        self.hsv_channels = hsv_channels
        self.physics_channels = physics_channels
        self.velocity_channels = velocity_channels
        self.output_classes = output_classes
        self.class_names = class_names
        self.normalize = normalize
        self.robust_scaling = robust_scaling
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.seed = seed
        self.augmentation_seed = (
            augmentation_seed if augmentation_seed is not None else seed
        )
        self.val_shuffle = val_shuffle
        self.shared_loader_generator = shared_loader_generator
        self._shared_generator = None
        if self.shared_loader_generator:
            self._shared_generator = torch.Generator()
            self._shared_generator.manual_seed(self.seed)

        if augmentations is not None:
            self.train_transform = self.create_augmentations(augmentations)
        else:
            self.train_transform = None

        self.val_transform = None

    @staticmethod
    def seed_worker(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def _generator(self) -> torch.Generator:
        if self._shared_generator is not None:
            return self._shared_generator
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return generator

    def create_augmentations(self, aug_opts: dict) -> ChwGeometricAugmentations:
        return ChwGeometricAugmentations(aug_opts, seed=self.augmentation_seed)

    def setup(self, stage: Optional[str] = None):
        self.band_names = load_band_names(self.processed_dir)
        self.use_channels = list(range(len(self.band_names)))

        if stage == "fit" or stage is None:
            self.train_dataset = GlacierDataset(
                self.processed_dir / "train",
                self.output_classes,
                self.normalize,
                robust_scaling=self.robust_scaling,
                transforms=self.train_transform,
            )
            self.val_dataset = GlacierDataset(
                self.processed_dir / "val",
                self.output_classes,
                self.normalize,
                robust_scaling=self.robust_scaling,
                transforms=self.val_transform,
            )

    def _dataloader(self, dataset, shuffle: bool) -> DataLoader:
        worker_opts = self._worker_opts()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            worker_init_fn=self.seed_worker,
            generator=self._generator(),
            **worker_opts,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, shuffle=self.val_shuffle)

    def _worker_opts(self) -> dict:
        if self.num_workers <= 0:
            return {}

        opts = {}
        if self.persistent_workers is not None:
            opts["persistent_workers"] = self.persistent_workers
        if self.prefetch_factor is not None:
            opts["prefetch_factor"] = self.prefetch_factor
        return opts
