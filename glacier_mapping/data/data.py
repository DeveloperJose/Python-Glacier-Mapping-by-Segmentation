#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 13:24:56 2021

@author: mibook
"""

import glob
import logging
import os
import pathlib
import random

import elasticdeform
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import model.functions as fn

BAND_NAMES = np.array(
    [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6_VCID1",
        "B6_VCID2",
        "B7",
        "elevation",
        "slope",
        "physics",
    ]
)


def fetch_loaders(
    processed_dir,
    batch_size=32,
    use_channels=[3, 2, 1],
    output_classes=[1],
    class_names=["BG", "CleanIce", "Debris"],
    physics_channel=10,
    normalize=False,
    train_folder="train",
    val_folder="val",
    test_folder="test",
    shuffle=True,
):
    """Function to fetch dataLoaders for the Training / Validation
    Args:
        processed_dir(str): Directory with the processed data
        batch_size(int): The size of each batch during training. Defaults to 32.
    Return:
        Returns train and val dataloaders
    """
    fn.log(
        logging.INFO,
        f"fetch_loaders() | Output classes: {[class_names[cl] for cl in output_classes]} | raw={output_classes}",
    )
    fn.log(logging.INFO, f"fetch_loaders() | Using channels {BAND_NAMES[use_channels]}")

    if isinstance(processed_dir, str):
        processed_dir = pathlib.Path(processed_dir)

    train_dataset = GlacierDataset(
        processed_dir / train_folder,
        use_channels,
        output_classes,
        physics_channel,
        normalize,
        transforms=transforms.Compose(
            [
                # DropoutChannels(0.5),
                FlipHorizontal(0.15),
                FlipVertical(0.15),
                Rot270(0.15),
                # ElasticDeform(1)
            ]
        ),
    )
    val_dataset = GlacierDataset(
        processed_dir / val_folder,
        use_channels,
        output_classes,
        physics_channel,
        normalize,
    )
    test_dataset = GlacierDataset(
        processed_dir / test_folder,
        use_channels,
        output_classes,
        physics_channel,
        normalize,
    )

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=8,
        shuffle=shuffle,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=8,
        shuffle=shuffle,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=8,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader


class GlacierDataset(Dataset):
    """Custom Dataset for Glacier Data
    Indexing the i^th element returns the underlying image and the associated
    binary mask
    """

    def __init__(
        self,
        folder_path,
        use_channels,
        output_classes,
        physics_channel,
        normalize,
        transforms=None,
    ):
        """Initialize dataset."""
        self.folder_path = folder_path
        self.use_channels = use_channels
        self.output_classes = np.array(output_classes, dtype=np.uint8)
        self.normalize = normalize
        self.transforms = transforms

        self.physics_channel = physics_channel
        self.use_physics = physics_channel in use_channels

        # Sanity checking
        assert isinstance(output_classes, list), "output_classes must be a list"
        assert len(set(output_classes)) == len(
            output_classes
        ), "output_classes cannot have duplicates"
        assert all(self.output_classes >= 0) and all(
            self.output_classes < 3
        ), "output_classes must be either 0 (BG), 1 (CleanIce), or 2 (Debris)"

        # Get image and mask files from provided folder path
        self.img_files = glob.glob(os.path.join(folder_path, "*tiff*"))
        self.mask_files = [s.replace("tiff", "mask") for s in self.img_files]

        # Load normalization arrays
        arr = np.load(folder_path.parent / "normalize_train.npy")
        if self.normalize == "min-max":
            self.min, self.max = arr[2][use_channels], arr[3][use_channels]
        if self.normalize == "mean-std":
            self.mean, self.std = arr[0], arr[1]
            self.mean, self.std = self.mean[use_channels], self.std[use_channels]

    def __getitem__(self, index):
        """getitem method to retrieve a single instance of the dataset
        Args:
            index(int): Index identifier of the data instance
        Return:
            data(x) and corresponding label(y)
        """
        file_data = np.load(self.img_files[index])
        data = file_data[:, :, self.use_channels]

        _mask = np.sum(data, axis=2) == 0
        if self.normalize == "min-max":
            data = np.clip(data, self.min, self.max)
            data = (data - self.min) / (self.max - self.min)
        elif self.normalize == "mean-std":
            # mean-std all channels except physics
            # if self.use_physics:
            #     data[:, :, :-1] = (data[:, :, :-1] - self.mean[:-1]) / self.std[:-1]
            #     data[:, :, -1] = (data[:, :, -1] - data[:, :, -1].mean()) / data[:, :, -1].std()
            # else:
            data = (data - self.mean) / self.std
        else:
            raise ValueError("normalize must be min-max or mean-std")
        label = np.expand_dims(np.load(self.mask_files[index]), axis=2)
        # ones = label == 1
        # twos = label == 2
        # zeros = np.invert(ones + twos)
        # label = np.concatenate((zeros, ones, twos), axis=2)
        # print('DEBUGGING', np.sum(label==0), np.sum(label==1), np.sum(label==2))

        # Set labels depending on problem (Binary vs Multi-Class)
        # fn.log(logging.INFO, f'label has unique={np.unique(label)}')
        if len(self.output_classes) == 1:
            binary_class = self.output_classes[0]
            # label = np.concatenate((label != binary_class, label == binary_class), axis=2)
            label = np.concatenate((label == 0, label == binary_class), axis=2)
        else:
            # label = np.concatenate((label == 0, label == 1, label==2), axis=2)
            label = np.concatenate([label == x for x in self.output_classes], axis=2)
        label[_mask] = 0

        if self.transforms:
            sample = {"image": data, "mask": label}
            sample = self.transforms(sample)
            data = torch.from_numpy(sample["image"].copy()).float()
            label = torch.from_numpy(sample["mask"].copy()).float()
        else:
            data = torch.from_numpy(data).float()
            label = torch.from_numpy(label).float()
        return data, label

    def __len__(self):
        """Function to return the length of the dataset
        Args:
            None
        Return:
            len(img_files)(int): The length of the dataset (img_files)
        """
        return len(self.img_files)


class FlipHorizontal(object):
    """Flip horizontal randomly the image in a sample.

    Args:
        p (float between 0 and 1): Probability of FlipHorizontal
    """

    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample["image"], sample["mask"]
        if torch.rand(1) < self.p:
            data = data[:, ::-1, :]
            label = label[:, ::-1, :]
        return {"image": data, "mask": label}


class FlipVertical(object):
    """Flip vertically randomly the image in a sample.

    Args:
        p (float between 0 and 1): Probability of FlipVertical
    """

    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample["image"], sample["mask"]
        if torch.rand(1) < self.p:
            data = data[::-1, :, :]
            label = label[::-1, :, :]
        return {"image": data, "mask": label}


class Rot270(object):
    """Flip vertically randomly the image in a sample.

    Args:
        p (float between 0 and 1): Probability of Rot270
    """

    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample["image"], sample["mask"]
        if torch.rand(1) < self.p:
            data = data.transpose((1, 0, 2))
            label = label.transpose((1, 0, 2))
        return {"image": data, "mask": label}


class DropoutChannels(object):
    """
    Apply Random channel dropouts
    """

    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample["image"], sample["mask"]
        if torch.rand(1) < self.p:
            rand_channel_index = np.random.randint(
                low=0, high=data.shape[2], size=int(data.shape[2] / 5)
            )
            data[:, :, rand_channel_index] = 0
        return {"image": data, "mask": label}


class ElasticDeform(object):
    """
    Apply Elasticdeform from U-Net
    """

    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample["image"], sample["mask"]
        label = label.astype(np.float32)
        if torch.rand(1) < self.p:
            [data, label] = elasticdeform.deform_random_grid([data, label], axis=(0, 1))
        label = np.round(label).astype(bool)
        return {"image": data, "mask": label}
