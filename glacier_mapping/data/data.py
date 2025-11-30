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

import glacier_mapping.model.functions as fn

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
        "NDVI",
        "NDWI",
        "NDSI",
        "H",
        "S",
        "V",
        "flow_accumulation",
        "slope_magnitude",
        "tpi",
        "roughness",
        "plan_curvature",
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
    """
    Build train/val/test dataloaders.

    Args:
        processed_dir: Root of prepared dataset (contains train/val/test subfolders)
        batch_size: Batch size for DataLoader
        use_channels: Indices into BAND_NAMES
        output_classes: 0=BG, 1=CleanIce, 2=Debris. If len==1 → binary (NOT~cls vs cls)
        normalize: "min-max" or "mean-std"
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
                # ElasticDeform(1),
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

    common_loader_kwargs = dict(
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=8,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **common_loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **common_loader_kwargs,
    )
    return train_loader, val_loader, test_loader


class GlacierDataset(Dataset):
    """
    Custom Dataset for Glacier Data.

    Returns:
        x        : float32 tensor (H, W, C_in) (normalized)
        y_onehot: float32 tensor (H, W, C_out) (one-hot)
        y_int   : int64   tensor (H, W, 1) with values {0,1,2,255}
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
        self.folder_path = folder_path
        self.use_channels = use_channels
        self.output_classes = np.array(output_classes, dtype=np.uint8)
        self.normalize = normalize
        self.transforms = transforms

        self.physics_channel = physics_channel
        self.use_physics = physics_channel in use_channels

        if isinstance(self.folder_path, str):
            self.folder_path = pathlib.Path(self.folder_path)

        assert isinstance(output_classes, list), "output_classes must be a list"
        assert len(set(output_classes)) == len(output_classes), (
            "output_classes cannot have duplicates"
        )
        assert all(self.output_classes >= 0) and all(self.output_classes < 3), (
            "output_classes must be either 0 (BG), 1 (CleanIce), or 2 (Debris)"
        )

        # Find image + mask files
        self.img_files = glob.glob(os.path.join(folder_path, "*tiff*"))
        self.mask_files = [s.replace("tiff", "mask") for s in self.img_files]

        # Normalization stats
        arr = np.load(folder_path.parent / "normalize_train.npy")
        if self.normalize == "min-max":
            self.min, self.max = arr[2][use_channels], arr[3][use_channels]
        elif self.normalize == "mean-std":
            self.mean, self.std = arr[0], arr[1]
            self.mean, self.std = self.mean[use_channels], self.std[use_channels]
        else:
            raise ValueError("normalize must be 'min-max' or 'mean-std'")

    def __getitem__(self, index):
        file_data = np.load(self.img_files[index])
        data = file_data[:, :, self.use_channels]

        if self.normalize == "min-max":
            data = np.clip(data, self.min, self.max)
            data = (data - self.min) / (self.max - self.min)
        elif self.normalize == "mean-std":
            data = (data - self.mean) / self.std

        label_int = np.load(self.mask_files[index]).astype(np.uint8)
        label_int = np.expand_dims(label_int, axis=2)

        if len(self.output_classes) == 1:
            binary_class = self.output_classes[0]
            label = np.concatenate(
                (label_int != binary_class, label_int == binary_class), axis=2
            )
        else:
            label = np.concatenate(
                [label_int == x for x in self.output_classes], axis=2
            )

        if self.transforms:
            sample = {"image": data, "mask": label}
            sample = self.transforms(sample)
            data = torch.from_numpy(sample["image"].copy()).float()
            label = torch.from_numpy(sample["mask"].copy()).float()
        else:
            data = torch.from_numpy(data).float()
            label = torch.from_numpy(label).float()

        return data, label, torch.from_numpy(label_int).long()

    def __len__(self):
        return len(self.img_files)


class FlipHorizontal(object):
    def __init__(self, p):
        if (p < 0) or (p > 1):
            raise ValueError("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample["image"], sample["mask"]
        if torch.rand(1) < self.p:
            data = data[:, ::-1, :]
            label = label[:, ::-1, :]
        return {"image": data, "mask": label}


class FlipVertical(object):
    def __init__(self, p):
        if (p < 0) or (p > 1):
            raise ValueError("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample["image"], sample["mask"]
        if torch.rand(1) < self.p:
            data = data[::-1, :, :]
            label = label[::-1, :, :]
        return {"image": data, "mask": label}


class Rot270(object):
    def __init__(self, p):
        if (p < 0) or (p > 1):
            raise ValueError("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample["image"], sample["mask"]
        if torch.rand(1) < self.p:
            data = data.transpose((1, 0, 2))
            label = label.transpose((1, 0, 2))
        return {"image": data, "mask": label}


class DropoutChannels(object):
    """Random channel dropout augmentation."""

    def __init__(self, p):
        if (p < 0) or (p > 1):
            raise ValueError("Probability should be between 0 and 1")
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
    """Elastic deformation augmentation."""

    def __init__(self, p):
        if (p < 0) or (p > 1):
            raise ValueError("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample["image"], sample["mask"]
        label = label.astype(np.float32)
        if torch.rand(1) < self.p:
            [data, label] = elasticdeform.deform_random_grid([data, label], axis=(0, 1))
        label = np.round(label).astype(bool)
        return {"image": data, "mask": label}
