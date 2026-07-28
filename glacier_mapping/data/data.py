import json
import logging
import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset

import glacier_mapping.utils.logging as fn

# Channel groups resolve from band_metadata.json by name at runtime.
CHANNEL_GROUP_DEFINITIONS = {
    "landsat": {
        "names": ["B1", "B2", "B3", "B4", "B5", "B6_VCID1", "B6_VCID2", "B7"],
        "description": "Landsat-7 spectral bands",
    },
    "dem": {
        "names": ["elevation", "slope_deg"],
        "description": "Digital Elevation Model features",
    },
    "spectral_indices": {
        "names": ["NDVI", "NDWI", "NDSI"],
        "description": "Spectral indices",
    },
    "hsv": {
        "names": ["H", "S", "V"],
        "description": "HSV color space channels",
    },
    "velocity": {
        "names": ["velocity", "velocity_x", "velocity_y", "velocity_mask"],
        "description": "Legacy ITS_LIVE velocity data (magnitude, vx, vy, mask)",
        "mandatory": ["velocity_mask"],
        "no_normalize": ["velocity_mask"],
    },
    "velocity_quality_v2": {
        "names": [
            "velocity_speed",
            "velocity_count",
            "velocity_error",
            "velocity_relative_error",
            "velocity_valid",
        ],
        "description": "Date-aware scalar ITS_LIVE speed and quality features",
        "mandatory": ["velocity_valid"],
        "no_normalize": ["velocity_valid"],
    },
    "physics": {
        "names": ["flow_accumulation", "tpi", "roughness", "plan_curvature"],
        "description": "Physics-based terrain features",
    },
}

# Channels requiring logarithmic scaling (0 to +inf -> 0 to 1)
LOG_CHANNELS = {
    "velocity",
    "velocity_speed",
    "velocity_count",
    "velocity_error",
    "velocity_relative_error",
    "flow_accumulation",
    "roughness",
}

# Channels requiring symmetric logarithmic scaling (-inf to +inf -> -1 to 1)
SYMLOG_CHANNELS = {"velocity_x", "velocity_y", "tpi", "plan_curvature"}


def load_band_names(processed_dir):
    """
    Load band names from band_metadata.json if available

    Args:
        processed_dir: Path to processed dataset directory

    Returns:
        np.ndarray: Array of band names
    """
    if isinstance(processed_dir, str):
        processed_dir = pathlib.Path(processed_dir)

    metadata_path = processed_dir / "band_metadata.json"

    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            band_names = np.array(metadata["band_names"])
            fn.log(
                logging.INFO,
                f"Loaded {len(band_names)} band names from {metadata_path}",
            )
            return band_names
        except Exception as e:
            fn.log(
                logging.WARNING,
                f"Failed to load band_metadata.json: {e}. Throwing exception.",
            )
            # return BAND_NAMES_LEGACY.copy()
            raise e
    else:
        fn.log(logging.INFO, "No band_metadata.json found. Throwing exception.")
        raise Exception("Failed to load band_metadata.json")
        # return BAND_NAMES_LEGACY.copy()


def get_no_normalize_channel_names():
    """
    Return set of channel names that should NOT be normalized.

    These are typically binary mask channels where normalization
    would destroy semantic meaning (e.g., velocity_mask is 0/1).

    Returns:
        Set[str]: Channel names to exclude from normalization
    """
    no_norm = set()
    for group in CHANNEL_GROUP_DEFINITIONS.values():
        if "no_normalize" in group:
            no_norm.update(group["no_normalize"])
    return no_norm


class GlacierDataset(Dataset):
    def __init__(
        self,
        folder_path,
        output_classes,
        normalize,
        robust_scaling=True,
        transforms=None,
    ):
        self.folder_path = folder_path
        self.output_classes = np.array(output_classes, dtype=np.uint8)
        self.normalize = normalize
        self.robust_scaling = robust_scaling
        self.transforms = transforms

        if isinstance(self.folder_path, str):
            self.folder_path = pathlib.Path(self.folder_path)

        assert isinstance(output_classes, list), "output_classes must be a list"
        assert len(set(output_classes)) == len(output_classes), (
            "output_classes cannot have duplicates"
        )
        assert all(self.output_classes >= 0) and all(self.output_classes < 3), (
            "output_classes must be either 0 (BG), 1 (CleanIce), or 2 (Debris)"
        )

        self.x_path = self.folder_path / "X.npy"
        self.y_path = self.folder_path / "y.npy"
        if not self.x_path.exists() or not self.y_path.exists():
            raise FileNotFoundError(
                f"comprehensive_v3 dataset requires {self.x_path} and {self.y_path}"
            )

        x_meta = np.load(self.x_path, mmap_mode="r")
        y_meta = np.load(self.y_path, mmap_mode="r")
        if x_meta.ndim != 4:
            raise ValueError(f"Expected X.npy shape [N,C,H,W], got {x_meta.shape}")
        if y_meta.ndim != 3:
            raise ValueError(f"Expected y.npy shape [N,H,W], got {y_meta.shape}")
        if x_meta.shape[0] != y_meta.shape[0] or x_meta.shape[2:] != y_meta.shape[1:]:
            raise ValueError(f"X/y shape mismatch: {x_meta.shape} vs {y_meta.shape}")

        self.num_samples = int(x_meta.shape[0])
        self.num_channels = int(x_meta.shape[1])
        self.spatial_shape = tuple(x_meta.shape[2:])
        self._x = None
        self._y = None

        band_names = load_band_names(self.folder_path.parent)
        if len(band_names) != self.num_channels:
            raise ValueError(
                f"Band metadata has {len(band_names)} bands but X.npy has "
                f"{self.num_channels} channels"
            )

        self.channel_names = band_names.tolist()
        velocity_mask_name = (
            "velocity_valid"
            if "velocity_valid" in self.channel_names
            else "velocity_mask"
        )
        self.velocity_mask_pos = (
            self.channel_names.index(velocity_mask_name)
            if velocity_mask_name in self.channel_names
            else None
        )
        self.velocity_value_positions = [
            idx
            for idx, name in enumerate(self.channel_names)
            if name
            in {
                "velocity",
                "velocity_x",
                "velocity_y",
                "velocity_speed",
                "velocity_count",
                "velocity_error",
                "velocity_relative_error",
            }
        ]

    def _lazy_open(self):
        if self._x is None:
            self._x = np.load(self.x_path, mmap_mode="r")
            self._y = np.load(self.y_path, mmap_mode="r")

    def __getitem__(self, index):
        self._lazy_open()
        data = np.array(self._x[index], dtype=np.float32, copy=True)
        label_int = np.array(self._y[index], dtype=np.uint8, copy=True)

        if self.transforms:
            data, label_int = self.transforms(data, label_int)

        data = torch.from_numpy(data)

        return data, torch.from_numpy(label_int)

    def __len__(self):
        return self.num_samples


def apply_chw_geometric_transform(
    image: np.ndarray, label_int: np.ndarray, transform_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an Albumentations-equivalent geometric transform to CHW arrays."""
    if transform_name == "h_flip":
        return (
            np.ascontiguousarray(image[:, :, ::-1]),
            np.ascontiguousarray(label_int[:, ::-1]),
        )
    if transform_name == "v_flip":
        return (
            np.ascontiguousarray(image[:, ::-1, :]),
            np.ascontiguousarray(label_int[::-1, :]),
        )
    if transform_name == "rotate90":
        return (
            np.ascontiguousarray(np.rot90(image, k=1, axes=(1, 2))),
            np.ascontiguousarray(np.rot90(label_int, k=1, axes=(0, 1))),
        )
    if transform_name == "transpose":
        return (
            np.ascontiguousarray(np.swapaxes(image, 1, 2)),
            np.ascontiguousarray(label_int.T),
        )
    raise ValueError(f"Unsupported transform: {transform_name}")


class ChwGeometricAugmentations:
    """Fast CHW replacement for the previous Albumentations geometric transforms."""

    def __init__(self, aug_opts: dict, seed: int):
        self.h_flip_prob = float(aug_opts.get("h_flip_prob", 0.0))
        self.v_flip_prob = float(aug_opts.get("v_flip_prob", 0.0))
        self.rotate90_prob = float(aug_opts.get("rotate90_prob", 0.0))
        self.transpose_prob = float(aug_opts.get("transpose_prob", 0.0))
        self.legacy_torch_rng = bool(aug_opts.get("legacy_torch_rng", False))
        self.seed = seed

    def _draw(self) -> float:
        if self.legacy_torch_rng:
            return float(torch.rand(1).item())
        return float(np.random.random())

    def __call__(
        self, image: np.ndarray, label_int: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._draw() < self.h_flip_prob:
            image, label_int = apply_chw_geometric_transform(image, label_int, "h_flip")
        if self._draw() < self.v_flip_prob:
            image, label_int = apply_chw_geometric_transform(image, label_int, "v_flip")
        if self._draw() < self.rotate90_prob:
            image, label_int = apply_chw_geometric_transform(
                image, label_int, "rotate90"
            )
        if self._draw() < self.transpose_prob:
            image, label_int = apply_chw_geometric_transform(
                image, label_int, "transpose"
            )
        return image, label_int
