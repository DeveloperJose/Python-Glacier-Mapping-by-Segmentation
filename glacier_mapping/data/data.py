import glob
import json
import logging
import os
import pathlib
import random

import elasticdeform
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import glacier_mapping.utils.logging as fn

# Legacy hardcoded band names (for backward compatibility)
BAND_NAMES_LEGACY = np.array(
    [
        "B1",  # 0
        "B2",  # 1
        "B3",  # 2
        "B4",  # 3
        "B5",  # 4
        "B6_VCID1",  # 5
        "B6_VCID2",  # 6
        "B7",  # 7
        "elevation",  # 8  (raw meters)
        "slope_deg",  # 9  (raw degrees)
        "NDVI",  # 10
        "NDWI",  # 11
        "NDSI",  # 12
        "H",  # 13
        "S",  # 14
        "V",  # 15
        "flow_accumulation",  # 16
        "tpi",  # 17
        "roughness",  # 18
        "plan_curvature",  # 19
    ]
)

# Global BAND_NAMES loaded dynamically
BAND_NAMES = BAND_NAMES_LEGACY.copy()

# Channel group definitions for semantic selection
CHANNEL_GROUP_DEFINITIONS = {
    "landsat": {
        "indices": [0, 1, 2, 3, 4, 5, 6, 7],
        "names": ["B1", "B2", "B3", "B4", "B5", "B6_VCID1", "B6_VCID2", "B7"],
        "description": "Landsat-7 spectral bands",
    },
    "dem": {
        "indices": [8, 9],
        "names": ["elevation", "slope_deg"],
        "description": "Digital Elevation Model features",
    },
    "spectral_indices": {
        "indices": [10, 11, 12],
        "names": ["NDVI", "NDWI", "NDSI"],
        "description": "Spectral indices",
    },
    "hsv": {
        "indices": [13, 14, 15],
        "names": ["H", "S", "V"],
        "description": "HSV color space channels",
    },
    "velocity": {
        "indices": [16, 17, 18, 19],
        "names": ["velocity", "velocity_x", "velocity_y", "velocity_mask"],
        "description": "ITS_LIVE glacier velocity data (magnitude, vx, vy, mask)",
        "mandatory_indices": [19],  # velocity_mask must always be included
    },
    "physics": {
        "indices": [20, 21, 22, 23],
        "names": ["flow_accumulation", "tpi", "roughness", "plan_curvature"],
        "description": "Physics-based terrain features",
    },
}


def load_band_names(processed_dir):
    """
    Load band names from band_metadata.json if available, otherwise fall back to legacy.
    
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
            fn.log(logging.INFO, f"Loaded {len(band_names)} band names from {metadata_path}")
            return band_names
        except Exception as e:
            fn.log(logging.WARNING, f"Failed to load band_metadata.json: {e}. Using legacy band names.")
            return BAND_NAMES_LEGACY.copy()
    else:
        fn.log(logging.INFO, "No band_metadata.json found. Using legacy band names.")
        return BAND_NAMES_LEGACY.copy()


def resolve_channel_selection(
    processed_dir,
    landsat_channels=None,
    dem_channels=None,
    spectral_indices_channels=None,
    hsv_channels=None,
    physics_channels=None,
    velocity_channels=None,
):
    """
    Resolve semantic channel group specifications to numerical indices.
    
    Args:
        processed_dir: Path to processed dataset directory
        landsat_channels: true (all), false/None/[] (skip), or list of indices/names
        dem_channels: true (all), false/None/[] (skip), or list of indices/names
        spectral_indices_channels: true (all), false/None/[] (skip), or list of indices/names
        hsv_channels: true (all), false/None/[] (skip), or list of indices/names
        physics_channels: true (all), false/None/[] (skip), or list of indices/names
        velocity_channels: true (all), false/None/[] (skip), or list of indices/names
        
    Returns:
        List[int]: Sorted list of channel indices to use
        
    Raises:
        ValueError: If no channels are selected
        
    Warnings:
        - Logs warning if requested channel not in dataset (graceful skip)
    """
    # Load band names to validate availability
    band_names = load_band_names(processed_dir)
    max_available_channels = len(band_names)
    
    fn.log(logging.INFO, f"Available channels in dataset: {max_available_channels}")
    fn.log(logging.INFO, f"Band names: {band_names.tolist()}")
    
    selected_channels = []
    velocity_selected = False  # Track if any velocity channel was selected
    
    # Process each channel group
    channel_groups = [
        ("landsat", landsat_channels),
        ("dem", dem_channels),
        ("spectral_indices", spectral_indices_channels),
        ("hsv", hsv_channels),
        ("velocity", velocity_channels),
        ("physics", physics_channels),
    ]
    
    for group_name, group_value in channel_groups:
        if group_value is None or group_value is False:
            fn.log(logging.DEBUG, f"Skipping channel group: {group_name}")
            continue
            
        if group_value == []:
            fn.log(logging.DEBUG, f"Skipping channel group (empty list): {group_name}")
            continue
        
        group_def = CHANNEL_GROUP_DEFINITIONS[group_name]
        
        if group_value is True:
            # Use all channels in this group (if available)
            fn.log(logging.INFO, f"Enabling all {group_name} channels")
            for idx in group_def["indices"]:
                if idx < max_available_channels:
                    selected_channels.append(idx)
                    if group_name == "velocity":
                        velocity_selected = True
                else:
                    channel_name = group_def["names"][group_def["indices"].index(idx)]
                    fn.log(
                        logging.WARNING,
                        f"Channel {idx} ({channel_name}) from {group_name} not available in dataset "
                        f"(only {max_available_channels} channels). Skipping."
                    )
                    
        elif isinstance(group_value, list):
            # Parse list of indices and/or names
            fn.log(logging.INFO, f"Enabling selected {group_name} channels: {group_value}")
            for item in group_value:
                if isinstance(item, int):
                    # Treat as index WITHIN the group (0-based)
                    if 0 <= item < len(group_def["indices"]):
                        channel_idx = group_def["indices"][item]
                        if channel_idx < max_available_channels:
                            selected_channels.append(channel_idx)
                            if group_name == "velocity":
                                velocity_selected = True
                        else:
                            fn.log(
                                logging.WARNING,
                                f"Channel index {channel_idx} from {group_name} not available in dataset. Skipping."
                            )
                    else:
                        fn.log(
                            logging.WARNING,
                            f"Index {item} out of range for {group_name} (valid: 0-{len(group_def['indices'])-1})"
                        )
                        
                elif isinstance(item, str):
                    # Channel name - resolve to index
                    if item in group_def["names"]:
                        idx_in_group = group_def["names"].index(item)
                        channel_idx = group_def["indices"][idx_in_group]
                        if channel_idx < max_available_channels:
                            selected_channels.append(channel_idx)
                            if group_name == "velocity":
                                velocity_selected = True
                        else:
                            fn.log(
                                logging.WARNING,
                                f"Channel '{item}' (index {channel_idx}) from {group_name} "
                                f"not available in dataset. Skipping."
                            )
                    else:
                        fn.log(
                            logging.WARNING,
                            f"Channel name '{item}' not found in {group_name} group. "
                            f"Valid names: {group_def['names']}"
                        )
                else:
                    fn.log(
                        logging.WARNING,
                        f"Invalid channel specification in {group_name}: {item}. "
                        f"Must be int (index) or str (name)."
                    )
    
    # CRITICAL: Add mandatory velocity mask if any velocity channel selected
    if velocity_selected:
        mask_idx = 19  # velocity_mask is always at index 19
        if mask_idx not in selected_channels:
            if mask_idx < max_available_channels:
                selected_channels.append(mask_idx)
                fn.log(logging.INFO, 
                       "✓ Auto-included mandatory velocity_mask channel (index 19)")
            else:
                fn.log(logging.WARNING,
                       "⚠ velocity_mask (index 19) not available in dataset!")
    
    # Remove duplicates and sort
    selected_channels = sorted(list(set(selected_channels)))
    
    if not selected_channels:
        raise ValueError(
            "No channels selected! At least one channel group must be enabled. "
            "Set landsat_channels, dem_channels, spectral_indices_channels, "
            "hsv_channels, physics_channels, or velocity_channels to true or provide a list."
        )
    
    fn.log(logging.INFO, f"✓ Resolved channel selection: {selected_channels}")
    fn.log(logging.INFO, f"✓ Selected band names: {band_names[selected_channels].tolist()}")
    fn.log(logging.INFO, f"✓ Total channels: {len(selected_channels)}")
    
    return selected_channels


def fetch_loaders(
    processed_dir,
    batch_size=32,
    landsat_channels=True,
    dem_channels=True,
    spectral_indices_channels=True,
    hsv_channels=True,
    physics_channels=False,
    velocity_channels=True,
    output_classes=[1],
    class_names=["BG", "CleanIce", "Debris"],
    normalize="mean-std",
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
        landsat_channels: Channel selection for Landsat bands (true/false/list)
        dem_channels: Channel selection for DEM features (true/false/list)
        spectral_indices_channels: Channel selection for spectral indices (true/false/list)
        hsv_channels: Channel selection for HSV channels (true/false/list)
        physics_channels: Channel selection for physics features (true/false/list)
        velocity_channels: Channel selection for velocity data (true/false/list)
        output_classes: 0=BG, 1=CleanIce, 2=Debris. If len==1 → binary (NOT~cls vs cls)
        normalize: "min-max" or "mean-std"
    """
    # Resolve semantic channel groups to numerical indices
    use_channels = resolve_channel_selection(
        processed_dir,
        landsat_channels=landsat_channels,
        dem_channels=dem_channels,
        spectral_indices_channels=spectral_indices_channels,
        hsv_channels=hsv_channels,
        physics_channels=physics_channels,
        velocity_channels=velocity_channels,
    )
    
    # Load band names dynamically
    global BAND_NAMES
    BAND_NAMES = load_band_names(processed_dir)
    
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
        normalize,
    )
    test_dataset = GlacierDataset(
        processed_dir / test_folder,
        use_channels,
        output_classes,
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
        normalize,
        transforms=None,
    ):
        self.folder_path = folder_path
        self.use_channels = use_channels
        self.output_classes = np.array(output_classes, dtype=np.uint8)
        self.normalize = normalize
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
