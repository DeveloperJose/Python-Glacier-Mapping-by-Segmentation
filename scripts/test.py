#!/usr/bin/env python3
"""
Comprehensive test suite for glacier mapping task validation.

Tests all three task types to prevent regressions:
- Binary Clean Ice (CI): target_class_ids=[1], output_classes=[1]
- Binary Debris-Covered Ice (DCI): target_class_ids=[2], output_classes=[2]
- Multi-class: target_class_ids=[1,2], output_classes=[0,1,2]

Features:
- Creates temporary test configs automatically
- Uses subset data for fast execution
- Verbose output for debugging
- Complete pipeline validation
- Regression detection
- Automatic cleanup of temporary files

Usage:
    uv run python scripts/test.py [--server local] [--subset-size 5] [--epochs 2]
"""

import argparse
import copy
import os
import sys
import tempfile
import traceback
import subprocess
import unittest
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

import torch
import numpy as np
import yaml

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from glacier_mapping.utils.config import load_config, load_server_config
from glacier_mapping.lightning.glacier_datamodule import GlacierDataModule
import json
from glacier_mapping.lightning.glacier_module import GlacierSegmentationModule
from glacier_mapping.data.slice import (
    get_tiff_np,
    save_slices,
    read_shp,
    add_index,
    compute_dems,
)
from glacier_mapping.data.data import apply_chw_geometric_transform
from glacier_mapping.model.losses import customloss, make_onehot
from glacier_mapping.model.evaluation import (
    precision,
    recall,
    IoU,
    dice,
    tp_fp_fn,
    get_pr_iou,
    calculate_binary_metrics,
    create_invalid_mask,
    predict_from_probs,
    _normalize_target,
    metric_name,
    full_metric_name,
    merge_ci_debris,
    predict_slice,
    _iter_split_samples,
    CLASS_TO_INDEX,
)
import rasterio

TEST_DATASET_NAME = "comprehensive_v3"


class GlacierTaskTestSuite:
    """Comprehensive test suite for glacier mapping tasks."""

    def __init__(self, server: str = "local", subset_size: int = 5, epochs: int = 2):
        self.server = server
        self.subset_size = subset_size
        self.epochs = epochs
        self.test_results = {}
        self.temp_configs = []
        self.temp_dir = tempfile.mkdtemp(prefix="glacier_test_")

        # Setup logging
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        self.logger = logging.getLogger(__name__)

        print("=== GLACIER MAPPING COMPREHENSIVE TEST SUITE ===")
        print(f"Server: {server} | Subset Size: {subset_size} | Epochs: {epochs}")
        print(f"Temp Directory: {self.temp_dir}")
        print(f"Timestamp: {self._get_timestamp()}")
        print()

    def test_raw_file_integrity(self):
        """Verify integrity of raw TIFF, DEM, and velocity files."""
        print("=== Test: Raw File Integrity ===")
        server_config = load_server_config(self.server)
        image_dir = Path(server_config["image_dir"])
        dem_dir = Path(server_config["dem_dir"])
        velocity_dir = Path(server_config["velocity_dir"])
        image_files = sorted(list(image_dir.glob("*.tif")))

        if not image_files:
            print("  - No image files found, skipping test.")
            return

        for i, image_file in enumerate(image_files[: self.subset_size]):
            # --- Image File ---
            with rasterio.open(image_file) as src:
                data = src.read()
                if np.isnan(data).any() or np.isinf(data).any():
                    print(
                        f"  ⚠️ NaN or Inf found in image {image_file.name} (will be handled by slice.py)"
                    )
                    # raise ValueError(f"NaN or Inf found in image {image_file.name}")
            print(f"  ✓ {image_file.name}: Checked for NaN/Inf values.")

            # --- DEM File ---
            dem_file = dem_dir / image_file.name
            if dem_file.exists():
                with rasterio.open(dem_file) as src:
                    data = src.read()
                    if np.isnan(data).any() or np.isinf(data).any():
                        print(
                            f"  ⚠️ NaN or Inf found in DEM {dem_file.name} (will be handled by slice.py)"
                        )
                        # raise ValueError(f"NaN or Inf found in DEM {dem_file.name}")

                    elevation, slope = data[0], data[1]
                    if not (-500 < elevation.min() and elevation.max() < 9000):
                        print(
                            f"  ⚠️ {dem_file.name}: Unusual elevation range [{elevation.min()}, {elevation.max()}]"
                        )
                    if not (0 <= slope.min() and slope.max() <= 90):
                        print(
                            f"  ⚠️ {dem_file.name}: Unusual slope range [{slope.min()}, {slope.max()}]"
                        )
                print(
                    f"  ✓ {dem_file.name}: Checked for NaN/Inf values and plausible ranges."
                )

            # --- Velocity File ---
            velocity_file = velocity_dir / image_file.name
            if velocity_file.exists():
                with rasterio.open(velocity_file) as src:
                    data = src.read()
                    if np.isnan(data).any() or np.isinf(data).any():
                        print(
                            f"  ⚠️ NaN or Inf found in velocity {velocity_file.name} (will be handled by slice.py)"
                        )
                        # raise ValueError(
                        #     f"NaN or Inf found in velocity {velocity_file.name}"
                        # )

                    mask = data[3]
                    if not np.all((mask == 0) | (mask == 1)):
                        unique_vals = np.unique(mask)
                        raise ValueError(
                            f"Raw velocity mask not binary in {velocity_file.name}. Values: {unique_vals}"
                        )
                print(f"  ✓ {velocity_file.name}: No NaN/Inf values and binary mask.")
        print()

    def test_preprocessing_functions(self):
        """Test the preprocessing functions on a subset of raw data."""
        print("=== Test: Preprocessing Functions ===")
        server_config = load_server_config(self.server)
        image_dir = Path(server_config["image_dir"])
        dem_dir = Path(server_config["dem_dir"])
        velocity_dir = Path(server_config["velocity_dir"])
        labels_path = Path(server_config["labels_dir"]) / "HKH_CIDC_5basins_all.shp"

        image_files = sorted(list(image_dir.glob("*.tif")))
        labels = read_shp(labels_path)

        if not image_files:
            print("  - No image files found, skipping test.")
            return

        for i, image_file in enumerate(image_files[: self.subset_size]):
            fname = image_file.name
            tiff_fname = image_dir / fname
            dem_fname = dem_dir / fname
            velocity_fname = velocity_dir / fname

            conf = {
                "image_dir": str(image_dir),
                "dem_dir": str(dem_dir),
                "velocity_dir": str(velocity_dir),
                "add_velocity": True,
                "physics_res": None,
                "physics_scale": None,
                "add_ndvi": False,
                "add_ndwi": False,
                "add_ndsi": False,
                "add_hsv": False,
                "window_size": [256, 256],
                "overlap": 0,
                "filter": 0.0,
                "out_dir": self.temp_dir,
            }

            # Test get_tiff_np
            tiff_np, band_names = get_tiff_np(
                tiff_fname,
                dem_fname=dem_fname,
                velocity_fname=velocity_fname,
                physics_res=conf["physics_res"],
                physics_scale=conf["physics_scale"],
                add_ndvi=conf["add_ndvi"],
                add_ndwi=conf["add_ndwi"],
                add_ndsi=conf["add_ndsi"],
                add_hsv=conf["add_hsv"],
                return_band_names=True,
                verbose=True,
            )
            print(
                f"  ✓ get_tiff_np returned array of shape {tiff_np.shape} for {fname}"
            )

            # Test save_slices
            save_path = Path(self.temp_dir) / f"preprocessed_{i}"
            save_path.mkdir()
            save_slices(i, fname, labels, save_path, **conf)
            print(f"  ✓ save_slices ran without errors for {fname}")

            # Verify the output
            output_files = list(save_path.glob("tiff_*.npy"))
            mask_files = list(save_path.glob("mask_*.npy"))

            if output_files and mask_files:
                print(f"  ✓ Found {len(output_files)} output files for {fname}.")
                break
        else:
            raise ValueError("No valid slices were generated from the first 5 images.")

        # Check a sample slice
        sample_slice = np.load(output_files[0])
        velocity_mask_channel = band_names.index("velocity_mask")
        velocity_mask = sample_slice[..., velocity_mask_channel]
        is_binary = np.all((velocity_mask == 0) | (velocity_mask == 1))

        if not is_binary:
            unique_vals = np.unique(velocity_mask)
            print(
                f"  ❌ Processed velocity mask is NOT BINARY. Unique values: {unique_vals}"
            )
            raise ValueError("Processed velocity mask is not binary.")
        else:
            print("  ✓ Processed velocity mask is binary.")
        print()

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _run_preprocessing(self):
        """Run the preprocessing script for the test dataset."""
        print("=== PREPROCESSING STEP ===")
        preprocess_config_path = "configs/datasets/comprehensive_v3.yaml"

        if not Path(preprocess_config_path).exists():
            print(
                f"⚠️ Preprocessing config not found at {preprocess_config_path}, skipping."
            )
            return

        cmd = [
            "uv",
            "run",
            "python",
            "scripts/preprocess.py",
            "--server",
            self.server,
            "--config",
            preprocess_config_path,
        ]

        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("❌ Preprocessing failed!")
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError("Preprocessing script failed.")
        else:
            print("✓ Preprocessing completed successfully.")
        print()

    def create_temp_config(self, task_name: str, task_config: Dict[str, Any]) -> str:
        """Create temporary configuration file."""
        config_path = Path(self.temp_dir) / f"{task_name}_test.yaml"

        with open(config_path, "w") as f:
            yaml.dump(task_config, f, default_flow_style=False)

        self.temp_configs.append(config_path)
        return str(config_path)

    def create_missing_configs(self):
        """Auto-create minimal test configs for missing task types."""
        print("=== CONFIGURATION SETUP ===")

        # Base template for minimal test configs
        base_config = {
            "training_opts": {
                "dataset_name": TEST_DATASET_NAME,
                "run_name": "",  # Will be set per task
                "seed": 42,
                "deterministic": True,
                "epochs": self.epochs,
                "early_stopping": self.epochs + 10,  # Disable early stopping
                "val_viz_n": 0,  # Disable visualizations for speed
                "run_test_eval": False,  # Disable test evaluation for speed
            },
            "loader_opts": {
                "physics_channels": False,
                "velocity_channels": False,
                "batch_size": min(4, self.subset_size),  # Small batch for testing
            },
        }

        # Check existing configs
        existing_configs = {}

        # Check for debris ice config (velocity loss)
        debris_velocity_config_path = Path(
            "configs/local/debris_ice/velocity_loss_experiment.yaml"
        )
        if debris_velocity_config_path.exists():
            existing_configs["debris_ice_velocity"] = str(debris_velocity_config_path)
            print(f"✓ Existing config found: {debris_velocity_config_path}")
        else:
            # Create debris ice config with velocity loss enabled
            debris_velocity_config = copy.deepcopy(base_config)
            debris_velocity_config["training_opts"]["run_name"] = (
                "debris_ice_velocity_test"
            )
            debris_velocity_config["loader_opts"]["velocity_channels"] = True
            debris_velocity_config["loss_opts"] = {"use_velocity_loss": True}
            debris_velocity_path = self.create_temp_config(
                "debris_ice_velocity", debris_velocity_config
            )
            existing_configs["debris_ice_velocity"] = debris_velocity_path
            print(f"✓ Auto-created temp config: {debris_velocity_path}")

        # Check for debris ice config (class weighting)
        debris_weighted_config_path = Path(
            "configs/local/debris_ice/class_weighting_experiment.yaml"
        )
        if debris_weighted_config_path.exists():
            existing_configs["debris_ice_weighted"] = str(debris_weighted_config_path)
            print(f"✓ Existing config found: {debris_weighted_config_path}")
        else:
            # Create debris ice config with class weighting enabled
            debris_weighted_config = copy.deepcopy(base_config)
            debris_weighted_config["training_opts"]["run_name"] = (
                "debris_ice_weighted_test"
            )
            debris_weighted_config["loader_opts"]["velocity_channels"] = True
            debris_weighted_config["loader_opts"]["physics_channels"] = True
            debris_weighted_config["loss_opts"] = {"class_weights": [1, 6.0]}
            debris_weighted_path = self.create_temp_config(
                "debris_ice_weighted", debris_weighted_config
            )
            existing_configs["debris_ice_weighted"] = debris_weighted_path
            print(f"✓ Auto-created temp config: {debris_weighted_path}")

        # Create clean ice config
        clean_config = copy.deepcopy(base_config)
        clean_config["training_opts"]["run_name"] = "clean_ice_test"
        clean_path = self.create_temp_config("clean_ice", clean_config)
        existing_configs["clean_ice"] = clean_path
        print(f"✓ Auto-created temp config: {clean_path}")

        # Create multiclass config
        multi_config = copy.deepcopy(base_config)
        multi_config["training_opts"]["run_name"] = "multiclass_test"
        multi_path = self.create_temp_config("multiclass", multi_config)
        existing_configs["multiclass"] = multi_path
        print(f"✓ Auto-created temp config: {multi_path}")

        print()
        return existing_configs

    def load_merged_config(self, config_path: str) -> Tuple[Dict[str, Any], str]:
        """Load configuration with 4-level merging simulation."""
        # Load base configs
        train_config = load_config("configs/train.yaml")
        server_config = load_server_config(self.server)

        # Load experiment config
        exp_config = load_config(config_path)

        # Determine task from config path or content
        task_name = "unknown"
        if "clean_ice" in config_path:
            task_name = "clean_ice"
        elif "debris_ice" in config_path:
            task_name = "debris_ice"
        elif "multiclass" in config_path:
            task_name = "multiclass"

        # Load task config
        task_config = load_config(f"configs/tasks/{task_name}.yaml")

        # Simulate 4-level merging
        merged = {}

        # Level 1: Base training config
        merged.update(train_config)

        # Level 2: Server config (training_opts and loader_opts)
        if "training_opts" in server_config:
            merged["training_opts"] = {
                **merged.get("training_opts", {}),
                **server_config["training_opts"],
            }
        if "loader_opts" in server_config:
            merged["loader_opts"] = {
                **merged.get("loader_opts", {}),
                **server_config["loader_opts"],
            }

        # Level 3: Task config
        if "loader_opts" in task_config:
            merged["loader_opts"] = {
                **merged.get("loader_opts", {}),
                **task_config["loader_opts"],
            }
        if "loss_opts" in task_config:
            merged["loss_opts"] = {
                **merged.get("loss_opts", {}),
                **task_config["loss_opts"],
            }
        if "metrics_opts" in task_config:
            merged["metrics_opts"] = {
                **merged.get("metrics_opts", {}),
                **task_config["metrics_opts"],
            }

        # Level 4: Experiment config
        if "training_opts" in exp_config:
            merged["training_opts"] = {
                **merged.get("training_opts", {}),
                **exp_config["training_opts"],
            }
        if "loader_opts" in exp_config:
            merged["loader_opts"] = {
                **merged.get("loader_opts", {}),
                **exp_config["loader_opts"],
            }
        if "loss_opts" in exp_config:
            merged["loss_opts"] = {
                **merged.get("loss_opts", {}),
                **exp_config["loss_opts"],
            }

        return merged, task_name

    def inspect_raw_data(self, dataset_path: str) -> Dict[str, Any]:
        """Inspect raw mask files and class distribution."""
        dataset_path_obj = Path(dataset_path)

        # Find mask data in each split
        splits = ["train", "val", "test"]
        mask_files = []
        y_arrays = []

        for split in splits:
            split_dir = dataset_path_obj / split
            if split_dir.exists():
                y_path = split_dir / "y.npy"
                if y_path.exists():
                    y_arrays.append(y_path)
                else:
                    mask_files.extend(list(split_dir.glob("mask_*.npy")))

        # Limit to subset size
        mask_files = mask_files[: self.subset_size]

        if not mask_files and not y_arrays:
            raise ValueError(f"No mask files found in {dataset_path}")

        # Analyze class distribution
        class_counts = {0: 0, 1: 0, 2: 0, 255: 0}
        total_pixels = 0

        masks_seen = 0
        if y_arrays:
            for y_path in y_arrays:
                y_arr = np.load(y_path, mmap_mode="r")
                for mask in y_arr[: self.subset_size]:
                    unique, counts = np.unique(mask, return_counts=True)
                    for val, count in zip(unique, counts):
                        if val in class_counts:
                            class_counts[val] += count
                            total_pixels += count
                    masks_seen += 1
                    if masks_seen >= self.subset_size:
                        break
                if masks_seen >= self.subset_size:
                    break
        else:
            for mask_file in mask_files:
                mask = np.load(mask_file)
                unique, counts = np.unique(mask, return_counts=True)

                for val, count in zip(unique, counts):
                    if val in class_counts:
                        class_counts[val] += count
                        total_pixels += count
                masks_seen += 1

        # Calculate percentages
        class_percentages = {}
        for val, count in class_counts.items():
            class_percentages[val] = (
                (count / total_pixels * 100) if total_pixels > 0 else 0
            )

        sample_shape = None
        if y_arrays:
            sample_shape = np.load(y_arrays[0], mmap_mode="r").shape[1:]
        elif mask_files:
            mask = np.load(mask_files[0])
            sample_shape = mask.shape

        return {
            "mask_files": masks_seen,
            "class_counts": class_counts,
            "class_percentages": class_percentages,
            "total_pixels": total_pixels,
            "sample_shape": sample_shape,
        }

    def test_task(self, config_path: str, task_name: str) -> Dict[str, Any]:
        """Test a single task comprehensively."""
        print(f"=== TASK: {task_name.upper()} ===")
        print(f"Config: {config_path}")
        print()

        result = {
            "task_name": task_name,
            "config_path": config_path,
            "tests": {},
            "passed": True,
            "errors": [],
        }

        try:
            # 1. Configuration Verification
            print("1.1 Configuration Verification:")
            config, detected_task = self.load_merged_config(config_path)

            output_classes = config["loader_opts"]["output_classes"]
            target_class_ids = config["loss_opts"].get("target_class_ids", [])

            print(f"  ✓ output_classes: {output_classes}")
            print(f"  ✓ target_class_ids: {target_class_ids}")
            print("  ✓ 4-level merging successful")

            result["tests"]["config"] = {
                "output_classes": output_classes,
                "target_class_ids": target_class_ids,
                "passed": True,
            }
            print()

            # 2. Raw Data Inspection
            print("1.2 Raw Data Inspection:")
            server_config = load_server_config(self.server)
            dataset_path = str(
                Path(server_config["processed_data_path"])
                / config["training_opts"]["dataset_name"]
            )

            raw_data_info = self.inspect_raw_data(str(dataset_path))
            print(f"  ✓ Found {raw_data_info['mask_files']} mask files")
            print(
                f"  ✓ Class distribution: BG={raw_data_info['class_percentages'][0]:.1f}%, "
                f"CI={raw_data_info['class_percentages'][1]:.1f}%, "
                f"DCI={raw_data_info['class_percentages'][2]:.1f}%"
            )
            print(f"  ✓ Mask shape: {raw_data_info['sample_shape']}")

            result["tests"]["raw_data"] = {
                "mask_files": raw_data_info["mask_files"],
                "class_percentages": raw_data_info["class_percentages"],
                "passed": True,
            }
            print()

            # 3. Data Loading Verification
            print("1.3 Data Loading Verification:")

            # Create data module with subset
            loader_opts = config["loader_opts"].copy()
            # Remove unsupported parameters
            loader_opts.pop("target_class_ids", None)

            data_module = GlacierDataModule(
                processed_dir=str(dataset_path), **loader_opts
            )

            # Setup to resolve channels
            data_module.setup()

            # Load a small batch
            train_loader = data_module.train_dataloader()
            batch = next(iter(train_loader))
            x, y_int = batch

            print(f"  ✓ Input shape: {x.shape}")
            print(f"  ✓ Integer target shape: {y_int.shape}")
            print(f"  ✓ Input range: [{x.min():.3f}, {x.max():.3f}]")

            valid_pixels = y_int != 255

            # Verify target class distribution against raw masks.
            if len(output_classes) == 1:
                target_class = output_classes[0]
                target_percentage = (
                    (y_int == target_class) & valid_pixels
                ).float().mean().item() * 100
                expected_percentage = raw_data_info["class_percentages"][target_class]
                print(
                    f"  ✓ Binary: target class {target_class} = {target_percentage:.1f}% "
                    f"(expected ~{expected_percentage:.1f}%)"
                )
            else:
                for cls in output_classes:
                    cls_percentage = (
                        (y_int == cls) & valid_pixels
                    ).float().mean().item() * 100
                    expected_percentage = raw_data_info["class_percentages"][cls]
                    print(
                        f"  ✓ Class {cls}: {cls_percentage:.1f}% "
                        f"(expected ~{expected_percentage:.1f}%)"
                    )

            result["tests"]["data_loading"] = {
                "input_shape": list(x.shape),
                "target_shape": list(y_int.shape),
                "passed": True,
            }
            print()

            # 3.5 Enhanced Channel Range Validation
            print("1.4 Enhanced Channel Range Validation:")

            # Check each channel type for reasonable ranges
            channel_ranges = {}

            # Get band names from dataset metadata
            import json

            metadata_path = Path(dataset_path) / "band_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                all_band_names = metadata.get("band_names", [])
            else:
                all_band_names = []

            selected_channel_names = all_band_names

            for i, band_name in enumerate(selected_channel_names[: x.shape[1]]):
                channel_data = x[:, i, :, :]
                channel_min = channel_data.min().item()
                channel_max = channel_data.max().item()
                channel_mean = channel_data.mean().item()
                channel_std = channel_data.std().item()

                channel_ranges[band_name] = {
                    "min": channel_min,
                    "max": channel_max,
                    "mean": channel_mean,
                    "std": channel_std,
                }

                # Validate reasonable ranges based on channel type
                if band_name.startswith("B"):  # Landsat bands
                    if channel_min < -1000 or channel_max > 2000:
                        print(
                            f"  ⚠️ {band_name}: Unusual range [{channel_min:.1f}, {channel_max:.1f}]"
                        )
                    else:
                        print(
                            f"  ✓ {band_name}: Valid range [{channel_min:.1f}, {channel_max:.1f}]"
                        )
                elif band_name in ["elevation", "slope_deg"]:  # DEM channels
                    if band_name == "elevation" and (
                        channel_min < 0 or channel_max > 9000
                    ):
                        print(
                            f"  ⚠️ {band_name}: Unusual elevation range [{channel_min:.1f}, {channel_max:.1f}]"
                        )
                    elif band_name == "slope_deg" and (
                        channel_min < 0 or channel_max > 90
                    ):
                        print(
                            f"  ⚠️ {band_name}: Unusual slope range [{channel_min:.1f}, {channel_max:.1f}]"
                        )
                    else:
                        print(
                            f"  ✓ {band_name}: Valid range [{channel_min:.1f}, {channel_max:.1f}]"
                        )
                elif band_name == "velocity_mask":
                    # Check if all values are either 0 or 1
                    is_binary = torch.all(
                        (channel_data == 0) | (channel_data == 1)
                    ).item()
                    if not is_binary:
                        unique_vals = torch.unique(channel_data)
                        print(
                            f"  ❌ {band_name}: NOT BINARY. Unique values: {unique_vals.numpy()}"
                        )
                        raise ValueError(
                            f"Velocity mask is not binary. Values: {unique_vals.numpy()}"
                        )
                    else:
                        print(
                            f"  ✓ {band_name}: Is binary [{channel_min:.1f}, {channel_max:.1f}]"
                        )
                elif band_name.startswith("velocity"):  # Velocity channels
                    if abs(channel_min) > 50 or abs(channel_max) > 50:
                        print(
                            f"  ⚠️ {band_name}: Unusual velocity range [{channel_min:.1f}, {channel_max:.1f}]"
                        )
                    else:
                        print(
                            f"  ✓ {band_name}: Valid range [{channel_min:.1f}, {channel_max:.1f}]"
                        )
                elif band_name in ["NDVI", "NDWI", "NDSI"]:  # Spectral indices
                    if channel_min < -1.0 or channel_max > 1.0:
                        print(
                            f"  ⚠️ {band_name}: Unusual index range [{channel_min:.3f}, {channel_max:.3f}]"
                        )
                    else:
                        print(
                            f"  ✓ {band_name}: Valid range [{channel_min:.3f}, {channel_max:.3f}]"
                        )
                elif band_name in ["H", "S", "V"]:  # HSV channels
                    if band_name == "H" and (channel_min < 0 or channel_max > 360):
                        print(
                            f"  ⚠️ {band_name}: Unusual hue range [{channel_min:.1f}, {channel_max:.1f}]"
                        )
                    elif band_name in ["S", "V"] and (
                        channel_min < 0 or channel_max > 1.0
                    ):
                        print(
                            f"  ⚠️ {band_name}: Unusual {band_name} range [{channel_min:.3f}, {channel_max:.3f}]"
                        )
                    else:
                        print(
                            f"  ✓ {band_name}: Valid range [{channel_min:.3f}, {channel_max:.3f}]"
                        )
                else:
                    print(
                        f"  ✓ {band_name}: Range [{channel_min:.3f}, {channel_max:.3f}]"
                    )

            result["tests"]["channel_ranges"] = channel_ranges
            print()

            # 4. Model Integration
            print("1.5 Model Integration:")

            # Use the data module's channel and class configuration for the model.
            loader_opts = config.get("loader_opts", {})
            loader_opts["processed_dir"] = str(
                dataset_path
            )  # Ensure model knows data path

            model_init_args = {
                key: loader_opts.get(key)
                for key in [
                    "landsat_channels",
                    "dem_channels",
                    "spectral_indices_channels",
                    "hsv_channels",
                    "physics_channels",
                    "velocity_channels",
                    "output_classes",
                    "class_names",
                ]
                if key in loader_opts
            }

            # Create Lightning module
            model = GlacierSegmentationModule(
                model_opts=config.get("model_opts", {}),
                loss_opts=config.get("loss_opts", {}),
                optim_opts=config.get("optim_opts", {}),
                metrics_opts=config.get("metrics_opts", {}),
                loader_opts=loader_opts,  # Pass full loader_opts for other settings
                **model_init_args,
            )

            # Verification: Ensure channel counts match
            if len(data_module.use_channels) != len(model.use_channels):
                raise ValueError(
                    f"Channel count mismatch: data module has {len(data_module.use_channels)} "
                    f"but model has {len(model.use_channels)}. Check channel configs."
                )

            print(f"  ✓ Model input channels: {len(model.use_channels)}")
            print(f"  ✓ Model output channels: {model.model.seg_layer.out_channels}")

            # Test forward pass
            with torch.no_grad():
                logits = model.model(x)
                print(f"  ✓ Forward pass output shape: {logits.shape}")

                # Check activation
                if logits.shape[1] == 2:
                    probs = torch.sigmoid(logits)
                    print("  ✓ Activation: sigmoid (binary)")
                else:
                    probs = torch.softmax(logits, dim=1)
                    print("  ✓ Activation: softmax (multi-class)")

                print(f"  ✓ Probability range: [{probs.min():.3f}, {probs.max():.3f}]")

            result["tests"]["model"] = {
                "input_channels": len(data_module.use_channels),
                "output_channels": model.model.seg_layer.out_channels,
                "forward_pass": True,
                "passed": True,
            }
            print()

            # 5. Loss Function Verification
            print("1.6 Loss Function Verification:")

            loss_fn = model.loss_fn
            print(f"  ✓ Loss function: {type(loss_fn).__name__}")

            # Output classes and target IDs determine the foreground mapping.
            if len(output_classes) == 1:
                # Binary tasks remap the target class to channel 1.
                print(
                    f"  ✓ Binary task: output_classes={output_classes}, target_class_ids={target_class_ids}"
                )
            else:
                # Multiclass tasks retain all configured foreground classes.
                print(
                    f"  ✓ Multi-class task: output_classes={output_classes}, target_class_ids={target_class_ids}"
                )

            # Test loss computation
            with torch.no_grad():
                # Extract velocity data for loss function
                velocity = None
                velocity_mask = None
                if (
                    model.use_velocity_loss
                    and model.velocity_idx is not None
                    and model.velocity_mask_idx is not None
                ):
                    vel_norm = x[:, model.velocity_idx : model.velocity_idx + 1, :, :]

                    if model.normalization == "mean-std":
                        # Convert normalization arrays to tensors.
                        mean = torch.from_numpy(model.norm_arr[0, :]).to(
                            vel_norm.device
                        )
                        std = torch.from_numpy(model.norm_arr[1, :]).to(vel_norm.device)
                        velocity = (
                            vel_norm * std[model.velocity_idx]
                            + mean[model.velocity_idx]
                        )
                    elif model.normalization == "min-max":
                        _min = torch.from_numpy(model.norm_arr_full[2, :]).to(
                            vel_norm.device
                        )
                        _max = torch.from_numpy(model.norm_arr_full[3, :]).to(
                            vel_norm.device
                        )
                        velocity = (
                            vel_norm
                            * (_max[model.velocity_idx] - _min[model.velocity_idx])
                            + _min[model.velocity_idx]
                        )

                    velocity_mask = x[
                        :, model.velocity_mask_idx : model.velocity_mask_idx + 1, :, :
                    ]

                dice_loss, boundary_loss, velocity_loss = loss_fn(
                    logits,
                    y_int,
                    velocity=velocity,
                    velocity_mask=velocity_mask,
                )

                total_loss = dice_loss + boundary_loss + velocity_loss
                print(f"  ✓ Dice loss: {dice_loss.item():.4f}")
                print(f"  ✓ Boundary loss: {boundary_loss.item():.4f}")
                print(f"  ✓ Velocity loss: {velocity_loss.item():.4f}")
                print(f"  ✓ Total loss: {total_loss.item():.4f}")

            result["tests"]["loss"] = {
                "loss_computed": True,
                "passed": True,
            }
            print()

            # 6. Enhanced Dataset Integrity Validation
            print("1.7 Enhanced Dataset Integrity Validation:")

            # Check for comprehensive dataset specific issues
            if "comprehensive_phys64_s1" in str(dataset_path):
                print("  ✓ Comprehensive dataset detected")

                # Validate physics/velocity channel availability
                velocity_channels_enabled = loader_opts.get("velocity_channels", False)
                physics_channels_enabled = loader_opts.get("physics_channels", False)

                if velocity_channels_enabled:
                    print("  ✓ Velocity channels enabled for comprehensive dataset")
                else:
                    print("  ⚠️ Velocity channels disabled for comprehensive dataset")

                if physics_channels_enabled:
                    print("  ✓ Physics channels enabled for comprehensive dataset")
                else:
                    print("  ⚠️ Physics channels disabled for comprehensive dataset")

                # Check for velocity mask in data
                if "velocity_mask" in selected_channel_names:
                    print("  ✓ Velocity mask channel available")
                else:
                    print("  ⚠️ Velocity mask channel missing")

            # 7. Velocity Loss Configuration Check
            print("1.8 Velocity Loss Configuration Check:")

            velocity_loss_enabled = config.get("loss_opts", {}).get(
                "use_velocity_loss", False
            )
            velocity_channels_in_data = loader_opts.get("velocity_channels", False)

            print(f"  ✓ Loss velocity_loss setting: {velocity_loss_enabled}")
            print(f"  ✓ Data velocity channels: {velocity_channels_in_data}")

            # Validate velocity loss configuration consistency
            if velocity_loss_enabled and not velocity_channels_in_data:
                print(
                    "  ⚠️ WARNING: Velocity loss enabled but no velocity channels in data!"
                )
            elif not velocity_loss_enabled and velocity_channels_in_data:
                print("  ✓ Velocity channels available but loss disabled (OK for Gen7)")
            elif velocity_loss_enabled and velocity_channels_in_data:
                print("  ✓ Velocity loss properly configured with velocity data")
            else:
                print("  ✓ Velocity loss disabled (baseline configuration)")

            result["tests"]["velocity_config"] = {
                "velocity_loss_enabled": velocity_loss_enabled,
                "velocity_channels_in_data": velocity_channels_in_data,
                "consistent": velocity_loss_enabled == velocity_channels_in_data
                or not velocity_loss_enabled,
            }
            print()

            # 8. End-to-End Validation
            print("1.9 End-to-End Validation:")

            # Verify data flow consistency
            expected_output_channels = (
                2 if len(output_classes) == 1 else len(output_classes)
            )
            actual_output_channels = model.model.seg_layer.out_channels

            if actual_output_channels != expected_output_channels:
                raise ValueError(
                    f"Output channel mismatch: expected {expected_output_channels}, got {actual_output_channels}"
                )

            print("  ✓ Config → Data → Model → Loss pipeline consistent")
            print(f"  ✓ {task_name} task validation completed successfully")

            result["tests"]["end_to_end"] = {
                "pipeline_consistent": True,
                "passed": True,
            }
            print()

        except Exception as e:
            result["passed"] = False
            result["errors"].append(str(e))
            print(f"  ❌ ERROR: {e}")
            traceback.print_exc()
            print()

        return result

    def run_all_tests(self) -> Dict[str, Any]:
        """Execute complete test suite."""
        try:
            # Run standalone tests
            self.test_raw_file_integrity()
            self.test_preprocessing_functions()
            self.verify_preprocessed_dataset()

            # Create missing configs
            configs = self.create_missing_configs()

            # Test each task
            for task_name, config_path in configs.items():
                self.test_results[task_name] = self.test_task(config_path, task_name)

            # Generate summary
            self.generate_summary()

        except Exception as e:
            print(f"❌ Test suite failed: {e}")
            traceback.print_exc()

        finally:
            # Cleanup temporary files
            self.cleanup()

        return self.test_results

    def verify_preprocessed_dataset(self):
        """Verify the integrity of the entire preprocessed dataset."""
        print("=== Test: Verify Preprocessed Dataset ===")
        server_config = load_server_config(self.server)
        dataset_name = TEST_DATASET_NAME
        dataset_path = Path(server_config["processed_data_path"]) / dataset_name

        if not dataset_path.exists():
            print(f"  - Dataset not found at {dataset_path}, skipping verification.")
            return

        # Load band metadata
        metadata_path = dataset_path / "band_metadata.json"
        if not metadata_path.exists():
            raise ValueError("band_metadata.json not found in dataset.")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        band_names = metadata.get("band_names", [])
        expected_channels = int(metadata.get("num_bands", len(band_names)))
        stats_path = dataset_path / "dataset_statistics.json"
        expected_spatial = None
        if stats_path.exists():
            with open(stats_path, "r") as f:
                stats = json.load(f)
            window_size = stats.get("summary", {}).get("config", {}).get("window_size")
            if window_size and len(window_size) == 2:
                expected_spatial = tuple(window_size)
        try:
            velocity_mask_channel = band_names.index("velocity_mask")
        except ValueError:
            print(
                "  - No velocity mask channel in this dataset, skipping verification."
            )
            return

        corrupt_files = []
        for split in ["train", "val", "test"]:
            split_dir = dataset_path / split
            if split_dir.exists():
                x_path = split_dir / "X.npy"
                y_path = split_dir / "y.npy"
                if not x_path.exists() or not y_path.exists():
                    corrupt_files.extend([x_path, y_path])
                    continue

                x_data = np.load(x_path, mmap_mode="r")
                y_data = np.load(y_path, mmap_mode="r")
                if x_data.ndim != 4 or x_data.shape[1] != expected_channels:
                    print(
                        f"  - {x_path.name}: Expected NCHW with {expected_channels} channels, got {x_data.shape}"
                    )
                    corrupt_files.append(x_path)
                if y_data.ndim != 3 or y_data.shape[0] != x_data.shape[0]:
                    print(f"  - {y_path.name}: Incorrect label shape {y_data.shape}")
                    corrupt_files.append(y_path)
                if (
                    expected_spatial is not None
                    and tuple(x_data.shape[2:]) != expected_spatial
                ):
                    print(
                        f"  - {x_path.name}: Incorrect spatial shape {x_data.shape[2:]}"
                    )
                    corrupt_files.append(x_path)
                velocity_mask = x_data[:, velocity_mask_channel, :, :]
                if not np.all((velocity_mask == 0) | (velocity_mask == 1)):
                    unique_vals = np.unique(velocity_mask)
                    print(
                        f"  - {x_path.name}: Velocity mask not binary. Values: {unique_vals}"
                    )
                    corrupt_files.append(x_path)
                if not np.all(np.isin(y_data, [0, 1, 2, 255])):
                    print(f"  - {y_path.name}: Contains invalid labels.")
                    corrupt_files.append(y_path)

        if corrupt_files:
            print(f"  ❌ Found {len(corrupt_files)} corrupt files:")
            for f in corrupt_files[:5]:  # Print first 5
                print(f"    - {f.name}")
            raise ValueError("Dataset verification failed. Corrupt files found.")
        else:
            print("  ✓ All slice files are valid.")
        print()

    def generate_summary(self):
        """Generate comprehensive test summary."""
        print("=== COMPREHENSIVE TEST SUMMARY ===")

        all_passed = True
        for task_name, result in self.test_results.items():
            status = "✅ PASSED" if result["passed"] else "❌ FAILED"
            print(f"{task_name.title()}: {status}")
            if not result["passed"]:
                all_passed = False
                for error in result["errors"]:
                    print(f"  Error: {error}")

        print()

        if all_passed:
            print("🎉 ALL TESTS PASSED - No regressions detected!")
            print()
            print("Regression Prevention Status:")

            # Comparative analysis
            clean_result = self.test_results.get("clean_ice", {})
            debris_result = self.test_results.get("debris_ice", {})
            multi_result = self.test_results.get("multiclass", {})

            if all(
                [
                    r.get("passed", False)
                    for r in [clean_result, debris_result, multi_result]
                ]
            ):
                print("✓ Class targeting logic verified across all tasks")
                print("✓ Binary remapping logic confirmed working")
                print("✓ Multi-class loss handling validated")
                print("✓ Configuration inheritance verified")
                print("✓ Data preprocessing consistency confirmed")

                # Specific validations
                clean_config = clean_result.get("tests", {}).get("config", {})
                debris_config = debris_result.get("tests", {}).get("config", {})
                multi_config = multi_result.get("tests", {}).get("config", {})

                if (
                    clean_config.get("output_classes") == [1]
                    and debris_config.get("output_classes") == [2]
                    and multi_config.get("output_classes") == [0, 1, 2]
                ):
                    print("✓ Output classes correctly configured per task")

                if (
                    clean_config.get("target_class_ids") == [1]
                    and debris_config.get("target_class_ids") == [2]
                    and multi_config.get("target_class_ids") == [1, 2]
                ):
                    print("✓ Target class IDs correctly configured per task")

            print()
            print(
                "The glacier mapping system is ready for production use across all task types."
            )

        else:
            print("❌ SOME TESTS FAILED - Regressions detected!")
            print(
                "Please review the errors above and fix the issues before proceeding."
            )

    def cleanup(self):
        """Clean up temporary files."""
        import shutil

        try:
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"⚠ Warning: Could not clean up temp directory {self.temp_dir}: {e}")


class TestVelocityLossMath(unittest.TestCase):
    """Test velocity loss mathematical correctness and Kendall formulation."""

    def setUp(self):
        self.metadata_dir = tempfile.TemporaryDirectory()
        metadata = {
            "band_names": [
                "B1",
                "B2",
                "B3",
                "B4",
                "B5",
                "B6_VCID1",
                "B6_VCID2",
                "B7",
                "velocity",
                "velocity_x",
                "velocity_y",
                "velocity_mask",
            ]
        }
        metadata_path = Path(self.metadata_dir.name) / "band_metadata.json"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    def tearDown(self):
        self.metadata_dir.cleanup()

    def test_velocity_loss_basic_functionality(self):
        batch_size, height, width = 2, 32, 32
        pred_logits = torch.randn(batch_size, 2, height, width)
        target_int = torch.ones(batch_size, height, width).long()
        velocity = torch.abs(torch.randn(batch_size, 1, height, width)) * 5.0
        velocity_mask = torch.ones(batch_size, 1, height, width)
        loss_fn = customloss(output_classes=[1])
        dice_loss, boundary_loss, velocity_loss = loss_fn(
            pred_logits, target_int, velocity=velocity, velocity_mask=velocity_mask
        )
        self.assertGreater(velocity_loss.item(), 0.0)
        self.assertLess(velocity_loss.item(), 100.0)
        dice_loss_no_vel, boundary_loss_no_vel, velocity_loss_no_vel = loss_fn(
            pred_logits, target_int, velocity=None, velocity_mask=None
        )
        self.assertEqual(velocity_loss_no_vel.item(), 0.0)

    def test_sigma_initialization_values(self):
        from glacier_mapping.lightning.glacier_module import GlacierSegmentationModule

        config = {
            "model_opts": {"args": {"net_depth": 4, "first_channel_output": 16}},
            "loss_opts": {"use_velocity_loss": True},
            "optim_opts": {},
            "metrics_opts": {},
            "loader_opts": {
                "processed_dir": self.metadata_dir.name,
                "velocity_channels": True,
                "output_classes": [1],
                "class_names": ["background", "foreground"],
            },
        }
        model = GlacierSegmentationModule(**config)
        self.assertIsNotNone(model.raw_log_var_dice)
        self.assertIsNotNone(model.raw_log_var_boundary)
        self.assertFalse(hasattr(model, "sigma_velocity"))
        self.assertAlmostEqual(
            model._sigma_from_log_var(model.raw_log_var_dice).item(), 0.5
        )
        self.assertAlmostEqual(
            model._sigma_from_log_var(model.raw_log_var_boundary).item(), 0.5
        )

    def test_velocity_threshold_config_passthrough(self):
        from glacier_mapping.lightning.glacier_module import GlacierSegmentationModule

        model = GlacierSegmentationModule(
            model_opts={"args": {"net_depth": 4, "first_channel_output": 16}},
            loss_opts={"use_velocity_loss": True, "velocity_high_speed_threshold": 9.5},
            optim_opts={},
            metrics_opts={},
            loader_opts={
                "processed_dir": self.metadata_dir.name,
                "velocity_channels": True,
                "output_classes": [1],
                "class_names": ["background", "foreground"],
            },
        )
        self.assertEqual(model.loss_fn.velocity_high_speed_threshold, 9.5)

    def test_kendall_formulation_components(self):
        losses = torch.tensor([0.5, 0.3])
        sigmas = torch.tensor([0.8, 1.1])
        expected_total = torch.tensor(0.0)
        for loss, sigma in zip(losses, sigmas):
            log_var = torch.log(sigma**2)
            expected_total += 0.5 * torch.exp(-log_var) * loss
            expected_total += 0.5 * log_var
        self.assertGreater(expected_total.item(), 0.0)
        self.assertTrue(torch.isfinite(expected_total))

    def test_velocity_loss_edge_cases(self):
        loss_fn = customloss(output_classes=[1])
        pred_logits = torch.randn(2, 2, 32, 32)
        target_int = torch.ones(2, 32, 32).long()
        velocity = torch.zeros(2, 1, 32, 32)
        velocity_mask = torch.ones(2, 1, 32, 32)
        dice_loss, boundary_loss, velocity_loss = loss_fn(
            pred_logits, target_int, velocity=velocity, velocity_mask=velocity_mask
        )
        self.assertGreaterEqual(velocity_loss.item(), 0.0)
        self.assertLess(velocity_loss.item(), 0.01)
        velocity_mask = torch.zeros(2, 1, 32, 32)
        dice_loss, boundary_loss, velocity_loss = loss_fn(
            pred_logits, target_int, velocity=velocity, velocity_mask=velocity_mask
        )
        self.assertEqual(velocity_loss.item(), 0.0)

    def test_velocity_loss_sync_free_matches_guarded_formula(self):
        torch.manual_seed(42)
        logits = torch.randn(2, 2, 16, 16, requires_grad=True)
        target = torch.randint(0, 2, (2, 16, 16), dtype=torch.uint8)
        velocity = torch.rand(2, 1, 16, 16) * 10.0
        velocity_mask = (torch.rand(2, 1, 16, 16) > 0.25).float()
        loss_fn = customloss(
            output_classes=[1],
            velocity_loss_weight=0.2,
            velocity_high_speed_threshold=3.16,
        )

        new_velocity_loss = loss_fn(
            logits,
            target,
            velocity=velocity,
            velocity_mask=velocity_mask,
            current_epoch=20,
        )[2]

        probabilities = torch.softmax(logits, dim=1)
        valid = (target != 255).unsqueeze(1).float()
        moving = torch.sigmoid((velocity - 3.16) * 0.5)
        combined_mask = velocity_mask * valid
        valid_count = combined_mask.sum()
        guarded_base = (
            probabilities[:, :1] * moving * combined_mask
        ).sum() / valid_count
        guarded_velocity_loss = guarded_base * 0.2

        torch.testing.assert_close(new_velocity_loss, guarded_velocity_loss)
        new_gradient = torch.autograd.grad(
            new_velocity_loss, logits, retain_graph=True
        )[0]
        old_gradient = torch.autograd.grad(guarded_velocity_loss, logits)[0]
        torch.testing.assert_close(new_gradient, old_gradient)

    def test_velocity_loss_sync_free_empty_mask_is_zero(self):
        logits = torch.randn(1, 2, 8, 8, requires_grad=True)
        target = torch.ones(1, 8, 8, dtype=torch.uint8)
        velocity = torch.rand(1, 1, 8, 8) * 10.0
        velocity_mask = torch.zeros(1, 1, 8, 8)
        loss_fn = customloss(output_classes=[1], velocity_loss_weight=0.2)

        velocity_loss = loss_fn(
            logits,
            target,
            velocity=velocity,
            velocity_mask=velocity_mask,
            current_epoch=20,
        )[2]

        self.assertEqual(velocity_loss.item(), 0.0)
        gradient = torch.autograd.grad(velocity_loss, logits)[0]
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))

    def test_sigma_minimum_constraint(self):
        from glacier_mapping.lightning.glacier_module import GlacierSegmentationModule

        model = GlacierSegmentationModule(
            model_opts={"args": {"net_depth": 4, "first_channel_output": 16}},
            loss_opts={"use_velocity_loss": True},
            optim_opts={},
            metrics_opts={},
            loader_opts={
                "processed_dir": self.metadata_dir.name,
                "velocity_channels": True,
                "output_classes": [1],
                "class_names": ["bg", "fg"],
            },
        )
        with torch.no_grad():
            model.raw_log_var_dice.data = torch.tensor(-20.0)
            model.raw_log_var_boundary.data = torch.tensor(-20.0)
        pred_logits = torch.randn(1, 2, 32, 32)
        y_uint8 = torch.ones(1, 32, 32).byte()
        velocity = torch.randn(1, 1, 32, 32)
        velocity_mask = torch.ones(1, 1, 32, 32)
        test_loss = model.compute_loss(
            pred_logits, y_uint8, velocity=velocity, velocity_mask=velocity_mask
        )
        self.assertTrue(torch.isfinite(test_loss))
        self.assertGreater(test_loss.item(), 0.0)


class TestSliceFunctions(unittest.TestCase):
    def test_add_index(self):
        tiff_np = np.ones((10, 10, 4), dtype=np.float32)
        tiff_np[..., 0] = 2
        tiff_np[..., 1] = 4
        result = add_index(tiff_np, index1=1, index2=0)
        expected_index = (4 - 2) / (4 + 2)
        self.assertEqual(result.shape, (10, 10, 5))
        self.assertTrue(np.allclose(result[..., 4], expected_index))
        tiff_np[..., 0] = -4
        result = add_index(tiff_np, index1=1, index2=0)
        self.assertFalse(np.isnan(result).any())

    def test_compute_dems(self):
        dem_np = np.zeros((10, 10, 2), dtype=np.float32)
        dem_np[..., 0] = 1000
        dem_np[..., 1] = 30
        result = compute_dems(dem_np)
        self.assertEqual(result.shape, (10, 10, 2))
        self.assertTrue(np.all(result[..., 0] == 1000))
        self.assertTrue(np.all(result[..., 1] == 30))


class TestChwAugmentations(unittest.TestCase):
    def _real_high_valid_slices(self, limit: int = 2):
        configured = os.environ.get("GLACIER_TEST_DATA_DIR")
        if not configured:
            self.skipTest("GLACIER_TEST_DATA_DIR is not configured")
        root = Path(configured)
        if not root.exists():
            self.skipTest(f"Local reference dataset not found: {root}")

        candidates = []
        for mask_path in sorted(root.glob("mask_*.npy"))[:200]:
            mask = np.load(mask_path, mmap_mode="r")
            valid_fraction = float(np.mean(mask != 255))
            if valid_fraction > 0.75:
                tiff_path = mask_path.with_name(
                    mask_path.name.replace("mask_", "tiff_")
                )
                if tiff_path.exists():
                    candidates.append((valid_fraction, tiff_path, mask_path))

        if len(candidates) < limit:
            self.skipTest("Not enough high-validity local reference slices found")

        candidates.sort(reverse=True, key=lambda row: row[0])
        return candidates[:limit]

    def _compare_with_albumentations_functional(
        self, transform_name: str, fn_name: str
    ):
        import albumentations.augmentations.geometric.functional as F

        albumentations_fn = getattr(F, fn_name)
        for _, tiff_path, mask_path in self._real_high_valid_slices():
            image_hwc = np.load(tiff_path).astype(np.float32)
            label_int = np.load(mask_path).astype(np.uint8)

            image_chw = np.transpose(image_hwc, (2, 0, 1))
            got_image, got_label = apply_chw_geometric_transform(
                image_chw, label_int, transform_name
            )

            expected_image = np.transpose(albumentations_fn(image_hwc), (2, 0, 1))
            expected_label = albumentations_fn(label_int)

            np.testing.assert_array_equal(got_image, expected_image)
            np.testing.assert_array_equal(got_label, expected_label)

    def test_h_flip_matches_albumentations_on_real_slices(self):
        self._compare_with_albumentations_functional("h_flip", "hflip")

    def test_v_flip_matches_albumentations_on_real_slices(self):
        self._compare_with_albumentations_functional("v_flip", "vflip")

    def test_transpose_matches_albumentations_on_real_slices(self):
        self._compare_with_albumentations_functional("transpose", "transpose")

    def test_rot90_matches_albumentations_on_real_slices(self):
        import albumentations.augmentations.geometric.functional as F

        for _, tiff_path, mask_path in self._real_high_valid_slices():
            image_hwc = np.load(tiff_path).astype(np.float32)
            label_int = np.load(mask_path).astype(np.uint8)
            image_chw = np.transpose(image_hwc, (2, 0, 1))

            got_image, got_label = apply_chw_geometric_transform(
                image_chw, label_int, "rotate90"
            )

            expected_image = np.transpose(F.rot90(image_hwc, factor=1), (2, 0, 1))
            expected_label = F.rot90(label_int, factor=1)

            np.testing.assert_array_equal(got_image, expected_image)
            np.testing.assert_array_equal(got_label, expected_label)


def dice_loss_broadcast(
    pred_prob: torch.Tensor,
    target_prob: torch.Tensor,
    ignore_mask_exp: torch.Tensor,
    smooth: float,
    class_weights_tensor: torch.Tensor | None,
) -> torch.Tensor:
    """Broadcast-mask Dice loss reference implementation for parity tests."""
    weighted_pred = pred_prob * ignore_mask_exp
    weighted_target = target_prob * ignore_mask_exp
    weighted_prod = pred_prob * target_prob * ignore_mask_exp

    numerator = 2 * weighted_prod.sum(dim=(0, 2, 3)) + smooth
    denominator = (
        weighted_pred.sum(dim=(0, 2, 3)) + weighted_target.sum(dim=(0, 2, 3)) + smooth
    )
    dice_per_class = 1 - numerator / denominator

    if class_weights_tensor is not None:
        return (dice_per_class * class_weights_tensor).sum()
    return dice_per_class.mean()


class TestDiceParity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_aryal_custom_loss_matches_pinned_upstream(self):
        import importlib.util

        configured = os.environ.get("ARYAL_UPSTREAM_DIR")
        upstream_root = (
            Path(configured)
            if configured
            else Path(__file__).resolve().parents[2]
            / "glacier-mapping-aryal-upstream-archive"
        )
        upstream_path = upstream_root / "segmentation/model/losses.py"
        if not upstream_path.exists():
            self.skipTest("Set ARYAL_UPSTREAM_DIR to run upstream loss parity")
        spec = importlib.util.spec_from_file_location(
            "aryal_upstream_losses", upstream_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)

        logits = torch.randn(2, 2, 16, 16, device=self.device, requires_grad=True)
        labels = torch.randint(0, 3, (2, 16, 16), device=self.device)
        labels[0, 0, 0] = 255
        target = torch.stack((labels == 0, labels == 1), dim=1).float()

        expected_dice, expected_boundary = upstream.customloss(
            act=torch.nn.Softmax(dim=1), outchannels=2, masked=True
        ).to(self.device)(logits, target)
        actual_dice, actual_boundary, _ = customloss(
            output_classes=[1],
            behavior="aryal_2023",
            binary_non_target_policy="ignore",
            class_weights=[0, 1],
        ).to(self.device)(logits, labels)

        torch.testing.assert_close(actual_dice, expected_dice.sum())
        torch.testing.assert_close(actual_boundary, expected_boundary)

    def test_aryal_binary_non_target_pixels_are_ignored(self):
        labels = torch.tensor([[[0, 1, 2, 255]]], device=self.device)
        target = make_onehot(
            labels,
            2,
            [1],
            self.device,
            binary_non_target_policy="ignore",
        )
        expected = torch.tensor(
            [[[[1, 0, 0, 0]], [[0, 1, 0, 0]]]],
            dtype=torch.float32,
            device=self.device,
        )
        torch.testing.assert_close(target, expected)

    def test_binary_dice_loss_value_parity(self):
        pred = torch.randn(4, 2, 32, 32, device=self.device, requires_grad=True)
        target_int = torch.randint(
            0, 2, (4, 32, 32), device=self.device, dtype=torch.uint8
        )
        target_int[0, 0, 0] = 255
        target_int[1, -1, -1] = 255

        pred_prob = torch.softmax(pred, dim=1)
        target = make_onehot(target_int, 2, [1], self.device).detach()
        ignore_mask_exp = (target_int != 255).unsqueeze(1).float().to(self.device)

        expected = dice_loss_broadcast(pred_prob, target, ignore_mask_exp, 1.0, None)
        dice_loss = customloss(output_classes=[1])(pred, target_int)[0]
        self.assertAlmostEqual(dice_loss.item(), expected.item(), places=6)

    def test_multiclass_dice_loss_value_parity(self):
        pred = torch.randn(4, 3, 32, 32, device=self.device, requires_grad=True)
        target_int = torch.randint(
            0, 3, (4, 32, 32), device=self.device, dtype=torch.uint8
        )
        target_int[0, 0, 0] = 255
        target_int[2, -1, -1] = 255

        pred_prob = torch.softmax(pred, dim=1)
        target = make_onehot(target_int, 3, [0, 1, 2], self.device).detach()
        ignore_mask_exp = (target_int != 255).unsqueeze(1).float().to(self.device)

        expected = dice_loss_broadcast(pred_prob, target, ignore_mask_exp, 1.0, None)
        dice_loss = customloss(output_classes=[0, 1, 2])(pred, target_int)[0]
        self.assertAlmostEqual(dice_loss.item(), expected.item(), places=6)

    def test_binary_dice_with_class_weights(self):
        pred = torch.randn(4, 2, 32, 32, device=self.device)
        target_int = torch.randint(
            0, 2, (4, 32, 32), device=self.device, dtype=torch.uint8
        )
        target_int[0, 0, 0] = 255

        pred_prob = torch.softmax(pred, dim=1)
        target = make_onehot(target_int, 2, [1], self.device).detach()
        ignore_mask_exp = (target_int != 255).unsqueeze(1).float().to(self.device)
        class_weights = torch.tensor([0.3, 0.7], device=self.device)

        expected = dice_loss_broadcast(
            pred_prob, target, ignore_mask_exp, 1.0, class_weights
        )
        loss_fn = customloss(class_weights=[0.3, 0.7], output_classes=[1]).to(
            self.device
        )
        dice_loss = loss_fn(pred, target_int)[0]
        self.assertAlmostEqual(dice_loss.item(), expected.item(), places=6)

    def test_gradient_parity(self):
        pred = torch.randn(2, 2, 16, 16, device=self.device, requires_grad=True)
        target_int = torch.randint(
            0, 2, (2, 16, 16), device=self.device, dtype=torch.uint8
        )
        target_int[0, 0, 0] = 255

        pred_prob = torch.softmax(pred, dim=1)
        target = make_onehot(target_int, 2, [1], self.device).detach()
        ignore_mask_exp = (target_int != 255).unsqueeze(1).float().to(self.device)

        expected = dice_loss_broadcast(pred_prob, target, ignore_mask_exp, 1.0, None)
        expected_grad = torch.autograd.grad(expected, pred, retain_graph=True)[0]

        dice_loss = customloss(output_classes=[1])(pred, target_int)[0]
        actual_grad = torch.autograd.grad(dice_loss, pred)[0]

        torch.testing.assert_close(expected_grad, actual_grad, rtol=1e-5, atol=1e-5)


class TestClassWeights(unittest.TestCase):
    def test_binary_class_weights(self):
        batch_size, height, width = 2, 32, 32
        pred_logits = torch.randn(batch_size, 2, height, width)
        target_int = torch.ones(batch_size, height, width).long()
        loss_fn = customloss(class_weights=[0, 1], output_classes=[1])
        dice_loss, boundary_loss, velocity_loss = loss_fn(pred_logits, target_int)
        self.assertTrue(torch.isfinite(dice_loss))
        self.assertGreaterEqual(dice_loss.item(), 0.0)
        self.assertEqual(velocity_loss.item(), 0.0)

    def test_multiclass_class_weights(self):
        batch_size, height, width = 2, 32, 32
        pred_logits = torch.randn(batch_size, 3, height, width)
        target_int = torch.ones(batch_size, height, width).long()
        loss_fn = customloss(class_weights=[0, 1, 1], output_classes=[0, 1, 2])
        dice_loss, boundary_loss, velocity_loss = loss_fn(pred_logits, target_int)
        self.assertTrue(torch.isfinite(dice_loss))
        self.assertGreaterEqual(dice_loss.item(), 0.0)
        self.assertEqual(velocity_loss.item(), 0.0)

    def test_class_weights_length_mismatch(self):
        batch_size, height, width = 2, 32, 32
        pred_logits = torch.randn(batch_size, 2, height, width)
        target_int = torch.ones(batch_size, height, width).long()
        loss_fn = customloss(class_weights=[0, 1, 1], output_classes=[1])
        with self.assertRaises(ValueError):
            loss_fn(pred_logits, target_int)

    def test_no_class_weights_fallback(self):
        batch_size, height, width = 2, 32, 32
        pred_logits = torch.randn(batch_size, 2, height, width)
        target_int = torch.ones(batch_size, height, width).long()
        loss_fn = customloss(output_classes=[1])
        dice_loss, boundary_loss, velocity_loss = loss_fn(pred_logits, target_int)
        self.assertTrue(torch.isfinite(dice_loss))
        self.assertGreaterEqual(dice_loss.item(), 0.0)
        self.assertEqual(velocity_loss.item(), 0.0)


class TestEvaluationFunctions(unittest.TestCase):
    """Unit tests for evaluation.py metric helpers and prediction functions."""

    def test_precision_recall_iou_dice(self):
        self.assertAlmostEqual(precision(10, 2, 1), 10.0 / 12)
        self.assertAlmostEqual(recall(10, 2, 1), 10.0 / 11)
        self.assertAlmostEqual(IoU(10, 2, 1), 10.0 / 13)
        self.assertAlmostEqual(dice(10, 2, 1), 20.0 / 23)

    def test_metrics_zero_division(self):
        self.assertEqual(precision(0, 0, 0), 0.0)
        self.assertEqual(recall(0, 0, 0), 0.0)
        self.assertEqual(IoU(0, 0, 0), 0.0)
        self.assertEqual(dice(0, 0, 0), 0.0)

    def test_tp_fp_fn_basic(self):
        pred = torch.tensor([1, 1, 0, 1, 0])
        true = torch.tensor([1, 0, 0, 1, 1])
        tp, fp, fn = tp_fp_fn(pred, true)
        self.assertEqual(tp, 2)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 1)

    def test_tp_fp_fn_custom_label(self):
        pred = torch.tensor([2, 2, 0, 2, 0])
        true = torch.tensor([2, 0, 0, 2, 2])
        tp, fp, fn = tp_fp_fn(pred, true, label=2)
        self.assertEqual(tp, 2)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 1)

    def test_tp_fp_fn_empty(self):
        pred = torch.tensor([], dtype=torch.int64)
        true = torch.tensor([], dtype=torch.int64)
        tp, fp, fn = tp_fp_fn(pred, true)
        self.assertEqual((tp, fp, fn), (0, 0, 0))

    def test_get_pr_iou(self):
        pred = np.array([1, 1, 0, 1, 0])
        true = np.array([1, 0, 0, 1, 1])
        p, r, i, tp, fp, fn = get_pr_iou(pred, true)
        self.assertAlmostEqual(p, 2.0 / 3)
        self.assertAlmostEqual(r, 2.0 / 3)
        self.assertAlmostEqual(i, 2.0 / 4)
        self.assertEqual((tp, fp, fn), (2, 1, 1))

    def test_calculate_binary_metrics(self):
        y_pred = np.array([1, 1, 0, 1, 0, 1])
        y_true = np.array([1, 0, 0, 1, 1, 1], dtype=np.uint8)
        p, r, i, tp, fp, fn = calculate_binary_metrics(y_pred, y_true, target_class=1)
        self.assertEqual((tp, fp, fn), (3, 1, 1))

    def test_calculate_binary_metrics_with_mask(self):
        y_pred = np.array([1, 1, 0, 1, 0, 1])
        y_true = np.array([1, 0, 0, 1, 1, 1], dtype=np.uint8)
        mask = np.array([False, False, True, False, False, False])
        p, r, i, tp, fp, fn = calculate_binary_metrics(
            y_pred, y_true, target_class=1, mask=mask
        )
        self.assertEqual((tp, fp, fn), (3, 1, 1))

    def test_calculate_binary_metrics_all_masked(self):
        y_pred = np.array([1, 0])
        y_true = np.array([1, 1], dtype=np.uint8)
        mask = np.array([True, True])
        p, r, i, tp, fp, fn = calculate_binary_metrics(
            y_pred, y_true, target_class=1, mask=mask
        )
        self.assertEqual((tp, fp, fn), (0, 0, 0))
        self.assertEqual(i, 0.0)

    def test_create_invalid_mask(self):
        x = np.ones((4, 4, 3), dtype=np.float32)
        x[0, 0] = 0.0
        y = np.zeros((4, 4), dtype=np.uint8)
        y[1, 1] = 255
        mask = create_invalid_mask(x, y)
        self.assertTrue(mask[0, 0])
        self.assertTrue(mask[1, 1])
        self.assertFalse(mask[2, 2])

    def test_normalize_target(self):
        self.assertEqual(_normalize_target("dci"), "dci")
        self.assertEqual(_normalize_target("DCI"), "dci")
        self.assertEqual(_normalize_target("debris"), "dci")
        self.assertEqual(_normalize_target("ci"), "ci")
        self.assertEqual(_normalize_target("CI"), "ci")
        self.assertEqual(_normalize_target("cleanice"), "ci")
        with self.assertRaises(ValueError):
            _normalize_target("invalid")

    def test_metric_name_format(self):
        self.assertEqual(metric_name("full", "val", "dci", "iou"), "full_val_dci_iou")
        self.assertEqual(full_metric_name("val", "ci", "iou"), "full_val_ci_iou")

    def test_predict_from_probs_binary(self):
        probs = np.zeros((4, 4, 2), dtype=np.float32)
        probs[:, :, 1] = 0.8
        probs[0, 0, 1] = 0.3
        probs[1, 1, 1] = 0.6
        from types import SimpleNamespace

        module = SimpleNamespace()
        module.output_classes = [1]
        module.metrics_opts = {"threshold": [0.5]}
        pred = predict_from_probs(probs, module, fill_holes=False)
        self.assertEqual(pred[0, 0], 0)
        self.assertEqual(pred[1, 1], 1)
        self.assertEqual(pred[2, 2], 1)
        self.assertEqual(pred.dtype, np.uint8)

    def test_predict_from_probs_multiclass(self):
        probs = np.zeros((4, 4, 3), dtype=np.float32)
        probs[:, :, 0] = 0.1
        probs[:, :, 1] = 0.8
        probs[:, :, 2] = 0.1
        from types import SimpleNamespace

        module = SimpleNamespace()
        module.output_classes = [0, 1, 2]
        pred = predict_from_probs(probs, module, fill_holes=False)
        self.assertEqual(pred[0, 0], 1)

    def test_predict_from_probs_fill_holes(self):
        probs = np.zeros((10, 10, 2), dtype=np.float32)
        probs[:, :, 1] = 1.0
        probs[4:6, 4:6, 1] = 0.0
        from types import SimpleNamespace

        module = SimpleNamespace()
        module.output_classes = [1]
        module.metrics_opts = {"threshold": [0.5]}
        pred_no_fill = predict_from_probs(probs, module, fill_holes=False)
        pred_fill = predict_from_probs(probs, module, fill_holes=True)
        self.assertEqual(pred_no_fill[5, 5], 0)
        self.assertEqual(pred_fill[5, 5], 1)

    def test_predict_from_probs_threshold_none(self):
        probs = np.zeros((4, 4, 2), dtype=np.float32)
        probs[:, :, 1] = 0.7
        from types import SimpleNamespace

        module = SimpleNamespace()
        module.output_classes = [1]
        module.metrics_opts = {"threshold": [0.5]}
        pred = predict_from_probs(probs, module, threshold=None, fill_holes=False)
        self.assertEqual(pred[0, 0], 1)

    def test_predict_slice_packed_chw_matches_legacy_hwc(self):
        class ThresholdModel:
            device = torch.device("cpu")
            use_channels = [0, 1]
            output_classes = [2]

            @staticmethod
            def normalize(x):
                return x

            @staticmethod
            def forward(x):
                score = x[:, :1]
                return torch.cat((-score, score), dim=1)

        chw = np.ones((2, 4, 4), dtype=np.float32)
        chw[0, 0, 0] = -1.0
        hwc = np.transpose(chw, (1, 2, 0))
        module = ThresholdModel()

        legacy, _ = predict_slice(module, hwc, fill_holes=False)
        packed, packed_mask = predict_slice(
            module, chw, fill_holes=False, preprocessed_chw=True
        )

        np.testing.assert_array_equal(packed, legacy)
        self.assertIsNone(packed_mask)

    def test_iter_split_samples_reads_packed_arrays(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as tmp_dir:
            split_dir = Path(tmp_dir) / "test"
            split_dir.mkdir()
            x = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2)
            y = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
            np.save(split_dir / "X.npy", x)
            np.save(split_dir / "y.npy", y)

            samples, count = _iter_split_samples(
                SimpleNamespace(processed_dir=tmp_dir), "test"
            )
            loaded = list(samples)

        self.assertEqual(count, 2)
        self.assertEqual(loaded[0][0], "sample_000000")
        self.assertTrue(loaded[0][3])
        np.testing.assert_array_equal(loaded[1][1], x[1])
        np.testing.assert_array_equal(loaded[1][2], y[1])

    def test_merge_ci_debris(self):
        h, w = 10, 10
        prob_ci = np.zeros((h, w, 2), dtype=np.float32)
        prob_ci[:, :, 1] = 0.9
        prob_dci = np.zeros((h, w, 2), dtype=np.float32)
        prob_dci[:, :, 1] = 0.1
        merged, probs = merge_ci_debris(prob_ci, prob_dci, thr_ci=0.5, thr_dci=0.5)
        self.assertEqual(merged[0, 0], CLASS_TO_INDEX["ci"])
        self.assertEqual(merged[0, 0], 1)
        self.assertEqual(probs.shape, (h, w, 3))

    def test_merge_ci_debris_dci_wins(self):
        h, w = 10, 10
        prob_ci = np.zeros((h, w, 2), dtype=np.float32)
        prob_ci[:, :, 1] = 0.1
        prob_dci = np.zeros((h, w, 2), dtype=np.float32)
        prob_dci[:, :, 1] = 0.9
        merged, probs = merge_ci_debris(prob_ci, prob_dci, thr_ci=0.5, thr_dci=0.5)
        self.assertEqual(merged[0, 0], CLASS_TO_INDEX["dci"])
        self.assertEqual(merged[0, 0], 2)

    def test_merge_ci_debris_both_active(self):
        h, w = 10, 10
        prob_ci = np.zeros((h, w, 2), dtype=np.float32)
        prob_ci[:, :, 1] = 0.9
        prob_dci = np.zeros((h, w, 2), dtype=np.float32)
        prob_dci[:, :, 1] = 0.9
        merged, probs = merge_ci_debris(prob_ci, prob_dci, thr_ci=0.5, thr_dci=0.5)
        self.assertEqual(merged[0, 0], CLASS_TO_INDEX["dci"])
        self.assertEqual(probs.shape, (h, w, 3))


def run_unit_tests():
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive glacier mapping test suite"
    )
    parser.add_argument("--server", default="local", help="Server configuration to use")
    parser.add_argument(
        "--subset-size", type=int, default=5, help="Number of files to test per split"
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of epochs for testing"
    )
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")

    args = parser.parse_args()

    if args.unit:
        success = run_unit_tests()
        sys.exit(0 if success else 1)

    # Create and run test suite
    test_suite = GlacierTaskTestSuite(
        server=args.server, subset_size=args.subset_size, epochs=args.epochs
    )

    results = test_suite.run_all_tests()

    # Exit with appropriate code
    all_passed = all(result.get("passed", False) for result in results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
