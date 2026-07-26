#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import multiprocessing
import multiprocessing.pool
import os
import random
import shutil
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from addict import Dict
from tqdm import tqdm

import glacier_mapping.data.slice as fn
from glacier_mapping.data.data import (
    CHANNEL_GROUP_DEFINITIONS,
    get_no_normalize_channel_names,
    load_band_names,
)

import matplotlib

matplotlib.use("Agg")


def remove_and_create(dirpath):
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath)


def istarmap(pool_self, func, iterable, chunksize=1):
    pool_self._check_running()
    if chunksize < 1:
        raise ValueError("Chunksize must be 1+, not {0:n}".format(chunksize))

    task_batches = multiprocessing.pool.Pool._get_tasks(func, iterable, chunksize)
    result = multiprocessing.pool.IMapIterator(pool_self)
    pool_self._taskqueue.put(
        (
            pool_self._guarded_task_generation(
                result._job, multiprocessing.pool.starmapstar, task_batches
            ),
            result._set_length,
        )
    )
    return (item for chunk in result for item in chunk)


multiprocessing.pool.Pool.istarmap = istarmap


PACKED_RECIPES: dict[str, list[str]] = {
    "comprehensive_v3_landsat": CHANNEL_GROUP_DEFINITIONS["landsat"]["names"],
    "comprehensive_v3_landsat_dem": (
        CHANNEL_GROUP_DEFINITIONS["landsat"]["names"]
        + CHANNEL_GROUP_DEFINITIONS["dem"]["names"]
    ),
    "comprehensive_v3_landsat_dem_flowacc": (
        CHANNEL_GROUP_DEFINITIONS["landsat"]["names"]
        + CHANNEL_GROUP_DEFINITIONS["dem"]["names"]
        + ["flow_accumulation"]
    ),
    "comprehensive_v3_landsat_dem_velmag": (
        CHANNEL_GROUP_DEFINITIONS["landsat"]["names"]
        + CHANNEL_GROUP_DEFINITIONS["dem"]["names"]
        + ["velocity", "velocity_mask"]
    ),
    "comprehensive_v3_landsat_dem_velocity_speed_v2": (
        CHANNEL_GROUP_DEFINITIONS["landsat"]["names"]
        + CHANNEL_GROUP_DEFINITIONS["dem"]["names"]
        + ["velocity_speed", "velocity_valid"]
    ),
    "comprehensive_v3_landsat_dem_velocity_quality_v2": (
        CHANNEL_GROUP_DEFINITIONS["landsat"]["names"]
        + CHANNEL_GROUP_DEFINITIONS["dem"]["names"]
        + CHANNEL_GROUP_DEFINITIONS["velocity_quality_v2"]["names"]
    ),
    "c02t1_dn_agreement_ldem_v2_baseline": (
        CHANNEL_GROUP_DEFINITIONS["landsat"]["names"]
        + CHANNEL_GROUP_DEFINITIONS["dem"]["names"]
    ),
    "c02t1_dn_agreement_ldem_v2_speed": (
        CHANNEL_GROUP_DEFINITIONS["landsat"]["names"]
        + CHANNEL_GROUP_DEFINITIONS["dem"]["names"]
        + ["velocity_speed", "velocity_valid"]
    ),
    "c02t1_dn_agreement_ldem_v2_quality": (
        CHANNEL_GROUP_DEFINITIONS["landsat"]["names"]
        + CHANNEL_GROUP_DEFINITIONS["dem"]["names"]
        + CHANNEL_GROUP_DEFINITIONS["velocity_quality_v2"]["names"]
    ),
    "comprehensive_v3_landsat_dem_flowacc_velmag": (
        CHANNEL_GROUP_DEFINITIONS["landsat"]["names"]
        + CHANNEL_GROUP_DEFINITIONS["dem"]["names"]
        + ["flow_accumulation", "velocity", "velocity_mask"]
    ),
    "comprehensive_v3_complete_no_hsv": (
        CHANNEL_GROUP_DEFINITIONS["landsat"]["names"]
        + CHANNEL_GROUP_DEFINITIONS["dem"]["names"]
        + CHANNEL_GROUP_DEFINITIONS["spectral_indices"]["names"]
        + CHANNEL_GROUP_DEFINITIONS["velocity"]["names"]
        + CHANNEL_GROUP_DEFINITIONS["physics"]["names"]
    ),
}


def load_config_with_server_paths(config_path, server_name="desktop"):
    slice_config = Dict(yaml.safe_load(open(config_path)))

    servers_cfg = Dict(yaml.safe_load(Path("configs/servers.yaml").read_text()))
    server = servers_cfg[server_name]

    if "image_dir" not in slice_config:
        slice_config.image_dir = server.image_dir
    if "dem_dir" not in slice_config:
        slice_config.dem_dir = server.dem_dir
    if "labels_dir" not in slice_config:
        slice_config.labels_dir = server.labels_dir
    slice_config.out_dir = f"{server.processed_data_path}/{slice_config.output_name}"

    if "velocity_dir" not in slice_config and "velocity_dir" in server:
        slice_config.velocity_dir = server.velocity_dir

    return slice_config


def mask_path_for_tiff(tiff_path: Path) -> Path:
    return tiff_path.with_name(tiff_path.name.replace("tiff_", "mask_", 1))


def compute_model_visible_normalization(
    split_dir: Path, band_names: list[str]
) -> np.ndarray:
    channel_count = len(band_names)
    counts = np.zeros(channel_count, dtype=np.int64)
    sums = np.zeros(channel_count, dtype=np.float64)
    sum_squares = np.zeros(channel_count, dtype=np.float64)
    mins = np.full(channel_count, np.inf, dtype=np.float64)
    maxs = np.full(channel_count, -np.inf, dtype=np.float64)

    velocity_value_names = {
        "velocity",
        "velocity_x",
        "velocity_y",
        "velocity_speed",
        "velocity_count",
        "velocity_error",
        "velocity_relative_error",
    }
    velocity_mask_name = (
        "velocity_valid" if "velocity_valid" in band_names else "velocity_mask"
    )
    velocity_mask_idx = (
        band_names.index(velocity_mask_name)
        if velocity_mask_name in band_names
        else None
    )

    tiff_files = sorted(split_dir.glob("tiff_*.npy"))
    for tiff_file in tqdm(tiff_files, desc=f"Computing stats for {split_dir.name}"):
        mask_file = mask_path_for_tiff(tiff_file)
        if not mask_file.exists():
            raise FileNotFoundError(f"Missing mask for {tiff_file}: {mask_file}")

        data = np.load(tiff_file)
        mask = np.load(mask_file)
        valid = mask != fn.IGNORE_LABEL

        if not np.any(valid):
            continue

        velocity_valid = valid
        if velocity_mask_idx is not None:
            velocity_valid = valid & (data[:, :, velocity_mask_idx] > 0.5)

        for channel_idx, band_name in enumerate(band_names):
            channel_valid = (
                velocity_valid if band_name in velocity_value_names else valid
            )
            if not np.any(channel_valid):
                continue

            values = data[:, :, channel_idx][channel_valid]
            finite = np.isfinite(values)
            if not np.any(finite):
                continue

            values = values[finite].astype(np.float64, copy=False)
            counts[channel_idx] += values.size
            sums[channel_idx] += values.sum()
            sum_squares[channel_idx] += np.square(values).sum()
            mins[channel_idx] = min(mins[channel_idx], values.min())
            maxs[channel_idx] = max(maxs[channel_idx], values.max())

    means = np.zeros(channel_count, dtype=np.float64)
    stds = np.ones(channel_count, dtype=np.float64)
    valid_channels = counts > 0
    means[valid_channels] = sums[valid_channels] / counts[valid_channels]
    variances = np.zeros(channel_count, dtype=np.float64)
    variances[valid_channels] = (
        sum_squares[valid_channels] / counts[valid_channels]
    ) - np.square(means[valid_channels])
    stds[valid_channels] = np.sqrt(np.maximum(variances[valid_channels], 0.0))
    stds[stds < 1e-6] = 1.0

    mins[~valid_channels] = 0.0
    maxs[~valid_channels] = 1.0

    stats = np.asarray((means, stds, mins, maxs), dtype=np.float32)
    return stats


def normalize_slice_for_v3(
    data: np.ndarray, norm_stats: np.ndarray, band_names: list[str]
) -> np.ndarray:
    """Apply train-stat mean/std normalization for v3 packed arrays."""
    mean = norm_stats[0]
    std = norm_stats[1]
    normalized = (data - mean) / std

    no_norm_names = get_no_normalize_channel_names()
    no_norm_mask = np.array([name in no_norm_names for name in band_names])
    if np.any(no_norm_mask):
        normalized[:, :, no_norm_mask] = data[:, :, no_norm_mask]

    velocity_mask_name = (
        "velocity_valid" if "velocity_valid" in band_names else "velocity_mask"
    )
    if velocity_mask_name in band_names:
        velocity_mask_idx = band_names.index(velocity_mask_name)
        velocity_value_indices = [
            idx
            for idx, name in enumerate(band_names)
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
        if velocity_value_indices:
            missing_velocity = normalized[:, :, velocity_mask_idx] <= 0.5
            for channel_idx in velocity_value_indices:
                normalized[:, :, channel_idx][missing_velocity] = 0.0

    return normalized.astype(np.float32, copy=False)


def pack_split_v3(
    split_dir: Path,
    band_names: list[str],
    train_norm_stats: np.ndarray,
    delete_slice_files: bool = True,
) -> dict:
    """Pack generated per-slice files into v3 split-level NCHW arrays."""
    tiff_files = sorted(split_dir.glob("tiff_*.npy"))
    if not tiff_files:
        raise FileNotFoundError(f"No tiff_*.npy files found in {split_dir}")

    first = np.load(tiff_files[0], mmap_mode="r")
    if first.ndim != 3:
        raise ValueError(f"Expected HWC slice, got {first.shape} in {tiff_files[0]}")

    height, width, channels = first.shape
    if channels != len(band_names):
        raise ValueError(
            f"Band metadata has {len(band_names)} bands but slices have {channels}"
        )

    x_path = split_dir / "X.npy"
    y_path = split_dir / "y.npy"
    x_arr = np.lib.format.open_memmap(
        x_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(tiff_files), channels, height, width),
    )
    y_arr = np.lib.format.open_memmap(
        y_path, mode="w+", dtype=np.uint8, shape=(len(tiff_files), height, width)
    )

    packed_records = []
    mask_files = []
    for idx, tiff_file in enumerate(tqdm(tiff_files, desc=f"Packing {split_dir.name}")):
        mask_file = mask_path_for_tiff(tiff_file)
        if not mask_file.exists():
            raise FileNotFoundError(f"Missing mask for {tiff_file}: {mask_file}")

        data = np.load(tiff_file)
        mask = np.load(mask_file).astype(np.uint8, copy=False)
        normalized = normalize_slice_for_v3(data, train_norm_stats, band_names)

        x_arr[idx] = np.transpose(normalized, (2, 0, 1))
        y_arr[idx] = mask
        packed_records.append(
            {
                "index": idx,
                "source_tiff_file": tiff_file.name,
                "source_mask_file": mask_file.name,
            }
        )
        mask_files.append(mask_file)

    x_arr.flush()
    y_arr.flush()

    manifest = {
        "format": "comprehensive_v3",
        "layout": "NCHW",
        "normalized": True,
        "normalization": "mean-std",
        "x": x_path.name,
        "y": y_path.name,
        "num_samples": len(tiff_files),
        "shape": [len(tiff_files), channels, height, width],
        "label_shape": [len(tiff_files), height, width],
        "dtype": {"x": "float32", "y": "uint8"},
        "records": packed_records,
    }
    with open(split_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if delete_slice_files:
        for path in [*tiff_files, *mask_files]:
            path.unlink()

    return manifest


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def load_full_dataset_config(source_dir: Path) -> dict:
    metadata_path = source_dir / "band_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing band metadata: {metadata_path}")
    with open(metadata_path, "r") as f:
        return json.load(f)


def pack_recipe_split(
    source_split_dir: Path,
    target_split_dir: Path,
    channel_indices: list[int],
    channel_names: list[str],
    x_dtype: np.dtype,
) -> dict:
    source_x_path = source_split_dir / "X.npy"
    source_y_path = source_split_dir / "y.npy"
    if not source_x_path.exists() or not source_y_path.exists():
        raise FileNotFoundError(f"Missing source X/y files in {source_split_dir}")

    source_x = np.load(source_x_path, mmap_mode="r")
    source_y = np.load(source_y_path, mmap_mode="r")
    if source_x.ndim != 4 or source_y.ndim != 3:
        raise ValueError(
            f"Expected source X [N,C,H,W] and y [N,H,W], got "
            f"{source_x.shape} and {source_y.shape}"
        )

    target_split_dir.mkdir(parents=True, exist_ok=True)
    target_x_path = target_split_dir / "X.npy"
    target_y_path = target_split_dir / "y.npy"

    n, _c, h, w = source_x.shape
    target_x = np.lib.format.open_memmap(
        target_x_path,
        mode="w+",
        dtype=x_dtype,
        shape=(n, len(channel_indices), h, w),
    )

    for sample_idx in tqdm(
        range(n),
        desc=f"Packing {target_split_dir.parent.name}/{target_split_dir.name}",
    ):
        target_x[sample_idx] = source_x[sample_idx, channel_indices, :, :].astype(
            x_dtype, copy=False
        )
    target_x.flush()

    label_storage = link_or_copy(source_y_path, target_y_path)

    manifest = {
        "format": "comprehensive_v3_packed",
        "layout": "NCHW",
        "normalized": True,
        "normalization": "mean-std",
        "x": target_x_path.name,
        "y": target_y_path.name,
        "num_samples": int(n),
        "shape": [int(n), len(channel_indices), int(h), int(w)],
        "label_shape": [
            int(source_y.shape[0]),
            int(source_y.shape[1]),
            int(source_y.shape[2]),
        ],
        "dtype": {"x": str(np.dtype(x_dtype)), "y": str(source_y.dtype)},
        "channels": channel_names,
        "source_channel_indices": channel_indices,
        "source_split": str(source_split_dir),
        "label_storage": label_storage,
    }
    with open(target_split_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def pack_recipe_dataset(
    source_dir: Path,
    output_root: Path,
    recipe_name: str,
    recipe_channels: list[str],
    x_dtype: np.dtype,
    dry_run: bool = False,
) -> None:
    source_band_names = load_band_names(source_dir).tolist()
    missing = [name for name in recipe_channels if name not in source_band_names]
    if missing:
        raise ValueError(f"{recipe_name} requested missing channels: {missing}")

    channel_indices = [source_band_names.index(name) for name in recipe_channels]
    target_dir = output_root / recipe_name

    print(f"\nRecipe: {recipe_name}")
    print(f"  Source: {source_dir}")
    print(f"  Target: {target_dir}")
    print(f"  Channels ({len(recipe_channels)}): {recipe_channels}")
    print(f"  Source indices: {channel_indices}")
    if dry_run:
        return

    remove_and_create(target_dir)

    source_metadata = load_full_dataset_config(source_dir)
    band_metadata = {
        "band_names": recipe_channels,
        "num_bands": len(recipe_channels),
        "source_dataset": source_dir.name,
        "source_channel_indices": channel_indices,
        "source_band_names": source_band_names,
        "source_config": source_metadata.get("config", {}),
        "x_dtype": str(np.dtype(x_dtype)),
    }
    with open(target_dir / "band_metadata.json", "w") as f:
        json.dump(band_metadata, f, indent=2)

    source_norm = source_dir / "normalize_train.npy"
    if source_norm.exists():
        norm_arr = np.load(source_norm)
        np.save(target_dir / "normalize_train.npy", norm_arr[:, channel_indices])

    for metadata_file in [
        "dataset_statistics.json",
        "slice_meta.csv",
        "skipped_slices_meta.csv",
    ]:
        src = source_dir / metadata_file
        if src.exists():
            shutil.copy2(src, target_dir / metadata_file)

    split_manifests = {}
    label_hashes = {}
    for split in ["train", "val", "test"]:
        split_manifests[split] = pack_recipe_split(
            source_dir / split,
            target_dir / split,
            channel_indices,
            recipe_channels,
            x_dtype,
        )
        label_hashes[split] = sha256_file(target_dir / split / "y.npy")

    recipe_manifest = {
        "format": "comprehensive_v3_packed",
        "recipe_name": recipe_name,
        "source_dataset": source_dir.name,
        "source_path": str(source_dir),
        "channels": recipe_channels,
        "source_channel_indices": channel_indices,
        "x_dtype": str(np.dtype(x_dtype)),
        "splits": split_manifests,
        "label_sha256": label_hashes,
    }
    with open(target_dir / "recipe_manifest.json", "w") as f:
        json.dump(recipe_manifest, f, indent=2)


def pack_recipe_datasets(
    source_dir: Path,
    output_root: Path,
    recipe_names: list[str],
    x_dtype: np.dtype,
    dry_run: bool = False,
) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source full dataset does not exist: {source_dir}")
    for recipe_name in recipe_names:
        if recipe_name not in PACKED_RECIPES:
            valid = ", ".join(sorted(PACKED_RECIPES))
            raise ValueError(f"Unknown recipe '{recipe_name}'. Valid recipes: {valid}")
        pack_recipe_dataset(
            source_dir=source_dir,
            output_root=output_root,
            recipe_name=recipe_name,
            recipe_channels=PACKED_RECIPES[recipe_name],
            x_dtype=x_dtype,
            dry_run=dry_run,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slice and preprocess glacier data")
    parser.add_argument(
        "--server",
        required=True,
        help="Server name from configs/servers.yaml (must be specified explicitly)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/datasets/comprehensive_v3.yaml",
        help="Path to preprocessing config file",
    )
    parser.add_argument(
        "--save-skipped-visualizations",
        action="store_true",
        help="Save PNG visualizations of skipped slices (overrides config file)",
    )
    parser.add_argument(
        "--regenerate-full",
        action="store_true",
        help="Regenerate the canonical full dataset from raw source files.",
    )
    parser.add_argument(
        "--source-dataset",
        type=str,
        default=None,
        help="Existing full dataset to pack from. Defaults to config output_name.",
    )
    parser.add_argument(
        "--recipes",
        type=str,
        default=",".join(PACKED_RECIPES.keys()),
        help="Comma-separated packed recipe dataset names to create, or 'all'.",
    )
    parser.add_argument(
        "--packed-dtype",
        choices=["float32", "float16"],
        default="float32",
        help="Feature dtype for packed recipe X.npy arrays.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print recipe packing actions without writing datasets.",
    )
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    warnings.filterwarnings("ignore")

    conf = load_config_with_server_paths(args.config, args.server)

    if args.save_skipped_visualizations:
        conf.save_skipped_visualizations = True
    else:
        conf.save_skipped_visualizations = conf.get(
            "save_skipped_visualizations", False
        )

    if not args.regenerate_full:
        source_dataset = args.source_dataset or conf.output_name
        servers_cfg = Dict(yaml.safe_load(Path("configs/servers.yaml").read_text()))
        output_root = Path(servers_cfg[args.server].processed_data_path)
        source_dir = output_root / source_dataset
        recipe_names = (
            list(PACKED_RECIPES.keys())
            if args.recipes == "all"
            else [name.strip() for name in args.recipes.split(",") if name.strip()]
        )
        pack_recipe_datasets(
            source_dir=source_dir,
            output_root=output_root,
            recipe_names=recipe_names,
            x_dtype=np.dtype(args.packed_dtype),
            dry_run=args.dry_run,
        )
        print("\nPacked recipe generation completed successfully.")
        raise SystemExit(0)

    saved_df = pd.DataFrame(
        columns=[
            "Landsat ID",
            "Image",
            "Slice",
            "Background",
            "Clean Ice",
            "Debris",
            "Masked",
            "Background Percentage",
            "Clean Ice Percentage",
            "Debris Percentage",
            "Masked Percentage",
            "split",
        ]
    )

    skipped_df = pd.DataFrame(
        columns=[
            "Landsat ID",
            "Image",
            "Slice",
            "Background",
            "Clean Ice",
            "Debris",
            "Masked",
            "Background Percentage",
            "Clean Ice Percentage",
            "Debris Percentage",
            "Masked Percentage",
            "split",
        ]
    )

    images = sorted(Path(conf.image_dir).glob("*.tif"))
    idx = np.random.permutation(len(images))
    splits = {
        "test": sorted([images[i] for i in idx[: int(conf.test * len(images))]]),
        "val": sorted(
            [
                images[i]
                for i in idx[
                    int(conf.test * len(images)) : int(
                        (conf.test + conf.val) * len(images)
                    )
                ]
            ]
        ),
        "train": sorted(
            [images[i] for i in idx[int((conf.test + conf.val) * len(images)) :]]
        ),
    }
    print(
        f"Splits: test ({len(splits['test'])}), val ({len(splits['val'])}), train ({len(splits['train'])})"
    )
    print("Output will be in", conf.out_dir)
    labels = fn.read_shp(Path(conf.labels_dir) / "HKH_CIDC_5basins_all.shp")
    remove_and_create(conf.out_dir)

    band_names_metadata = None

    split_norm_stats = {}
    with tqdm(total=1, desc="temp") as pbar:
        for split, meta in splits.items():
            savepath = Path(conf["out_dir"]) / split
            slice_conf = dict(conf)
            if split == "test":
                slice_conf["overlap"] = 0
            fn_process = partial(
                fn.save_slices, labels=labels, savepath=savepath, **slice_conf
            )
            remove_and_create(savepath)

            pbar.set_description(f"Processing dataset {split}")
            pbar.reset(len(meta))
            cores = multiprocessing.cpu_count()
            workers = max(1, int(cores * 0.75))
            print(f"Using {workers}/{cores} CPU cores")
            with multiprocessing.Pool(workers) as pool:
                for result in pool.istarmap(fn_process, enumerate(meta)):
                    _mu, _s, _mi, _ma, df_rows, skipped_rows, band_names = result
                    for row in df_rows:
                        saved_df.loc[len(saved_df.index)] = row
                    for row in skipped_rows:
                        skipped_df.loc[len(skipped_df.index)] = row

                    if band_names_metadata is None and band_names is not None:
                        band_names_metadata = band_names

                    pbar.update(1)

            if band_names_metadata is None:
                raise ValueError("No band metadata returned during preprocessing.")

            norm_stats = compute_model_visible_normalization(
                savepath, band_names_metadata
            )
            split_norm_stats[split] = norm_stats
            np.save(
                Path(conf["out_dir"]) / f"normalize_{split}",
                norm_stats,
            )
            print(f"Saved normalization stats for {split}")

    train_norm_stats = split_norm_stats["train"]
    v3_manifests = {}
    for split in ["train", "val", "test"]:
        v3_manifests[split] = pack_split_v3(
            Path(conf["out_dir"]) / split,
            band_names_metadata,
            train_norm_stats,
            delete_slice_files=True,
        )

    saved_df.to_csv(
        Path(conf["out_dir"]) / "slice_meta.csv", encoding="utf-8", index=False
    )
    skipped_df.to_csv(
        Path(conf["out_dir"]) / "skipped_slices_meta.csv", encoding="utf-8", index=False
    )

    statistics = {}

    print("\n" + "=" * 80)
    print("DATASET STATISTICS BY SPLIT")
    print("=" * 80)

    for split in ["train", "val", "test"]:
        split_df = saved_df[saved_df["split"] == split]

        if len(split_df) == 0:
            continue

        total_bg = split_df["Background"].sum()
        total_ci = split_df["Clean Ice"].sum()
        total_debris = split_df["Debris"].sum()
        total_masked = split_df["Masked"].sum()
        total_valid = total_bg + total_ci + total_debris
        total_all = total_valid + total_masked

        pct_bg = (total_bg / total_all) * 100 if total_all > 0 else 0
        pct_ci = (total_ci / total_all) * 100 if total_all > 0 else 0
        pct_debris = (total_debris / total_all) * 100 if total_all > 0 else 0
        pct_masked = (total_masked / total_all) * 100 if total_all > 0 else 0

        pct_ci_valid = (total_ci / total_valid) * 100 if total_valid > 0 else 0
        pct_debris_valid = (total_debris / total_valid) * 100 if total_valid > 0 else 0
        pct_bg_valid = (total_bg / total_valid) * 100 if total_valid > 0 else 0
        statistics[split] = {
            "images": int(len(split_df["Image"].unique())),
            "slices": int(len(split_df)),
            "total_pixels": int(total_all),
            "pixels": {
                "background": int(total_bg),
                "clean_ice": int(total_ci),
                "debris_ice": int(total_debris),
                "masked_invalid": int(total_masked),
            },
            "percentages_all_pixels": {
                "background": float(pct_bg),
                "clean_ice": float(pct_ci),
                "debris_ice": float(pct_debris),
                "masked_invalid": float(pct_masked),
            },
            "percentages_valid_pixels": {
                "background": float(pct_bg_valid),
                "clean_ice": float(pct_ci_valid),
                "debris_ice": float(pct_debris_valid),
            },
        }

        print(f"\n{split.upper()} SET:")
        print(f"  Images: {statistics[split]['images']}")
        print(f"  Slices: {statistics[split]['slices']}")
        print(f"  Total pixels: {total_all:,}")
        print("\n  Pixel Distribution (all pixels):")
        print(f"    Background:        {total_bg:12,} ({pct_bg:5.2f}%)")
        print(f"    Clean Ice:         {total_ci:12,} ({pct_ci:5.2f}%)")
        print(f"    Debris Ice:        {total_debris:12,} ({pct_debris:5.2f}%)")
        print(f"    Masked/Invalid:    {total_masked:12,} ({pct_masked:5.2f}%)")
        print("\n  Pixel Distribution (valid pixels only):")
        print(f"    Background:        {pct_bg_valid:5.2f}%")
        print(f"    Clean Ice:         {pct_ci_valid:5.2f}%")
        print(f"    Debris Ice:        {pct_debris_valid:5.2f}%")

    statistics["skipped"] = {}

    print("\n" + "=" * 80)
    print("SKIPPED SLICES STATISTICS (due to filtering)")
    print("=" * 80)
    print(
        f"Filter threshold: {conf.filter * 100}% minimum glacier pixels (CI + Debris)"
    )

    for split in ["train", "val", "test"]:
        split_skipped_df = skipped_df[skipped_df["split"] == split]

        if len(split_skipped_df) == 0:
            continue

        total_bg = split_skipped_df["Background"].sum()
        total_ci = split_skipped_df["Clean Ice"].sum()
        total_debris = split_skipped_df["Debris"].sum()
        total_masked = split_skipped_df["Masked"].sum()
        total_valid = total_bg + total_ci + total_debris
        total_all = total_valid + total_masked

        pct_bg = (total_bg / total_all) * 100 if total_all > 0 else 0
        pct_ci = (total_ci / total_all) * 100 if total_all > 0 else 0
        pct_debris = (total_debris / total_all) * 100 if total_all > 0 else 0
        pct_masked = (total_masked / total_all) * 100 if total_all > 0 else 0

        pct_ci_valid = (total_ci / total_valid) * 100 if total_valid > 0 else 0
        pct_debris_valid = (total_debris / total_valid) * 100 if total_valid > 0 else 0
        pct_bg_valid = (total_bg / total_valid) * 100 if total_valid > 0 else 0

        statistics["skipped"][split] = {
            "slices": int(len(split_skipped_df)),
            "total_pixels": int(total_all),
            "pixels": {
                "background": int(total_bg),
                "clean_ice": int(total_ci),
                "debris_ice": int(total_debris),
                "masked_invalid": int(total_masked),
            },
            "percentages_all_pixels": {
                "background": float(pct_bg),
                "clean_ice": float(pct_ci),
                "debris_ice": float(pct_debris),
                "masked_invalid": float(pct_masked),
            },
            "percentages_valid_pixels": {
                "background": float(pct_bg_valid),
                "clean_ice": float(pct_ci_valid),
                "debris_ice": float(pct_debris_valid),
            },
        }

        print(f"\n{split.upper()} SET (SKIPPED):")
        print(f"  Skipped slices: {len(split_skipped_df)}")
        print(f"  Total pixels: {total_all:,}")
        print("\n  Pixel Distribution (all pixels):")
        print(f"    Background:        {total_bg:12,} ({pct_bg:5.2f}%)")
        print(f"    Clean Ice:         {total_ci:12,} ({pct_ci:5.2f}%)")
        print(f"    Debris Ice:        {total_debris:12,} ({pct_debris:5.2f}%)")
        print(f"    Masked/Invalid:    {total_masked:12,} ({pct_masked:5.2f}%)")
        print("\n  Pixel Distribution (valid pixels only):")
        print(f"    Background:        {pct_bg_valid:5.2f}%")
        print(f"    Clean Ice:         {pct_ci_valid:5.2f}%")
        print(f"    Debris Ice:        {pct_debris_valid:5.2f}%")

    total_kept_slices = sum(
        stats["slices"]
        for stats in statistics.values()
        if isinstance(stats, dict) and "slices" in stats
    )
    total_skipped_slices = sum(
        stats["slices"]
        for stats in statistics.get("skipped", {}).values()
        if isinstance(stats, dict)
    )
    total_all_slices = total_kept_slices + total_skipped_slices
    kept_percentage = (
        (total_kept_slices / total_all_slices * 100) if total_all_slices > 0 else 0
    )
    skipped_percentage = (
        (total_skipped_slices / total_all_slices * 100) if total_all_slices > 0 else 0
    )

    statistics["summary"] = {
        "total_images": sum(
            stats["images"]
            for stats in statistics.values()
            if isinstance(stats, dict) and "images" in stats
        ),
        "total_slices_kept": total_kept_slices,
        "total_slices_skipped": total_skipped_slices,
        "total_slices_processed": total_all_slices,
        "kept_percentage": float(kept_percentage),
        "skipped_percentage": float(skipped_percentage),
        "split_ratios": {
            "test": float(conf.test),
            "val": float(conf.val),
            "train": 1.0 - float(conf.test) - float(conf.val),
        },
        "config": {
            "window_size": conf.window_size,
            "overlap": int(conf.overlap),
            "filter_threshold": float(conf.filter),
        },
    }

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY:")
    print(f"  Total slices processed: {total_all_slices:,}")
    print(f"  Slices kept:            {total_kept_slices:,} ({kept_percentage:.2f}%)")
    print(
        f"  Slices skipped:         {total_skipped_slices:,} ({skipped_percentage:.2f}%)"
    )
    print("=" * 80)

    stats_path = Path(conf["out_dir"]) / "dataset_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(statistics, f, indent=2)

    band_metadata_path = None
    if band_names_metadata is not None:
        band_metadata = {
            "band_names": band_names_metadata,
            "num_bands": len(band_names_metadata),
            "config": {
                "use_dem": bool(hasattr(conf, "dem_dir") and conf.dem_dir),
                "use_velocity": bool(conf.get("add_velocity", False)),
                "use_indices": {
                    "ndvi": bool(conf.get("add_ndvi", False)),
                    "ndwi": bool(conf.get("add_ndwi", False)),
                    "ndsi": bool(conf.get("add_ndsi", False)),
                    "hsv": bool(conf.get("add_hsv", False)),
                },
                "use_physics": bool(conf.get("physics_res") not in [None, "None"]),
            },
        }
        band_metadata_path = Path(conf["out_dir"]) / "band_metadata.json"
        with open(band_metadata_path, "w") as f:
            json.dump(band_metadata, f, indent=2)
        print("\nGenerated band metadata:")
        print(f"  Bands: {band_names_metadata}")
        print(f"  Total: {len(band_names_metadata)} channels")

    print("\n" + "=" * 80)
    print("FILES SAVED:")
    print(f"  Dataset statistics:     {stats_path}")
    if band_metadata_path is not None:
        print(f"  Band metadata:          {band_metadata_path}")
    print(f"  Kept slices metadata:   {Path(conf['out_dir']) / 'slice_meta.csv'}")
    print(
        f"  Skipped slices metadata: {Path(conf['out_dir']) / 'skipped_slices_meta.csv'}"
    )
    print("\nProcessing completed successfully!")
    print("=" * 80)
