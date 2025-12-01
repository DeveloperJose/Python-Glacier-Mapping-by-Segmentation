#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import multiprocessing
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
import glacier_mapping.utils.parallel

import matplotlib

matplotlib.use("Agg")


def remove_and_create(dirpath):
    """Remove and recreate directory."""
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath)


# Monkey patch istarmap into multiprocessing.Pool
multiprocessing.pool.Pool.istarmap = glacier_mapping.utils.parallel.istarmap


def load_config_with_server_paths(config_path, server_name="desktop"):
    """Load config and construct absolute paths from servers.yaml"""
    # Load slice config
    slice_config = Dict(yaml.safe_load(open(config_path)))

    # Load server paths
    servers_cfg = Dict(yaml.safe_load(Path("conf/servers.yaml").read_text()))
    server = servers_cfg[server_name]  # defaults to desktop

    # Use absolute paths directly from servers.yaml
    slice_config.image_dir = server.image_dir
    slice_config.dem_dir = server.dem_dir
    slice_config.labels_dir = server.labels_dir
    slice_config.out_dir = f"{server.processed_data_path}/{slice_config.output_name}"

    return slice_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slice and preprocess glacier data")
    parser.add_argument(
        "--server",
        default="desktop",
        choices=["desktop", "bilbo", "frodo"],
        help="Server name (default: desktop)",
    )
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    warnings.filterwarnings("ignore")

    # Load config with server paths
    conf = load_config_with_server_paths(
        "./conf/slice_and_preprocess.yaml", args.server
    )

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

    images = sorted(os.listdir(Path(conf.image_dir)))
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
    print("Splits:", splits)
    # breakpoint()
    labels = fn.read_shp(Path(conf.labels_dir) / "HKH_CIDC_5basins_all.shp")
    remove_and_create(conf.out_dir)

    with tqdm(total=1, desc="temp") as pbar:
        for split, meta in splits.items():
            means, stds, mins, maxs = [], [], [], []
            savepath = Path(conf["out_dir"]) / split
            fn_process = partial(
                fn.save_slices, labels=labels, savepath=savepath, **conf
            )
            remove_and_create(savepath)

            pbar.set_description(f"Processing dataset {split}")
            pbar.reset(len(meta))
            cores = multiprocessing.cpu_count()
            workers = max(1, int(cores * 0.75))
            print(f"Using {workers}/{cores} CPU cores")
            with multiprocessing.Pool(workers) as pool:
                for result in pool.istarmap(fn_process, enumerate(meta)):
                    mu, s, mi, ma, df_rows, skipped_rows = result
                    means.append(mu)
                    stds.append(s)
                    mins.append(mi)
                    maxs.append(ma)
                    for row in df_rows:
                        saved_df.loc[len(saved_df.index)] = row
                    for row in skipped_rows:
                        skipped_df.loc[len(skipped_df.index)] = row
                    pbar.update(1)

            means_agg = np.mean(np.asarray(means), axis=0)
            stds_agg = np.mean(np.asarray(stds), axis=0)
            mins_agg = np.min(np.asarray(mins), axis=0)
            maxs_agg = np.max(np.asarray(maxs), axis=0)

            np.save(
                Path(conf["out_dir"]) / f"normalize_{split}",
                np.asarray((means_agg, stds_agg, mins_agg, maxs_agg)),
            )
            print(f"Saved normalization stats for {split}")

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

        # Print statistics
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

    # ============================================================================
    # Compute statistics for SKIPPED slices
    # ============================================================================
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

        # Calculate total pixels in skipped slices
        total_bg = split_skipped_df["Background"].sum()
        total_ci = split_skipped_df["Clean Ice"].sum()
        total_debris = split_skipped_df["Debris"].sum()
        total_masked = split_skipped_df["Masked"].sum()
        total_valid = total_bg + total_ci + total_debris
        total_all = total_valid + total_masked

        # Percentages
        pct_bg = (total_bg / total_all) * 100 if total_all > 0 else 0
        pct_ci = (total_ci / total_all) * 100 if total_all > 0 else 0
        pct_debris = (total_debris / total_all) * 100 if total_all > 0 else 0
        pct_masked = (total_masked / total_all) * 100 if total_all > 0 else 0

        pct_ci_valid = (total_ci / total_valid) * 100 if total_valid > 0 else 0
        pct_debris_valid = (total_debris / total_valid) * 100 if total_valid > 0 else 0
        pct_bg_valid = (total_bg / total_valid) * 100 if total_valid > 0 else 0

        # Store statistics
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

        # Print statistics
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

    # Add summary statistics
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

    # Save statistics to JSON
    stats_path = Path(conf["out_dir"]) / "dataset_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(statistics, f, indent=2)

    print("\n" + "=" * 80)
    print("FILES SAVED:")
    print(f"  Dataset statistics:     {stats_path}")
    print(f"  Kept slices metadata:   {Path(conf['out_dir']) / 'slice_meta.csv'}")
    print(
        f"  Skipped slices metadata: {Path(conf['out_dir']) / 'skipped_slices_meta.csv'}"
    )
    print("\nProcessing completed successfully!")
    print("=" * 80)
