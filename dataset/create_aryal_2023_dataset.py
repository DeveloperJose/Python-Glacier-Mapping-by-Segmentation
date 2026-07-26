#!/usr/bin/env python3
"""Reproduce Aryal (2022 thesis / 2023 paper) HKH preprocessing.

This intentionally preserves published-code behavior that differs from the modern
pipeline: seed-42 cell split, overlap on every split, square-height column loop,
10% all-glacier filtering, image-level averaged normalization statistics, and
invalid Landsat pixels represented as ignored labels at load time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import rasterio
import yaml
from tqdm import tqdm

from glacier_mapping.data.slice import get_mask

WINDOW_SIZE = 512
OVERLAP = 64
STEP = WINDOW_SIZE - OVERLAP
FILTER_FRACTION = 0.1
IGNORE_LABEL = 255
BAND_NAMES = ["B1", "B2", "B3", "B4", "B5", "B6_VCID1", "B6_VCID2", "B7"]
UPSTREAM_COMMIT = "378c053194285e5166526cb1ba981ef82a938fde"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pad_2d(array: np.ndarray, fill: int = 0) -> np.ndarray:
    result = np.full((WINDOW_SIZE, WINDOW_SIZE), fill, dtype=array.dtype)
    result[: array.shape[0], : array.shape[1]] = array
    return result


def pad_hwc(array: np.ndarray) -> np.ndarray:
    result = np.zeros((WINDOW_SIZE, WINDOW_SIZE, array.shape[2]), dtype=array.dtype)
    result[: array.shape[0], : array.shape[1], :] = array
    return result


def build_split(images: list[Path]) -> dict[str, list[Path]]:
    # np.random.seed(42); np.random.permutation(...) from upstream script.
    permutation = np.random.RandomState(42).permutation(len(images))
    test_end = int(0.2 * len(images))
    val_end = int(0.3 * len(images))
    return {
        "test": sorted(images[index] for index in permutation[:test_end]),
        "val": sorted(images[index] for index in permutation[test_end:val_end]),
        "train": sorted(images[index] for index in permutation[val_end:]),
    }


def multiclass_mask(image_path: Path, labels) -> np.ndarray:
    one_hot = get_mask(image_path, labels)
    mask = np.zeros(one_hot.shape[:2], dtype=np.uint8)
    for index in range(one_hot.shape[2]):
        mask[one_hot[:, :, index] == 1] = index + 1
    return mask


def process_image(
    image_path: Path,
    dem_path: Path,
    labels,
    split_dir: Path,
    image_number: int,
    split: str,
) -> tuple[list[dict], list[tuple[Path, Path]], np.ndarray]:
    if not dem_path.exists():
        raise FileNotFoundError(f"Missing aligned DEM: {dem_path}")

    with rasterio.open(image_path) as image, rasterio.open(dem_path) as dem:
        if (image.width, image.height, image.crs, image.transform) != (
            dem.width,
            dem.height,
            dem.crs,
            dem.transform,
        ):
            raise ValueError(f"Landsat/DEM grids differ: {image_path.name}")
        landsat = np.nan_to_num(np.transpose(image.read(), (1, 2, 0))).astype(
            np.float32
        )

    if landsat.shape[2] != 8:
        raise ValueError(
            f"Expected 8 Landsat bands in {image_path}, got {landsat.shape}"
        )

    mask = multiclass_mask(image_path, labels)
    image_stats = np.asarray(
        [
            landsat.mean(axis=(0, 1)),
            landsat.std(axis=(0, 1)),
            landsat.min(axis=(0, 1)),
            landsat.max(axis=(0, 1)),
        ],
        dtype=np.float32,
    )

    rows: list[dict] = []
    records: list[tuple[Path, Path]] = []
    slice_number = 0

    # Preserve upstream bug: columns iterate over image height, not width.
    for row in range(0, landsat.shape[0], STEP):
        for column in range(0, landsat.shape[0], STEP):
            y_slice = pad_2d(
                mask[row : row + WINDOW_SIZE, column : column + WINDOW_SIZE]
            )
            glacier_fraction = np.count_nonzero(y_slice) / y_slice.size
            if glacier_fraction < FILTER_FRACTION:
                slice_number += 1
                continue

            x_unpadded = landsat[
                row : row + WINDOW_SIZE, column : column + WINDOW_SIZE, :
            ]
            x_slice = pad_hwc(x_unpadded)

            # Upstream image filter uses all generated channels. Lat/lon channels
            # make every real raster pixel nonzero, so this is exactly real extent.
            real_pixels = x_unpadded.shape[0] * x_unpadded.shape[1]
            if real_pixels / (WINDOW_SIZE * WINDOW_SIZE) < 0.5:
                slice_number += 1
                continue

            # Upstream zeroes every channel when the first seven Landsat bands sum
            # to zero. Loader then masks pixels where all selected bands are zero.
            invalid = np.sum(x_slice[:, :, :7], axis=2) == 0
            x_slice[invalid] = 0
            y_packed = y_slice.copy()
            y_packed[invalid] = IGNORE_LABEL

            x_path = split_dir / f"tiff_{image_number}_slice_{slice_number}.npy"
            y_path = split_dir / f"mask_{image_number}_slice_{slice_number}.npy"
            np.save(x_path, x_slice)
            np.save(y_path, y_packed)
            records.append((x_path, y_path))

            # Upstream slice_meta calls pixels masked only when every generated
            # channel is zero. Real pixels have nonzero lat/lon auxiliaries, so
            # this metadata mask is padding only. Landsat gaps are masked later
            # by the loader and are encoded as 255 in y_packed above.
            padded = np.ones((WINDOW_SIZE, WINDOW_SIZE), dtype=bool)
            padded[: x_unpadded.shape[0], : x_unpadded.shape[1]] = False
            bg = int(np.count_nonzero((y_slice == 0) & ~padded))
            ci = int(np.count_nonzero((y_slice == 1) & ~padded))
            debris = int(np.count_nonzero((y_slice == 2) & ~padded))
            masked = int(np.count_nonzero(padded))
            total = bg + ci + debris + masked
            rows.append(
                {
                    "Landsat ID": image_path.name,
                    "Image": image_number,
                    "Slice": slice_number,
                    "Background": bg,
                    "Clean Ice": ci,
                    "Debris": debris,
                    "Masked": masked,
                    "Background Percentage": bg / total,
                    "Clean Ice Percentage": ci / total,
                    "Debris Percentage": debris / total,
                    "Masked Percentage": masked / total,
                    "split": split,
                }
            )
            slice_number += 1

    return rows, records, image_stats


def pack_split(
    split_dir: Path,
    records: list[tuple[Path, Path]],
    train_stats: np.ndarray,
) -> dict:
    if not records:
        raise RuntimeError(f"No retained slices in {split_dir}")

    sample_count = len(records)
    x_out = np.lib.format.open_memmap(
        split_dir / "X.npy",
        mode="w+",
        dtype=np.float32,
        shape=(sample_count, 8, WINDOW_SIZE, WINDOW_SIZE),
    )
    y_out = np.lib.format.open_memmap(
        split_dir / "y.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(sample_count, WINDOW_SIZE, WINDOW_SIZE),
    )

    mean, std = train_stats[0], train_stats[1]
    if np.any(std == 0):
        raise ValueError("Aryal train normalization contains zero standard deviation")

    manifest_records = []
    for index, (x_path, y_path) in enumerate(
        tqdm(records, desc=f"Packing {split_dir.name}")
    ):
        raw = np.load(x_path)
        normalized = (raw - mean) / std
        x_out[index] = np.transpose(normalized, (2, 0, 1))
        y_out[index] = np.load(y_path)
        manifest_records.append(
            {
                "index": index,
                "source_tiff_file": x_path.name,
                "source_mask_file": y_path.name,
            }
        )
        x_path.unlink()
        y_path.unlink()

    x_out.flush()
    y_out.flush()
    manifest = {
        "format": "aryal_2023_packed",
        "layout": "NCHW",
        "normalized": True,
        "normalization": "aryal_image_mean_of_means_mean_std",
        "num_samples": sample_count,
        "shape": [sample_count, 8, WINDOW_SIZE, WINDOW_SIZE],
        "label_shape": [sample_count, WINDOW_SIZE, WINDOW_SIZE],
        "dtype": {"x": "float32", "y": "uint8"},
        "records": manifest_records,
    }
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="desktop")
    parser.add_argument("--output-name", default="aryal_2023_landsat8_reproduction")
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(
            "/home/devj/local-arch/data/HKH_raw/labels/HKH_CIDC_5basins_all.shp"
        ),
    )
    parser.add_argument(
        "--max-images-per-split",
        type=int,
        default=None,
        help="Fixture mode only; process first N images in each established split.",
    )
    args = parser.parse_args()

    servers = yaml.safe_load(Path("configs/servers.yaml").read_text())
    server = servers[args.server]
    image_dir = Path(server["image_dir"])
    dem_dir = Path(server["dem_dir"])
    output_dir = Path(server["processed_data_path"]) / args.output_name
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite immutable reproduction dataset: {output_dir}"
        )

    images = sorted(image_dir.glob("*.tif"))
    if len(images) != 202:
        raise ValueError(f"Expected 202 Aryal Landsat inputs, found {len(images)}")
    missing_dems = [path.name for path in images if not (dem_dir / path.name).exists()]
    if missing_dems:
        raise FileNotFoundError(f"Missing DEM pairs: {missing_dems}")

    labels = pyogrio.read_dataframe(args.labels, on_invalid="fix")
    if len(labels) != 30096:
        raise ValueError(f"Expected 30096 Aryal label features, found {len(labels)}")
    classes = sorted(labels["Glaciers"].unique().tolist())
    if classes != ["Clean Ice", "Debris covered"]:
        raise ValueError(f"Unexpected label classes: {classes}")

    splits = build_split(images)
    full_split_names = {
        key: [path.name for path in value] for key, value in splits.items()
    }
    output_dir.mkdir(parents=True)

    all_rows: list[dict] = []
    split_records: dict[str, list[tuple[Path, Path]]] = {}
    split_image_stats: dict[str, list[np.ndarray]] = {}

    for split in ["test", "val", "train"]:
        split_dir = output_dir / split
        split_dir.mkdir()
        selected = splits[split]
        if args.max_images_per_split is not None:
            selected = selected[: args.max_images_per_split]
        split_records[split] = []
        split_image_stats[split] = []
        for image_number, image_path in enumerate(
            tqdm(selected, desc=f"Slicing {split}")
        ):
            rows, records, stats = process_image(
                image_path,
                dem_dir / image_path.name,
                labels,
                split_dir,
                image_number,
                split,
            )
            all_rows.extend(rows)
            split_records[split].extend(records)
            split_image_stats[split].append(stats)

    normalization = {}
    for split, stats_list in split_image_stats.items():
        stats = np.stack(stats_list)
        aggregate = np.asarray(
            [
                stats[:, 0].mean(axis=0),
                stats[:, 1].mean(axis=0),
                stats[:, 2].min(axis=0),
                stats[:, 3].max(axis=0),
            ],
            dtype=np.float32,
        )
        normalization[split] = aggregate
        np.save(output_dir / f"normalize_{split}.npy", aggregate)

    manifests = {}
    for split in ["train", "val", "test"]:
        manifests[split] = pack_split(
            output_dir / split, split_records[split], normalization["train"]
        )

    pd.DataFrame(all_rows).to_csv(output_dir / "slice_meta.csv", index=False)
    (output_dir / "band_metadata.json").write_text(
        json.dumps(
            {
                "band_names": BAND_NAMES,
                "num_bands": 8,
                "source_dataset": "Aryal Landsat7_2005 + DEM pairing validation",
                "upstream_commit": UPSTREAM_COMMIT,
            },
            indent=2,
        )
    )
    provenance = {
        "protocol": "Aryal 2022 thesis / arXiv:2301.11454v1 / public code",
        "upstream_commit": UPSTREAM_COMMIT,
        "image_dir": str(image_dir),
        "dem_dir": str(dem_dir),
        "labels": str(args.labels),
        "raw_image_count": len(images),
        "reported_publication_cell_count": 201,
        "observed_split_counts": {key: len(value) for key, value in splits.items()},
        "full_split_filenames": full_split_names,
        "fixture_max_images_per_split": args.max_images_per_split,
        "retained_slice_counts": {
            key: manifests[key]["num_samples"] for key in manifests
        },
        "label_sha256": {
            split: sha256_file(output_dir / split / "y.npy") for split in manifests
        },
        "intentional_compatibility_adaptations": [
            "Malformed original shapefile rings are read with pyogrio on_invalid=fix.",
            "Only the eight model-visible Landsat bands are stored.",
            "Loader-time all-zero one-hot masking is encoded as label 255.",
            "Raw slices are normalized during packing with upstream train statistics.",
        ],
        "preserved_upstream_behaviors": [
            "NumPy seed-42 geographic split",
            "40 test / 20 validation / 142 train cells for 202 supplied inputs",
            "512 window, 64 overlap on all splits",
            "column iteration bounded by raster height",
            "10% glacier fraction over full padded window",
            "normalization mean/std averaged over per-image statistics",
            "first-seven-band all-zero invalid mask",
        ],
    }
    (output_dir / "aryal_reproduction_provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )
    print(json.dumps(provenance["retained_slice_counts"], indent=2))
    print(f"Created immutable Aryal reproduction dataset: {output_dir}")


if __name__ == "__main__":
    main()
