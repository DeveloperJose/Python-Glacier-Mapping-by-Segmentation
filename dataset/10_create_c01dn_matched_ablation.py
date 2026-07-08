#!/usr/bin/env python3
"""Create C01-DN-matched relaxed-valid ablation dataset.

Evidence base:
- Legacy GEE script exports LANDSAT/LE07/C01/T1_RT and calls .uint8().
- New full8 relaxed datasets are C02 Tier-1 TOA float bands (optical TOA reflectance; thermal brightness temperature).
- Forensics show legacy optical bands have byte-domain clipping/saturation.

This ablation tests whether remaining legacy advantage is mainly radiometric/domain:
robustly map relaxed agreement Landsat bands to legacy byte-like C01 DN distribution
using train-set valid-pixel p1/p99, then recompute Landsat-derived indices/HSV.
No labels, masks, auxiliary channels, splits, or sample order are changed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from skimage.color import rgb2hsv
from tqdm import tqdm

from glacier_mapping.data.slice import IGNORE_LABEL
from scripts.preprocess import normalize_slice_for_v3

DATA_ROOT = Path("/home/devj/local-arch/data/HKH")
SOURCE_DIR = DATA_ROOT / "comprehensive_v3_hkh_full8_agreement_quality_step3_relaxed_valid"
LEGACY_DIR = DATA_ROOT / "comprehensive_v3"
TARGET_DIR = DATA_ROOT / "comprehensive_v3_hkh_full8_agreement_quality_step3_relaxed_valid_c01dn_matched"
OUT_DIR = Path("dataset/outputs")

BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6_VCID1",
    "B6_VCID2",
    "B7",
    "elevation",
    "slope_deg",
    "velocity",
    "velocity_x",
    "velocity_y",
    "velocity_mask",
    "NDVI",
    "NDWI",
    "NDSI",
    "H",
    "S",
    "V",
    "flow_accumulation",
    "tpi",
    "roughness",
    "plan_curvature",
]
LANDSAT_BANDS = BANDS[:8]
DERIVED_BANDS = ["NDVI", "NDWI", "NDSI", "H", "S", "V"]
VALUE_BANDS_FOR_VELOCITY_MASK = {"velocity", "velocity_x", "velocity_y"}


def load_norm(dataset_dir: Path) -> np.ndarray:
    return np.load(dataset_dir / "normalize_train.npy")


def denorm_sample(x_chw: np.ndarray, norm: np.ndarray) -> np.ndarray:
    return x_chw * norm[1, :, None, None] + norm[0, :, None, None]


def valid_values_by_band(
    dataset_dir: Path,
    band_indices: list[int],
    max_per_sample: int = 8000,
    seed: int = 123,
) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    norm = load_norm(dataset_dir)
    x = np.load(dataset_dir / "train" / "X.npy", mmap_mode="r")
    y = np.load(dataset_dir / "train" / "y.npy", mmap_mode="r")
    chunks: dict[int, list[np.ndarray]] = {idx: [] for idx in band_indices}
    for sample_idx in tqdm(range(x.shape[0]), desc=f"Sample stats {dataset_dir.name}"):
        mask = y[sample_idx] != IGNORE_LABEL
        flat = np.flatnonzero(mask.ravel())
        if flat.size == 0:
            continue
        take = rng.choice(flat, size=min(max_per_sample, flat.size), replace=False)
        rows = take // y.shape[1]
        cols = take % y.shape[2]
        for band_idx in band_indices:
            vals = (
                x[sample_idx, band_idx, rows, cols].astype(np.float32)
                * norm[1, band_idx]
                + norm[0, band_idx]
            )
            vals = vals[np.isfinite(vals)]
            if vals.size:
                chunks[band_idx].append(vals.astype(np.float32, copy=False))
    return {
        idx: np.concatenate(parts) if parts else np.array([], dtype=np.float32)
        for idx, parts in chunks.items()
    }


def compute_mapping() -> dict[str, dict[str, float]]:
    src_vals = valid_values_by_band(SOURCE_DIR, list(range(8)), seed=123)
    legacy_vals = valid_values_by_band(LEGACY_DIR, list(range(8)), seed=456)
    mapping: dict[str, dict[str, float]] = {}
    for band_idx, band in enumerate(LANDSAT_BANDS):
        s = src_vals[band_idx]
        l = legacy_vals[band_idx]
        src_p1, src_p99 = np.quantile(s, [0.01, 0.99])
        leg_p1, leg_p99 = np.quantile(l, [0.01, 0.99])
        scale = (leg_p99 - leg_p1) / max(src_p99 - src_p1, 1e-6)
        offset = leg_p1 - src_p1 * scale
        mapping[band] = {
            "source_p1": float(src_p1),
            "source_p99": float(src_p99),
            "legacy_p1": float(leg_p1),
            "legacy_p99": float(leg_p99),
            "scale": float(scale),
            "offset": float(offset),
        }
    return mapping


def transform_landsat(data: np.ndarray, mapping: dict[str, dict[str, float]]) -> None:
    for band_idx, band in enumerate(LANDSAT_BANDS):
        m = mapping[band]
        dn = data[band_idx] * m["scale"] + m["offset"]
        data[band_idx] = np.rint(np.clip(dn, 0.0, 255.0)).astype(np.float32)


def recompute_indices_hsv(data: np.ndarray) -> None:
    b1, b2, b3, b4, b5, _b61, _b62, b7 = range(8)

    def ratio(i1: int, i2: int) -> np.ndarray:
        num = data[i1] - data[i2]
        den = data[i1] + data[i2]
        out = np.divide(num, den, out=np.zeros_like(num, dtype=np.float32), where=den != 0)
        return np.nan_to_num(out).clip(-1, 1)

    data[BANDS.index("NDVI")] = ratio(b4, b3)
    data[BANDS.index("NDWI")] = ratio(b2, b4)
    data[BANDS.index("NDSI")] = ratio(b2, b5)

    # Match slice.py: rgb_img = [B5, B4, B2], then rgb2hsv(rgb_img[:, :, [2,1,0]])
    rgb = np.stack([data[b5], data[b4], data[b2]], axis=-1).astype(np.float32)
    rgb = np.clip(rgb / 255.0, 0.0, 1.0)
    hsv = rgb2hsv(rgb[:, :, [2, 1, 0]])
    data[BANDS.index("H")] = hsv[:, :, 0]
    data[BANDS.index("S")] = hsv[:, :, 1]
    data[BANDS.index("V")] = hsv[:, :, 2]


def transformed_sample(source_x: np.ndarray, norm: np.ndarray, mapping: dict[str, dict[str, float]]) -> np.ndarray:
    data = denorm_sample(source_x, norm).astype(np.float32, copy=True)
    transform_landsat(data, mapping)
    recompute_indices_hsv(data)
    return data


def compute_train_stats(mapping: dict[str, dict[str, float]]) -> np.ndarray:
    source_norm = load_norm(SOURCE_DIR)
    x = np.load(SOURCE_DIR / "train" / "X.npy", mmap_mode="r")
    y = np.load(SOURCE_DIR / "train" / "y.npy", mmap_mode="r")
    c = x.shape[1]
    counts = np.zeros(c, dtype=np.int64)
    sums = np.zeros(c, dtype=np.float64)
    sum_squares = np.zeros(c, dtype=np.float64)
    mins = np.full(c, np.inf, dtype=np.float64)
    maxs = np.full(c, -np.inf, dtype=np.float64)
    vel_mask_idx = BANDS.index("velocity_mask")
    for sample_idx in tqdm(range(x.shape[0]), desc="Compute C01DN train stats"):
        data = transformed_sample(x[sample_idx], source_norm, mapping)
        mask = y[sample_idx] != IGNORE_LABEL
        velocity_valid = mask & (data[vel_mask_idx] > 0.5)
        for channel_idx, band in enumerate(BANDS):
            channel_valid = velocity_valid if band in VALUE_BANDS_FOR_VELOCITY_MASK else mask
            if not np.any(channel_valid):
                continue
            vals = data[channel_idx][channel_valid]
            vals = vals[np.isfinite(vals)].astype(np.float64, copy=False)
            if vals.size == 0:
                continue
            counts[channel_idx] += vals.size
            sums[channel_idx] += vals.sum()
            sum_squares[channel_idx] += np.square(vals).sum()
            mins[channel_idx] = min(mins[channel_idx], vals.min())
            maxs[channel_idx] = max(maxs[channel_idx], vals.max())
    means = np.zeros(c, dtype=np.float64)
    stds = np.ones(c, dtype=np.float64)
    valid = counts > 0
    means[valid] = sums[valid] / counts[valid]
    variances = np.zeros(c, dtype=np.float64)
    variances[valid] = sum_squares[valid] / counts[valid] - means[valid] ** 2
    stds[valid] = np.sqrt(np.maximum(variances[valid], 0.0))
    stds[stds < 1e-6] = 1.0
    mins[~valid] = 0.0
    maxs[~valid] = 1.0
    return np.asarray((means, stds, mins, maxs), dtype=np.float32)


def build_split(split: str, mapping: dict[str, dict[str, float]], target_norm: np.ndarray) -> dict[str, Any]:
    source_norm = load_norm(SOURCE_DIR)
    src_split = SOURCE_DIR / split
    dst_split = TARGET_DIR / split
    dst_split.mkdir(parents=True, exist_ok=True)
    x = np.load(src_split / "X.npy", mmap_mode="r")
    y = np.load(src_split / "y.npy", mmap_mode="r")
    out_x = np.lib.format.open_memmap(
        dst_split / "X.npy", mode="w+", dtype=np.float32, shape=x.shape
    )
    for sample_idx in tqdm(range(x.shape[0]), desc=f"Build {split}"):
        data_chw = transformed_sample(x[sample_idx], source_norm, mapping)
        data_hwc = np.transpose(data_chw, (1, 2, 0))
        normed_hwc = normalize_slice_for_v3(data_hwc, target_norm, BANDS)
        out_x[sample_idx] = np.transpose(normed_hwc, (2, 0, 1))
    out_x.flush()
    try:
        if (dst_split / "y.npy").exists():
            (dst_split / "y.npy").unlink()
        os_link = getattr(Path, "hardlink_to", None)
        if os_link is not None:
            (dst_split / "y.npy").hardlink_to(src_split / "y.npy")
            label_storage = "hardlink"
        else:
            raise OSError
    except OSError:
        shutil.copy2(src_split / "y.npy", dst_split / "y.npy")
        label_storage = "copy"

    source_manifest = json.loads((src_split / "manifest.json").read_text())
    manifest = dict(source_manifest)
    manifest.update(
        {
            "format": "comprehensive_v3_c01dn_matched",
            "normalized": True,
            "normalization": "mean-std",
            "x": "X.npy",
            "y": "y.npy",
            "transform": "robust p1/p99 C02-to-legacy C01 DN mapping; Landsat indices/HSV recomputed",
            "label_storage": label_storage,
        }
    )
    (dst_split / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_dataset_statistics() -> None:
    stats: dict[str, Any] = {}
    for split in ["train", "val", "test"]:
        y = np.load(TARGET_DIR / split / "y.npy", mmap_mode="r")
        total = int(y.size)
        counts = {
            "background": int((y == 0).sum()),
            "clean_ice": int((y == 1).sum()),
            "debris_ice": int((y == 2).sum()),
            "masked_invalid": int((y == IGNORE_LABEL).sum()),
        }
        valid = total - counts["masked_invalid"]
        stats[split] = {
            "slices": int(y.shape[0]),
            "total_pixels": total,
            "pixels": counts,
            "percentages_all_pixels": {k: v / total * 100 for k, v in counts.items()},
            "percentages_valid_pixels": {
                "background": counts["background"] / valid * 100 if valid else 0.0,
                "clean_ice": counts["clean_ice"] / valid * 100 if valid else 0.0,
                "debris_ice": counts["debris_ice"] / valid * 100 if valid else 0.0,
            },
        }
    (TARGET_DIR / "dataset_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if TARGET_DIR.exists():
        if not args.force:
            raise FileExistsError(f"Target exists: {TARGET_DIR}. Use --force.")
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mapping = compute_mapping()
    (TARGET_DIR / "c01dn_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    (OUT_DIR / "c01dn_matched_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    target_norm = compute_train_stats(mapping)
    np.save(TARGET_DIR / "normalize_train.npy", target_norm)
    for split in ["train", "val", "test"]:
        np.save(TARGET_DIR / f"normalize_{split}.npy", target_norm)

    for split in ["train", "val", "test"]:
        build_split(split, mapping, target_norm)

    shutil.copy2(SOURCE_DIR / "band_metadata.json", TARGET_DIR / "band_metadata.json")
    write_dataset_statistics()

    print(f"Created {TARGET_DIR}")
    print(f"Mapping: {TARGET_DIR / 'c01dn_mapping.json'}")


if __name__ == "__main__":
    main()
