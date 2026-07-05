#!/usr/bin/env python3
"""Local gapfill tournament for aligned HKH raw target/donor exports.

No GEE. Uses full-scene exports as context, but validates on sampled full-scene
windows with simulated SLC-like holdouts. Goal is to answer which donor family
and single-vs-multi policy works best before producing full-scene fills.

Outputs:
  dataset/outputs/5_gapfill_tournament/{id}/validation_metrics.csv
  dataset/outputs/5_gapfill_tournament/{id}/summary.json
  dataset/outputs/5_gapfill_tournament/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom
from numba import njit, prange
from sklearn.cluster import MiniBatchKMeans

# Reuse audited local NSPI implementation. This is local-only; no GEE.
sys.path.insert(0, str(Path("scripts").resolve()))
from run_hkh_nspi_local import _nspi_single, compute_similarity_threshold  # noqa: E402

TARGETS_JSON = Path("dataset/outputs/1_targets.json")
SLATE_CSV = Path("dataset/outputs/3_donor_slate_narrow.csv")
TARGET_DIR = Path("dataset/raw/targets")
DONOR_DIR = Path("dataset/raw/donors")
TARGET_DIR_FULL8 = Path("dataset/raw_full8/targets")
DONOR_DIR_FULL8 = Path("dataset/raw_full8/donors")
OUTPUT_ROOT = Path("dataset/outputs/5_gapfill_tournament")

OPTICAL_BANDS = ["B1", "B2", "B3", "B4", "B5", "B7"]
FULL8_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6_VCID_1", "B6_VCID_2", "B7"]
BAND_GROUPS = {
    "optical6_legacy": {"indices": [0, 1, 2, 3, 4, 5], "names": OPTICAL_BANDS},
    "optical6": {"indices": [0, 1, 2, 3, 4, 5], "names": OPTICAL_BANDS},
    "full8": {"indices": list(range(8)), "names": FULL8_BANDS},
    "optical6_from_full8": {"indices": [0, 1, 2, 3, 4, 7], "names": OPTICAL_BANDS},
    "thermal2": {"indices": [5, 6], "names": ["B6_VCID_1", "B6_VCID_2"]},
}
KIND_CODE = {"lt05": 1, "le07_slc_on": 2, "le07_slc_off": 3}

# Defaults chosen to get fast pilot signal without pretending it is final truth.
DEFAULT_IDS = "04,16,26"
DEFAULT_WINDOW_SIZE = 384
DEFAULT_MAX_WINDOWS = 8
DEFAULT_MIN_HOLDOUT_PIXELS = 4_000
DEFAULT_MAX_HOLDOUT_PIXELS_PER_WINDOW = 30_000
DEFAULT_GLOBAL_SAMPLE_PIXELS = 250_000
DEFAULT_SEED = 42

NSPI_MIN_SIMILAR = 20
NSPI_MAX_WINDOW = 8  # radius; full window = 17x17. Paper multi default can be 15 later.
NSPI_NUM_CLASS = 5
DN_MIN = 0.0
DN_MAX = 1.0
EPS = 1e-6


@dataclass(frozen=True)
class DonorInfo:
    kind: str
    path: Path
    date: str
    score: float


@dataclass(frozen=True)
class HoldoutSpec:
    mask_type: str
    name: str
    mask: np.ndarray


def parse_ids(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        json.dump(data, tmp, indent=2)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for key in sorted({k for row in rows for k in row}):
        fields.append(key)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def row_band_group(row: dict[str, Any]) -> str:
    return str(row.get("band_group") or "optical6_legacy")


def row_value_mode(row: dict[str, Any]) -> str:
    return str(row.get("value_mode") or "source")


def metric_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str, str, str, str, str]:
    return (
        str(row.get("target_id", "")),
        str(row.get("window", "")),
        str(row.get("method", "")),
        str(row.get("donor_policy", "")),
        str(row.get("metric_scope", "")),
        str(row.get("comparison_group", "")),
        str(row.get("mask_type", "")),
        str(row.get("mask_name", "")),
        row_band_group(row),
        row_value_mode(row),
    )


def has_metric(
    existing_keys: set[tuple[str, str, str, str, str, str, str, str, str, str]],
    target_id: int,
    window: int,
    method: str,
    donor_policy: str,
    mask_type: str,
    mask_name: str,
    scope: str = "own",
    group: str = "own_valid_pixels",
    band_group: str = "optical6_legacy",
    value_mode: str = "source",
) -> bool:
    return (
        str(target_id),
        str(window),
        method,
        donor_policy,
        scope,
        group,
        mask_type,
        mask_name,
        band_group,
        value_mode,
    ) in existing_keys


def has_all_band_metrics(
    existing_keys: set[tuple[str, str, str, str, str, str, str, str, str, str]],
    target_id: int,
    window: int,
    method: str,
    donor_policy: str,
    mask_type: str,
    mask_name: str,
    band_groups: list[str],
    scope: str = "own",
    group: str = "own_valid_pixels",
    value_mode: str = "source",
) -> bool:
    return all(
        has_metric(
            existing_keys,
            target_id,
            window,
            method,
            donor_policy,
            mask_type,
            mask_name,
            scope,
            group,
            band_group,
            value_mode,
        )
        for band_group in band_groups
    )


def load_targets_meta() -> dict[int, dict[str, Any]]:
    return {int(row["id"]): row for row in read_json(TARGETS_JSON)}


def load_slate(donor_dir: Path = DONOR_DIR) -> dict[int, list[DonorInfo]]:
    out: dict[int, list[DonorInfo]] = defaultdict(list)
    with SLATE_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tid = int(row["target_id"])
            date = row["donor_date"].replace("-", "")
            path = donor_dir / f"{tid:02d}_donor_{row['donor_kind']}_{date}.tif"
            if not path.exists():
                continue
            out[tid].append(
                DonorInfo(
                    kind=row["donor_kind"],
                    path=path,
                    date=row["donor_date"],
                    score=float(row.get("family_score") or 0.0),
                )
            )
    for donors in out.values():
        donors.sort(key=lambda d: d.score, reverse=True)
    return out


def target_path(target: dict[str, Any], target_dir: Path = TARGET_DIR) -> Path:
    return target_dir / target.get("target_filename", target["filename_target"])


def stack_schema(band_mode: str) -> tuple[int, int, int, int, list[str]]:
    if band_mode == "full8":
        return 8, 10, 11, 12, FULL8_BANDS
    return 6, 8, 9, 10, OPTICAL_BANDS


def band_groups_for_mode(band_mode: str) -> list[str]:
    if band_mode == "full8":
        return ["full8", "optical6_from_full8", "thermal2"]
    return ["optical6_legacy"]


def value_scale_for_band_group(band_group: str, value_mode: str) -> str:
    if value_mode == "scene_minmax":
        return "per_scene_per_band_minmax_0_1_nonholdout_target_donors"
    if band_group in {"full8", "thermal2"}:
        return "source_TOA_values_unscaled"
    return "optical_TOA_reflectance"


def load_stack(path: Path, band_mode: str = "optical6") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        profile = src.profile.copy()
    image_bands, data_idx, clear_idx, gap_idx, _names = stack_schema(band_mode)
    if arr.shape[0] <= gap_idx:
        raise ValueError(f"{path} has {arr.shape[0]} bands; {band_mode} expects at least {gap_idx + 1}")
    image = arr[:image_bands]
    data_present = arr[data_idx] > 0.5
    clear_valid = arr[clear_idx] > 0.5
    slc_gap = arr[gap_idx] > 0.5
    finite = np.isfinite(image).all(axis=0)
    data_present &= finite
    clear_valid &= finite
    return image, data_present, clear_valid, slc_gap, profile


def compute_scene_minmax_scales(
    target: np.ndarray,
    target_mask: np.ndarray,
    donor_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    domain_mask: np.ndarray,
    holdout_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-scene, per-band min/max from non-holdout valid target+donors.

    Data-driven scaling only; no assumed units. Values are clipped to [0, 1].
    Holdout target truth is excluded from scale fitting to avoid validation leakage.
    """
    bands = target.shape[0]
    mins = np.full(bands, np.inf, dtype=np.float32)
    maxs = np.full(bands, -np.inf, dtype=np.float32)
    base_mask = domain_mask & ~holdout_mask
    masks_and_arrays: list[tuple[np.ndarray, np.ndarray]] = [(target, target_mask & base_mask)]
    masks_and_arrays.extend((arr, valid & base_mask) for arr, valid, _gap in donor_arrays.values())
    for b in range(bands):
        for arr, mask in masks_and_arrays:
            vals = arr[b][mask & np.isfinite(arr[b])]
            if vals.size == 0:
                continue
            mins[b] = min(mins[b], float(np.nanmin(vals)))
            maxs[b] = max(maxs[b], float(np.nanmax(vals)))
        if not np.isfinite(mins[b]) or not np.isfinite(maxs[b]) or maxs[b] <= mins[b]:
            mins[b] = 0.0
            maxs[b] = 1.0
    return mins, maxs


def apply_scene_minmax(arr: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    denom = np.maximum(maxs - mins, EPS).astype(np.float32)
    return np.clip((arr - mins[:, None, None]) / denom[:, None, None], DN_MIN, DN_MAX).astype(np.float32)


def rasterize_domain(target: dict[str, Any], profile: dict) -> np.ndarray:
    # target_domain_geojson from Step 1 is lon/lat. Exports are in target UTM CRS.
    geom = transform_geom("EPSG:4326", profile["crs"], target["target_domain_geojson"])
    height = int(profile["height"])
    width = int(profile["width"])
    transform = profile["transform"]
    return geometry_mask([geom], out_shape=(height, width), transform=transform, invert=True)


def shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    height, width = mask.shape
    mat = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(mask.astype(np.uint8), mat, (width, height), flags=cv2.INTER_NEAREST, borderValue=0).astype(bool)


def load_external_slc_gap_masks(
    target_shape: tuple[int, int],
    target_dir: Path,
    donor_dir: Path,
    slc_gap_band_1based: int,
) -> list[tuple[str, np.ndarray]]:
    """Use all currently downloaded post-SLC gap rasters as shape-library masks.

    These are pixel-space SLC stripe patterns, resized if needed. They are used
    only for simulated validation holdouts, never as geospatial truth.
    """
    masks: list[tuple[str, np.ndarray]] = []
    paths = sorted(target_dir.glob("*_target_*.tif")) + sorted(donor_dir.glob("*_donor_le07_slc_off_*.tif"))
    out_h, out_w = target_shape
    for path in paths:
        try:
            with rasterio.open(path) as src:
                if src.count < slc_gap_band_1based:
                    continue
                slc = src.read(slc_gap_band_1based).astype(np.float32) > 0.5
        except Exception:
            continue
        if int(slc.sum()) == 0:
            continue
        if slc.shape != target_shape:
            slc = cv2.resize(slc.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        masks.append((path.stem, slc))
    return masks


def make_holdout_specs(
    target_clear: np.ndarray,
    target_gap: np.ndarray,
    domain: np.ndarray,
    donor_gap_masks: dict[str, np.ndarray],
    rng: np.random.Generator,
    target_dir: Path = TARGET_DIR,
    donor_dir: Path = DONOR_DIR,
    slc_gap_band_1based: int = 11,
) -> list[HoldoutSpec]:
    """Build named realistic validation masks from real SLC geometries."""
    specs: list[HoldoutSpec] = []
    height, width = target_clear.shape
    shifts = [(80, 0), (-80, 0), (160, 0), (-160, 0), (240, 0), (-240, 0), (160, 120), (-160, -120)]

    shifted_union = np.zeros_like(target_clear, dtype=bool)
    for dx, dy in shifts:
        shifted_union |= shift_mask(target_gap, dx, dy) & target_clear & domain
    if int(shifted_union.sum()) > 0:
        specs.append(HoldoutSpec("slc_shifted_target_gap", "target_gap_shifted_union", shifted_union))

    for kind, gap in donor_gap_masks.items():
        donor_base = gap & target_clear & domain
        if int(donor_base.sum()) > 0:
            specs.append(HoldoutSpec("slc_donor_gap_pattern", f"{kind}_gap_unshifted", donor_base))
        donor_shifted = np.zeros_like(target_clear, dtype=bool)
        for dx, dy in [(80, 0), (-80, 0), (160, 0), (-160, 0), (80, 80), (-80, -80)]:
            donor_shifted |= shift_mask(gap, dx, dy) & target_clear & domain
        if int(donor_shifted.sum()) > 0:
            specs.append(HoldoutSpec("slc_donor_gap_pattern", f"{kind}_gap_shifted", donor_shifted))

    external = load_external_slc_gap_masks(target_clear.shape, target_dir, donor_dir, slc_gap_band_1based)
    if external:
        chosen = rng.choice(len(external), size=min(8, len(external)), replace=False)
        for idx in chosen:
            name, gap = external[int(idx)]
            lib_shifted = np.zeros_like(target_clear, dtype=bool)
            for dx, dy in [(0, 0), (80, 0), (-80, 0), (160, 80), (-160, -80)]:
                lib_shifted |= shift_mask(gap, dx, dy) & target_clear & domain
            if int(lib_shifted.sum()) > 0:
                specs.append(HoldoutSpec("slc_library_gap_pattern", name, lib_shifted))

    gap_u8 = target_gap.astype(np.uint8)
    for radius in [3, 5, 9, 15]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        dilated = cv2.dilate(gap_u8, kernel).astype(bool)
        edge = dilated & ~target_gap & target_clear & domain
        if int(edge.sum()) > 0:
            specs.append(HoldoutSpec("slc_gap_edge", f"target_gap_edge_r{radius}", edge))

    patch = np.zeros_like(target_clear, dtype=bool)
    candidates = np.argwhere(target_clear & domain)
    if len(candidates):
        n = min(80, max(12, len(candidates) // 400_000))
        chosen = candidates[rng.choice(len(candidates), size=n, replace=False)]
        for y, x in chosen:
            r = int(rng.integers(12, 32))
            y1, y2 = max(0, y - r), min(height, y + r)
            x1, x2 = max(0, x - r), min(width, x + r)
            patch[y1:y2, x1:x2] |= target_clear[y1:y2, x1:x2] & domain[y1:y2, x1:x2]
    if int(patch.sum()) > 0:
        specs.append(HoldoutSpec("stratified_patch", "random_clear_patches", patch))

    return specs


def pick_windows(
    holdout: np.ndarray,
    window_size: int,
    max_windows: int,
    min_holdout_pixels: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int, int]]:
    height, width = holdout.shape
    ys, xs = np.where(holdout)
    if len(xs) == 0:
        return []
    windows: list[tuple[int, int, int, int]] = []
    attempts = 0
    order = rng.permutation(len(xs))
    for idx in order:
        if len(windows) >= max_windows:
            break
        y = int(ys[idx])
        x = int(xs[idx])
        y1 = max(0, min(height - window_size, y - window_size // 2))
        x1 = max(0, min(width - window_size, x - window_size // 2))
        y2 = min(height, y1 + window_size)
        x2 = min(width, x1 + window_size)
        attempts += 1
        if int(holdout[y1:y2, x1:x2].sum()) < min_holdout_pixels:
            continue
        # Avoid near duplicates.
        if any(abs(y1 - wy1) < window_size // 2 and abs(x1 - wx1) < window_size // 2 for wy1, wx1, _, _ in windows):
            continue
        windows.append((y1, x1, y2, x2))
        if attempts > max_windows * 200:
            break
    return windows


def spectral_classes(image: np.ndarray, band_names: list[str]) -> dict[str, np.ndarray]:
    name_to_idx = {name: idx for idx, name in enumerate(band_names)}
    b2 = image[name_to_idx["B2"]]
    b3 = image[name_to_idx["B3"]]
    b4 = image[name_to_idx["B4"]]
    b5 = image[name_to_idx["B5"]]
    b7_idx = name_to_idx.get("B7")
    optical_indices = [name_to_idx[name] for name in OPTICAL_BANDS if name in name_to_idx]
    ndvi = (b4 - b3) / (b4 + b3 + EPS)
    ndsi = (b2 - b5) / (b2 + b5 + EPS)
    brightness = np.nanmean(image[optical_indices], axis=0) if optical_indices else np.nanmean(image, axis=0)
    return {
        "snow_ice": (ndsi > 0.35) & (brightness > 0.12),
        "vegetation": ndvi > 0.25,
        "dark_shadow": brightness < 0.10,
        "rock_debris_other": np.ones_like(brightness, dtype=bool),
    }


def fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    good = np.isfinite(x) & np.isfinite(y)
    if int(good.sum()) < 128:
        return 1.0, 0.0
    xg = x[good]
    yg = y[good]
    # Two-pass robust least-squares via residual clipping.
    a, b = np.polyfit(xg, yg, 1)
    pred = a * xg + b
    resid = np.abs(yg - pred)
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med))) + EPS
    keep = resid <= med + 4.0 * 1.4826 * mad
    if int(keep.sum()) >= 128:
        a, b = np.polyfit(xg[keep], yg[keep], 1)
    if not np.isfinite(a) or not np.isfinite(b) or a < 0.0 or a > 4.0:
        return 1.0, float(np.nanmedian(yg - xg))
    return float(a), float(b)


def sample_global_fit(
    target: np.ndarray,
    donor: np.ndarray,
    train_mask: np.ndarray,
    max_samples: int,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    ys, xs = np.where(train_mask)
    if len(xs) > max_samples:
        idx = rng.choice(len(xs), size=max_samples, replace=False)
        ys = ys[idx]
        xs = xs[idx]
    coefs = []
    for b in range(target.shape[0]):
        coefs.append(fit_affine(donor[b, ys, xs], target[b, ys, xs]))
    return coefs


def predict_affine(donor: np.ndarray, coefs: list[tuple[float, float]]) -> np.ndarray:
    out = np.empty_like(donor, dtype=np.float32)
    for b, (a, off) in enumerate(coefs):
        out[b] = np.clip(a * donor[b] + off, DN_MIN, DN_MAX)
    return out


def box_sum(arr: np.ndarray, radius: int) -> np.ndarray:
    k = 2 * radius + 1
    return cv2.boxFilter(arr.astype(np.float32), ddepth=-1, ksize=(k, k), normalize=False, borderType=cv2.BORDER_REFLECT)


def roberts_edge(img: np.ndarray) -> np.ndarray:
    k1 = np.array([[1, 0], [0, -1]], dtype=np.float32)
    k2 = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    return np.abs(cv2.filter2D(img.astype(np.float32), -1, k1, borderType=cv2.BORDER_REFLECT)) + np.abs(
        cv2.filter2D(img.astype(np.float32), -1, k2, borderType=cv2.BORDER_REFLECT)
    )


def lbp_pattern(img: np.ndarray, tolerance: float = 0.005) -> np.ndarray:
    """8-neighbor LBP pattern matching XZhu APA metric orientation/weights."""
    center = img.astype(np.float32)
    out = np.zeros(center.shape, dtype=np.float32)
    neighbors = [
        ((0, -1), 16),
        ((-1, -1), 8),
        ((-1, 0), 4),
        ((-1, 1), 2),
        ((0, 1), 1),
        ((1, 1), 128),
        ((1, 0), 64),
        ((1, -1), 32),
    ]
    for (dy, dx), weight in neighbors:
        shifted = np.roll(np.roll(center, dy, axis=0), dx, axis=1)
        out += (shifted > center + tolerance).astype(np.float32) * weight
    out[[0, -1], :] = 0
    out[:, [0, -1]] = 0
    return out


def apa_edge_lbp_metrics(pred: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    """APA-inspired edge/LBP normalized differences.

    Source reference archived at:
    dataset/reference_code/xzhu_lab/.../optimal_accuracy_metrics.py
    """
    interior = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    if int(interior.sum()) < 128:
        return {"edge_nd_mean": math.nan, "lbp_nd_mean": math.nan}
    edge_vals: list[float] = []
    lbp_vals: list[float] = []
    for b in range(pred.shape[0]):
        te = roberts_edge(truth[b])
        pe = roberts_edge(pred[b])
        vals = te[interior]
        if vals.size >= 128:
            thresh = float(np.quantile(vals, 0.9))
            edge_mask = interior & (te >= thresh)
            if int(edge_mask.sum()) > 0:
                edge_vals.append(float(np.mean((pe[edge_mask] - te[edge_mask]) / (np.abs(pe[edge_mask] + te[edge_mask]) + 1e-5))))
        tl = lbp_pattern(truth[b]) / 255.0
        pl = lbp_pattern(pred[b]) / 255.0
        lbp_vals.append(float(np.mean((pl[interior] - tl[interior]) / (np.abs(pl[interior] + tl[interior]) + 1e-5))))
    return {
        "edge_nd_mean": float(np.mean(edge_vals)) if edge_vals else math.nan,
        "lbp_nd_mean": float(np.mean(lbp_vals)) if lbp_vals else math.nan,
    }


def predict_mnspi_class_affine(
    target: np.ndarray,
    donor: np.ndarray,
    train_valid: np.ndarray,
    donor_valid: np.ndarray,
    fallback_coefs: list[tuple[float, float]],
    num_class: int = 4,
    min_class_pixels: int = 256,
    max_kmeans_pixels: int = 50_000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """MNSPI-inspired class-wise affine donor->target comparator.

    Reference idea from archived `CLOUD_REMOVE_FAST.pro`: cluster clear input,
    fit per-class temporal relation, fallback when slope outside [1/3, 3]. This
    lightweight version skips per-cloud object morphology and spatial blend, so
    it is a baseline/idea test, not exact MNSPI.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    features = np.moveaxis(donor, 0, -1).reshape(-1, donor.shape[0])
    flat_valid = donor_valid.reshape(-1) & np.isfinite(features).all(axis=1)
    idx = np.flatnonzero(flat_valid)
    if len(idx) < max(num_class * 100, min_class_pixels):
        return predict_affine(donor, fallback_coefs), donor_valid
    if len(idx) > max_kmeans_pixels:
        idx_fit = rng.choice(idx, size=max_kmeans_pixels, replace=False)
    else:
        idx_fit = idx
    km = MiniBatchKMeans(n_clusters=num_class, random_state=0, batch_size=4096, n_init=3, max_iter=50)
    km.fit(features[idx_fit])
    labels = np.full(features.shape[0], -1, dtype=np.int16)
    labels[idx] = km.predict(features[idx])
    labels2d = labels.reshape(donor.shape[1:])

    out = predict_affine(donor, fallback_coefs)
    for cls in range(num_class):
        cls_train = train_valid & donor_valid & (labels2d == cls)
        if int(cls_train.sum()) < min_class_pixels:
            continue
        cls_apply = donor_valid & (labels2d == cls)
        for b in range(target.shape[0]):
            a, off = fit_affine(donor[b, cls_train], target[b, cls_train])
            # MNSPI fast rejects unstable relation and falls back to mean difference.
            if a < (1.0 / 3.0) or a > 3.0 or not np.isfinite(a):
                a = 1.0
                off = float(np.nanmean(target[b, cls_train] - donor[b, cls_train]))
            out[b, cls_apply] = np.clip(a * donor[b, cls_apply] + off, DN_MIN, DN_MAX)
    return out, donor_valid & np.isfinite(out).all(axis=0)


def predict_usgs_local(
    target: np.ndarray,
    donor: np.ndarray,
    train_valid: np.ndarray,
    donor_valid: np.ndarray,
    kernel_radius: int = 5,
    min_neighbors: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Local port of USGS/Gorelick/Donchyts SLC-off regression fill.

    Per band: local linear fit target = scale * donor + offset. If scale is
    invalid, fallback to local mean/std scaling; if still invalid, fallback to
    mean-difference offset. Prediction valid only where donor valid and enough
    common neighbors exist.
    """
    common = train_valid & donor_valid
    count = box_sum(common.astype(np.float32), kernel_radius)
    valid_pred = donor_valid & (count >= float(min_neighbors))
    out = np.full_like(donor, np.nan, dtype=np.float32)
    cnt = np.maximum(count, 1.0)
    min_scale = 1.0 / 3.0
    max_scale = 3.0
    for b in range(target.shape[0]):
        x = np.where(common, donor[b], 0.0).astype(np.float32)
        y = np.where(common, target[b], 0.0).astype(np.float32)
        sx = box_sum(x, kernel_radius)
        sy = box_sum(y, kernel_radius)
        sxx = box_sum(x * x, kernel_radius)
        syy = box_sum(y * y, kernel_radius)
        sxy = box_sum(x * y, kernel_radius)
        denom = cnt * sxx - sx * sx
        scale = np.where(np.abs(denom) > EPS, (cnt * sxy - sx * sy) / (denom + EPS), np.nan)
        offset = (sy - scale * sx) / cnt

        mean_x = sx / cnt
        mean_y = sy / cnt
        var_x = np.maximum(sxx / cnt - mean_x * mean_x, 0.0)
        var_y = np.maximum(syy / cnt - mean_y * mean_y, 0.0)
        scale2 = np.sqrt(var_y) / (np.sqrt(var_x) + EPS)
        offset2 = mean_y - scale2 * mean_x

        invalid = (~np.isfinite(scale)) | (scale < min_scale) | (scale > max_scale)
        scale = np.where(invalid, scale2, scale)
        offset = np.where(invalid, offset2, offset)

        invalid2 = (~np.isfinite(scale)) | (scale < min_scale) | (scale > max_scale)
        scale = np.where(invalid2, 1.0, scale)
        offset = np.where(invalid2, mean_y - mean_x, offset)

        pred = np.clip(scale * donor[b] + offset, DN_MIN, DN_MAX)
        out[b] = np.where(valid_pred, pred, np.nan)
    return out, valid_pred & np.isfinite(out).all(axis=0)


def metrics_for_prediction(
    pred: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    classes: dict[str, np.ndarray],
    denominator_mask: np.ndarray | None = None,
    band_indices: list[int] | None = None,
    band_names: list[str] | None = None,
) -> dict[str, Any]:
    if band_indices is None:
        band_indices = list(range(pred.shape[0]))
    if band_names is None:
        band_names = [f"B{i + 1}" for i in range(len(band_indices))]
    pred_eval = pred[band_indices]
    truth_eval = truth[band_indices]
    valid = mask & np.isfinite(pred_eval).all(axis=0) & np.isfinite(truth_eval).all(axis=0)
    denominator = mask if denominator_mask is None else denominator_mask
    candidate_pixels = int(mask.sum())
    holdout_pixels = int(denominator.sum())
    valid_pixels = int(valid.sum())
    out: dict[str, Any] = {
        "holdout_pixels": holdout_pixels,
        "candidate_pixels": candidate_pixels,
        "valid_pixels": valid_pixels,
        "candidate_fill_fraction": float(valid_pixels / candidate_pixels) if candidate_pixels else 0.0,
        "holdout_fill_fraction": float(valid_pixels / holdout_pixels) if holdout_pixels else 0.0,
    }
    if valid_pixels == 0:
        out.update(
            {
                "mean_mae": math.nan,
                "mean_rmse": math.nan,
                "effective_rmse": math.nan,
                "sam_deg": math.nan,
                "ndsi_mae": math.nan,
                "ndvi_mae": math.nan,
                "edge_nd_mean": math.nan,
                "lbp_nd_mean": math.nan,
            }
        )
        return out
    diff = pred_eval[:, valid] - truth_eval[:, valid]
    mae = np.mean(np.abs(diff), axis=1)
    rmse = np.sqrt(np.mean(diff * diff, axis=1))
    for b, name in enumerate(band_names):
        out[f"mae_{name}"] = float(mae[b])
        out[f"rmse_{name}"] = float(rmse[b])
    out["mean_mae"] = float(np.mean(mae))
    out["mean_rmse"] = float(np.mean(rmse))
    out["effective_rmse"] = float(out["mean_rmse"] / math.sqrt(max(out["holdout_fill_fraction"], EPS)))

    p = pred_eval[:, valid]
    t = truth_eval[:, valid]
    dot = np.sum(p * t, axis=0)
    norm = np.linalg.norm(p, axis=0) * np.linalg.norm(t, axis=0) + EPS
    out["sam_deg"] = float(np.degrees(np.mean(np.arccos(np.clip(dot / norm, -1, 1)))))

    name_to_eval_idx = {name: idx for idx, name in enumerate(band_names)}
    if all(name in name_to_eval_idx for name in ["B2", "B3", "B4", "B5"]):
        def ndsi(a: np.ndarray) -> np.ndarray:
            return (a[name_to_eval_idx["B2"]] - a[name_to_eval_idx["B5"]]) / (
                a[name_to_eval_idx["B2"]] + a[name_to_eval_idx["B5"]] + EPS
            )

        def ndvi(a: np.ndarray) -> np.ndarray:
            return (a[name_to_eval_idx["B4"]] - a[name_to_eval_idx["B3"]]) / (
                a[name_to_eval_idx["B4"]] + a[name_to_eval_idx["B3"]] + EPS
            )

        out["ndsi_mae"] = float(np.mean(np.abs(ndsi(p) - ndsi(t))))
        out["ndvi_mae"] = float(np.mean(np.abs(ndvi(p) - ndvi(t))))
    else:
        out["ndsi_mae"] = math.nan
        out["ndvi_mae"] = math.nan
    out["brightness_mae"] = float(np.mean(np.abs(np.mean(p, axis=0) - np.mean(t, axis=0))))
    out.update(apa_edge_lbp_metrics(pred_eval, truth_eval, valid))

    for cname, cmask in classes.items():
        cvalid = valid & cmask
        if int(cvalid.sum()) == 0:
            continue
        cdiff = pred_eval[:, cvalid] - truth_eval[:, cvalid]
        out[f"{cname}_pixels"] = int(cvalid.sum())
        out[f"{cname}_mean_rmse"] = float(np.mean(np.sqrt(np.mean(cdiff * cdiff, axis=1))))
    return out


def crop(arr: np.ndarray, win: tuple[int, int, int, int]) -> np.ndarray:
    y1, x1, y2, x2 = win
    if arr.ndim == 3:
        return arr[:, y1:y2, x1:x2]
    return arr[y1:y2, x1:x2]


def date_to_doy(date: str) -> int:
    return int(np.datetime64(date, "D").astype(object).strftime("%j"))


def classify_timeseries_images(
    images: np.ndarray,
    masks: np.ndarray,
    num_class: int,
) -> np.ndarray:
    """Faithful k-means class step from NSPI_fillSINGLE_useTIMESERIES.py.

    Original: for each image, set bad-mask pixels to 0, run MiniBatchKMeans
    with n_clusters=num_class+1, random_state=0, max_iter=20,
    reassignment_ratio=0.02, then labels+1.
    """
    n_image, _bands, height, width = images.shape
    classes = np.zeros((n_image, height, width), dtype=np.int16)
    for i_img in range(n_image):
        imagei = images[i_img].copy()
        bad = masks[i_img] != 0
        imagei[:, bad] = 0.0
        image_hw = np.moveaxis(imagei, 0, -1)
        new_shape = (image_hw.shape[0] * image_hw.shape[1], 6)
        new_imagei = np.maximum(image_hw[:, :, :6].reshape(new_shape), 0)
        clf = MiniBatchKMeans(n_clusters=num_class + 1, random_state=0, max_iter=20, reassignment_ratio=0.02)
        labels = clf.fit(new_imagei).labels_.reshape(height, width)
        classes[i_img] = labels.astype(np.int16) + 1
    return classes


@njit(cache=True)
def _nspi_timeseries_fillsingle_kernel(
    images: np.ndarray,
    masks: np.ndarray,
    classes: np.ndarray,
    doys: np.ndarray,
    target_index: int,
    fill_mask: np.ndarray,
    target_train_valid: np.ndarray,
    min_similar: int,
    dn_min: float,
    dn_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_image, num_band, nl, ns = images.shape
    fine0 = images[target_index].copy()
    gap = np.empty((nl, ns), dtype=np.int16)
    for j in range(nl):
        for i in range(ns):
            if fill_mask[j, i]:
                gap[j, i] = 1
            elif target_train_valid[j, i]:
                gap[j, i] = 0
            else:
                gap[j, i] = -1
    mark = np.zeros((nl, ns), dtype=np.int16)

    temp_dis = np.empty(n_image, dtype=np.int32)
    for k in range(n_image):
        d = doys[k] - doys[target_index]
        if d < 0:
            d = -d
        temp_dis[k] = d
    order = np.argsort(temp_dis)
    max_window = int(np.round(max(nl, ns) * 0.25))
    init_w = int(np.ceil(0.5 * (np.sqrt(min_similar) - 1.0)))

    note_finish = 0
    i_input = 1
    max_candidates = (2 * max_window + 1) * (2 * max_window + 1)
    cand_y = np.empty(max_candidates, dtype=np.int32)
    cand_x = np.empty(max_candidates, dtype=np.int32)
    rmsei = np.empty(max_candidates, dtype=np.float32)
    rmse12 = np.empty(max_candidates, dtype=np.float32)
    disi = np.empty(max_candidates, dtype=np.float32)

    while note_finish != 1 and i_input <= n_image - 1:
        donor_idx = order[i_input]
        mask2 = masks[donor_idx]
        class2 = classes[donor_idx]

        for i in range(ns):
            for j in range(nl):
                if not (0 < gap[j, i] < 10):
                    continue
                if mask2[j, i] != 0:
                    continue

                classij = class2[j, i]
                # find possible largest window using row/column profiles
                num_outcloud = 0
                min_spatial = 2147483647
                for x in range(ns):
                    if gap[j, x] == 0 and mask2[j, x] == 0:
                        dd = x - i
                        if dd < 0:
                            dd = -dd
                        if dd < min_spatial:
                            min_spatial = dd
                        num_outcloud += 1
                for y in range(nl):
                    if gap[y, i] == 0 and mask2[y, i] == 0:
                        dd = y - j
                        if dd < 0:
                            dd = -dd
                        if dd < min_spatial:
                            min_spatial = dd
                        num_outcloud += 1
                if num_outcloud > 0:
                    max_window_p = min_spatial + 50
                    if max_window_p > max_window:
                        max_window_p = max_window
                else:
                    max_window_p = max_window

                start_w = init_w
                # Original fillSINGLE code uses b2=max(nl-1, j+max_window),
                # which clips to bottom in Python slicing. Preserve that effect.
                if max_window_p == max_window:
                    a1p = i - max_window
                    if a1p < 0:
                        a1p = 0
                    a2p = i + max_window
                    if a2p > ns - 1:
                        a2p = ns - 1
                    b1p = j - max_window
                    if b1p < 0:
                        b1p = 0
                    b2p = nl - 1
                    c_pre = 0
                    for yy in range(b1p, b2p + 1):
                        for xx in range(a1p, a2p + 1):
                            if gap[yy, xx] == 0 and mask2[yy, xx] == 0 and class2[yy, xx] == classij:
                                c_pre += 1
                    if c_pre < 2.0 * min_similar:
                        start_w = max_window_p

                end_w = max_window_p
                ind_success = 0
                mid = start_w
                a1 = 0
                a2 = 0
                b1 = 0
                b2 = 0
                c_common = 0
                while ind_success == 0 and start_w <= end_w:
                    mid = int(np.floor(start_w + (end_w - start_w) / 2.0))
                    a1 = i - mid
                    if a1 < 0:
                        a1 = 0
                    a2 = i + mid
                    if a2 > ns - 1:
                        a2 = ns - 1
                    b1 = j - mid
                    if b1 < 0:
                        b1 = 0
                    b2 = j + mid
                    if b2 > nl - 1:
                        b2 = nl - 1
                    c_common = 0
                    for yy in range(b1, b2 + 1):
                        for xx in range(a1, a2 + 1):
                            if gap[yy, xx] == 0 and mask2[yy, xx] == 0 and class2[yy, xx] == classij:
                                c_common += 1
                    if c_common > 3.0 * min_similar:
                        end_w = mid - 1
                    else:
                        if c_common < 2.0 * min_similar:
                            start_w = mid + 1
                        else:
                            ind_success = 1

                idx = 0
                for yy in range(b1, b2 + 1):
                    for xx in range(a1, a2 + 1):
                        if gap[yy, xx] == 0 and mask2[yy, xx] == 0 and class2[yy, xx] == classij:
                            cand_y[idx] = yy
                            cand_x[idx] = xx
                            dy = (j - b1) - (yy - b1)
                            dx = (i - a1) - (xx - a1)
                            disi[idx] = np.sqrt(float(dx * dx + dy * dy))
                            s1 = 0.0
                            s2 = 0.0
                            for ib in range(num_band):
                                on_common = images[donor_idx, ib, yy, xx]
                                d1 = on_common - images[donor_idx, ib, j, i]
                                d2 = on_common - images[target_index, ib, yy, xx]
                                s1 += d1 * d1
                                s2 += d2 * d2
                            rmsei[idx] = np.sqrt(s1 / num_band) + 0.0001
                            rmse12[idx] = np.sqrt(s2 / num_band) + 0.0001
                            idx += 1
                c_common = idx

                if c_common >= 2.0 * min_similar:
                    order_rmse = np.argsort(rmsei[:c_common])
                    use_n = min_similar
                    weight_sum = 0.0
                    t1 = 0.0
                    t2 = 0.0
                    for kk in range(use_n):
                        ci = order_rmse[kk]
                        cd = rmsei[ci] * disi[ci]
                        w = np.inf if cd == 0.0 else 1.0 / cd
                        disi[kk] = w
                        cand_y[kk] = cand_y[ci]
                        cand_x[kk] = cand_x[ci]
                        rmsei[kk] = rmsei[ci]
                        rmse12[kk] = rmse12[ci]
                        weight_sum += w
                        t1 += rmsei[kk]
                        t2 += rmse12[kk]
                    for kk in range(use_n):
                        disi[kk] = np.nan if weight_sum == 0.0 else disi[kk] / weight_sum
                    t1 = t1 / use_n
                    t2 = t2 / use_n
                    denom_t = t1 + t2
                    w_t1 = np.nan if denom_t == 0.0 else t2 / denom_t
                    w_t2 = np.nan if denom_t == 0.0 else t1 / denom_t
                    for ib in range(num_band):
                        predict_1 = 0.0
                        delta = 0.0
                        for kk in range(use_n):
                            yy = cand_y[kk]
                            xx = cand_x[kk]
                            w = disi[kk]
                            similar_off = images[target_index, ib, yy, xx]
                            similar_on = images[donor_idx, ib, yy, xx]
                            predict_1 += similar_off * w
                            delta += (similar_off - similar_on) * w
                        predict_2 = images[donor_idx, ib, j, i] + delta
                        if dn_min < predict_2 < dn_max:
                            fine0[ib, j, i] = w_t1 * predict_1 + w_t2 * predict_2
                        else:
                            fine0[ib, j, i] = predict_1
                    mark[j, i] = 1 + 10 * i_input
                    gap[j, i] = -2

                else:
                    if c_common >= 3 and mid >= max_window:
                        weight_sum = 0.0
                        t1 = 0.0
                        t2 = 0.0
                        for kk in range(c_common):
                            cd = rmsei[kk] * disi[kk]
                            w = np.inf if cd == 0.0 else 1.0 / cd
                            disi[kk] = w
                            weight_sum += w
                            t1 += rmsei[kk]
                            t2 += rmse12[kk]
                        for kk in range(c_common):
                            disi[kk] = np.nan if weight_sum == 0.0 else disi[kk] / weight_sum
                        t1 = t1 / c_common
                        t2 = t2 / c_common
                        denom_t = t1 + t2
                        w_t1 = np.nan if denom_t == 0.0 else t2 / denom_t
                        w_t2 = np.nan if denom_t == 0.0 else t1 / denom_t
                        for ib in range(num_band):
                            predict_1 = 0.0
                            delta = 0.0
                            for kk in range(c_common):
                                yy = cand_y[kk]
                                xx = cand_x[kk]
                                w = disi[kk]
                                similar_off = images[target_index, ib, yy, xx]
                                similar_on = images[donor_idx, ib, yy, xx]
                                predict_1 += similar_off * w
                                delta += (similar_off - similar_on) * w
                            predict_2 = images[donor_idx, ib, j, i] + delta
                            if dn_min < predict_2 < dn_max:
                                fine0[ib, j, i] = w_t1 * predict_1 + w_t2 * predict_2
                            else:
                                fine0[ib, j, i] = predict_1
                        mark[j, i] = 2 + 10 * i_input
                        gap[j, i] = -2
                    else:
                        first_half = target_index - 1
                        if first_half < 0:
                            first_half = 0
                        second_half = target_index + 1
                        if second_half > n_image - 1:
                            second_half = n_image - 1
                        num_before = 0
                        i_before = -1
                        for kk in range(0, first_half + 1):
                            if masks[kk, j, i] == 0:
                                num_before += 1
                                i_before = kk
                        num_after = 0
                        i_after = -1
                        for kk in range(second_half, n_image):
                            if masks[kk, j, i] == 0:
                                if num_after == 0:
                                    i_after = kk
                                num_after += 1
                        if num_before > 0 and num_after > 0:
                            T1 = doys[target_index] - doys[i_before]
                            T2 = doys[i_after] - doys[target_index]
                            for ib in range(num_band):
                                before_band = images[i_before, ib, j, i]
                                after_band = images[i_after, ib, j, i]
                                denom_time = T2 + T1
                                fine0[ib, j, i] = np.nan if denom_time == 0 else (T2 * before_band + T1 * after_band) / denom_time
                        else:
                            for ib in range(num_band):
                                fine0[ib, j, i] = images[donor_idx, ib, j, i]
                        mark[j, i] = 3 + 10 * i_input
                        gap[j, i] = -2

        c_unfilled = 0
        for yy in range(nl):
            for xx in range(ns):
                if gap[yy, xx] > 0:
                    c_unfilled += 1
        if c_unfilled == 0:
            note_finish = 1
        else:
            i_input += 1

    for j in range(nl):
        for i in range(ns):
            if fill_mask[j, i] and gap[j, i] > 0:
                for ib in range(num_band):
                    fine0[ib, j, i] = np.nan
    return fine0, mark


def run_nspi_timeseries_faithful_crop(
    target: np.ndarray,
    target_train_valid: np.ndarray,
    fill_mask: np.ndarray,
    donors: list[tuple[DonorInfo, np.ndarray, np.ndarray]],
    target_date: str,
) -> tuple[np.ndarray, np.ndarray]:
    stack_items: list[tuple[str, np.ndarray, np.ndarray]] = [(target_date, target, target_train_valid)]
    stack_items.extend((d.date, dopt, dvalid) for d, dopt, dvalid in donors)
    stack_items.sort(key=lambda item: np.datetime64(item[0]))
    target_index = [i for i, item in enumerate(stack_items) if item[0] == target_date and item[1] is target][0]
    images = np.stack([item[1] for item in stack_items]).astype(np.float32)
    masks = np.stack([np.where(item[2], 0, 1).astype(np.uint8) for item in stack_items])
    doys = np.array([date_to_doy(item[0]) for item in stack_items], dtype=np.int32)
    classes = classify_timeseries_images(images, masks, NSPI_NUM_CLASS)
    return _nspi_timeseries_fillsingle_kernel(
        images,
        masks,
        classes,
        doys,
        target_index,
        fill_mask,
        target_train_valid,
        NSPI_MIN_SIMILAR,
        DN_MIN,
        DN_MAX,
    )


def run_nspi_single_crop(
    target: np.ndarray,
    donor: np.ndarray,
    target_train_valid: np.ndarray,
    donor_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    th = compute_similarity_threshold(donor, donor_valid, NSPI_NUM_CLASS)
    return _nspi_single(target, donor, target_train_valid, donor_valid, th, NSPI_MIN_SIMILAR, NSPI_MAX_WINDOW, DN_MIN, DN_MAX)


def run_nspi_multi_crop(
    target: np.ndarray,
    donors: list[tuple[DonorInfo, np.ndarray, np.ndarray]],
    target_train_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    filled = target.copy()
    quality = np.zeros(target.shape[1:], dtype=np.uint8)
    remaining = ~target_train_valid
    for idx, (_info, donor, donor_valid) in enumerate(donors, start=1):
        donor_out, donor_q = run_nspi_single_crop(target, donor, target_train_valid, donor_valid)
        usable = remaining & np.isin(donor_q, np.array([1, 2, 3], dtype=np.uint8))
        for b in range(target.shape[0]):
            filled[b, usable] = donor_out[b, usable]
        quality[usable] = donor_q[usable] + 10 * idx
        remaining[usable] = False
    quality[remaining] = 5
    for b in range(target.shape[0]):
        filled[b, remaining] = np.nan
    return filled, quality


def donor_order_variants(donors: list[DonorInfo], target_date: str) -> list[tuple[str, list[DonorInfo]]]:
    by_kind = {d.kind: d for d in donors}
    variants: list[tuple[str, list[DonorInfo]]] = []
    score_order = sorted(donors, key=lambda d: d.score, reverse=True)
    variants.append(("score_all3", score_order))
    # date closeness in days, string compare ok only not enough; use numpy datetime64.
    td = np.datetime64(target_date)
    variants.append(("date_all3", sorted(donors, key=lambda d: abs((np.datetime64(d.date) - td).astype(int)))))
    for names in [
        ("lt05_on_off", ["lt05", "le07_slc_on", "le07_slc_off"]),
        ("on_lt05_off", ["le07_slc_on", "lt05", "le07_slc_off"]),
        ("off_lt05_on", ["le07_slc_off", "lt05", "le07_slc_on"]),
    ]:
        label, kinds = names
        seq = [by_kind[k] for k in kinds if k in by_kind]
        if len(seq) >= 2:
            variants.append((label, seq))
    # Pairs answer whether all3 beats simpler multi.
    for a, b in [("lt05", "le07_slc_on"), ("lt05", "le07_slc_off"), ("le07_slc_on", "le07_slc_off")]:
        seq = [by_kind[k] for k in (a, b) if k in by_kind]
        if len(seq) == 2:
            variants.append((f"pair_{a}_then_{b}", seq))
    # Deduplicate same label/order.
    seen = set()
    out = []
    for label, seq in variants:
        key = (label, tuple(d.kind for d in seq))
        if key not in seen:
            seen.add(key)
            out.append((label, seq))
    return out


def doy_distance_days(date_a: str, date_b: str) -> int:
    a = np.datetime64(date_a, "D").astype(object)
    b = np.datetime64(date_b, "D").astype(object)
    da = int(a.strftime("%j"))
    db = int(b.strftime("%j"))
    d = abs(da - db)
    return min(d, 366 - d)


def date_distance_days(date_a: str, date_b: str) -> int:
    return int(abs((np.datetime64(date_a) - np.datetime64(date_b)).astype(int)))


def make_best_of_single_nspi_candidates(
    nspi_candidates: list[dict[str, Any]],
    truth: np.ndarray,
    target_date: str,
) -> list[dict[str, Any]]:
    """Build oracle and deployable per-pixel selectors from independent NSPI runs."""
    if len(nspi_candidates) < 2:
        return []
    shape = truth.shape[1:]
    preds = [c["pred"] for c in nspi_candidates]
    valids = [c["valid"] for c in nspi_candidates]
    qualities = [c.get("quality") for c in nspi_candidates]

    def empty_pred() -> np.ndarray:
        out = np.full_like(truth, np.nan, dtype=np.float32)
        return out

    out: list[dict[str, Any]] = []

    # Truth oracle: upper bound on whether donors are complementary.
    oracle_pred = empty_pred()
    oracle_valid = np.zeros(shape, dtype=bool)
    best_err = np.full(shape, np.inf, dtype=np.float32)
    for pred, valid in zip(preds, valids):
        err = np.full(shape, np.inf, dtype=np.float32)
        if valid.any():
            diff = pred[:, valid] - truth[:, valid]
            err[valid] = np.sqrt(np.nanmean(diff * diff, axis=0)).astype(np.float32)
        take = valid & (err < best_err)
        for b in range(truth.shape[0]):
            oracle_pred[b, take] = pred[b, take]
        best_err[take] = err[take]
        oracle_valid |= take
    out.append(
        {
            "method": "nspi_best_of_single_oracle",
            "donor_policy": "oracle_min_error",
            "donor_order": "per_pixel_truth_oracle",
            "pred": oracle_pred,
            "valid": oracle_valid,
            "real_gap_candidate_fraction": max(c.get("real_gap_candidate_fraction", 0.0) for c in nspi_candidates),
        }
    )

    # Deployable v1: prefer NSPI quality 1, then 2, then 3; ties by Step 3 score.
    quality_pred = empty_pred()
    quality_valid = np.zeros(shape, dtype=bool)
    best_rank = np.full(shape, np.inf, dtype=np.float32)
    for cand, pred, valid, q in zip(nspi_candidates, preds, valids, qualities):
        if q is None:
            q_rank = np.where(valid, 9.0, np.inf).astype(np.float32)
        else:
            # smaller quality better; tie breaker via donor score.
            q_rank = q.astype(np.float32) - 0.01 * float(cand.get("donor_score", 0.0))
        take = valid & (q_rank < best_rank)
        for b in range(truth.shape[0]):
            quality_pred[b, take] = pred[b, take]
        best_rank[take] = q_rank[take]
        quality_valid |= take
    out.append(
        {
            "method": "nspi_best_of_single_policy",
            "donor_policy": "quality_then_step3",
            "donor_order": "per_pixel_quality",
            "pred": quality_pred,
            "valid": quality_valid,
            "real_gap_candidate_fraction": max(c.get("real_gap_candidate_fraction", 0.0) for c in nspi_candidates),
        }
    )

    # Deployable v2: choose valid prediction closest to cross-donor consensus,
    # with NSPI quality and Step3 score tie-breaks. No truth used.
    agreement_pred = empty_pred()
    agreement_valid = np.zeros(shape, dtype=bool)
    stack = np.stack(preds, axis=0)  # donor,b,y,x
    valid_stack = np.stack(valids, axis=0)
    masked = np.where(valid_stack[:, None, :, :], stack, np.nan)
    median_pred = np.nanmedian(masked, axis=0)
    best_agree = np.full(shape, np.inf, dtype=np.float32)
    for cand, pred, valid, q in zip(nspi_candidates, preds, valids, qualities):
        dist = np.sqrt(np.nanmean((pred - median_pred) ** 2, axis=0)).astype(np.float32)
        q_penalty = np.where(q is None, 2.0, np.clip(q.astype(np.float32), 1, 5) * 0.002)
        score = dist + q_penalty - 0.001 * float(cand.get("donor_score", 0.0))
        take = valid & (score < best_agree)
        for b in range(truth.shape[0]):
            agreement_pred[b, take] = pred[b, take]
        best_agree[take] = score[take]
        agreement_valid |= take
    out.append(
        {
            "method": "nspi_best_of_single_policy",
            "donor_policy": "agreement_quality_step3",
            "donor_order": "per_pixel_agreement",
            "pred": agreement_pred,
            "valid": agreement_valid,
            "real_gap_candidate_fraction": max(c.get("real_gap_candidate_fraction", 0.0) for c in nspi_candidates),
        }
    )

    # NSPI time-series inspired deployable blend. Original upstream Python code
    # sorts available dates by DOY proximity and blends spatial/temporal NSPI
    # estimates with quality/fallback flags. Exact script is UI/temp-file/per-pixel
    # heavy, so this tournament comparator uses independent NSPI predictions and
    # weights valid donors by season proximity, acquisition proximity, NSPI quality,
    # Step3 score, and real target-gap coverage. No truth used.
    ts_weight_sum = np.zeros(shape, dtype=np.float32)
    ts_pred = np.zeros_like(truth, dtype=np.float32)
    for cand, pred, valid, q in zip(nspi_candidates, preds, valids, qualities):
        donor_date = str(cand.get("donor_date") or target_date)
        doy_w = math.exp(-doy_distance_days(target_date, donor_date) / 45.0)
        date_w = math.exp(-date_distance_days(target_date, donor_date) / 3650.0)
        score_w = 0.5 + max(0.0, float(cand.get("donor_score", 0.0)))
        cover_w = 0.25 + float(cand.get("real_gap_candidate_fraction", 0.0))
        base = float(doy_w * date_w * score_w * cover_w)
        if q is None:
            q_w = np.where(valid, 0.3, 0.0).astype(np.float32)
        else:
            q_w = np.where(q == 1, 1.0, np.where(q == 2, 0.65, np.where(q == 3, 0.35, 0.0))).astype(np.float32)
        w = np.where(valid, base * q_w, 0.0).astype(np.float32)
        ts_weight_sum += w
        for b in range(truth.shape[0]):
            ts_pred[b] += np.where(w > 0, pred[b] * w, 0.0)
    ts_valid = ts_weight_sum > 0
    for b in range(truth.shape[0]):
        ts_pred[b] = np.where(ts_valid, ts_pred[b] / np.maximum(ts_weight_sum, EPS), np.nan)
    out.append(
        {
            "method": "nspi_timeseries_weighted",
            "donor_policy": "doy_quality_step3_blend",
            "donor_order": "per_pixel_weighted_blend",
            "pred": ts_pred,
            "valid": ts_valid & np.isfinite(ts_pred).all(axis=0),
            "real_gap_candidate_fraction": max(c.get("real_gap_candidate_fraction", 0.0) for c in nspi_candidates),
        }
    )

    # Deployable v0: global Step 3 donor score among valid independent NSPI predictions.
    score_pred = empty_pred()
    score_valid = np.zeros(shape, dtype=bool)
    best_score = np.full(shape, -np.inf, dtype=np.float32)
    for cand, pred, valid in zip(nspi_candidates, preds, valids):
        score = float(cand.get("donor_score", 0.0))
        take = valid & (score > best_score)
        for b in range(truth.shape[0]):
            score_pred[b, take] = pred[b, take]
        best_score[take] = score
        score_valid |= take
    out.append(
        {
            "method": "nspi_best_of_single_policy",
            "donor_policy": "step3_score",
            "donor_order": "per_pixel_step3",
            "pred": score_pred,
            "valid": score_valid,
            "real_gap_candidate_fraction": max(c.get("real_gap_candidate_fraction", 0.0) for c in nspi_candidates),
        }
    )
    return out


def append_candidate_metrics(
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    truth: np.ndarray,
    classes: dict[str, np.ndarray],
    denominator: np.ndarray,
    target_id: int,
    window_index: int,
    scope: str,
    group: str,
    existing_keys: set[tuple[str, str, str, str, str, str, str, str, str, str]] | None = None,
    band_groups: list[str] | None = None,
    value_mode: str = "source",
) -> None:
    if not candidates:
        return
    if band_groups is None:
        band_groups = ["optical6"]

    def append_one(cand: dict[str, Any], eval_mask: np.ndarray) -> None:
        for band_group in band_groups or ["optical6"]:
            spec = BAND_GROUPS[band_group]
            base = {
                "target_id": target_id,
                "window": window_index,
                "method": cand["method"],
                "donor_policy": cand["donor_policy"],
                "donor_order": cand["donor_order"],
                "mask_type": cand.get("mask_type", "unknown"),
                "mask_name": cand.get("mask_name", "unknown"),
                "metric_scope": scope,
                "comparison_group": group,
                "band_group": band_group,
                "value_mode": value_mode,
                "value_scale": value_scale_for_band_group(band_group, value_mode),
                "real_gap_candidate_fraction": cand.get("real_gap_candidate_fraction", ""),
            }
            if existing_keys is not None and metric_key(base) in existing_keys:
                continue
            row = metrics_for_prediction(
                cand["pred"],
                truth,
                eval_mask,
                classes,
                denominator,
                band_indices=spec["indices"],
                band_names=spec["names"],
            )
            row.update(base)
            rows.append(row)
            if existing_keys is not None:
                existing_keys.add(metric_key(row))

    if scope == "own":
        for cand in candidates:
            append_one(cand, cand["valid"])
        return

    common = denominator.copy()
    for cand in candidates:
        common &= cand["valid"]
    if int(common.sum()) < 500:
        return
    for cand in candidates:
        append_one(cand, common)


def run_for_id(
    target_id: int,
    target_meta: dict[str, Any],
    donors: list[DonorInfo],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    out_dir = OUTPUT_ROOT / f"{target_id:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "validation_metrics.csv"
    target_dir = TARGET_DIR_FULL8 if args.band_mode == "full8" else TARGET_DIR
    donor_dir = DONOR_DIR_FULL8 if args.band_mode == "full8" else DONOR_DIR
    _image_bands, _data_idx, _clear_idx, gap_idx, band_names = stack_schema(args.band_mode)
    band_groups = band_groups_for_mode(args.band_mode)
    rows = [] if args.overwrite else read_csv_rows(metrics_path)
    existing_keys = set() if args.overwrite else {metric_key(row) for row in rows}
    if rows:
        print(f"ID {target_id:02d}: resume with {len(rows)} existing metric rows", flush=True)
    tpath = target_path(target_meta, target_dir)
    print(f"ID {target_id:02d}: load target {tpath}", flush=True)
    target, target_present, target_clear, target_slc_gap, profile = load_stack(tpath, args.band_mode)
    domain = rasterize_domain(target_meta, profile)
    target_clear = target_clear & domain
    target_gap = target_slc_gap & domain

    donor_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    donor_gap_masks: dict[str, np.ndarray] = {}
    for d in donors:
        print(f"ID {target_id:02d}: load donor {d.kind} {d.path.name}", flush=True)
        dopt, _dp, dclear, dgap, _prof = load_stack(d.path, args.band_mode)
        donor_arrays[d.kind] = (dopt, dclear & domain, dgap & domain)
        # Donor-gap-pattern validation should use real SLC-off stripe masks,
        # not pre-SLC footprint/domain differences.
        if d.kind == "le07_slc_off" and int((dgap & domain).sum()) > 0:
            donor_gap_masks[d.kind] = dgap & domain

    specs = make_holdout_specs(
        target_clear,
        target_gap,
        domain,
        donor_gap_masks,
        np.random.default_rng(args.seed + target_id),
        target_dir,
        donor_dir,
        gap_idx + 1,
    )
    rng_windows = np.random.default_rng(args.seed + 1000 + target_id)
    rng_windows.shuffle(specs)
    per_spec = max(1, math.ceil(args.max_windows / max(1, len(specs))))
    window_specs: list[tuple[HoldoutSpec, tuple[int, int, int, int]]] = []
    for spec in specs:
        for win in pick_windows(spec.mask, args.window_size, per_spec, args.min_holdout_pixels, rng_windows):
            window_specs.append((spec, win))
            if len(window_specs) >= args.max_windows:
                break
        if len(window_specs) >= args.max_windows:
            break
    by_type = defaultdict(int)
    for spec, _win in window_specs:
        by_type[spec.mask_type] += 1
    print(
        f"ID {target_id:02d}: donors={len(donors)} masks={len(specs)} windows={len(window_specs)} "
        f"by_type={dict(by_type)}",
        flush=True,
    )
    if not window_specs:
        return []

    # Global per-donor affine models use full-scene sampled overlap excluding validation masks.
    global_holdout = np.zeros_like(target_clear, dtype=bool)
    for spec, _win in window_specs:
        global_holdout |= spec.mask
    train_global = target_clear & ~global_holdout
    if args.value_mode == "scene_minmax":
        mins, maxs = compute_scene_minmax_scales(target, train_global, donor_arrays, domain, global_holdout)
        target = apply_scene_minmax(target, mins, maxs)
        donor_arrays = {
            kind: (apply_scene_minmax(dopt, mins, maxs), dvalid, dgap)
            for kind, (dopt, dvalid, dgap) in donor_arrays.items()
        }
        print(
            f"ID {target_id:02d}: scene_minmax mins={np.round(mins, 6).tolist()} maxs={np.round(maxs, 6).tolist()}",
            flush=True,
        )
    global_coefs: dict[str, list[tuple[float, float]]] = {}
    for d in donors:
        dopt, dvalid, _dgap = donor_arrays[d.kind]
        global_coefs[d.kind] = sample_global_fit(
            target,
            dopt,
            train_global & dvalid,
            args.global_sample_pixels,
            np.random.default_rng(args.seed + 2000 + target_id + KIND_CODE[d.kind]),
        )

    real_gap_pixels = int(target_gap.sum())
    donor_real_gap_fraction = {
        kind: (float((target_gap & dvalid).sum() / real_gap_pixels) if real_gap_pixels else 0.0)
        for kind, (_dopt, dvalid, _dgap) in donor_arrays.items()
    }

    rows_before = len(rows)
    for wi, (spec, win) in enumerate(window_specs):
        print(
            f"ID {target_id:02d}: window {wi + 1}/{len(window_specs)} {spec.mask_type}/{spec.name} {win}",
            flush=True,
        )
        t = crop(target, win)
        tclear = crop(target_clear, win)
        h = crop(spec.mask, win)
        train_valid = tclear & ~h
        cls = spectral_classes(t, band_names)
        legacy_specs: list[tuple[str, str]] = []
        for d in donors:
            legacy_specs.extend(
                [
                    ("affine_global", d.kind),
                    ("affine_window", d.kind),
                    ("mnspi_class_affine", d.kind),
                    ("usgs_local", d.kind),
                ]
            )
            if not args.skip_nspi:
                legacy_specs.append(("nspi_single", d.kind))
        if not args.skip_nspi:
            for label, _seq in donor_order_variants(donors, target_meta["date"]):
                legacy_specs.append(("nspi_multi", label))
            if len(donors) >= 2:
                legacy_specs.extend(
                    [
                        ("nspi_best_of_single_oracle", "oracle_min_error"),
                        ("nspi_best_of_single_policy", "quality_then_step3"),
                        ("nspi_best_of_single_policy", "agreement_quality_step3"),
                        ("nspi_timeseries_weighted", "doy_quality_step3_blend"),
                        ("nspi_best_of_single_policy", "step3_score"),
                    ]
                )
        need_legacy = args.overwrite or any(
            not has_all_band_metrics(
                existing_keys,
                target_id,
                wi,
                method,
                policy,
                spec.mask_type,
                spec.name,
                band_groups,
                value_mode=args.value_mode,
            )
            for method, policy in legacy_specs
        )
        faithful_policy = "original_fillsingle"
        need_faithful = (not args.skip_nspi) and (
            args.overwrite
            or not has_all_band_metrics(
                existing_keys,
                target_id,
                wi,
                "nspi_timeseries_faithful",
                faithful_policy,
                spec.mask_type,
                spec.name,
                band_groups,
                value_mode=args.value_mode,
            )
        )
        if not need_legacy and not need_faithful:
            print(f"ID {target_id:02d}: window {wi + 1} all metric rows exist; skip compute", flush=True)
            continue

        candidates: list[dict[str, Any]] = []
        nspi_single_candidates: list[dict[str, Any]] = []

        # Single-donor regression and NSPI.
        for d in donors:
            dopt_full, dvalid_full, _dgap_full = donor_arrays[d.kind]
            dopt = crop(dopt_full, win)
            dvalid = crop(dvalid_full, win)
            usable_h = h & dvalid
            if not need_legacy:
                continue

            pred_global = predict_affine(dopt, global_coefs[d.kind])
            candidates.append(
                {
                    "method": "affine_global",
                    "donor_policy": d.kind,
                    "donor_order": d.kind,
                    "pred": pred_global,
                    "valid": usable_h & np.isfinite(pred_global).all(axis=0),
                    "real_gap_candidate_fraction": donor_real_gap_fraction.get(d.kind, 0.0),
                }
            )

            local_coefs = []
            local_train = train_valid & dvalid
            for b in range(t.shape[0]):
                local_coefs.append(fit_affine(dopt[b, local_train], t[b, local_train]))
            pred_local = predict_affine(dopt, local_coefs)
            candidates.append(
                {
                    "method": "affine_window",
                    "donor_policy": d.kind,
                    "donor_order": d.kind,
                    "pred": pred_local,
                    "valid": usable_h & np.isfinite(pred_local).all(axis=0),
                    "real_gap_candidate_fraction": donor_real_gap_fraction.get(d.kind, 0.0),
                }
            )

            pred_mnspi, mnspi_valid = predict_mnspi_class_affine(
                t,
                dopt,
                train_valid,
                dvalid,
                global_coefs[d.kind],
                rng=np.random.default_rng(args.seed + target_id * 1000 + wi),
            )
            candidates.append(
                {
                    "method": "mnspi_class_affine",
                    "donor_policy": d.kind,
                    "donor_order": d.kind,
                    "pred": pred_mnspi,
                    "valid": h & mnspi_valid,
                    "real_gap_candidate_fraction": donor_real_gap_fraction.get(d.kind, 0.0),
                }
            )

            pred_usgs, usgs_valid = predict_usgs_local(t, dopt, train_valid, dvalid)
            candidates.append(
                {
                    "method": "usgs_local",
                    "donor_policy": d.kind,
                    "donor_order": d.kind,
                    "pred": pred_usgs,
                    "valid": h & usgs_valid,
                    "real_gap_candidate_fraction": donor_real_gap_fraction.get(d.kind, 0.0),
                }
            )

            if args.skip_nspi:
                continue
            nspi_pred, nspi_q = run_nspi_single_crop(t, dopt, train_valid, dvalid)
            nspi_valid = usable_h & np.isin(nspi_q, np.array([1, 2, 3], dtype=np.uint8))
            nspi_candidate = {
                "method": "nspi_single",
                "donor_policy": d.kind,
                "donor_order": d.kind,
                "pred": nspi_pred,
                "valid": nspi_valid & np.isfinite(nspi_pred).all(axis=0),
                "quality": nspi_q,
                "donor_score": d.score,
                "donor_date": d.date,
                "real_gap_candidate_fraction": donor_real_gap_fraction.get(d.kind, 0.0),
            }
            candidates.append(nspi_candidate)
            nspi_single_candidates.append(nspi_candidate)

        if need_legacy and not args.skip_nspi:
            # Multi-donor NSPI variants.
            for label, seq in donor_order_variants(donors, target_meta["date"]):
                seq_crops = []
                valid_any = np.zeros_like(h, dtype=bool)
                for d in seq:
                    dopt_full, dvalid_full, _dgap_full = donor_arrays[d.kind]
                    dopt = crop(dopt_full, win)
                    dvalid = crop(dvalid_full, win)
                    seq_crops.append((d, dopt, dvalid))
                    valid_any |= dvalid
                pred, q = run_nspi_multi_crop(t, seq_crops, train_valid)
                usable = h & valid_any & (q != 5)
                candidates.append(
                    {
                        "method": "nspi_multi",
                        "donor_policy": label,
                        "donor_order": ">".join(d.kind for d in seq),
                        "pred": pred,
                        "valid": usable & np.isfinite(pred).all(axis=0),
                        "real_gap_candidate_fraction": max(donor_real_gap_fraction.get(d.kind, 0.0) for d in seq),
                    }
                )

            candidates.extend(make_best_of_single_nspi_candidates(nspi_single_candidates, t, target_meta["date"]))

        if need_faithful:
            seq_crops = []
            valid_any = np.zeros_like(h, dtype=bool)
            for d in donors:
                dopt_full, dvalid_full, _dgap_full = donor_arrays[d.kind]
                dopt = crop(dopt_full, win)
                dvalid = crop(dvalid_full, win)
                seq_crops.append((d, dopt, dvalid))
                valid_any |= dvalid
            pred, mark = run_nspi_timeseries_faithful_crop(t, train_valid, h, seq_crops, target_meta["date"])
            candidates.append(
                {
                    "method": "nspi_timeseries_faithful",
                    "donor_policy": faithful_policy,
                    "donor_order": "original_doy_order",
                    "pred": pred,
                    "valid": h & valid_any & (mark > 0) & np.isfinite(pred).all(axis=0),
                    "real_gap_candidate_fraction": max(donor_real_gap_fraction.get(d.kind, 0.0) for d in donors),
                }
            )

        for cand in candidates:
            cand["mask_type"] = spec.mask_type
            cand["mask_name"] = spec.name
        append_candidate_metrics(
            rows,
            candidates,
            t,
            cls,
            h,
            target_id,
            wi,
            "own",
            "own_valid_pixels",
            existing_keys,
            band_groups,
            args.value_mode,
        )
        append_candidate_metrics(
            rows,
            [c for c in candidates if c["method"] == "nspi_single"],
            t,
            cls,
            h,
            target_id,
            wi,
            "common",
            "nspi_single_donors",
            existing_keys,
            band_groups,
            args.value_mode,
        )
        append_candidate_metrics(
            rows,
            [c for c in candidates if c["method"] == "nspi_multi"],
            t,
            cls,
            h,
            target_id,
            wi,
            "common",
            "nspi_multi_policies",
            existing_keys,
            band_groups,
            args.value_mode,
        )
        append_candidate_metrics(
            rows,
            [c for c in candidates if c["method"] in {"nspi_single", "nspi_multi", "nspi_timeseries_weighted"}],
            t,
            cls,
            h,
            target_id,
            wi,
            "common",
            "nspi_all",
            existing_keys,
            band_groups,
            args.value_mode,
        )

    added = len(rows) - rows_before
    args.added_total = getattr(args, "added_total", 0) + added
    if args.overwrite or added > 0:
        write_csv(metrics_path, rows)
        summary = summarize_rows(rows)
        write_json(out_dir / "summary.json", summary)
        print(f"ID {target_id:02d}: wrote {metrics_path} (+{added} rows)", flush=True)
    else:
        print(f"ID {target_id:02d}: no missing metric rows; left files unchanged", flush=True)
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("valid_pixels", 0) and not math.isnan(float(row.get("mean_rmse", math.nan))):
            for mask_group in ("all", str(row.get("mask_type", "unknown"))):
                groups[
                    (
                        mask_group,
                        str(row.get("metric_scope", "own")),
                        str(row.get("comparison_group", "own_valid_pixels")),
                        str(row["method"]),
                        str(row["donor_policy"]),
                        row_band_group(row),
                        row_value_mode(row),
                    )
                ].append(row)
    scored = []
    for (mask_group, scope, group, method, policy, band_group, value_mode), items in groups.items():
        weights = np.array([float(r["valid_pixels"]) for r in items], dtype=np.float64)
        rmses = np.array([float(r["mean_rmse"]) for r in items], dtype=np.float64)
        maes = np.array([float(r["mean_mae"]) for r in items], dtype=np.float64)
        eff = np.array([float(r.get("effective_rmse", math.nan)) for r in items], dtype=np.float64)
        holdout_pixels = np.array([float(r.get("holdout_pixels", 0)) for r in items], dtype=np.float64)
        valid_pixels = np.array([float(r.get("valid_pixels", 0)) for r in items], dtype=np.float64)
        real_gap_fracs = np.array(
            [float(r.get("real_gap_candidate_fraction") or 0.0) for r in items], dtype=np.float64
        )
        if weights.sum() <= 0:
            continue
        total_holdout = float(holdout_pixels.sum())
        total_valid = float(valid_pixels.sum())
        scored.append(
            {
                "mask_type": mask_group,
                "metric_scope": scope,
                "comparison_group": group,
                "method": method,
                "donor_policy": policy,
                "band_group": band_group,
                "value_mode": value_mode,
                "windows": len(items),
                "valid_pixels": int(total_valid),
                "holdout_pixels": int(total_holdout),
                "holdout_fill_fraction": total_valid / total_holdout if total_holdout else 0.0,
                "weighted_mean_rmse": float(np.average(rmses, weights=weights)),
                "weighted_mean_mae": float(np.average(maes, weights=weights)),
                "weighted_effective_rmse": float(np.average(eff, weights=weights)),
                "real_gap_candidate_fraction": float(np.average(real_gap_fracs, weights=weights)),
            }
        )
    scored.sort(key=lambda r: (r["mask_type"] != "all", r["metric_scope"] != "own", r["weighted_effective_rmse"]))
    own = [row for row in scored if row["metric_scope"] == "own" and row["mask_type"] == "all"]
    return {"ranked": scored, "best": own[0] if own else (scored[0] if scored else None)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local gapfill donor/method tournament")
    parser.add_argument("--ids", default=DEFAULT_IDS, help="Comma-separated IDs, default 04,16,26")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--max-windows", type=int, default=DEFAULT_MAX_WINDOWS)
    parser.add_argument("--min-holdout-pixels", type=int, default=DEFAULT_MIN_HOLDOUT_PIXELS)
    parser.add_argument("--global-sample-pixels", type=int, default=DEFAULT_GLOBAL_SAMPLE_PIXELS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--band-mode",
        choices=["optical6", "full8"],
        default="optical6",
        help="Use original 11-band optical6 raw stacks or new 13-band full8 raw_full8 stacks",
    )
    parser.add_argument(
        "--value-mode",
        choices=["source", "scene_minmax"],
        default="source",
        help="Evaluate source units or per-scene per-band min-max normalized values",
    )
    parser.add_argument("--skip-nspi", action="store_true", help="Run only affine baselines")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild per-ID validation_metrics.csv instead of appending missing rows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.added_total = 0
    targets = load_targets_meta()
    donor_dir = DONOR_DIR_FULL8 if args.band_mode == "full8" else DONOR_DIR
    slate = load_slate(donor_dir)
    all_rows: list[dict[str, Any]] = []
    for target_id in parse_ids(args.ids):
        if target_id not in targets:
            raise SystemExit(f"unknown target id {target_id}")
        donors = slate.get(target_id, [])
        if not donors:
            raise SystemExit(f"no downloaded slate donors for ID {target_id:02d}")
        rows = run_for_id(target_id, targets[target_id], donors, args)
        all_rows.extend(rows)
    if args.overwrite or args.added_total > 0:
        global_rows: list[dict[str, Any]] = []
        for path in sorted(OUTPUT_ROOT.glob("??/validation_metrics.csv")):
            global_rows.extend(read_csv_rows(path))
        summary_rows = []
        by_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in global_rows:
            by_id[int(row["target_id"])].append(row)
        for target_id, rows in sorted(by_id.items()):
            summary = summarize_rows(rows)
            for rank, item in enumerate(summary["ranked"], start=1):
                summary_rows.append({"target_id": target_id, "rank": rank, **item})
        write_csv(OUTPUT_ROOT / "validation_metrics_all.csv", global_rows)
        write_csv(OUTPUT_ROOT / "summary.csv", summary_rows)
        write_json(
            OUTPUT_ROOT / "summary.json",
            {"ids": sorted(by_id), "ranked_by_id": summary_rows[:200]},
        )
        print(f"wrote {OUTPUT_ROOT / 'summary.csv'}")
        print(f"wrote {OUTPUT_ROOT / 'validation_metrics_all.csv'}")
    else:
        print("no missing metric rows; left global summary files unchanged")


if __name__ == "__main__":
    main()
