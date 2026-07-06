#!/usr/bin/env python3
"""Create local HKH fishnet GeoTIFF dataset variants from full8 raw scenes.

Self-contained dataset workflow inputs live under dataset/ except optional legacy
Landsat7_2005 rasters used as exact grid templates. Fishnet copy:
  dataset/hkh_fishnet.geojson

Outputs one folder per dataset directly under /home/devj/local-arch/data/HKH_raw/:
  image0.tif, image1.tif, ...

Filled variants run NSPI in per-scene min-max normalized space (same evidence base
as the normalized full8 tournament), then inverse-transform predictions back to
raw source values for GeoTIFF output. No ML normalization is applied here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from numba import njit, prange
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds, transform as window_transform
from rasterio.warp import reproject, transform_bounds, transform_geom

DATASET_DIR = Path(__file__).resolve().parent
TARGETS_JSON = DATASET_DIR / "outputs/1_targets.json"
SLATE_CSV = DATASET_DIR / "outputs/3_donor_slate_narrow.csv"
SUMMARY_CSV = DATASET_DIR / "outputs/5_gapfill_tournament/summary.csv"
FISHNET_GEOJSON = DATASET_DIR / "hkh_fishnet.geojson"
TARGET_DIR_FULL8 = DATASET_DIR / "raw_full8/targets"
DONOR_DIR_FULL8 = DATASET_DIR / "raw_full8/donors"
DEFAULT_TEMPLATE_DIR = Path("/home/devj/local-arch/data/HKH_raw/Landsat7_2005")
DEFAULT_OUTPUT_ROOT = Path("/home/devj/local-arch/data/HKH_raw")

FULL8_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6_VCID_1", "B6_VCID_2", "B7"]
PROVENANCE_BANDS = [
    "target_id",
    "target_year",
    "target_doy",
    "target_valid",
    "pixel_source",
    "source_year",
    "source_doy",
    "donor_kind",
    "fill_quality",
    "donor_bitmask",
]
PROVENANCE_PIXEL_SOURCE = {
    "nodata": 0,
    "target": 1,
    "nspi_selected": 2,
    "nspi_weighted_blend": 3,
}
DONOR_KIND_CODE = {"none": 0, "lt05": 1, "le07_slc_on": 2, "le07_slc_off": 3}
DONOR_BITMASK = {"lt05": 1, "le07_slc_on": 2, "le07_slc_off": 4}
VARIANTS = {
    "raw_target": "HKH_full8_raw_target",
    "nspi_timeseries_weighted": "HKH_full8_nspi_timeseries_weighted",
    "agreement_quality_step3": "HKH_full8_agreement_quality_step3",
    "nspi_multi_score_all3": "HKH_full8_nspi_multi_score_all3",
}
EPS = 1e-6
NSPI_MIN_SIMILAR = 20
NSPI_MAX_WINDOW = 8
NSPI_NUM_CLASS = 5


@dataclass(frozen=True)
class DonorInfo:
    kind: str
    path: Path
    date: str
    score: float


@dataclass(frozen=True)
class TileTemplate:
    index: int
    path: Path
    crs: Any
    transform: Any
    width: int
    height: int
    bounds: Any
    profile: dict[str, Any]


@njit(parallel=True, cache=True)
def _nspi_single_fill(
    target: np.ndarray,
    donor: np.ndarray,
    train_valid: np.ndarray,
    donor_valid: np.ndarray,
    fill_mask: np.ndarray,
    similar_th: float,
    min_similar: int,
    max_window: int,
    dn_min: float,
    dn_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    bands, height, width = target.shape
    out = target.copy()
    quality = np.zeros((height, width), dtype=np.uint8)
    max_candidates = (2 * max_window + 1) * (2 * max_window + 1)
    init_extent = int(np.ceil(0.5 * (np.sqrt(min_similar) - 1.0)))
    if init_extent < 1:
        init_extent = 1

    for y in prange(height):
        rmse = np.empty(max_candidates, dtype=np.float32)
        rmse12 = np.empty(max_candidates, dtype=np.float32)
        dist = np.empty(max_candidates, dtype=np.float32)
        cand_y = np.empty(max_candidates, dtype=np.int32)
        cand_x = np.empty(max_candidates, dtype=np.int32)
        for x in range(width):
            if not fill_mask[y, x]:
                quality[y, x] = 0
                continue
            if not donor_valid[y, x]:
                quality[y, x] = 5
                for b in range(bands):
                    out[b, y, x] = np.nan
                continue

            filled = False
            extent = init_extent
            while extent <= max_window and not filled:
                y1 = max(0, y - extent)
                y2 = min(height - 1, y + extent)
                x1 = max(0, x - extent)
                x2 = min(width - 1, x + extent)

                c_common = 0
                for yy in range(y1, y2 + 1):
                    for xx in range(x1, x2 + 1):
                        if train_valid[yy, xx] and donor_valid[yy, xx]:
                            diff_sq = 0.0
                            diff_sq2 = 0.0
                            good = True
                            for b in range(bands):
                                dv = donor[b, yy, xx]
                                dp = donor[b, y, x]
                                tv = target[b, yy, xx]
                                if not np.isfinite(dv) or not np.isfinite(dp) or not np.isfinite(tv):
                                    good = False
                                    break
                                d = dv - dp
                                diff_sq += d * d
                                d12 = dv - tv
                                diff_sq2 += d12 * d12
                            if good:
                                rmse[c_common] = np.sqrt(diff_sq / bands) + 0.0001
                                rmse12[c_common] = np.sqrt(diff_sq2 / bands) + 0.0001
                                dy = float(y - yy)
                                dx = float(x - xx)
                                dd = np.sqrt(dy * dy + dx * dx)
                                if dd < 1e-6:
                                    dd = 1e-6
                                dist[c_common] = dd
                                cand_y[c_common] = yy
                                cand_x[c_common] = xx
                                c_common += 1

                if c_common > min_similar:
                    c_similar = 0
                    for i in range(c_common):
                        if rmse[i] <= similar_th:
                            if c_similar != i:
                                rmse[c_similar] = rmse[i]
                                rmse12[c_similar] = rmse12[i]
                                dist[c_similar] = dist[i]
                                cand_y[c_similar] = cand_y[i]
                                cand_x[c_similar] = cand_x[i]
                            c_similar += 1

                    if c_similar < min_similar and extent < max_window:
                        extent += 1
                        continue

                    if c_similar > 0:
                        use_n = c_similar
                        qcode = 1 if c_similar >= min_similar else 2
                        weight_sum = 0.0
                        r1 = 0.0
                        r2 = 0.0
                        for i in range(use_n):
                            cd = rmse[i] * dist[i]
                            if cd < 1e-6:
                                cd = 1e-6
                            w = 1.0 / cd
                            dist[i] = w
                            weight_sum += w
                            r1 += rmse[i]
                            r2 += rmse12[i]
                        if weight_sum <= 0:
                            quality[y, x] = 5
                            break
                        for i in range(use_n):
                            dist[i] /= weight_sum
                        r1 /= use_n
                        r2 /= use_n
                        denom = r1 + r2
                        wt1 = 0.5 if denom <= 1e-6 else r2 / denom
                        wt2 = 0.5 if denom <= 1e-6 else r1 / denom

                        for b in range(bands):
                            predict1 = 0.0
                            delta = 0.0
                            for i in range(use_n):
                                yy = cand_y[i]
                                xx = cand_x[i]
                                w = dist[i]
                                predict1 += target[b, yy, xx] * w
                                delta += (target[b, yy, xx] - donor[b, yy, xx]) * w
                            predict2 = donor[b, y, x] + delta
                            if predict2 > dn_min and predict2 < dn_max:
                                out[b, y, x] = wt1 * predict1 + wt2 * predict2
                            else:
                                out[b, y, x] = predict1
                        quality[y, x] = qcode
                        filled = True
                        break

                    if c_similar == 0 and extent >= max_window:
                        if c_common > 0:
                            for b in range(bands):
                                delta_sum = 0.0
                                for i in range(c_common):
                                    yy = cand_y[i]
                                    xx = cand_x[i]
                                    delta_sum += target[b, yy, xx] - donor[b, yy, xx]
                                pred = donor[b, y, x] + delta_sum / c_common
                                if pred < dn_min:
                                    pred = dn_min
                                if pred > dn_max:
                                    pred = dn_max
                                out[b, y, x] = pred
                            quality[y, x] = 3
                            filled = True
                            break
                        quality[y, x] = 5
                        break
                else:
                    if extent < max_window:
                        extent += 1
                        continue
                    if c_common > 0:
                        for b in range(bands):
                            delta_sum = 0.0
                            for i in range(c_common):
                                yy = cand_y[i]
                                xx = cand_x[i]
                                delta_sum += target[b, yy, xx] - donor[b, yy, xx]
                            pred = donor[b, y, x] + delta_sum / c_common
                            if pred < dn_min:
                                pred = dn_min
                            if pred > dn_max:
                                pred = dn_max
                            out[b, y, x] = pred
                        quality[y, x] = 3
                        filled = True
                        break
                    quality[y, x] = 5
                    break

            if not filled and quality[y, x] == 0:
                quality[y, x] = 5
                for b in range(bands):
                    out[b, y, x] = np.nan
    return out, quality


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_int_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def load_targets_meta() -> dict[int, dict[str, Any]]:
    return {int(row["id"]): row for row in read_json(TARGETS_JSON)}


def load_slate() -> dict[int, list[DonorInfo]]:
    out: dict[int, list[DonorInfo]] = defaultdict(list)
    for row in read_csv_rows(SLATE_CSV):
        tid = int(row["target_id"])
        date = row["donor_date"].replace("-", "")
        path = DONOR_DIR_FULL8 / f"{tid:02d}_donor_{row['donor_kind']}_{date}.tif"
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


def target_path(target: dict[str, Any]) -> Path:
    return TARGET_DIR_FULL8 / target.get("target_filename", target["filename_target"])


def load_full8_stack(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        profile = src.profile.copy()
    if arr.shape[0] < 13:
        raise ValueError(f"{path} has {arr.shape[0]} bands; expected full8 13-band raw stack")
    image = arr[:8]
    data_present = arr[10] > 0.5
    clear_valid = arr[11] > 0.5
    slc_gap = arr[12] > 0.5
    finite = np.isfinite(image).all(axis=0)
    data_present &= finite
    clear_valid &= finite
    return image, data_present, clear_valid, slc_gap, profile


def rasterize_domain_array(
    target: dict[str, Any],
    crs: Any,
    transform: Any,
    height: int,
    width: int,
) -> np.ndarray:
    geom = transform_geom("EPSG:4326", crs, target["target_domain_geojson"])
    return geometry_mask(
        [geom],
        out_shape=(height, width),
        transform=transform,
        invert=True,
    )


def rasterize_domain(target: dict[str, Any], profile: dict[str, Any]) -> np.ndarray:
    return rasterize_domain_array(
        target,
        profile["crs"],
        profile["transform"],
        int(profile["height"]),
        int(profile["width"]),
    )


def expand_window(win: Window, pad: int, width: int, height: int) -> Window:
    col = max(0, int(math.floor(win.col_off)) - pad)
    row = max(0, int(math.floor(win.row_off)) - pad)
    col2 = min(width, int(math.ceil(win.col_off + win.width)) + pad)
    row2 = min(height, int(math.ceil(win.row_off + win.height)) + pad)
    return Window(col, row, max(0, col2 - col), max(0, row2 - row))


def tile_window_in_scene(tile: TileTemplate, scene_profile: dict[str, Any], pad: int = NSPI_MAX_WINDOW) -> Window | None:
    bounds = transform_bounds(tile.crs, scene_profile["crs"], *tile.bounds, densify_pts=21)
    with rasterio.io.MemoryFile() as mem:
        profile = {
            "driver": "GTiff",
            "width": int(scene_profile["width"]),
            "height": int(scene_profile["height"]),
            "count": 1,
            "dtype": "uint8",
            "crs": scene_profile["crs"],
            "transform": scene_profile["transform"],
        }
        with mem.open(**profile) as ds:
            win = from_bounds(*bounds, transform=ds.transform)
    win = expand_window(win, pad, int(scene_profile["width"]), int(scene_profile["height"]))
    if win.width <= 0 or win.height <= 0:
        return None
    return win


def load_full8_window(path: Path, win: Window) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as src:
        arr = src.read(indexes=list(range(1, 9)) + [11, 12, 13], window=win).astype(np.float32)
        profile = src.profile.copy()
        profile.update(
            width=int(win.width),
            height=int(win.height),
            transform=window_transform(win, src.transform),
        )
    image = arr[:8]
    data_present = arr[8] > 0.5
    clear_valid = arr[9] > 0.5
    slc_gap = arr[10] > 0.5
    finite = np.isfinite(image).all(axis=0)
    data_present &= finite
    clear_valid &= finite
    return image, data_present, clear_valid, slc_gap, profile


def compute_scene_minmax_stream(
    target_path_: Path,
    target_meta: dict[str, Any],
    donors: list[DonorInfo],
) -> tuple[np.ndarray, np.ndarray, dict[str, float], int]:
    mins = np.full(8, np.inf, dtype=np.float32)
    maxs = np.full(8, -np.inf, dtype=np.float32)
    gap_total = 0
    donor_gap_clear = {d.kind: 0 for d in donors}
    donor_datasets = []
    with rasterio.open(target_path_) as target_ds:
        for d in donors:
            donor_datasets.append((d, rasterio.open(d.path)))
        try:
            for _idx, win in target_ds.block_windows(1):
                t_img = target_ds.read(indexes=list(range(1, 9)), window=win).astype(np.float32)
                t_clear = target_ds.read(12, window=win).astype(np.float32) > 0.5
                t_gap = target_ds.read(13, window=win).astype(np.float32) > 0.5
                transform = window_transform(win, target_ds.transform)
                domain = rasterize_domain_array(target_meta, target_ds.crs, transform, int(win.height), int(win.width))
                t_mask = t_clear & domain & np.isfinite(t_img).all(axis=0)
                gap = t_gap & domain
                gap_total += int(gap.sum())
                for b in range(8):
                    vals = t_img[b][t_mask & np.isfinite(t_img[b])]
                    if vals.size:
                        mins[b] = min(mins[b], float(np.nanmin(vals)))
                        maxs[b] = max(maxs[b], float(np.nanmax(vals)))
                for d, ds in donor_datasets:
                    d_img = ds.read(indexes=list(range(1, 9)), window=win).astype(np.float32)
                    d_clear = ds.read(12, window=win).astype(np.float32) > 0.5
                    d_mask = d_clear & domain & np.isfinite(d_img).all(axis=0)
                    donor_gap_clear[d.kind] += int((gap & d_mask).sum())
                    for b in range(8):
                        vals = d_img[b][d_mask & np.isfinite(d_img[b])]
                        if vals.size:
                            mins[b] = min(mins[b], float(np.nanmin(vals)))
                            maxs[b] = max(maxs[b], float(np.nanmax(vals)))
        finally:
            for _d, ds in donor_datasets:
                ds.close()
    for b in range(8):
        if not np.isfinite(mins[b]) or not np.isfinite(maxs[b]) or maxs[b] <= mins[b]:
            mins[b] = 0.0
            maxs[b] = 1.0
    coverage = {kind: float(count / max(gap_total, 1)) for kind, count in donor_gap_clear.items()}
    return mins, maxs, coverage, gap_total


def compute_scene_minmax(
    target: np.ndarray,
    target_mask: np.ndarray,
    donor_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    domain: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mins = np.full(target.shape[0], np.inf, dtype=np.float32)
    maxs = np.full(target.shape[0], -np.inf, dtype=np.float32)
    arrays: list[tuple[np.ndarray, np.ndarray]] = [(target, target_mask & domain)]
    arrays.extend((arr, valid & domain) for arr, valid, _gap in donor_arrays.values())
    for b in range(target.shape[0]):
        for arr, mask in arrays:
            vals = arr[b][mask & np.isfinite(arr[b])]
            if vals.size == 0:
                continue
            mins[b] = min(mins[b], float(np.nanmin(vals)))
            maxs[b] = max(maxs[b], float(np.nanmax(vals)))
        if not np.isfinite(mins[b]) or not np.isfinite(maxs[b]) or maxs[b] <= mins[b]:
            mins[b] = 0.0
            maxs[b] = 1.0
    return mins, maxs


def apply_minmax(arr: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    denom = np.maximum(maxs - mins, EPS).astype(np.float32)
    return np.clip((arr - mins[:, None, None]) / denom[:, None, None], 0.0, 1.0).astype(np.float32)


def invert_minmax(arr: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    return (arr * (maxs - mins)[:, None, None] + mins[:, None, None]).astype(np.float32)


def compute_similarity_threshold(donor: np.ndarray, donor_valid: np.ndarray, num_class: int) -> float:
    vals = donor[:, donor_valid]
    if vals.size == 0:
        return 0.05
    per_band = np.nanstd(vals, axis=1) * 2.0 / float(num_class)
    th = float(np.nanmean(per_band))
    return th if np.isfinite(th) and th > 0 else 0.05


def date_distance_days(date_a: str, date_b: str) -> int:
    return int(abs((np.datetime64(date_a) - np.datetime64(date_b)).astype(int)))


def doy_distance_days(date_a: str, date_b: str) -> int:
    a = np.datetime64(date_a, "D").astype(object)
    b = np.datetime64(date_b, "D").astype(object)
    d = abs(int(a.strftime("%j")) - int(b.strftime("%j")))
    return min(d, 366 - d)


def run_nspi_single(
    target_norm: np.ndarray,
    donor_norm: np.ndarray,
    train_valid: np.ndarray,
    donor_valid: np.ndarray,
    fill_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    th = compute_similarity_threshold(donor_norm, donor_valid, NSPI_NUM_CLASS)
    return _nspi_single_fill(
        target_norm,
        donor_norm,
        train_valid,
        donor_valid,
        fill_mask,
        th,
        NSPI_MIN_SIMILAR,
        NSPI_MAX_WINDOW,
        0.0,
        1.0,
    )


def make_base_provenance(
    valid: np.ndarray, target_meta: dict[str, Any]
) -> np.ndarray:
    target_year, target_doy = target_date_parts(target_meta)
    target_id = int(target_meta["id"])
    provenance = np.zeros((len(PROVENANCE_BANDS), *valid.shape), dtype=np.uint16)
    provenance[0, valid] = target_id
    provenance[1, valid] = target_year
    provenance[2, valid] = target_doy
    provenance[3, valid] = 1
    provenance[4, valid] = PROVENANCE_PIXEL_SOURCE["target"]
    provenance[5, valid] = target_year
    provenance[6, valid] = target_doy
    return provenance


def donor_date_parts(donor: DonorInfo) -> tuple[int, int]:
    dt = datetime.strptime(str(donor.date), "%Y-%m-%d")
    return int(dt.year), int(dt.strftime("%j"))


def build_variant_predictions(
    target_raw: np.ndarray,
    target_norm: np.ndarray,
    base_valid: np.ndarray,
    fill_mask: np.ndarray,
    donors: list[DonorInfo],
    donor_arrays_norm: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    target_meta: dict[str, Any],
    mins: np.ndarray,
    maxs: np.ndarray,
    donor_gap_coverage: dict[str, float] | None = None,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    donor_gap_coverage = donor_gap_coverage or {}
    target_date = str(target_meta["date"])
    outputs = {"raw_target": target_raw.copy()}
    valids = {"raw_target": base_valid.copy()}
    provenances = {"raw_target": make_base_provenance(base_valid, target_meta)}
    meta: dict[str, Any] = {"donors": [d.__dict__ | {"path": str(d.path)} for d in donors]}
    if not fill_mask.any() or not donors:
        for v in ("nspi_timeseries_weighted", "agreement_quality_step3", "nspi_multi_score_all3"):
            outputs[v] = target_raw.copy()
            valids[v] = base_valid.copy()
            provenances[v] = provenances["raw_target"].copy()
        meta["note"] = "no fill mask or no donors; filled variants equal raw target"
        return outputs, valids, provenances, meta

    single: list[dict[str, Any]] = []
    for d in donors:
        donor_norm, donor_valid, _donor_gap = donor_arrays_norm[d.kind]
        pred_norm, q = run_nspi_single(target_norm, donor_norm, base_valid, donor_valid, fill_mask)
        valid = fill_mask & np.isin(q, np.array([1, 2, 3], dtype=np.uint8)) & np.isfinite(pred_norm).all(axis=0)
        single.append(
            {
                "donor": d,
                "pred_norm": pred_norm,
                "quality": q,
                "valid": valid,
                "donor_score": d.score,
                "donor_year": donor_date_parts(d)[0],
                "donor_doy": donor_date_parts(d)[1],
                "donor_kind_code": DONOR_KIND_CODE[d.kind],
                "donor_bitmask": DONOR_BITMASK[d.kind],
            }
        )
    meta["single_fill_pixels"] = {c["donor"].kind: int(c["valid"].sum()) for c in single}

    # score_all3 NSPI cascade: highest Step3 family score first, original multi-donor behavior.
    multi_norm = np.full_like(target_norm, np.nan, dtype=np.float32)
    multi_valid = np.zeros(fill_mask.shape, dtype=bool)
    multi_kind = np.zeros(fill_mask.shape, dtype=np.uint16)
    multi_year = np.zeros(fill_mask.shape, dtype=np.uint16)
    multi_doy = np.zeros(fill_mask.shape, dtype=np.uint16)
    multi_quality = np.zeros(fill_mask.shape, dtype=np.uint16)
    multi_bitmask = np.zeros(fill_mask.shape, dtype=np.uint16)
    remaining = fill_mask.copy()
    for c in sorted(single, key=lambda x: x["donor"].score, reverse=True):
        take = remaining & c["valid"]
        for b in range(target_norm.shape[0]):
            multi_norm[b, take] = c["pred_norm"][b, take]
        multi_valid[take] = True
        multi_kind[take] = c["donor_kind_code"]
        multi_year[take] = c["donor_year"]
        multi_doy[take] = c["donor_doy"]
        multi_quality[take] = c["quality"][take].astype(np.uint16)
        multi_bitmask[take] = c["donor_bitmask"]
        remaining[take] = False

    # agreement_quality_step3: closest to cross-donor median, with quality + Step3 tie-break.
    agree_norm = np.full_like(target_norm, np.nan, dtype=np.float32)
    agree_valid = np.zeros(fill_mask.shape, dtype=bool)
    agree_kind = np.zeros(fill_mask.shape, dtype=np.uint16)
    agree_year = np.zeros(fill_mask.shape, dtype=np.uint16)
    agree_doy = np.zeros(fill_mask.shape, dtype=np.uint16)
    agree_quality = np.zeros(fill_mask.shape, dtype=np.uint16)
    agree_bitmask = np.zeros(fill_mask.shape, dtype=np.uint16)
    if len(single) == 1:
        agree_norm = single[0]["pred_norm"].copy()
        agree_valid = single[0]["valid"].copy()
        agree_kind[agree_valid] = single[0]["donor_kind_code"]
        agree_year[agree_valid] = single[0]["donor_year"]
        agree_doy[agree_valid] = single[0]["donor_doy"]
        agree_quality[agree_valid] = single[0]["quality"][agree_valid].astype(np.uint16)
        agree_bitmask[agree_valid] = single[0]["donor_bitmask"]
    else:
        stack = np.stack([c["pred_norm"] for c in single], axis=0)
        valid_stack = np.stack([c["valid"] for c in single], axis=0)
        masked = np.where(valid_stack[:, None, :, :], stack, np.nan)
        with np.errstate(all="ignore"):
            median_pred = np.nanmedian(masked, axis=0)
        best = np.full(fill_mask.shape, np.inf, dtype=np.float32)
        for c in single:
            with np.errstate(all="ignore"):
                dist = np.sqrt(np.nanmean((c["pred_norm"] - median_pred) ** 2, axis=0)).astype(np.float32)
            q = c["quality"]
            q_penalty = np.clip(q.astype(np.float32), 1, 5) * 0.002
            score = dist + q_penalty - 0.001 * float(c["donor_score"])
            take = c["valid"] & (score < best)
            for b in range(target_norm.shape[0]):
                agree_norm[b, take] = c["pred_norm"][b, take]
            best[take] = score[take]
            agree_valid[take] = True
            agree_kind[take] = c["donor_kind_code"]
            agree_year[take] = c["donor_year"]
            agree_doy[take] = c["donor_doy"]
            agree_quality[take] = c["quality"][take].astype(np.uint16)
            agree_bitmask[take] = c["donor_bitmask"]

    # Time-series weighted blend: tournament deployable winner after full8 scene-minmax eval.
    weight_sum = np.zeros(fill_mask.shape, dtype=np.float32)
    ts_norm = np.zeros_like(target_norm, dtype=np.float32)
    ts_year_sum = np.zeros(fill_mask.shape, dtype=np.float32)
    ts_doy_sum = np.zeros(fill_mask.shape, dtype=np.float32)
    ts_quality_sum = np.zeros(fill_mask.shape, dtype=np.float32)
    ts_dominant_weight = np.zeros(fill_mask.shape, dtype=np.float32)
    ts_kind = np.zeros(fill_mask.shape, dtype=np.uint16)
    ts_bitmask = np.zeros(fill_mask.shape, dtype=np.uint16)
    for c in single:
        d = c["donor"]
        doy_w = math.exp(-doy_distance_days(target_date, d.date) / 45.0)
        date_w = math.exp(-date_distance_days(target_date, d.date) / 3650.0)
        score_w = 0.5 + max(0.0, float(d.score))
        cover_w = 0.25 + float(donor_gap_coverage.get(d.kind, c["valid"].sum() / max(int(fill_mask.sum()), 1)))
        base = float(doy_w * date_w * score_w * cover_w)
        q = c["quality"]
        q_w = np.where(q == 1, 1.0, np.where(q == 2, 0.65, np.where(q == 3, 0.35, 0.0))).astype(np.float32)
        w = np.where(c["valid"], base * q_w, 0.0).astype(np.float32)
        weight_sum += w
        active = w > 0
        ts_year_sum += np.where(active, float(c["donor_year"]) * w, 0.0)
        ts_doy_sum += np.where(active, float(c["donor_doy"]) * w, 0.0)
        ts_quality_sum += np.where(active, c["quality"].astype(np.float32) * w, 0.0)
        ts_bitmask[active] |= np.uint16(c["donor_bitmask"])
        dominant = active & (w > ts_dominant_weight)
        ts_dominant_weight[dominant] = w[dominant]
        ts_kind[dominant] = c["donor_kind_code"]
        for b in range(target_norm.shape[0]):
            ts_norm[b] += np.where(w > 0, c["pred_norm"][b] * w, 0.0)
    ts_valid = weight_sum > 0
    for b in range(target_norm.shape[0]):
        ts_norm[b] = np.where(ts_valid, ts_norm[b] / np.maximum(weight_sum, EPS), np.nan)
    ts_valid &= np.isfinite(ts_norm).all(axis=0)

    provenance_specs = {
        "nspi_multi_score_all3": (
            multi_kind,
            multi_year,
            multi_doy,
            multi_quality,
            multi_bitmask,
            PROVENANCE_PIXEL_SOURCE["nspi_selected"],
        ),
        "agreement_quality_step3": (
            agree_kind,
            agree_year,
            agree_doy,
            agree_quality,
            agree_bitmask,
            PROVENANCE_PIXEL_SOURCE["nspi_selected"],
        ),
        "nspi_timeseries_weighted": (
            ts_kind,
            np.rint(ts_year_sum / np.maximum(weight_sum, EPS)).astype(np.uint16),
            np.rint(ts_doy_sum / np.maximum(weight_sum, EPS)).astype(np.uint16),
            np.rint(ts_quality_sum / np.maximum(weight_sum, EPS)).astype(np.uint16),
            ts_bitmask,
            PROVENANCE_PIXEL_SOURCE["nspi_weighted_blend"],
        ),
    }

    target_id = int(target_meta["id"])
    target_year, target_doy = target_date_parts(target_meta)
    for name, pred_norm, valid_fill in [
        ("nspi_multi_score_all3", multi_norm, multi_valid),
        ("agreement_quality_step3", agree_norm, agree_valid),
        ("nspi_timeseries_weighted", ts_norm, ts_valid),
    ]:
        out = target_raw.copy()
        pred_raw = invert_minmax(np.clip(pred_norm, 0.0, 1.0), mins, maxs)
        for b in range(out.shape[0]):
            out[b, valid_fill] = pred_raw[b, valid_fill]
        outputs[name] = out
        valids[name] = base_valid | valid_fill
        provenance = provenances["raw_target"].copy()
        kind, year, doy, quality, bitmask, source_code = provenance_specs[name]
        provenance[0, valid_fill] = target_id
        provenance[1, valid_fill] = target_year
        provenance[2, valid_fill] = target_doy
        provenance[3, valid_fill] = 1
        provenance[4, valid_fill] = source_code
        provenance[5, valid_fill] = year[valid_fill]
        provenance[6, valid_fill] = doy[valid_fill]
        provenance[7, valid_fill] = kind[valid_fill]
        provenance[8, valid_fill] = quality[valid_fill]
        provenance[9, valid_fill] = bitmask[valid_fill]
        provenances[name] = provenance
        meta[f"{name}_fill_pixels"] = int(valid_fill.sum())

    return outputs, valids, provenances, meta


def load_templates(template_dir: Path, tile_indices: list[int] | None, max_tiles: int | None) -> list[TileTemplate]:
    paths = sorted(template_dir.glob("image*.tif"), key=lambda p: int(p.stem.replace("image", "")))
    if not paths:
        raise FileNotFoundError(f"No image*.tif templates found in {template_dir}")
    want = set(tile_indices) if tile_indices is not None else None
    out: list[TileTemplate] = []
    for path in paths:
        idx = int(path.stem.replace("image", ""))
        if want is not None and idx not in want:
            continue
        with rasterio.open(path) as src:
            out.append(
                TileTemplate(
                    index=idx,
                    path=path,
                    crs=src.crs,
                    transform=src.transform,
                    width=src.width,
                    height=src.height,
                    bounds=src.bounds,
                    profile=src.profile.copy(),
                )
            )
        if max_tiles is not None and len(out) >= max_tiles:
            break
    return out


def scene_intersects_tile(scene_profile: dict[str, Any], tile: TileTemplate) -> bool:
    sb = rasterio.coords.BoundingBox(
        left=scene_profile["transform"].c,
        top=scene_profile["transform"].f,
        right=scene_profile["transform"].c + scene_profile["transform"].a * scene_profile["width"],
        bottom=scene_profile["transform"].f + scene_profile["transform"].e * scene_profile["height"],
    )
    tb = transform_bounds(tile.crs, scene_profile["crs"], *tile.bounds, densify_pts=21)
    return not (tb[2] <= sb.left or tb[0] >= sb.right or tb[3] <= sb.bottom or tb[1] >= sb.top)


def initialize_outputs(
    output_root: Path,
    variants: list[str],
    templates: list[TileTemplate],
    overwrite: bool,
    write_provenance: bool = True,
    init_images: bool = True,
    overwrite_provenance: bool = False,
) -> None:
    for variant in variants:
        folder = output_root / VARIANTS[variant]
        if init_images and overwrite and folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
        provenance_folder = folder / "provenance"
        if write_provenance:
            provenance_folder.mkdir(parents=True, exist_ok=True)
        for tile in templates:
            if init_images:
                out_path = folder / f"image{tile.index}.tif"
                if not out_path.exists() or overwrite:
                    profile = tile.profile.copy()
                    profile.update(
                        driver="GTiff",
                        count=8,
                        dtype="float32",
                        nodata=0.0,
                        compress="deflate",
                        predictor=3,
                        tiled=True,
                        blockxsize=256,
                        blockysize=256,
                        sparse_ok=True,
                    )
                    with rasterio.open(out_path, "w", **profile) as dst:
                        # Sparse empty GeoTIFF: unwritten blocks read as nodata=0.
                        # Avoid writing full zero arrays for 4 variants x 202 tiles.
                        for i, band in enumerate(FULL8_BANDS, start=1):
                            dst.set_band_description(i, band)

            if not write_provenance:
                continue

            provenance_path = provenance_folder / f"image{tile.index}.tif"
            if provenance_path.exists() and not (overwrite or overwrite_provenance):
                continue
            provenance_profile = tile.profile.copy()
            provenance_profile.update(
                driver="GTiff",
                count=len(PROVENANCE_BANDS),
                dtype="uint16",
                nodata=0,
                compress="deflate",
                predictor=2,
                tiled=True,
                blockxsize=256,
                blockysize=256,
                sparse_ok=True,
            )
            with rasterio.open(provenance_path, "w", **provenance_profile) as dst:
                for i, band in enumerate(PROVENANCE_BANDS, start=1):
                    dst.set_band_description(i, band)


def update_one_tile(
    tile: TileTemplate,
    variant_folder: Path,
    scene_arr: np.ndarray,
    scene_valid: np.ndarray,
    scene_profile: dict[str, Any],
) -> dict[str, Any]:
    out_path = variant_folder / f"image{tile.index}.tif"
    dst_data = np.zeros((8, tile.height, tile.width), dtype=np.float32)
    dst_mask = np.zeros((tile.height, tile.width), dtype=np.uint8)
    src_data = np.where(scene_valid[None, :, :], scene_arr, 0.0).astype(np.float32, copy=False)
    reproject(
        source=src_data,
        destination=dst_data,
        src_transform=scene_profile["transform"],
        src_crs=scene_profile["crs"],
        dst_transform=tile.transform,
        dst_crs=tile.crs,
        src_nodata=0.0,
        dst_nodata=0.0,
        resampling=Resampling.nearest,
    )
    reproject(
        source=scene_valid.astype(np.uint8),
        destination=dst_mask,
        src_transform=scene_profile["transform"],
        src_crs=scene_profile["crs"],
        dst_transform=tile.transform,
        dst_crs=tile.crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    valid = dst_mask > 0
    if not valid.any():
        return {"tile": tile.index, "updated_pixels": 0}
    with rasterio.open(out_path, "r+") as dst:
        existing = dst.read()
        existing_mask = dst.dataset_mask() > 0
        for b in range(8):
            existing[b, valid] = dst_data[b, valid]
        combined_mask = existing_mask | valid
        dst.write(existing.astype(np.float32, copy=False))
        dst.write_mask((combined_mask.astype(np.uint8) * 255))
    return {"tile": tile.index, "updated_pixels": int(valid.sum())}


def target_date_parts(target_meta: dict[str, Any]) -> tuple[int, int]:
    dt = datetime.strptime(str(target_meta["date"]), "%Y-%m-%d")
    return int(dt.year), int(dt.strftime("%j"))


def update_one_provenance(
    tile: TileTemplate,
    variant_folder: Path,
    scene_provenance: np.ndarray,
    scene_profile: dict[str, Any],
) -> dict[str, Any]:
    provenance_path = variant_folder / "provenance" / f"image{tile.index}.tif"
    if not provenance_path.exists():
        raise FileNotFoundError(f"Missing provenance raster: {provenance_path}")
    if scene_provenance.shape[0] != len(PROVENANCE_BANDS):
        raise ValueError(
            f"Expected {len(PROVENANCE_BANDS)} provenance bands, "
            f"got {scene_provenance.shape[0]}"
        )

    dst_provenance = np.zeros(
        (len(PROVENANCE_BANDS), tile.height, tile.width), dtype=np.uint16
    )
    dst_mask = np.zeros((tile.height, tile.width), dtype=np.uint8)
    source_valid = scene_provenance[3] > 0
    reproject(
        source=scene_provenance.astype(np.uint16, copy=False),
        destination=dst_provenance,
        src_transform=scene_profile["transform"],
        src_crs=scene_profile["crs"],
        dst_transform=tile.transform,
        dst_crs=tile.crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    reproject(
        source=source_valid.astype(np.uint8),
        destination=dst_mask,
        src_transform=scene_profile["transform"],
        src_crs=scene_profile["crs"],
        dst_transform=tile.transform,
        dst_crs=tile.crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )

    valid = dst_mask > 0
    if not valid.any():
        return {"tile": tile.index, "provenance_updated_pixels": 0}

    with rasterio.open(provenance_path, "r+") as dst:
        existing = dst.read()
        existing_mask = dst.dataset_mask() > 0
        existing[:, valid] = dst_provenance[:, valid]
        combined_mask = existing_mask | valid
        dst.write(existing.astype(np.uint16, copy=False))
        dst.write_mask((combined_mask.astype(np.uint8) * 255))

    return {"tile": tile.index, "provenance_updated_pixels": int(valid.sum())}


def update_tiles_for_scene(
    outputs: dict[str, np.ndarray],
    valids: dict[str, np.ndarray],
    scene_profile: dict[str, Any],
    templates: list[TileTemplate],
    output_root: Path,
    variants: list[str],
    tile_workers: int,
) -> list[dict[str, Any]]:
    tiles = [t for t in templates if scene_intersects_tile(scene_profile, t)]
    rows: list[dict[str, Any]] = []
    for variant in variants:
        folder = output_root / VARIANTS[variant]
        if tile_workers <= 1 or len(tiles) <= 1:
            for tile in tiles:
                r = update_one_tile(tile, folder, outputs[variant], valids[variant], scene_profile)
                r.update({"variant": variant})
                rows.append(r)
        else:
            with ThreadPoolExecutor(max_workers=tile_workers) as ex:
                futs = [ex.submit(update_one_tile, tile, folder, outputs[variant], valids[variant], scene_profile) for tile in tiles]
                for fut in as_completed(futs):
                    r = fut.result()
                    r.update({"variant": variant})
                    rows.append(r)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def write_variant_metadata(output_root: Path, variants: list[str], templates: list[TileTemplate], args: argparse.Namespace) -> None:
    fish = read_json(FISHNET_GEOJSON)
    feature_count = len(fish.get("features", []))
    for variant in variants:
        folder = output_root / VARIANTS[variant]
        policy = {
            "variant": variant,
            "folder": folder.name,
            "bands": FULL8_BANDS,
            "dtype": "float32",
            "output_values": "raw_source_values",
            "method_space_for_filled_variants": "per_scene_per_band_minmax_0_1_then_inverse_to_raw",
            "fishnet_geojson_copy": str(FISHNET_GEOJSON),
            "fishnet_features": feature_count,
            "template_dir": str(args.template_dir),
            "tournament_evidence": str(SUMMARY_CSV),
            "provenance_enabled": not bool(getattr(args, "no_provenance", False)),
            "provenance_bands": PROVENANCE_BANDS,
            "provenance_semantics": {
                "target_id": "ID from dataset/outputs/1_targets.json for the target Landsat scene whose domain produced this pixel",
                "target_year": "Acquisition year of the target scene; intended temporal key for date-aware velocity products",
                "target_doy": "Acquisition day-of-year of the target scene",
                "target_valid": "1 where this variant has a valid generated pixel, else 0",
                "pixel_source": "0=nodata, 1=original clear target pixel, 2=single selected NSPI donor fill, 3=weighted NSPI donor blend",
                "source_year": "For target pixels: target year. For selected donor fills: donor year. For weighted blends: rounded weighted-mean donor year.",
                "source_doy": "For target pixels: target DOY. For selected donor fills: donor DOY. For weighted blends: rounded weighted-mean donor DOY.",
                "donor_kind": "0=none/target, 1=LT05, 2=LE07 SLC-on, 3=LE07 SLC-off. For weighted blends, dominant donor by weight.",
                "fill_quality": "NSPI quality code for filled pixels; 0 for original target pixels. Weighted blends store rounded weighted mean quality.",
                "donor_bitmask": "Bitmask of donor kinds contributing to pixel: LT05=1, LE07 SLC-on=2, LE07 SLC-off=4.",
            },
        }
        (folder / "policy.json").write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
        rows = [
            {
                "image": f"image{t.index}.tif",
                "tile_index": t.index,
                "template": str(t.path),
                "provenance": str(folder / "provenance" / f"image{t.index}.tif"),
            }
            for t in templates
        ]
        write_csv(folder / "manifest.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR)
    parser.add_argument(
        "--variants",
        default="raw_target,nspi_timeseries_weighted,agreement_quality_step3,nspi_multi_score_all3",
        help="Comma-separated variant keys",
    )
    parser.add_argument("--ids", default=None, help="Comma-separated target IDs; default all")
    parser.add_argument("--tile-indices", default=None, help="Comma-separated fishnet tile indices; default all")
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--tile-workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite-provenance",
        action="store_true",
        help="Overwrite provenance rasters without deleting/recreating image GeoTIFFs.",
    )
    parser.add_argument(
        "--no-provenance",
        action="store_true",
        help="Do not write per-variant provenance rasters (target_id/year/doy/valid).",
    )
    parser.add_argument(
        "--provenance-only",
        action="store_true",
        help=(
            "Backfill coarse target-date provenance without rewriting image GeoTIFFs. "
            "Does not reconstruct donor/fill provenance."
        ),
    )
    parser.add_argument(
        "--exact-provenance-only",
        action="store_true",
        help=(
            "Re-run full generation logic and write exact provenance rasters only; "
            "do not rewrite existing image GeoTIFFs. This preserves donor/fill provenance."
        ),
    )
    parser.add_argument(
        "--provenance-valid-mode",
        choices=("domain", "raw-valid"),
        default="domain",
        help=(
            "Validity mask used by --provenance-only. domain assigns scene date to all "
            "pixels in the target scene domain (best for later velocity dating of filled "
            "variants). raw-valid restricts to clear target pixels only. Exact filled-variant "
            "provenance is written during normal generation, not provenance-only."
        ),
    )
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Choices: {sorted(VARIANTS)}")
    if (args.provenance_only or args.exact_provenance_only) and args.no_provenance:
        raise ValueError("Provenance-only modes cannot be combined with --no-provenance")
    if args.provenance_only and args.exact_provenance_only:
        raise ValueError("Choose either --provenance-only or --exact-provenance-only")
    if args.provenance_only and args.overwrite:
        raise ValueError(
            "Use --overwrite-provenance with --provenance-only; --overwrite is reserved "
            "for image-regeneration runs."
        )
    if args.exact_provenance_only and args.overwrite:
        raise ValueError(
            "Use --overwrite-provenance with --exact-provenance-only; --overwrite is "
            "reserved for image-regeneration runs."
        )
    if not FISHNET_GEOJSON.exists():
        raise FileNotFoundError(f"Missing fishnet copy: {FISHNET_GEOJSON}")

    targets = load_targets_meta()
    slate = load_slate()
    ids = parse_int_list(args.ids) or sorted(targets)
    if args.max_scenes is not None:
        ids = ids[: args.max_scenes]
    templates = load_templates(args.template_dir, parse_int_list(args.tile_indices), args.max_tiles)
    print(f"variants={variants}")
    print(f"targets={ids}")
    print(f"tiles={len(templates)} indices={[t.index for t in templates[:10]]}{'...' if len(templates) > 10 else ''}")
    print(f"output_root={args.output_root}")
    if args.dry_run:
        return

    print("initializing sparse output rasters", flush=True)
    initialize_outputs(
        args.output_root,
        variants,
        templates,
        args.overwrite,
        write_provenance=not args.no_provenance,
        init_images=not (args.provenance_only or args.exact_provenance_only),
        overwrite_provenance=args.overwrite_provenance,
    )
    write_variant_metadata(args.output_root, variants, templates, args)
    print("initialization done", flush=True)

    all_rows: list[dict[str, Any]] = []
    for tid in ids:
        target_meta = targets[tid]
        path = target_path(target_meta)
        donor_infos = slate.get(tid, [])
        with rasterio.open(path) as src:
            scene_profile = src.profile.copy()
        scene_tiles = [tile for tile in templates if scene_intersects_tile(scene_profile, tile)]

        if args.provenance_only:
            print(
                f"ID {tid:02d}: provenance-only {path.name} "
                f"tiles={len(scene_tiles)} mode={args.provenance_valid_mode}",
                flush=True,
            )
            scene_rows: list[dict[str, Any]] = []
            for tile in scene_tiles:
                win = tile_window_in_scene(tile, scene_profile, pad=0)
                if win is None:
                    continue
                chunk_profile = scene_profile.copy()
                chunk_profile.update(
                    width=int(win.width),
                    height=int(win.height),
                    transform=window_transform(win, scene_profile["transform"]),
                )
                domain = rasterize_domain_array(
                    target_meta,
                    chunk_profile["crs"],
                    chunk_profile["transform"],
                    int(chunk_profile["height"]),
                    int(chunk_profile["width"]),
                )
                if args.provenance_valid_mode == "raw-valid":
                    _target_raw, _data_present, target_clear, _target_gap, _profile = load_full8_window(path, win)
                    provenance_valid = target_clear & domain
                else:
                    provenance_valid = domain
                provenance = make_base_provenance(provenance_valid, target_meta)
                for variant in variants:
                    folder = args.output_root / VARIANTS[variant]
                    r = update_one_provenance(
                        tile,
                        folder,
                        provenance,
                        chunk_profile,
                    )
                    r.update(
                        {
                            "variant": variant,
                            "target_id": tid,
                            "scene": target_meta["scene"],
                            "provenance_only": True,
                            "provenance_valid_mode": args.provenance_valid_mode,
                        }
                    )
                    scene_rows.append(r)
            all_rows.extend(scene_rows)
            print(f"ID {tid:02d}: provenance rows={len(scene_rows)}", flush=True)
            continue

        print(f"ID {tid:02d}: scan scene minmax {path.name} donors={[d.kind for d in donor_infos]}", flush=True)
        mins, maxs, donor_gap_coverage, scene_fill_pixels = compute_scene_minmax_stream(path, target_meta, donor_infos)
        print(
            f"ID {tid:02d}: tiles={len(scene_tiles)} fill_pixels={scene_fill_pixels} "
            f"coverage={{{', '.join(f'{k}: {v:.3f}' for k, v in donor_gap_coverage.items())}}} "
            f"mins={np.round(mins, 4).tolist()} maxs={np.round(maxs, 4).tolist()}",
            flush=True,
        )
        scene_rows: list[dict[str, Any]] = []
        for tile in scene_tiles:
            win = tile_window_in_scene(tile, scene_profile, pad=NSPI_MAX_WINDOW)
            if win is None:
                continue
            target_raw, _data_present, target_clear, target_gap, chunk_profile = load_full8_window(path, win)
            domain = rasterize_domain_array(
                target_meta,
                chunk_profile["crs"],
                chunk_profile["transform"],
                int(chunk_profile["height"]),
                int(chunk_profile["width"]),
            )
            base_valid = target_clear & domain
            fill_mask = target_gap & domain
            donor_arrays_raw: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for d in donor_infos:
                d_raw, _d_data, d_clear, d_gap, _d_profile = load_full8_window(d.path, win)
                donor_arrays_raw[d.kind] = (d_raw, d_clear & domain, d_gap)
            target_norm = apply_minmax(target_raw, mins, maxs)
            donor_arrays_norm = {
                kind: (apply_minmax(arr, mins, maxs), valid, gap)
                for kind, (arr, valid, gap) in donor_arrays_raw.items()
            }
            outputs, valids, provenances, meta = build_variant_predictions(
                target_raw,
                target_norm,
                base_valid,
                fill_mask,
                donor_infos,
                donor_arrays_norm,
                target_meta,
                mins,
                maxs,
                donor_gap_coverage,
            )
            for variant in variants:
                folder = args.output_root / VARIANTS[variant]
                if args.exact_provenance_only:
                    r = {
                        "tile": tile.index,
                        "updated_pixels": 0,
                        "image_update_skipped": True,
                    }
                else:
                    r = update_one_tile(
                        tile, folder, outputs[variant], valids[variant], chunk_profile
                    )
                if not args.no_provenance:
                    r.update(
                        update_one_provenance(
                            tile,
                            folder,
                            provenances[variant],
                            chunk_profile,
                        )
                    )
                r.update({"variant": variant, "target_id": tid, "scene": target_meta["scene"]})
                scene_rows.append(r)
            if len(scene_rows) % max(1, 4 * len(variants)) == 0:
                print(f"ID {tid:02d}: processed {len(scene_rows) // len(variants)}/{len(scene_tiles)} tiles", flush=True)
            del target_raw, target_norm, donor_arrays_raw, donor_arrays_norm, outputs, valids, provenances
        all_rows.extend(scene_rows)
        by_variant = defaultdict(int)
        for r in scene_rows:
            by_variant[r["variant"]] += int(r.get("updated_pixels") or 0)
        print(
            f"ID {tid:02d}: updated tiles={len(scene_tiles)} "
            + " ".join(f"{k}_px={v}" for k, v in sorted(by_variant.items())),
            flush=True,
        )

    for variant in variants:
        rows = [r for r in all_rows if r["variant"] == variant]
        write_csv(args.output_root / VARIANTS[variant] / "generation_updates.csv", rows)
    print("done", flush=True)


if __name__ == "__main__":
    main()
