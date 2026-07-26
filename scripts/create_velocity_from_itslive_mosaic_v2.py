#!/usr/bin/env python3
"""Create date-aware ITS_LIVE velocity products using HKH generation provenance.

This v2 pipeline is intentionally separate from fishnet image generation.  It uses
per-pixel provenance rasters written by dataset/6_create_fishnet_datasets.py to
choose the temporal velocity window for each output pixel, then writes a compact
quality-aware velocity stack aligned to the selected image variant.

Default product bands:
  1. velocity_speed          Inverse-variance/temporal-weighted annual speed
  2. velocity_count          Sum of annual ITS_LIVE observation counts used
  3. velocity_error          Weighted annual speed uncertainty
  4. velocity_relative_error velocity_error / max(velocity_speed, 1 m/yr)
  5. velocity_valid          1 where at least one annual velocity passed filters

All learned value bands are scalars. Directional vx/vy are deliberately excluded:
existing geometric augmentation does not transform vector components when images
are flipped or rotated, so using them would silently corrupt direction semantics.

The product is meant for later ML ablation, not direct training execution.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from affine import Affine
from pyproj import Transformer
from rasterio.enums import Resampling
from tqdm import tqdm

from glacier_mapping.utils.config import load_server_config
from scripts.create_velocity_from_itslive_mosaic import (
    CATALOG_PATH,
    DEFAULT_SKIP_THRESHOLD,
    VELOCITY_RESOLUTION,
    find_overlapping_datacubes,
    get_image_bbox_latlon,
    load_catalog,
    load_itslive_mosaic,
    resample_to_target,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

RAW_ROOT = Path("/home/devj/local-arch/data/HKH_raw")
VARIANT_DIRS = {
    "raw_target": "HKH_full8_raw_target",
    "nspi_timeseries_weighted": "HKH_full8_nspi_timeseries_weighted",
    "agreement_quality_step3": "HKH_full8_agreement_quality_step3",
    "nspi_multi_score_all3": "HKH_full8_nspi_multi_score_all3",
}

PROVENANCE_BAND_INDEX = {
    "target_id": 1,
    "target_year": 2,
    "target_doy": 3,
    "target_valid": 4,
    "pixel_source": 5,
    "source_year": 6,
    "source_doy": 7,
    "donor_kind": 8,
    "fill_quality": 9,
    "donor_bitmask": 10,
}

OUTPUT_BANDS = [
    "velocity_speed",
    "velocity_count",
    "velocity_error",
    "velocity_relative_error",
    "velocity_valid",
]


@dataclass(frozen=True)
class VelocityFilters:
    min_count: float
    max_error: float | None
    max_speed: float
    err_floor: float


@dataclass(frozen=True)
class ProcessingOptions:
    window_years: int
    year_weight_tau: float
    date_mode: str
    filters: VelocityFilters
    resampling: Resampling


@dataclass
class YearGrid:
    v: np.ndarray
    count: np.ndarray
    error: np.ndarray
    quality_weight: np.ndarray


def parse_int_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def sort_image_paths(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda p: int(p.stem.replace("image", "")))


def output_dir_for_variant(
    output_root: Path, variant: str, date_mode: str, window: int
) -> Path:
    return output_root / f"Velocity_v2_{variant}_{date_mode}_pm{window}yr"


def is_valid_velocity_file(
    path: Path, threshold: float = DEFAULT_SKIP_THRESHOLD
) -> bool:
    if not path.exists():
        return False
    try:
        with rasterio.open(path) as src:
            data = src.read()
            if data.shape[0] != len(OUTPUT_BANDS):
                return False
            if np.nanmax(np.abs(data[0])) >= threshold:
                return False
        return True
    except Exception:
        return False


def provenance_path_for_image(image_path: Path, provenance_dir: Path) -> Path:
    return provenance_dir / image_path.name


def validate_same_grid(a: rasterio.DatasetReader, b: rasterio.DatasetReader) -> None:
    if a.crs != b.crs or a.width != b.width or a.height != b.height:
        raise ValueError(f"Grid mismatch: {a.name} vs {b.name}")
    if not np.allclose(tuple(a.transform), tuple(b.transform), atol=1e-6):
        raise ValueError(f"Transform mismatch: {a.name} vs {b.name}")


def read_provenance_years(
    provenance_path: Path, image_path: Path, date_mode: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not provenance_path.exists():
        raise FileNotFoundError(f"Missing provenance raster: {provenance_path}")

    year_band = (
        PROVENANCE_BAND_INDEX["target_year"]
        if date_mode == "target"
        else PROVENANCE_BAND_INDEX["source_year"]
    )
    with (
        rasterio.open(image_path) as image_src,
        rasterio.open(provenance_path) as prov_src,
    ):
        validate_same_grid(image_src, prov_src)
        years = prov_src.read(year_band).astype(np.int16)
        valid = prov_src.read(PROVENANCE_BAND_INDEX["target_valid"]) > 0
        pixel_source = prov_src.read(PROVENANCE_BAND_INDEX["pixel_source"])
        donor_kind = prov_src.read(PROVENANCE_BAND_INDEX["donor_kind"])

    valid &= years > 0
    unique_years, counts = np.unique(years[valid], return_counts=True)
    source_values, source_counts = np.unique(pixel_source[valid], return_counts=True)
    donor_values, donor_counts = np.unique(donor_kind[valid], return_counts=True)
    stats = {
        "date_mode": date_mode,
        "unique_years": {str(int(y)): int(c) for y, c in zip(unique_years, counts)},
        "pixel_source_counts": {
            str(int(v)): int(c) for v, c in zip(source_values, source_counts)
        },
        "donor_kind_counts": {
            str(int(v)): int(c) for v, c in zip(donor_values, donor_counts)
        },
        "valid_provenance_pixels": int(valid.sum()),
    }
    return years, valid, stats


def bounds_in_datacube_crs(
    landsat_bounds: Any, landsat_epsg: int, datacube_epsg: int
) -> tuple[float, float, float, float]:
    if datacube_epsg == landsat_epsg:
        return (
            landsat_bounds.left,
            landsat_bounds.bottom,
            landsat_bounds.right,
            landsat_bounds.top,
        )

    transformer = Transformer.from_crs(
        f"EPSG:{landsat_epsg}", f"EPSG:{datacube_epsg}", always_xy=True
    )
    xs = [landsat_bounds.left, landsat_bounds.right]
    ys = [landsat_bounds.bottom, landsat_bounds.top]
    out_x: list[float] = []
    out_y: list[float] = []
    for x in xs:
        for y in ys:
            tx, ty = transformer.transform(x, y)
            out_x.append(tx)
            out_y.append(ty)
    return min(out_x), min(out_y), max(out_x), max(out_y)


def affine_from_centers(x: np.ndarray, y: np.ndarray) -> Affine:
    dx = float(x[1] - x[0]) if len(x) > 1 else float(VELOCITY_RESOLUTION)
    dy = float(y[1] - y[0]) if len(y) > 1 else -float(VELOCITY_RESOLUTION)
    return Affine(dx, 0.0, float(x[0] - dx / 2.0), 0.0, dy, float(y[0] - dy / 2.0))


def finite_or_nan(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32, copy=False)
    out = np.where(np.isfinite(out), out, np.nan).astype(np.float32, copy=False)
    # ITS_LIVE fill values are large sentinels; handle both signed conventions.
    out = np.where(np.abs(out) >= 32000, np.nan, out).astype(np.float32, copy=False)
    return out


def extract_year_from_dataset(
    ds: xr.Dataset,
    year: int,
    bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Affine] | None:
    minx, miny, maxx, maxy = bounds
    ds_cropped = ds.sel(x=slice(minx, maxx), y=slice(maxy, miny))
    year_mask = ds_cropped.time.dt.year == year
    if int(year_mask.sum()) == 0:
        return None
    ds_year = ds_cropped.isel(time=year_mask)
    if len(ds_year.x) == 0 or len(ds_year.y) == 0:
        return None

    # Annual composite should have one time slice per year; median is only a guard
    # against duplicate time entries in future ITS_LIVE releases.
    v = finite_or_nan(ds_year["v"].median(dim="time").values)
    count = finite_or_nan(ds_year["count"].median(dim="time").values)
    if "v_error" in ds_year:
        error = finite_or_nan(ds_year["v_error"].median(dim="time").values)
    else:
        error = np.full_like(v, np.nan, dtype=np.float32)
    transform = affine_from_centers(ds_year.x.values, ds_year.y.values)
    return v, count, error, transform


def merge_year_grid(
    merged: dict[int, YearGrid],
    year: int,
    v: np.ndarray,
    count: np.ndarray,
    error: np.ndarray,
    filters: VelocityFilters,
) -> None:
    valid = np.isfinite(v) & np.isfinite(count) & np.isfinite(error)
    valid &= count >= filters.min_count
    valid &= v >= 0.0
    valid &= v <= filters.max_speed
    if filters.max_error is not None:
        valid &= error <= filters.max_error

    # Annual ITS_LIVE composites are already generated from error-weighted fits.
    # Multiplying inverse variance by count again would double-weight observation
    # support. Count is retained as a validity filter and explicit output feature.
    quality_weight = np.where(
        valid,
        1.0 / np.square(np.maximum(error, filters.err_floor)),
        0.0,
    ).astype(np.float32)

    if year not in merged:
        merged[year] = YearGrid(
            v=np.where(valid, v, np.nan).astype(np.float32),
            count=np.where(valid, count, np.nan).astype(np.float32),
            error=np.where(valid, error, np.nan).astype(np.float32),
            quality_weight=quality_weight,
        )
        return

    old = merged[year]
    take = quality_weight > old.quality_weight
    old.v[take] = v[take]
    old.count[take] = count[take]
    old.error[take] = error[take]
    old.quality_weight[take] = quality_weight[take]


def extract_year_grids(
    datacubes: list[dict[str, Any]],
    years_needed: set[int],
    landsat_bounds: Any,
    landsat_epsg: int,
    landsat_shape: tuple[int, int],
    landsat_transform: Affine,
    landsat_crs: Any,
    options: ProcessingOptions,
) -> tuple[dict[int, YearGrid], list[dict[str, Any]]]:
    merged: dict[int, YearGrid] = {}
    extraction_rows: list[dict[str, Any]] = []

    for dc in datacubes:
        datacube_epsg = int(dc["epsg"])
        datacube_url = dc["composite_url"]
        dc_bounds = bounds_in_datacube_crs(landsat_bounds, landsat_epsg, datacube_epsg)
        try:
            ds = load_itslive_mosaic(datacube_url)
        except Exception as exc:
            extraction_rows.append(
                {
                    "datacube_url": datacube_url,
                    "datacube_epsg": datacube_epsg,
                    "status": "failed_open",
                    "error": str(exc),
                }
            )
            continue

        datacube_crs = f"EPSG:{datacube_epsg}"
        years_ok = 0
        for year in sorted(years_needed):
            try:
                extracted = extract_year_from_dataset(ds, year, dc_bounds)
            except Exception as exc:
                extraction_rows.append(
                    {
                        "datacube_url": datacube_url,
                        "datacube_epsg": datacube_epsg,
                        "year": year,
                        "status": "failed_extract",
                        "error": str(exc),
                    }
                )
                continue
            if extracted is None:
                continue
            v_src, count_src, error_src, src_transform = extracted
            v = resample_to_target(
                v_src,
                src_transform,
                datacube_crs,
                landsat_shape,
                landsat_transform,
                str(landsat_crs),
                resampling=options.resampling,
            )
            count = resample_to_target(
                count_src,
                src_transform,
                datacube_crs,
                landsat_shape,
                landsat_transform,
                str(landsat_crs),
                resampling=options.resampling,
            )
            error = resample_to_target(
                error_src,
                src_transform,
                datacube_crs,
                landsat_shape,
                landsat_transform,
                str(landsat_crs),
                resampling=options.resampling,
            )
            merge_year_grid(merged, year, v, count, error, options.filters)
            years_ok += 1

        extraction_rows.append(
            {
                "datacube_url": datacube_url,
                "datacube_epsg": datacube_epsg,
                "status": "success",
                "years_loaded": years_ok,
                "cross_zone_reproj": datacube_epsg != landsat_epsg,
                "overlap_pct": float(dc.get("overlap_pct", 0.0)),
            }
        )

    return merged, extraction_rows


def compose_from_year_grids(
    provenance_years: np.ndarray,
    provenance_valid: np.ndarray,
    year_grids: dict[int, YearGrid],
    options: ProcessingOptions,
) -> tuple[np.ndarray, dict[str, Any]]:
    shape = provenance_years.shape
    v_out = np.zeros(shape, dtype=np.float32)
    count_out = np.zeros(shape, dtype=np.float32)
    error_out = np.zeros(shape, dtype=np.float32)
    relative_error_out = np.zeros(shape, dtype=np.float32)
    valid_out = np.zeros(shape, dtype=np.float32)

    unique_years = sorted(
        int(y) for y in np.unique(provenance_years[provenance_valid]) if y > 0
    )
    per_year_stats: dict[str, Any] = {}
    for base_year in unique_years:
        pixel_mask = provenance_valid & (provenance_years == base_year)
        if not np.any(pixel_mask):
            continue

        weight_sum = np.zeros(shape, dtype=np.float32)
        v_weighted = np.zeros(shape, dtype=np.float32)
        error_weighted = np.zeros(shape, dtype=np.float32)
        count_sum = np.zeros(shape, dtype=np.float32)
        used_years: list[int] = []

        for year in range(
            base_year - options.window_years, base_year + options.window_years + 1
        ):
            grid = year_grids.get(year)
            if grid is None:
                continue
            temporal_weight = float(
                np.exp(-abs(year - base_year) / max(options.year_weight_tau, 1e-6))
            )
            w = grid.quality_weight * temporal_weight
            candidate_valid = pixel_mask & (w > 0)
            if not np.any(candidate_valid):
                continue
            used_years.append(year)
            weight_sum[candidate_valid] += w[candidate_valid]
            v_weighted[candidate_valid] += grid.v[candidate_valid] * w[candidate_valid]
            error_weighted[candidate_valid] += (
                grid.error[candidate_valid] * w[candidate_valid]
            )
            count_sum[candidate_valid] += grid.count[candidate_valid]

        assigned = pixel_mask & (weight_sum > 0)
        if np.any(assigned):
            v_out[assigned] = v_weighted[assigned] / weight_sum[assigned]
            error_out[assigned] = error_weighted[assigned] / weight_sum[assigned]
            count_out[assigned] = count_sum[assigned]
            relative_error_out[assigned] = error_out[assigned] / np.maximum(
                v_out[assigned], 1.0
            )
            valid_out[assigned] = 1.0

        per_year_stats[str(base_year)] = {
            "provenance_pixels": int(pixel_mask.sum()),
            "valid_velocity_pixels": int(assigned.sum()),
            "used_velocity_years": used_years,
        }

    stack = np.stack(
        [v_out, count_out, error_out, relative_error_out, valid_out]
    ).astype(np.float32)
    stats = {
        "per_provenance_year": per_year_stats,
        "valid_velocity_pixels": int(valid_out.sum()),
        "total_provenance_pixels": int(provenance_valid.sum()),
        "velocity_coverage_percent": float(
            valid_out.sum() / max(provenance_valid.sum(), 1) * 100.0
        ),
    }
    valid_values = v_out[valid_out > 0.5]
    if valid_values.size:
        stats["velocity_stats"] = {
            "mean": float(np.mean(valid_values)),
            "median": float(np.median(valid_values)),
            "min": float(np.min(valid_values)),
            "max": float(np.max(valid_values)),
            "std": float(np.std(valid_values)),
        }
    else:
        stats["velocity_stats"] = None
    return stack, stats


def process_single_image(
    args: tuple[int, Path, Path, Path, list[dict[str, Any]], ProcessingOptions],
) -> dict[str, Any]:
    image_idx, image_path, provenance_dir, output_dir, catalog, options = args
    output_path = output_dir / image_path.name
    stats_path = output_dir / f"{image_path.stem}_stats.json"

    try:
        provenance_path = provenance_path_for_image(image_path, provenance_dir)
        years, provenance_valid, provenance_stats = read_provenance_years(
            provenance_path, image_path, options.date_mode
        )
        if not np.any(provenance_valid):
            raise ValueError(f"No valid provenance pixels in {provenance_path}")

        base_years = {int(y) for y in np.unique(years[provenance_valid]) if y > 0}
        years_needed = {
            y
            for base in base_years
            for y in range(base - options.window_years, base + options.window_years + 1)
        }

        with rasterio.open(image_path) as src:
            landsat_meta = src.meta.copy()
            landsat_bounds = src.bounds
            landsat_shape = (src.height, src.width)
            landsat_transform = src.transform
            landsat_crs = src.crs
        landsat_epsg = int(str(landsat_crs).split(":")[-1])

        image_bbox_latlon = get_image_bbox_latlon(landsat_bounds, landsat_epsg)
        datacubes = find_overlapping_datacubes(image_bbox_latlon, catalog)
        if not datacubes:
            raise ValueError(f"No overlapping ITS_LIVE datacubes for {image_path.name}")

        year_grids, extraction_rows = extract_year_grids(
            datacubes,
            years_needed,
            landsat_bounds,
            landsat_epsg,
            landsat_shape,
            landsat_transform,
            landsat_crs,
            options,
        )
        # A tile can legitimately have no annual observations near its provenance
        # years. Preserve it as an explicit all-invalid product instead of leaving
        # a missing file that preprocessing could confuse with an I/O failure.
        output_data, compose_stats = compose_from_year_grids(
            years, provenance_valid, year_grids, options
        )

        output_meta = landsat_meta.copy()
        output_meta.update(
            {
                "count": len(OUTPUT_BANDS),
                "dtype": "float32",
                "nodata": None,
                "compress": "deflate",
                "predictor": 3,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
            }
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **output_meta) as dst:
            dst.write(output_data)
            for idx, band_name in enumerate(OUTPUT_BANDS, start=1):
                dst.set_band_description(idx, band_name)

        stats = {
            "image_name": image_path.stem,
            "image_index": image_idx,
            "status": "success",
            "image_path": str(image_path),
            "provenance_path": str(provenance_path),
            "output_path": str(output_path),
            "landsat_epsg": landsat_epsg,
            "date_mode": options.date_mode,
            "window_years": options.window_years,
            "year_weight_tau": options.year_weight_tau,
            "filters": {
                "min_count": options.filters.min_count,
                "max_error": options.filters.max_error,
                "max_speed": options.filters.max_speed,
                "err_floor": options.filters.err_floor,
            },
            "base_years": sorted(base_years),
            "years_needed": sorted(years_needed),
            "provenance_stats": provenance_stats,
            "compose_stats": compose_stats,
            "datacube_extractions": extraction_rows,
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        return stats

    except Exception as exc:
        logger.error("Failed %s: %s", image_path.name, exc)
        return {
            "image_name": image_path.stem,
            "image_index": image_idx,
            "status": "failed",
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate date-aware velocity products from ITS_LIVE mosaics and HKH provenance"
    )
    parser.add_argument(
        "--server", default="desktop", help="Server name from configs/servers.yaml"
    )
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANT_DIRS),
        default="agreement_quality_step3",
        help="HKH full8 variant whose images/provenance should be used",
    )
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--provenance-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--date-mode", choices=("target", "source"), default="target")
    parser.add_argument("--window-years", type=int, default=1)
    parser.add_argument("--year-weight-tau", type=float, default=1.0)
    parser.add_argument("--min-count", type=float, default=1.0)
    parser.add_argument("--max-error", type=float, default=200.0)
    parser.add_argument("--no-max-error", action="store_true")
    parser.add_argument("--max-speed", type=float, default=1000.0)
    parser.add_argument("--err-floor", type=float, default=1.0)
    parser.add_argument(
        "--resampling", choices=("nearest", "bilinear"), default="nearest"
    )
    parser.add_argument(
        "--tile-indices", default=None, help="Comma-separated tile indices"
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-threshold", type=float, default=DEFAULT_SKIP_THRESHOLD)
    args = parser.parse_args()

    server_config = load_server_config(args.server)
    if args.image_dir is None:
        args.image_dir = args.output_root / VARIANT_DIRS[args.variant]
    if args.provenance_dir is None:
        args.provenance_dir = args.image_dir / "provenance"
    if args.output_dir is None:
        args.output_dir = output_dir_for_variant(
            args.output_root, args.variant, args.date_mode, args.window_years
        )

    if not args.image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {args.image_dir}")
    if not args.provenance_dir.exists():
        raise FileNotFoundError(f"Missing provenance directory: {args.provenance_dir}")

    logger.info(
        "Server config loaded for %s; raw velocity_dir=%s",
        args.server,
        server_config.get("velocity_dir"),
    )
    logger.info("Image dir: %s", args.image_dir)
    logger.info("Provenance dir: %s", args.provenance_dir)
    logger.info("Output dir: %s", args.output_dir)
    logger.info("Loading catalog: %s", args.catalog)
    catalog = load_catalog(args.catalog)
    logger.info("Loaded %d datacubes", len(catalog))

    images = sort_image_paths(list(args.image_dir.glob("image*.tif")))
    wanted = parse_int_list(args.tile_indices)
    if wanted is not None:
        wanted_set = set(wanted)
        images = [p for p in images if int(p.stem.replace("image", "")) in wanted_set]
    if args.max_images is not None:
        images = images[: args.max_images]
    if args.skip_existing:
        before = len(images)
        images = [
            p
            for p in images
            if not is_valid_velocity_file(args.output_dir / p.name, args.skip_threshold)
        ]
        logger.info("skip-existing removed %d images", before - len(images))
    if not images:
        logger.info("No images to process")
        return

    resampling = (
        Resampling.nearest if args.resampling == "nearest" else Resampling.bilinear
    )
    options = ProcessingOptions(
        window_years=int(args.window_years),
        year_weight_tau=float(args.year_weight_tau),
        date_mode=args.date_mode,
        filters=VelocityFilters(
            min_count=float(args.min_count),
            max_error=None if args.no_max_error else float(args.max_error),
            max_speed=float(args.max_speed),
            err_floor=float(args.err_floor),
        ),
        resampling=resampling,
    )

    process_args = [
        (idx, image_path, args.provenance_dir, args.output_dir, catalog, options)
        for idx, image_path in enumerate(images)
    ]
    logger.info("Processing %d images with %d workers", len(process_args), args.workers)

    results: list[dict[str, Any]] = []
    if args.workers <= 1:
        for item in tqdm(process_args, desc="Velocity v2"):
            results.append(process_single_image(item))
    else:
        with multiprocessing.Pool(args.workers) as pool:
            for result in tqdm(
                pool.imap_unordered(process_single_image, process_args),
                total=len(process_args),
                desc="Velocity v2",
            ):
                results.append(result)

    success = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "velocity_v2_summary.csv"
    rows = []
    for r in success:
        compose = r.get("compose_stats", {})
        vel = compose.get("velocity_stats") or {}
        rows.append(
            {
                "image_name": r["image_name"],
                "date_mode": r["date_mode"],
                "window_years": r["window_years"],
                "base_years": ";".join(str(y) for y in r.get("base_years", [])),
                "valid_velocity_pixels": compose.get("valid_velocity_pixels"),
                "total_provenance_pixels": compose.get("total_provenance_pixels"),
                "coverage_percent": compose.get("velocity_coverage_percent"),
                "mean_velocity": vel.get("mean"),
                "median_velocity": vel.get("median"),
                "max_velocity": vel.get("max"),
                "std_velocity": vel.get("std"),
            }
        )
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    failure_path = args.output_dir / "velocity_v2_failures.json"
    with open(failure_path, "w") as f:
        json.dump(failed, f, indent=2)

    logger.info("Successful: %d Failed: %d", len(success), len(failed))
    if rows:
        df = pd.DataFrame(rows)
        logger.info("Mean coverage: %.2f%%", df["coverage_percent"].mean())
        logger.info("Median coverage: %.2f%%", df["coverage_percent"].median())
    logger.info("Summary: %s", summary_path)
    if failed:
        logger.warning("Failures: %s", failure_path)


if __name__ == "__main__":
    main()
