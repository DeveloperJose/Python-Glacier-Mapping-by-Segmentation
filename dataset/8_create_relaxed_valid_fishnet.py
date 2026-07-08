#!/usr/bin/env python3
"""Create full8 fishnet datasets with relaxed output-valid masks.

Purpose: test whether full8 underperformed because strict C02 QA clear masks
converted many existing glacier pixels to ignore.

Policy:
- Keep current report-faithful scene set/order.
- For native target pixels, output-valid = data_present & target domain.
- For SLC gap pixels, preserve values already filled by existing variants.
- Do not use cloud/shadow/RADSAT QA as a hard output mask.

This does not recompute NSPI. It patches existing generated variants by adding
raw target data-present pixels back into output, then preserving existing fills
where target data are absent.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = Path("/home/devj/local-arch/data/HKH_raw")
TEMPLATE_DIR = RAW_ROOT / "Landsat7_2005"
VARIANT_SUFFIXES = {
    "raw_target_relaxed_valid": "raw_target",
    "agreement_quality_step3_relaxed_valid": "agreement_quality_step3",
    "nspi_timeseries_weighted_relaxed_valid": "nspi_timeseries_weighted",
}
SOURCE_PREFIX = "HKH_full8"
OUTPUT_PREFIX = "HKH_full8"
SOURCE_VARIANTS = {
    key: RAW_ROOT / f"{SOURCE_PREFIX}_{suffix}"
    for key, suffix in VARIANT_SUFFIXES.items()
}
OUTPUT_VARIANTS = {
    key: RAW_ROOT / f"{OUTPUT_PREFIX}_{suffix}_relaxed_valid"
    for key, suffix in VARIANT_SUFFIXES.items()
}
FULL8_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6_VCID_1", "B6_VCID_2", "B7"]


def configure_variants(source_prefix: str, output_prefix: str) -> None:
    global SOURCE_PREFIX, OUTPUT_PREFIX, SOURCE_VARIANTS, OUTPUT_VARIANTS
    SOURCE_PREFIX = source_prefix
    OUTPUT_PREFIX = output_prefix
    SOURCE_VARIANTS = {
        key: RAW_ROOT / f"{source_prefix}_{suffix}"
        for key, suffix in VARIANT_SUFFIXES.items()
    }
    OUTPUT_VARIANTS = {
        key: RAW_ROOT / f"{output_prefix}_{suffix}_relaxed_valid"
        for key, suffix in VARIANT_SUFFIXES.items()
    }


def load_fishnet6():
    path = REPO_ROOT / "dataset/6_create_fishnet_datasets.py"
    spec = importlib.util.spec_from_file_location("fishnet6", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fishnet6"] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def initialize_outputs(templates, overwrite: bool) -> None:
    for key, folder in OUTPUT_VARIANTS.items():
        if overwrite and folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for tile in templates:
            out_path = folder / f"image{tile.index}.tif"
            if out_path.exists() and not overwrite:
                continue
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
                for band_idx, band in enumerate(FULL8_BANDS, start=1):
                    dst.set_band_description(band_idx, band)
        policy = {
            "variant": key,
            "source_variant": str(SOURCE_VARIANTS[key]),
            "bands": FULL8_BANDS,
            "dtype": "float32",
            "output_values": "raw target data_present pixels plus existing successful SLC fills",
            "valid_mask_policy": "data_present_or_existing_fill_not_strict_qa_clear",
            "qa_policy": "QA_PIXEL/QA_RADSAT not used as hard output mask",
            "template_dir": str(TEMPLATE_DIR),
            "source_prefix": SOURCE_PREFIX,
            "output_prefix": OUTPUT_PREFIX,
        }
        (folder / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")


def write_target_data_present_to_tile(fishnet6, tile, target_meta, target_path: Path) -> int:
    with rasterio.open(target_path) as src:
        scene_profile = src.profile.copy()
    if not fishnet6.scene_intersects_tile(scene_profile, tile):
        return 0
    win = fishnet6.tile_window_in_scene(tile, scene_profile, pad=0)
    if win is None:
        return 0
    with rasterio.open(target_path) as src:
        arr = src.read(indexes=list(range(1, 9)) + [11], window=win).astype(np.float32)
        chunk_profile = src.profile.copy()
        chunk_profile.update(
            width=int(win.width),
            height=int(win.height),
            transform=fishnet6.window_transform(win, src.transform),
        )
    image = arr[:8]
    data_present = arr[8] > 0.5
    finite = np.isfinite(image).all(axis=0)
    domain = fishnet6.rasterize_domain_array(
        target_meta,
        chunk_profile["crs"],
        chunk_profile["transform"],
        int(chunk_profile["height"]),
        int(chunk_profile["width"]),
    )
    valid = data_present & finite & domain
    if not valid.any():
        return 0

    dst_data = np.zeros((8, tile.height, tile.width), dtype=np.float32)
    dst_mask = np.zeros((tile.height, tile.width), dtype=np.uint8)
    src_data = np.where(valid[None, :, :], image, 0.0).astype(np.float32, copy=False)
    reproject(
        source=src_data,
        destination=dst_data,
        src_transform=chunk_profile["transform"],
        src_crs=chunk_profile["crs"],
        dst_transform=tile.transform,
        dst_crs=tile.crs,
        src_nodata=0.0,
        dst_nodata=0.0,
        resampling=Resampling.nearest,
    )
    reproject(
        source=valid.astype(np.uint8),
        destination=dst_mask,
        src_transform=chunk_profile["transform"],
        src_crs=chunk_profile["crs"],
        dst_transform=tile.transform,
        dst_crs=tile.crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    take = dst_mask > 0
    if not take.any():
        return 0
    for folder in OUTPUT_VARIANTS.values():
        out_path = folder / f"image{tile.index}.tif"
        with rasterio.open(out_path, "r+") as dst:
            existing = dst.read()
            existing_mask = dst.dataset_mask() > 0
            existing[:, take] = dst_data[:, take]
            combined = existing_mask | take
            dst.write(existing.astype(np.float32, copy=False))
            dst.write_mask((combined.astype(np.uint8) * 255))
    return int(take.sum())


def preserve_existing_fills(tile) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, out_folder in OUTPUT_VARIANTS.items():
        src_path = SOURCE_VARIANTS[key] / f"image{tile.index}.tif"
        out_path = out_folder / f"image{tile.index}.tif"
        with rasterio.open(src_path) as src, rasterio.open(out_path, "r+") as dst:
            src_data = src.read().astype(np.float32)
            src_mask = src.dataset_mask() > 0
            dst_data = dst.read().astype(np.float32)
            dst_mask = dst.dataset_mask() > 0
            take = src_mask & ~dst_mask
            if take.any():
                dst_data[:, take] = src_data[:, take]
                dst_mask |= take
                dst.write(dst_data)
                dst.write_mask((dst_mask.astype(np.uint8) * 255))
            counts[key] = int(take.sum())
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--source-prefix", default="HKH_full8")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--tile-indices", default=None)
    args = parser.parse_args()
    if args.output_prefix is None:
        args.output_prefix = args.source_prefix
    configure_variants(args.source_prefix, args.output_prefix)

    fishnet6 = load_fishnet6()
    targets = fishnet6.load_targets_meta()
    ids = sorted(targets)
    tile_indices = None
    if args.tile_indices:
        tile_indices = [int(x) for x in args.tile_indices.split(",") if x.strip()]
    templates = fishnet6.load_templates(TEMPLATE_DIR, tile_indices, args.max_tiles)
    initialize_outputs(templates, args.overwrite)

    rows: list[dict[str, object]] = []
    for tid in ids:
        target_meta = targets[tid]
        target_path = fishnet6.target_path(target_meta)
        with rasterio.open(target_path) as src:
            scene_profile = src.profile.copy()
        scene_tiles = [t for t in templates if fishnet6.scene_intersects_tile(scene_profile, t)]
        print(f"ID {tid:02d} {target_path.name}: tiles={len(scene_tiles)}", flush=True)
        for tile in scene_tiles:
            written = write_target_data_present_to_tile(fishnet6, tile, target_meta, target_path)
            rows.append(
                {
                    "target_id": tid,
                    "scene": target_meta["scene"],
                    "tile": tile.index,
                    "data_present_pixels_written": written,
                }
            )

    fill_rows: list[dict[str, object]] = []
    for tile in templates:
        counts = preserve_existing_fills(tile)
        row = {"tile": tile.index}
        row.update({f"{k}_existing_fill_pixels_preserved": v for k, v in counts.items()})
        fill_rows.append(row)

    for folder in OUTPUT_VARIANTS.values():
        write_csv(folder / "target_data_present_updates.csv", rows)
        write_csv(folder / "existing_fill_preservation.csv", fill_rows)
        manifest_rows = [
            {"image": f"image{tile.index}.tif", "tile_index": tile.index, "template": str(tile.path)}
            for tile in templates
        ]
        write_csv(folder / "manifest.csv", manifest_rows)
    print("done", flush=True)


if __name__ == "__main__":
    main()
