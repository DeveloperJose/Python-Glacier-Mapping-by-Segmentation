#!/usr/bin/env python3
"""Export aligned raw donor stacks from a completed donor plan CSV.

This script is separate from plan_raw_donors.py. It never replans or scores.
It reads selected_raw_donors.csv and queues Drive exports for chosen raw donors.

Exports are aligned to the local HKH_rebuild target rasters:
- region = full local target rectangle, not HKH fishnet union
- crs/crsTransform = target GeoTIFF metadata
- filename scheme = 01_donor_lt05, 01_donor_le07_slc_on, ...

Band schema matches the existing raw target stacks:
B1,B2,B3,B4,B5,B7,QA_PIXEL,QA_RADSAT,data_present,clear_valid,slc_gap.
All bands are exported Float32 to avoid mixed dtype export errors; QA uint16
values are exactly representable in Float32.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import ee

from google_earth_engine.export_hkh_gapfill import initialize_ee
from google_earth_engine.plan_raw_donors import DONOR_KINDS, QA_BANDS, qa_clear_mask, simple_valid_mask

DEFAULT_PLAN_CSV = Path("output/raw_donor_plans/selected_raw_donors.csv")
DEFAULT_REBUILD_DIR = Path("/home/devj/local-arch/data/HKH_raw/HKH_rebuild")
DRIVE_FOLDER = "HKH_rebuild"
OPTICAL_BANDS = ["B1", "B2", "B3", "B4", "B5", "B7"]
STACK_BANDS = OPTICAL_BANDS + QA_BANDS + ["data_present", "clear_valid", "slc_gap"]


def parse_subset(value: str | None) -> set[str]:
    return set(x.strip() for x in value.split(",")) if value else set()


def parse_ids(value: str | None) -> set[int]:
    return {int(x.strip()) for x in value.split(",")} if value else set()


def metadata_dir(rebuild_dir: Path) -> Path:
    return rebuild_dir / "metadata"


def load_rebuild_manifest(rebuild_dir: Path) -> dict[str, dict[str, Any]]:
    manifest_path = metadata_dir(rebuild_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {row["scene"]: row for row in manifest}


def gee_transform_from_raster_transform(transform: list[float]) -> list[float]:
    if len(transform) == 6:
        return transform
    if len(transform) == 9:
        return transform[:6]
    raise ValueError(f"Unsupported transform length: {len(transform)}")


def target_region(meta: dict[str, Any]) -> ee.Geometry:
    left, bottom, right, top = meta["raster"]["bounds"]
    crs = meta["raster"]["crs"]
    return ee.Geometry.Rectangle([left, bottom, right, top], proj=crs, geodesic=False)


def stack_for_export(img: ee.Image) -> ee.Image:
    optical = img.select(OPTICAL_BANDS)
    data_present = simple_valid_mask(img).rename("data_present")
    clear_valid = qa_clear_mask(img).rename("clear_valid")
    slc_gap = data_present.Not().rename("slc_gap")
    return (
        optical.addBands(img.select(QA_BANDS))
        .addBands(data_present)
        .addBands(clear_valid)
        .addBands(slc_gap)
        .select(STACK_BANDS)
        .toFloat()
    )


def queue_export(
    donor: ee.Image,
    desc: str,
    folder: str,
    region: ee.Geometry,
    crs: str,
    crs_transform: list[float],
) -> ee.batch.Task:
    task = ee.batch.Export.image.toDrive(
        image=stack_for_export(donor),
        description=desc,
        folder=folder,
        region=region,
        crs=crs,
        crsTransform=crs_transform,
        maxPixels=1e9,
    )
    task.start()
    return task


def write_donor_metadata(
    rebuild_dir: Path,
    target_meta: dict[str, Any],
    donor_row: dict[str, Any],
    export_name: str,
    folder: str,
) -> None:
    meta_dir = metadata_dir(rebuild_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    out = meta_dir / f"{export_name}_metadata.json"
    meta = {
        "id": target_meta["id"],
        "role": "donor",
        "filename": f"{export_name}.tif",
        "target_filename": target_meta["filename"],
        "target_scene": target_meta["scene"],
        "target_path_row": target_meta["path_row"],
        "target_date": target_meta["date"],
        "donor_kind": donor_row["donor_kind"],
        "donor_sensor": donor_row["donor_sensor"],
        "donor_date": donor_row["donor_date"],
        "donor_product_id": donor_row["donor_product_id"],
        "drive_folder": folder,
        "export_description": export_name,
        "bands": STACK_BANDS,
        "dtype": "float32",
        "alignment": {
            "source": "local target GeoTIFF metadata",
            "crs": target_meta["raster"]["crs"],
            "crsTransform": gee_transform_from_raster_transform(target_meta["raster"]["transform"]),
            "bounds": target_meta["raster"]["bounds"],
            "width": target_meta["raster"]["width"],
            "height": target_meta["raster"]["height"],
        },
        "planner_metrics": {
            k: donor_row.get(k, "")
            for k in [
                "score",
                "simple_gap_coverage",
                "qa_clear_gap_coverage",
                "simple_overlap_coverage",
                "qa_clear_overlap_coverage",
                "simple_balanced",
                "qa_balanced",
                "cloud_cover",
                "doy_diff",
                "year_diff",
                "abs_date_diff",
            ]
        },
    }
    out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export full-target raw donor stacks from selected_raw_donors.csv")
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN_CSV)
    parser.add_argument("--rebuild-dir", type=Path, default=DEFAULT_REBUILD_DIR)
    parser.add_argument("--subset", type=str, default=None, help="Comma-separated target path/rows")
    parser.add_argument("--ids", type=str, default=None, help="Comma-separated numeric target IDs, e.g. 04,16,26")
    parser.add_argument("--folder", default=DRIVE_FOLDER)
    parser.add_argument("--name-prefix", default="")
    parser.add_argument("--name-suffix", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initialize_ee()
    subset = parse_subset(args.subset)
    ids = parse_ids(args.ids)
    manifest = load_rebuild_manifest(args.rebuild_dir)

    with args.plan_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    queued = 0
    for row in rows:
        if not row.get("donor_product_id"):
            continue
        if subset and row["target_pr"] not in subset:
            continue
        target_meta = manifest[row["target_scene"]]
        target_id = int(target_meta["id"])
        if ids and target_id not in ids:
            continue

        kind = DONOR_KINDS[row["donor_kind"]]
        donor = ee.ImageCollection(kind.collection).filter(
            ee.Filter.eq("LANDSAT_PRODUCT_ID", row["donor_product_id"])
        ).first()

        id_prefix = f"{target_id:02d}"
        donor_date = row["donor_date"].replace("-", "")
        export_name = f"{args.name_prefix}{id_prefix}_donor_{row['donor_kind']}{args.name_suffix}_{donor_date}"
        region = target_region(target_meta)
        crs = target_meta["raster"]["crs"]
        crs_transform = gee_transform_from_raster_transform(target_meta["raster"]["transform"])

        print(
            f"{id_prefix} {row['target_pr']} {row['target_scene']} -> "
            f"{export_name} ({row['donor_date']})"
        )
        if not args.dry_run:
            queue_export(donor, export_name, args.folder, region, crs, crs_transform)
            write_donor_metadata(args.rebuild_dir, target_meta, row, export_name, args.folder)
            queued += 1

    if args.dry_run:
        print("dry run only")
    else:
        print(f"queued {queued} exports")


if __name__ == "__main__":
    main()
