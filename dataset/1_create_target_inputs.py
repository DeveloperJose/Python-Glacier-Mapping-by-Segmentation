#!/usr/bin/env python3
"""Create canonical target inputs for the HKH donor workflow.

This is step 1 of the new dataset workflow. It intentionally depends only on
hardcoded 41 scene definitions plus Google Earth Engine metadata. It does not
read old manifests, old local raw folders, fishnets, YAML configs, or prior
pipeline outputs.

Outputs live inside this repo:
  dataset/outputs/1_targets.json
  dataset/outputs/1_targets.csv

The target rectangle is for export/alignment. The target domain geometry is the
actual Landsat scene footprint and must be used by later metrics so outside-scene
nodata is not counted as SLC gap.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import ee

PROJECT = "hkh-glacier-mapping"
TARGET_COLLECTION = "LANDSAT/LE07/C02/T1_TOA"
OUTPUT_DIR = Path("dataset/outputs")
TARGETS_JSON = OUTPUT_DIR / "1_targets.json"
TARGETS_CSV = OUTPUT_DIR / "1_targets.csv"
RAW_TARGET_DIR = Path("dataset/raw/targets")

# C02 LE07 target bands needed by later raw export/metrics steps.
OPTICAL_BANDS = ["B1", "B2", "B3", "B4", "B5", "B7"]
QA_BANDS = ["QA_PIXEL", "QA_RADSAT"]
TARGET_STACK_BANDS = OPTICAL_BANDS + QA_BANDS + [
    "data_present",
    "clear_valid",
    "slc_gap",
]


@dataclass(frozen=True)
class TargetSeed:
    id: int
    path: int
    row: int
    report_date: str
    report_sensor: str
    target_date: str
    note: str

    @property
    def path_row(self) -> str:
        return f"{self.path:03d}-{self.row:03d}"

    @property
    def scene(self) -> str:
        return f"{self.path:03d}{self.row:03d}_{self.target_date.replace('-', '')}"

    @property
    def filename(self) -> str:
        return f"{self.id:02d}_target_{self.target_date.replace('-', '')}.tif"


# Base = Bibek ids.js order, with six dates corrected to match ICIMOD report.
# Report LT05 rows are replaced by LE07 proxy dates so all targets are LE07.
TARGET_SEEDS = [
    TargetSeed(1, 153, 36, "2006-09-24", "LE07", "2006-09-24", "matches report/Bibek corrected scene"),
    TargetSeed(2, 152, 35, "2006-09-17", "LE07", "2006-09-17", "matches report/Bibek corrected scene"),
    TargetSeed(3, 152, 34, "2006-07-31", "LE07", "2006-07-31", "matches report/Bibek corrected scene"),
    TargetSeed(4, 152, 33, "2006-07-31", "LE07", "2006-07-31", "matches report/Bibek corrected scene"),
    TargetSeed(5, 151, 35, "2006-07-08", "LE07", "2006-07-08", "matches report/Bibek corrected scene"),
    TargetSeed(6, 151, 34, "2005-08-22", "LE07", "2005-08-22", "matches report/Bibek corrected scene"),
    TargetSeed(7, 150, 36, "2005-09-16", "LE07", "2005-09-16", "matches report/Bibek corrected scene"),
    TargetSeed(8, 150, 35, "2007-11-09", "LE07", "2007-11-09", "matches report/Bibek corrected scene"),
    TargetSeed(9, 150, 34, "2005-09-16", "LE07", "2005-09-16", "matches report/Bibek corrected scene"),
    TargetSeed(10, 149, 37, "2004-10-24", "LE07", "2004-10-24", "matches report/Bibek corrected scene"),
    TargetSeed(11, 149, 36, "2007-11-02", "LE07", "2007-11-02", "matches report/Bibek corrected scene"),
    TargetSeed(12, 149, 35, "2007-09-15", "LE07", "2007-09-15", "matches report/Bibek corrected scene"),
    TargetSeed(13, 149, 34, "2006-07-26", "LE07", "2006-07-26", "matches report/Bibek corrected scene"),
    TargetSeed(14, 148, 37, "2007-11-27", "LE07", "2007-11-27", "matches report/Bibek corrected scene"),
    TargetSeed(15, 148, 36, "2005-09-02", "LE07", "2005-09-02", "matches report/Bibek corrected scene"),
    TargetSeed(16, 148, 35, "2006-11-08", "LE07", "2006-11-08", "matches report/Bibek corrected scene; known tile 189 scene"),
    TargetSeed(17, 147, 38, "2004-09-08", "LE07", "2004-09-08", "matches report/Bibek corrected scene"),
    TargetSeed(18, 147, 37, "2006-09-30", "LE07", "2006-09-30", "matches report/Bibek corrected scene"),
    TargetSeed(19, 147, 36, "2006-09-30", "LE07", "2006-09-30", "matches report/Bibek corrected scene"),
    TargetSeed(20, 147, 35, "2005-08-26", "LE07", "2005-08-26", "matches report/Bibek corrected scene"),
    TargetSeed(21, 146, 39, "2005-11-23", "LE07", "2005-11-23", "matches report/Bibek corrected scene"),
    TargetSeed(22, 146, 38, "2006-09-23", "LE07", "2006-09-23", "matches report/Bibek corrected scene"),
    TargetSeed(23, 146, 37, "2007-12-31", "LE07", "2007-12-31", "matches report/Bibek corrected scene; hard pilot scene"),
    TargetSeed(24, 146, 36, "2009-08-14", "LE07", "2009-08-14", "matches report/Bibek corrected scene"),
    TargetSeed(25, 145, 39, "2001-10-20", "LE07", "2001-10-20", "matches report/Bibek corrected scene"),
    TargetSeed(26, 144, 39, "2005-12-11", "LE07", "2005-12-11", "matches report/Bibek corrected scene; known tile 161 scene"),
    TargetSeed(27, 143, 39, "2008-12-12", "LE07", "2008-12-12", "matches report/Bibek corrected scene"),
    TargetSeed(28, 143, 40, "2005-10-17", "LE07", "2005-10-17", "matches report/Bibek corrected scene"),
    TargetSeed(29, 142, 40, "2008-11-03", "LE07", "2008-11-03", "matches report/Bibek corrected scene; hard pilot scene"),
    TargetSeed(30, 141, 40, "2005-11-12", "LT05", "2005-11-04", "report row was LT05 2005-11-12; replaced with LE07 proxy 2005-11-04"),
    TargetSeed(31, 141, 41, "2005-11-12", "LT05", "2005-11-04", "report row was LT05 2005-11-12; replaced with LE07 proxy 2005-11-04"),
    TargetSeed(32, 140, 41, "2007-12-21", "LE07", "2007-12-21", "matches report/Bibek corrected scene"),
    TargetSeed(33, 139, 41, "2007-12-14", "LE07", "2007-12-14", "matches report/Bibek corrected scene"),
    TargetSeed(34, 138, 41, "2007-12-23", "LE07", "2007-12-23", "matches report/Bibek corrected scene"),
    TargetSeed(35, 137, 41, "2006-01-27", "LE07", "2006-01-27", "matches report/Bibek corrected scene"),
    TargetSeed(36, 136, 41, "2006-07-31", "LE07", "2006-07-31", "matches report/Bibek corrected scene"),
    TargetSeed(37, 136, 40, "2008-11-09", "LE07", "2008-11-09", "matches report/Bibek corrected scene"),
    TargetSeed(38, 135, 40, "2008-12-04", "LE07", "2008-12-04", "matches report/Bibek corrected scene"),
    TargetSeed(39, 134, 40, "2009-09-27", "LE07", "2009-09-27", "matches report/Bibek corrected scene"),
    TargetSeed(40, 133, 40, "2004-12-11", "LE07", "2004-12-11", "matches report/Bibek corrected scene"),
    TargetSeed(41, 133, 41, "2009-11-07", "LE07", "2009-11-07", "matches report/Bibek corrected scene"),
]

CSV_FIELDS = [
    "id",
    "scene",
    "path",
    "row",
    "path_row",
    "date",
    "target_sensor",
    "target_collection",
    "gee_product_id",
    "gee_system_index",
    "gee_image_id",
    "cloud_cover",
    "gee_native_crs",
    "gee_native_transform",
    "target_domain_bounds",
    "target_domain_geojson",
    "export_grid_policy",
    "export_crs",
    "export_crs_transform",
    "export_region_bounds",
    "export_region_geojson",
    "export_width_estimate",
    "export_height_estimate",
    "filename_target",
    "report_date",
    "report_sensor",
    "source_note",
]


def initialize_ee() -> None:
    try:
        ee.Initialize(project=PROJECT)
    except Exception:
        ee.Authenticate(auth_mode="localhost")
        ee.Initialize(project=PROJECT)


def parse_ids(value: str | None) -> set[int]:
    if not value:
        return set()
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def target_by_id() -> dict[int, TargetSeed]:
    return {target.id: target for target in TARGET_SEEDS}


def load_existing() -> list[dict[str, Any]]:
    if not TARGETS_JSON.exists():
        return []
    return json.loads(TARGETS_JSON.read_text(encoding="utf-8"))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def write_outputs(rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda row: int(row["id"]))
    atomic_write_text(TARGETS_JSON, json.dumps(rows, indent=2) + "\n")

    TARGETS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        newline="",
        encoding="utf-8",
        dir=TARGETS_CSV.parent,
        delete=False,
    ) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in CSV_FIELDS})
        tmp_path = Path(tmp.name)
    tmp_path.replace(TARGETS_CSV)


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return value


def date_window(target_date: str) -> tuple[str, str]:
    # GEE filterDate end is exclusive.
    day = date.fromisoformat(target_date)
    next_day = date.fromordinal(day.toordinal() + 1)
    return target_date, next_day.isoformat()


def find_target_image(seed: TargetSeed) -> ee.Image:
    start, end = date_window(seed.target_date)
    coll = (
        ee.ImageCollection(TARGET_COLLECTION)
        .filter(ee.Filter.eq("WRS_PATH", seed.path))
        .filter(ee.Filter.eq("WRS_ROW", seed.row))
        .filterDate(start, end)
    )
    count = int(coll.size().getInfo())
    if count != 1:
        product_ids = coll.aggregate_array("LANDSAT_PRODUCT_ID").getInfo()
        raise RuntimeError(
            f"Expected exactly one target for ID {seed.id:02d} "
            f"{seed.path_row} {seed.target_date}; found {count}: {product_ids}"
        )
    return ee.Image(coll.first())


def bounds_info_in_projection(
    geometry: ee.Geometry,
    projection: ee.Projection,
) -> tuple[list[float], dict[str, Any]]:
    """Return numeric bounds and GeoJSON rectangle in target projection."""
    bounds_geojson = geometry.bounds(maxError=1, proj=projection).getInfo()
    coords = bounds_geojson["coordinates"][0]
    xs = [float(coord[0]) for coord in coords]
    ys = [float(coord[1]) for coord in coords]
    return [min(xs), min(ys), max(xs), max(ys)], bounds_geojson


def estimate_shape(bounds: list[float], transform: list[float]) -> tuple[int, int]:
    x_res = abs(float(transform[0]))
    y_res = abs(float(transform[4]))
    width = int(math.ceil((bounds[2] - bounds[0]) / x_res))
    height = int(math.ceil((bounds[3] - bounds[1]) / y_res))
    return width, height


def create_row(seed: TargetSeed) -> dict[str, Any]:
    img = find_target_image(seed)
    projection = img.select("B1").projection()
    projection_info = projection.getInfo()
    transform = [float(x) for x in projection_info["transform"]]
    domain_geometry = img.geometry()
    bounds, bounds_geojson = bounds_info_in_projection(domain_geometry, projection)
    width, height = estimate_shape(bounds, transform)
    system_index = str(img.get("system:index").getInfo())
    product_id = str(img.get("LANDSAT_PRODUCT_ID").getInfo())
    cloud_cover = float(img.get("CLOUD_COVER").getInfo())
    domain_geojson = domain_geometry.getInfo()

    return {
        "id": seed.id,
        "scene": seed.scene,
        "path": seed.path,
        "row": seed.row,
        "path_row": seed.path_row,
        "date": seed.target_date,
        "target_sensor": "LE07",
        "target_collection": TARGET_COLLECTION,
        "gee_product_id": product_id,
        "gee_system_index": system_index,
        "gee_image_id": f"{TARGET_COLLECTION}/{system_index}",
        "cloud_cover": cloud_cover,
        "gee_native_crs": projection_info["crs"],
        "gee_native_transform": transform,
        "target_domain_bounds": bounds,
        "target_domain_geojson": domain_geojson,
        "export_grid_policy": "gee_native_target_projection",
        "export_crs": projection_info["crs"],
        "export_crs_transform": transform,
        "export_region_bounds": bounds,
        "export_region_geojson": bounds_geojson,
        "export_width_estimate": width,
        "export_height_estimate": height,
        "filename_target": seed.filename,
        "target_stack_bands": TARGET_STACK_BANDS,
        "report_date": seed.report_date,
        "report_sensor": seed.report_sensor,
        "source_note": seed.note,
    }


def rebuild_csv() -> None:
    rows = load_existing()
    write_outputs(rows)
    print(f"rebuilt {TARGETS_CSV} from {TARGETS_JSON}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create canonical target inputs from hardcoded 41 scene IDs")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Process all 41 targets")
    group.add_argument("--ids", type=str, help="Comma-separated target IDs, e.g. 04,16,26")
    parser.add_argument("--force", action="store_true", help="Recompute requested IDs even if cached")
    parser.add_argument("--rebuild-csv", action="store_true", help="Rebuild CSV from existing JSON and exit")
    parser.add_argument("--list", action="store_true", help="List hardcoded targets and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        for seed in TARGET_SEEDS:
            print(f"{seed.id:02d} {seed.scene} report={seed.report_sensor} {seed.report_date} note={seed.note}")
        return
    if args.rebuild_csv:
        rebuild_csv()
        return

    requested = set(target_by_id()) if args.all else parse_ids(args.ids)
    if not requested:
        raise SystemExit("Use --all, --ids, --list, or --rebuild-csv")

    unknown = requested - set(target_by_id())
    if unknown:
        raise SystemExit(f"Unknown target IDs: {sorted(unknown)}")

    initialize_ee()
    existing_rows = load_existing()
    rows_by_id = {int(row["id"]): row for row in existing_rows}

    for target_id in sorted(requested):
        seed = target_by_id()[target_id]
        if target_id in rows_by_id and not args.force:
            print(f"skip {target_id:02d} {seed.scene} (cached)")
            continue
        print(f"query {target_id:02d} {seed.scene}")
        rows_by_id[target_id] = create_row(seed)
        write_outputs(list(rows_by_id.values()))
        print(f"wrote {TARGETS_JSON} ({len(rows_by_id)} targets cached)")

    RAW_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"done: {TARGETS_JSON}")
    print(f"done: {TARGETS_CSV}")


if __name__ == "__main__":
    main()
