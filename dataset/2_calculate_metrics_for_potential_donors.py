#!/usr/bin/env python3
"""Calculate GEE metrics for potential raw donors.

Step 2 reads only dataset/outputs/1_targets.json from Step 1. It extracts
stable evidence metrics for all strict candidate donors and writes repo-local
CSV/JSONL outputs. It does not pick final donors.

Outputs:
  dataset/outputs/2_donor_metrics.jsonl
  dataset/outputs/2_donor_metrics.csv
  dataset/outputs/2_donor_metrics_progress.json
  dataset/outputs/2_target_summary.csv

Resume model:
- each target/donor candidate row is appended to JSONL after calculation
- existing rows are reused on restart
- complete targets are skipped unless forced
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import ee

PROJECT = "hkh-glacier-mapping"
TARGETS_JSON = Path("dataset/outputs/1_targets.json")
OUTPUT_DIR = Path("dataset/outputs")
METRICS_JSONL = OUTPUT_DIR / "2_donor_metrics.jsonl"
METRICS_CSV = OUTPUT_DIR / "2_donor_metrics.csv"
PROGRESS_JSON = OUTPUT_DIR / "2_donor_metrics_progress.json"
SUMMARY_CSV = OUTPUT_DIR / "2_target_summary.csv"

METRIC_VERSION = "strict_full_scene_v1"

LT05_COLLECTION = "LANDSAT/LT05/C02/T1_TOA"
LE07_COLLECTION = "LANDSAT/LE07/C02/T1_TOA"
LE07_SLC_ON_END_EXCLUSIVE = "2003-05-31"
LE07_SLC_OFF_START = "2003-06-01"

CLOUD_MAX = 10.0
LT05_YEAR_WINDOW = 3
LT05_DOY_MAX = 45
LE07_SLC_ON_DOY_MAX = 60
LE07_SLC_OFF_YEAR_WINDOW = 3
LE07_SLC_OFF_DOY_MAX = 60

# Full-quality first pass. If GEE proves too slow, change sampling later with
# explicit metric version bump; do not silently downsample.
METRIC_SCALE_LABEL = "native_30m_target_grid"
TILE_SCALE = 8
MAX_PIXELS = 1_000_000_000_000
EPS = 1e-6

OPTICAL_BANDS = ["B1", "B2", "B3", "B4", "B5", "B7"]
QA_BANDS = ["QA_PIXEL", "QA_RADSAT"]

DONOR_KINDS = {
    "lt05": {"collection": LT05_COLLECTION, "sensor": "LT05"},
    "le07_slc_on": {"collection": LE07_COLLECTION, "sensor": "LE07"},
    "le07_slc_off": {"collection": LE07_COLLECTION, "sensor": "LE07"},
}

BASE_FIELDS = [
    "metric_version",
    "metric_scale",
    "target_id",
    "target_scene",
    "target_pr",
    "target_date",
    "target_product_id",
    "target_filename",
    "donor_kind",
    "donor_sensor",
    "donor_collection",
    "donor_product_id",
    "donor_date",
    "cloud_cover",
    "abs_date_diff_days",
    "doy_diff",
    "year_diff",
    "target_domain_pixel_count",
    "target_present_pixel_count",
    "target_gap_pixel_count",
    "target_clear_pixel_count",
    "donor_present_pixel_count",
    "donor_clear_pixel_count",
    "simple_gap_coverage",
    "qa_gap_coverage",
    "simple_overlap_coverage",
    "qa_overlap_coverage",
    "simple_balanced",
    "qa_balanced",
    "gap_collision",
    "qa_gap_collision",
    "spectral_overlap_pixel_count",
    "spectral_overlap_fraction",
    "mean_median_abs_residual",
    "mean_median_norm_residual",
    "brightness_residual",
    "ndsi_residual",
    "ndvi_residual",
    "spectral_low_confidence",
    "candidate_key",
]
SPECTRAL_FIELDS = [
    *(f"median_abs_residual_{band}" for band in OPTICAL_BANDS),
    *(f"median_norm_residual_{band}" for band in OPTICAL_BANDS),
]
CSV_FIELDS = BASE_FIELDS + SPECTRAL_FIELDS


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


def load_targets(overrides_json: Path | None = None) -> list[dict[str, Any]]:
    if not TARGETS_JSON.exists():
        raise FileNotFoundError(f"Run Step 1 first: {TARGETS_JSON}")
    rows = json.loads(TARGETS_JSON.read_text(encoding="utf-8"))
    targets = {int(row["id"]): row for row in rows}
    if overrides_json is not None:
        overrides = json.loads(overrides_json.read_text(encoding="utf-8"))
        for row in overrides:
            target_id = int(row["id"])
            if target_id not in targets:
                raise ValueError(f"Override target id not in base targets: {target_id}")
            targets[target_id] = row
    return sorted(targets.values(), key=lambda row: int(row["id"]))


def configure_output_dir(output_dir: Path) -> None:
    global OUTPUT_DIR, METRICS_JSONL, METRICS_CSV, PROGRESS_JSON, SUMMARY_CSV
    OUTPUT_DIR = output_dir
    METRICS_JSONL = OUTPUT_DIR / "2_donor_metrics.jsonl"
    METRICS_CSV = OUTPUT_DIR / "2_donor_metrics.csv"
    PROGRESS_JSON = OUTPUT_DIR / "2_donor_metrics_progress.json"
    SUMMARY_CSV = OUTPUT_DIR / "2_target_summary.csv"


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as tmp:
        json.dump(data, tmp, indent=2)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def load_progress() -> dict[str, Any]:
    if not PROGRESS_JSON.exists():
        return {"metric_version": METRIC_VERSION, "targets": {}}
    data = json.loads(PROGRESS_JSON.read_text(encoding="utf-8"))
    if data.get("metric_version") != METRIC_VERSION:
        return {"metric_version": METRIC_VERSION, "targets": {}}
    return data


def save_progress(progress: dict[str, Any]) -> None:
    atomic_write_json(PROGRESS_JSON, progress)


def iter_metric_rows() -> list[dict[str, Any]]:
    if not METRICS_JSONL.exists():
        return []
    rows = []
    with METRICS_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("metric_version") == METRIC_VERSION:
                rows.append(row)
    return rows


def existing_metric_keys() -> set[str]:
    return {row["candidate_key"] for row in iter_metric_rows()}


def append_metric_row(row: dict[str, Any]) -> None:
    METRICS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def rewrite_jsonl_excluding_targets(target_ids: set[int]) -> None:
    if not METRICS_JSONL.exists() or not target_ids:
        return
    kept = [
        row for row in iter_metric_rows() if int(row["target_id"]) not in target_ids
    ]
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=METRICS_JSONL.parent,
        delete=False,
    ) as tmp:
        for row in kept:
            tmp.write(json.dumps(row, sort_keys=True) + "\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(METRICS_JSONL)


def add_years(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year + years)
    except ValueError:
        return day.replace(month=2, day=28, year=day.year + years)


def date_window(center: str, years: int) -> tuple[str, str]:
    day = date.fromisoformat(center)
    start = add_years(day, -years)
    # GEE filterDate end is exclusive, so include last day by adding one ordinal.
    end_inclusive = add_years(day, years)
    end_exclusive = date.fromordinal(end_inclusive.toordinal() + 1)
    return start.isoformat(), end_exclusive.isoformat()


def add_temporal_props(img: ee.Image, target_date: str) -> ee.Image:
    target = ee.Date(target_date)
    target_doy = ee.Number(target.getRelative("day", "year"))
    target_year = ee.Number(target.get("year"))
    img_date = ee.Date(img.get("system:time_start"))
    img_doy = ee.Number(img_date.getRelative("day", "year"))
    img_year = ee.Number(img_date.get("year"))
    doy_diff = img_doy.subtract(target_doy).abs()
    doy_wrap = ee.Number(365).subtract(doy_diff)
    return (
        img.set("_doy_diff", doy_diff.min(doy_wrap))
        .set("_year_diff", target_year.subtract(img_year).abs())
        .set("_abs_date_diff", img_date.difference(target, "day").abs())
    )


def candidate_collection(kind: str, target: dict[str, Any]) -> ee.ImageCollection:
    info = DONOR_KINDS[kind]
    coll = (
        ee.ImageCollection(info["collection"])
        .filter(ee.Filter.eq("WRS_PATH", int(target["path"])))
        .filter(ee.Filter.eq("WRS_ROW", int(target["row"])))
        .filter(ee.Filter.lte("CLOUD_COVER", CLOUD_MAX))
        .map(lambda img: add_temporal_props(img, target["date"]))
    )

    if kind == "lt05":
        start, end = date_window(target["date"], LT05_YEAR_WINDOW)
        coll = coll.filterDate(start, end).filter(
            ee.Filter.lte("_doy_diff", LT05_DOY_MAX)
        )
    elif kind == "le07_slc_on":
        coll = coll.filterDate("1999-01-01", LE07_SLC_ON_END_EXCLUSIVE).filter(
            ee.Filter.lte("_doy_diff", LE07_SLC_ON_DOY_MAX)
        )
    elif kind == "le07_slc_off":
        start, end = date_window(target["date"], LE07_SLC_OFF_YEAR_WINDOW)
        start = max(start, LE07_SLC_OFF_START)
        coll = (
            coll.filterDate(start, end)
            .filter(ee.Filter.lte("_doy_diff", LE07_SLC_OFF_DOY_MAX))
            .filter(ee.Filter.neq("LANDSAT_PRODUCT_ID", target["gee_product_id"]))
            .filter(ee.Filter.neq("DATE_ACQUIRED", target["date"]))
        )
    else:
        raise ValueError(kind)
    return coll.sort("DATE_ACQUIRED")


def list_candidate_rows(kind: str, target: dict[str, Any]) -> list[dict[str, Any]]:
    coll = candidate_collection(kind, target)
    count = int(coll.size().getInfo())
    rows = []
    image_list = coll.toList(count)
    for idx in range(count):
        img = ee.Image(image_list.get(idx))
        product_id = str(img.get("LANDSAT_PRODUCT_ID").getInfo())
        rows.append(
            {
                "donor_kind": kind,
                "donor_sensor": DONOR_KINDS[kind]["sensor"],
                "donor_collection": DONOR_KINDS[kind]["collection"],
                "donor_product_id": product_id,
                "donor_date": str(img.get("DATE_ACQUIRED").getInfo()),
                "cloud_cover": float(img.get("CLOUD_COVER").getInfo()),
                "doy_diff": float(img.get("_doy_diff").getInfo()),
                "year_diff": float(img.get("_year_diff").getInfo()),
                "abs_date_diff_days": float(img.get("_abs_date_diff").getInfo()),
            }
        )
    return rows


def target_metric_collection(target: dict[str, Any]) -> str:
    """Use TOA consistently for target/donor selection metrics.

    Export product domain may be raw C02/T1 DN, but residual thresholds and donor
    candidate collections in this script are defined in TOA reflectance units.
    """
    sensor = str(target.get("target_sensor", ""))
    if sensor == "LE07" or str(target["gee_product_id"]).startswith("LE07"):
        return LE07_COLLECTION
    if sensor == "LT05" or str(target["gee_product_id"]).startswith("LT05"):
        return LT05_COLLECTION
    raise ValueError(f"Cannot infer TOA metric collection for target {target['id']}")


def image_by_product(collection: str, product_id: str) -> ee.Image:
    coll = ee.ImageCollection(collection).filter(
        ee.Filter.eq("LANDSAT_PRODUCT_ID", product_id)
    )
    return ee.Image(coll.first())


def simple_valid_mask(img: ee.Image) -> ee.Image:
    return img.select(OPTICAL_BANDS).mask().reduce(ee.Reducer.min()).unmask(0).gt(0)


def qa_clear_mask(img: ee.Image) -> ee.Image:
    optical = simple_valid_mask(img)
    qa_pixel = img.select("QA_PIXEL")
    qa_radsat = img.select("QA_RADSAT")
    bad_pixel_bits = (1 << 0) | (1 << 1) | (1 << 3) | (1 << 4)
    bad_radsat_bits = (
        (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 6) | (1 << 9)
    )
    qa_clear = qa_pixel.bitwiseAnd(bad_pixel_bits).eq(0)
    unsaturated = qa_radsat.bitwiseAnd(bad_radsat_bits).eq(0)
    return optical.And(qa_clear).And(unsaturated).unmask(0).gt(0)


def bool_to_float(img: ee.Image, name: str) -> ee.Image:
    return img.unmask(0).gt(0).rename(name).toFloat()


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def reduce_counts(
    target_img: ee.Image,
    donor_img: ee.Image,
    domain: ee.Geometry,
    crs: str,
    transform: list[float],
) -> dict[str, float]:
    target_present = simple_valid_mask(target_img)
    target_clear = qa_clear_mask(target_img)
    target_gap = target_present.Not().unmask(0).gt(0)
    donor_present = simple_valid_mask(donor_img)
    donor_clear = qa_clear_mask(donor_img)

    spectral_overlap = target_clear.And(donor_clear)
    count_img = ee.Image.cat(
        ee.Image.constant(1).rename("target_domain_pixel_count").toFloat(),
        bool_to_float(target_present, "target_present_pixel_count"),
        bool_to_float(target_gap, "target_gap_pixel_count"),
        bool_to_float(target_clear, "target_clear_pixel_count"),
        bool_to_float(donor_present, "donor_present_pixel_count"),
        bool_to_float(donor_clear, "donor_clear_pixel_count"),
        bool_to_float(target_gap.And(donor_present), "gap_present_pixel_count"),
        bool_to_float(target_gap.And(donor_clear), "gap_clear_pixel_count"),
        bool_to_float(target_clear.And(donor_present), "overlap_present_pixel_count"),
        bool_to_float(target_clear.And(donor_clear), "overlap_clear_pixel_count"),
        bool_to_float(spectral_overlap, "spectral_overlap_pixel_count"),
    )
    stats = count_img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=domain,
        crs=crs,
        crsTransform=transform,
        maxPixels=MAX_PIXELS,
        tileScale=TILE_SCALE,
    ).getInfo()
    return {
        key: float(stats.get(key) or 0.0) for key in count_img.bandNames().getInfo()
    }


def normalized_difference(img: ee.Image, a: str, b: str) -> ee.Image:
    band_a = img.select(a)
    band_b = img.select(b)
    return band_a.subtract(band_b).divide(band_a.add(band_b).abs().add(EPS))


def residual_image(target_img: ee.Image, donor_img: ee.Image) -> ee.Image:
    spectral_mask = qa_clear_mask(target_img).And(qa_clear_mask(donor_img))
    residuals = []
    abs_names = []
    norm_names = []
    for band in OPTICAL_BANDS:
        target_band = target_img.select(band)
        donor_band = donor_img.select(band)
        abs_name = f"median_abs_residual_{band}"
        norm_name = f"median_norm_residual_{band}"
        abs_residual = donor_band.subtract(target_band).abs().rename(abs_name)
        norm_residual = abs_residual.divide(
            donor_band.add(target_band).abs().add(EPS)
        ).rename(norm_name)
        residuals.extend([abs_residual, norm_residual])
        abs_names.append(abs_name)
        norm_names.append(norm_name)

    target_brightness = target_img.select(OPTICAL_BANDS).reduce(ee.Reducer.mean())
    donor_brightness = donor_img.select(OPTICAL_BANDS).reduce(ee.Reducer.mean())
    residuals.append(
        donor_brightness.subtract(target_brightness).abs().rename("brightness_residual")
    )

    target_ndsi = normalized_difference(target_img, "B2", "B5")
    donor_ndsi = normalized_difference(donor_img, "B2", "B5")
    residuals.append(donor_ndsi.subtract(target_ndsi).abs().rename("ndsi_residual"))

    target_ndvi = normalized_difference(target_img, "B4", "B3")
    donor_ndvi = normalized_difference(donor_img, "B4", "B3")
    residuals.append(donor_ndvi.subtract(target_ndvi).abs().rename("ndvi_residual"))

    return ee.Image.cat(residuals).updateMask(spectral_mask)


def reduce_spectral(
    target_img: ee.Image,
    donor_img: ee.Image,
    domain: ee.Geometry,
    crs: str,
    transform: list[float],
) -> dict[str, float]:
    img = residual_image(target_img, donor_img)
    stats = img.reduceRegion(
        reducer=ee.Reducer.median(),
        geometry=domain,
        crs=crs,
        crsTransform=transform,
        maxPixels=MAX_PIXELS,
        tileScale=TILE_SCALE,
    ).getInfo()
    values = {
        name: float(stats[name]) if stats.get(name) is not None else math.nan
        for name in img.bandNames().getInfo()
    }
    abs_vals = [
        values.get(f"median_abs_residual_{band}", math.nan) for band in OPTICAL_BANDS
    ]
    norm_vals = [
        values.get(f"median_norm_residual_{band}", math.nan) for band in OPTICAL_BANDS
    ]
    values["mean_median_abs_residual"] = mean_ignore_nan(abs_vals)
    values["mean_median_norm_residual"] = mean_ignore_nan(norm_vals)
    return values


def mean_ignore_nan(values: list[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    if not valid:
        return math.nan
    return float(sum(valid) / len(valid))


def candidate_key(target_id: int, donor_kind: str, donor_product_id: str) -> str:
    return f"{target_id:02d}|{donor_kind}|{donor_product_id}"


def calculate_metrics(
    target: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    target_img = image_by_product(
        target_metric_collection(target), target["gee_product_id"]
    )
    donor_img = image_by_product(
        candidate["donor_collection"], candidate["donor_product_id"]
    )
    domain = ee.Geometry(target["target_domain_geojson"])
    crs = target["export_crs"]
    transform = [float(x) for x in target["export_crs_transform"]]

    counts = reduce_counts(target_img, donor_img, domain, crs, transform)
    spectral = reduce_spectral(target_img, donor_img, domain, crs, transform)

    target_gap = counts["target_gap_pixel_count"]
    target_clear = counts["target_clear_pixel_count"]
    spectral_overlap = counts["spectral_overlap_pixel_count"]

    simple_gap = safe_div(counts["gap_present_pixel_count"], target_gap)
    qa_gap = safe_div(counts["gap_clear_pixel_count"], target_gap)
    simple_overlap = safe_div(counts["overlap_present_pixel_count"], target_clear)
    qa_overlap = safe_div(counts["overlap_clear_pixel_count"], target_clear)

    row = {
        "metric_version": METRIC_VERSION,
        "metric_scale": METRIC_SCALE_LABEL,
        "target_id": int(target["id"]),
        "target_scene": target["scene"],
        "target_pr": target["path_row"],
        "target_date": target["date"],
        "target_product_id": target["gee_product_id"],
        "target_filename": target["filename_target"],
        **candidate,
        "candidate_key": candidate_key(
            int(target["id"]), candidate["donor_kind"], candidate["donor_product_id"]
        ),
        **{
            key: counts[key]
            for key in [
                "target_domain_pixel_count",
                "target_present_pixel_count",
                "target_gap_pixel_count",
                "target_clear_pixel_count",
                "donor_present_pixel_count",
                "donor_clear_pixel_count",
                "spectral_overlap_pixel_count",
            ]
        },
        "simple_gap_coverage": simple_gap,
        "qa_gap_coverage": qa_gap,
        "simple_overlap_coverage": simple_overlap,
        "qa_overlap_coverage": qa_overlap,
        "simple_balanced": simple_gap * simple_overlap,
        "qa_balanced": qa_gap * qa_overlap,
        "gap_collision": 1.0 - simple_gap,
        "qa_gap_collision": 1.0 - qa_gap,
        "spectral_overlap_fraction": safe_div(spectral_overlap, target_clear),
        "spectral_low_confidence": spectral_overlap < 10_000,
        **spectral,
    }
    return row


def write_metrics_csv(rows: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_fields = list(
        dict.fromkeys(CSV_FIELDS + sorted({key for row in rows for key in row}))
    )
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=OUTPUT_DIR, delete=False
    ) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=all_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: csv_value(row.get(field, "")) for field in all_fields}
            )
        tmp_path = Path(tmp.name)
    tmp_path.replace(METRICS_CSV)


def csv_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return value


def pool_status(count: int, complete: bool) -> str:
    if not complete:
        return "not_run"
    return "ok" if count else "weak_empty"


def write_summary(
    targets: list[dict[str, Any]], rows: list[dict[str, Any]], progress: dict[str, Any]
) -> None:
    counts_by_target_kind: dict[int, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for row in rows:
        counts_by_target_kind[int(row["target_id"])][row["donor_kind"]] += 1

    fields = [
        "target_id",
        "target_scene",
        "target_pr",
        "target_date",
        "lt05_candidate_count",
        "le07_slc_on_candidate_count",
        "le07_slc_off_candidate_count",
        "lt05_pool_status",
        "le07_slc_on_pool_status",
        "le07_slc_off_pool_status",
        "complete",
    ]
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=OUTPUT_DIR, delete=False
    ) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fields)
        writer.writeheader()
        for target in targets:
            target_id = int(target["id"])
            counts = counts_by_target_kind[target_id]
            complete = bool(
                progress.get("targets", {}).get(f"{target_id:02d}", {}).get("complete")
            )
            row = {
                "target_id": target_id,
                "target_scene": target["scene"],
                "target_pr": target["path_row"],
                "target_date": target["date"],
                "lt05_candidate_count": counts.get("lt05", 0),
                "le07_slc_on_candidate_count": counts.get("le07_slc_on", 0),
                "le07_slc_off_candidate_count": counts.get("le07_slc_off", 0),
                "lt05_pool_status": pool_status(counts.get("lt05", 0), complete),
                "le07_slc_on_pool_status": pool_status(
                    counts.get("le07_slc_on", 0), complete
                ),
                "le07_slc_off_pool_status": pool_status(
                    counts.get("le07_slc_off", 0), complete
                ),
                "complete": complete,
            }
            writer.writerow(row)
        tmp_path = Path(tmp.name)
    tmp_path.replace(SUMMARY_CSV)


def rebuild_aggregate_outputs(overrides_json: Path | None = None) -> None:
    targets = load_targets(overrides_json)
    progress = load_progress()
    rows = iter_metric_rows()
    write_metrics_csv(rows)
    write_summary(targets, rows, progress)
    print(f"rebuilt {METRICS_CSV} ({len(rows)} rows)")
    print(f"rebuilt {SUMMARY_CSV}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate donor candidate metrics in GEE"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Process all targets")
    group.add_argument(
        "--ids", type=str, help="Comma-separated target IDs, e.g. 04,16,26"
    )
    parser.add_argument(
        "--force-ids", type=str, help="Comma-separated target IDs to recompute"
    )
    parser.add_argument(
        "--rebuild-csv",
        action="store_true",
        help="Rebuild CSV/summary from JSONL and exit",
    )
    parser.add_argument(
        "--list", action="store_true", help="List target IDs from Step 1 and exit"
    )
    parser.add_argument(
        "--target-overrides-json",
        type=Path,
        default=None,
        help="Optional target metadata overrides keyed by id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Isolated output directory; avoids overwriting canonical donor evidence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_output_dir(args.output_dir)
    targets = load_targets(args.target_overrides_json)
    targets_by_id = {int(row["id"]): row for row in targets}

    if args.list:
        for target in targets:
            print(
                f"{int(target['id']):02d} {target['scene']} {target['gee_product_id']}"
            )
        return

    if args.rebuild_csv:
        rebuild_aggregate_outputs(args.target_overrides_json)
        return

    requested = set(targets_by_id) if args.all else parse_ids(args.ids)
    force_ids = parse_ids(args.force_ids)
    requested |= force_ids
    if not requested:
        raise SystemExit("Use --all, --ids, --force-ids, --list, or --rebuild-csv")

    unknown = requested - set(targets_by_id)
    if unknown:
        raise SystemExit(f"Unknown target IDs: {sorted(unknown)}")

    if force_ids:
        rewrite_jsonl_excluding_targets(force_ids)

    initialize_ee()
    progress = load_progress()
    progress.setdefault("targets", {})
    for target_id in force_ids:
        progress["targets"].pop(f"{target_id:02d}", None)
    save_progress(progress)

    keys = existing_metric_keys()

    for target_id in sorted(requested):
        target = targets_by_id[target_id]
        target_key = f"{target_id:02d}"
        state = progress["targets"].get(target_key, {})
        if state.get("complete") and target_id not in force_ids:
            print(f"skip {target_key} {target['scene']} (complete)")
            continue

        print(f"target {target_key} {target['scene']}")
        candidates = []
        for kind in DONOR_KINDS:
            kind_rows = list_candidate_rows(kind, target)
            print(f"  {kind}: {len(kind_rows)} candidates")
            candidates.extend(kind_rows)

        completed_for_target = 0
        for idx, candidate in enumerate(candidates, start=1):
            key = candidate_key(
                target_id, candidate["donor_kind"], candidate["donor_product_id"]
            )
            if key in keys:
                completed_for_target += 1
                continue
            print(
                f"  metric {idx}/{len(candidates)} "
                f"{candidate['donor_kind']} {candidate['donor_date']}"
            )
            row = calculate_metrics(target, candidate)
            append_metric_row(row)
            keys.add(key)
            completed_for_target += 1

        progress["targets"][target_key] = {
            "complete": True,
            "metric_version": METRIC_VERSION,
            "completed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "candidate_count": len(candidates),
            "completed_candidate_count": completed_for_target,
        }
        save_progress(progress)
        print(f"complete {target_key}: {completed_for_target}/{len(candidates)}")

    rebuild_aggregate_outputs(args.target_overrides_json)


if __name__ == "__main__":
    main()
