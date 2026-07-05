#!/usr/bin/env python3
"""Plan raw Landsat donors for local HKH SLC-off gap-filling experiments.

This script is separate from export_hkh_gapfill.py. It does not run GEE gapfill.
It chooses one raw donor per target for each donor type:

- lt05: Landsat 5 TM, no SLC gaps, close temporal candidate.
- le07_slc_on: Landsat 7 ETM+ before SLC failure, same sensor/no gaps.
- le07_slc_off: Landsat 7 ETM+ after SLC failure, same sensor/close date,
  only useful if its gaps complement the target gaps.

Planning ROI is target WRS footprint intersected with the HKH fishnet union, not
one fishnet tile and not the full WRS scene. Metrics include both simple support
(native band mask) and strict QA-clear support, to avoid mixing policy choices.

Optional --export-selected exports raw target/donor stacks aligned to the target
B1 projection/transform. Bands are cast to Float32 so spectral and QA bands can
live in one stack; QA uint16 values are exactly representable in Float32.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import ee

from google_earth_engine.export_hkh_gapfill import (
    EXPORT_FOLDER,
    FISHNET_PATH,
    PROJECT,
    export_description,
    export_scenes,
    image_id,
    initialize_ee,
    is_slc_off,
    pr_string,
)

LE07_COLLECTION = "LANDSAT/LE07/C02/T1_TOA"
LT05_COLLECTION = "LANDSAT/LT05/C02/T1_TOA"
LE07_SLC_FAILURE = datetime(2003, 5, 31)

LE07_SPECTRAL_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6_VCID_1", "B6_VCID_2", "B7"]
LT05_SPECTRAL_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]
OPTICAL_BANDS = ["B1", "B2", "B3", "B4", "B5", "B7"]
QA_BANDS = ["QA_PIXEL", "QA_RADSAT"]
PLANNING_SCALE = 300
DEFAULT_OUT_DIR = Path("output/raw_donor_plans")


@dataclass(frozen=True)
class TargetScene:
    path: int
    row: int
    year: int
    month: int
    day: int

    @property
    def date(self) -> datetime:
        return datetime(self.year, self.month, self.day)

    @property
    def pr(self) -> str:
        return pr_string(self.path, self.row)

    @property
    def desc(self) -> str:
        return export_description(self.path, self.row, self.year, self.month, self.day)


@dataclass(frozen=True)
class DonorKind:
    name: str
    collection: str
    sensor: str
    spectral_bands: list[str]


DONOR_KINDS = {
    "lt05": DonorKind("lt05", LT05_COLLECTION, "LT05", LT05_SPECTRAL_BANDS),
    "le07_slc_on": DonorKind("le07_slc_on", LE07_COLLECTION, "LE07", LE07_SPECTRAL_BANDS),
    "le07_slc_off": DonorKind("le07_slc_off", LE07_COLLECTION, "LE07", LE07_SPECTRAL_BANDS),
}


def ee_feature_collection_from_geojson(path: Path) -> ee.FeatureCollection:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return ee.FeatureCollection(data)


def target_image(target: TargetScene) -> ee.Image:
    return ee.Image(image_id(target.path, target.row, target.year, target.month, target.day))


def hkh_roi_for_target(target: ee.Image, fishnet: ee.FeatureCollection) -> ee.Geometry:
    """HKH fishnet union clipped to target WRS footprint."""
    target_geom = target.geometry()
    hkh_geom = fishnet.filterBounds(target_geom).geometry()
    return hkh_geom.intersection(target_geom, ee.ErrorMargin(30))


def simple_valid_mask(img: ee.Image) -> ee.Image:
    return img.select(OPTICAL_BANDS).mask().reduce(ee.Reducer.min())


def qa_clear_mask(img: ee.Image) -> ee.Image:
    """Strict clear-valid mask: optical data, no fill/cloud/shadow/saturation.

    Snow/ice bit is intentionally not rejected.
    """
    optical = simple_valid_mask(img)
    qa_pixel = img.select("QA_PIXEL")
    qa_radsat = img.select("QA_RADSAT")
    bad_pixel_bits = (1 << 0) | (1 << 1) | (1 << 3) | (1 << 4)
    # Optical bands B1-B5 and B7. Works for LT05/LE07 C02 QA_RADSAT bits.
    bad_radsat_bits = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 6) | (1 << 9)
    qa_clear = qa_pixel.bitwiseAnd(bad_pixel_bits).eq(0)
    unsaturated = qa_radsat.bitwiseAnd(bad_radsat_bits).eq(0)
    return optical.And(qa_clear).And(unsaturated)


def target_gap_mask(target: ee.Image) -> ee.Image:
    return simple_valid_mask(target).Not()


def mask_fraction(numer: ee.Image, denom: ee.Image, roi: ee.Geometry) -> float:
    numer01 = numer.unmask(0).gt(0)
    denom01 = denom.unmask(0).gt(0)
    img = ee.Image.cat(
        numer01.And(denom01).rename("num").toFloat(),
        denom01.rename("den").toFloat(),
    )
    stats = img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi,
        scale=PLANNING_SCALE,
        bestEffort=True,
        maxPixels=1e8,
    ).getInfo()
    den = stats.get("den") or 0
    if den == 0:
        return 0.0
    return float((stats.get("num") or 0) / den)


def add_temporal_props(img: ee.Image, target_date: datetime) -> ee.Image:
    t = ee.Date(f"{target_date:%Y-%m-%d}")
    target_doy = ee.Number(t.getRelative("day", "year"))
    target_year = ee.Number(t.get("year"))
    img_date = ee.Date(img.get("system:time_start"))
    img_doy = ee.Number(img_date.getRelative("day", "year"))
    img_year = ee.Number(img_date.get("year"))
    doy_diff = img_doy.subtract(target_doy).abs()
    doy_wrap = ee.Number(365).subtract(doy_diff)
    return (
        img.set("_doy_diff", doy_diff.min(doy_wrap))
        .set("_year_diff", target_year.subtract(img_year).abs())
        .set("_abs_date_diff", img_date.difference(t, "day").abs())
    )


def date_range(center: datetime, years: int) -> tuple[str, str]:
    start = center - timedelta(days=365 * years)
    end = center + timedelta(days=365 * years + 1)
    return f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}"


def candidate_pool(
    kind: DonorKind,
    target: TargetScene,
    cloud_max: int,
    doy_max: int,
    year_window: int | None,
) -> ee.ImageCollection:
    coll = (
        ee.ImageCollection(kind.collection)
        .filter(ee.Filter.eq("WRS_PATH", target.path))
        .filter(ee.Filter.eq("WRS_ROW", target.row))
        .filter(ee.Filter.lt("CLOUD_COVER", cloud_max))
        .map(lambda img: add_temporal_props(img, target.date))
        .filter(ee.Filter.lte("_doy_diff", doy_max))
    )
    if kind.name == "le07_slc_on":
        coll = coll.filterDate("1999-04-15", "2003-05-31")
    elif kind.name == "le07_slc_off":
        start, end = date_range(target.date, year_window or 3)
        coll = coll.filterDate(max(start, "2003-06-01"), end)
        coll = coll.filter(ee.Filter.neq("DATE_ACQUIRED", f"{target.date:%Y-%m-%d}"))
    elif year_window is not None:
        start, end = date_range(target.date, year_window)
        coll = coll.filterDate(start, end)
    return coll


def get_candidate_images(
    kind: DonorKind,
    target: TargetScene,
    max_candidates: int,
    cloud_max: int,
    allow_cloud_fallback: bool,
) -> tuple[list[ee.Image], str]:
    """Return a manageable candidate list with conservative cloud defaults."""
    if kind.name == "lt05":
        filters = [(cloud_max, 60, 2), (cloud_max, 90, 3)]
        fallback_filters = [(30, 120, 5)]
    elif kind.name == "le07_slc_on":
        filters = [(cloud_max, 120, None)]
        fallback_filters = [(30, 160, None)]
    else:
        filters = [(cloud_max, 90, 2), (cloud_max, 120, 4)]
        fallback_filters = [(30, 180, 6)]
    if allow_cloud_fallback:
        filters += fallback_filters

    for this_cloud_max, doy_max, year_window in filters:
        pool = candidate_pool(kind, target, this_cloud_max, doy_max, year_window)
        count = int(pool.size().getInfo())
        if count == 0:
            continue
        # Sort by metadata before expensive per-image scoring.
        pool = pool.sort("CLOUD_COVER").sort("_abs_date_diff").sort("_doy_diff")
        limited = min(count, max_candidates)
        imgs = [ee.Image(pool.toList(limited).get(i)) for i in range(limited)]
        return imgs, f"cloud<{this_cloud_max},doy<={doy_max},years={year_window}"
    return [], "none"


def score_candidate(
    img: ee.Image,
    kind: DonorKind,
    target_img: ee.Image,
    roi: ee.Geometry,
) -> dict[str, Any]:
    target_gap = target_gap_mask(target_img)
    target_simple = simple_valid_mask(target_img)
    target_qa = qa_clear_mask(target_img)
    donor_simple = simple_valid_mask(img)
    donor_qa = qa_clear_mask(img)

    simple_gap = mask_fraction(donor_simple, target_gap, roi)
    qa_gap = mask_fraction(donor_qa, target_gap, roi)
    simple_overlap = mask_fraction(donor_simple, target_simple, roi)
    qa_overlap = mask_fraction(donor_qa, target_qa, roi)
    doy_diff = float(img.get("_doy_diff").getInfo())
    year_diff = float(img.get("_year_diff").getInfo())
    abs_date_diff = float(img.get("_abs_date_diff").getInfo())
    cloud = float(img.get("CLOUD_COVER").getInfo())
    date = str(img.get("DATE_ACQUIRED").getInfo())
    product_id = str(img.get("LANDSAT_PRODUCT_ID").getInfo())

    season_score = max(0.0, 1.0 - doy_diff / 180.0)
    cloud_score = max(0.0, 1.0 - cloud / 100.0)
    simple_balanced = simple_gap * simple_overlap
    qa_balanced = qa_gap * qa_overlap

    if kind.name == "lt05":
        score = 0.30 * qa_overlap + 0.25 * simple_overlap + 0.25 * season_score + 0.20 * cloud_score
    elif kind.name == "le07_slc_on":
        score = 0.55 * qa_balanced + 0.25 * simple_balanced + 0.15 * season_score + 0.05 * cloud_score
    else:
        # Post-SLC donors are only useful if they cover target gaps.
        score = 0.45 * simple_gap + 0.25 * qa_gap + 0.15 * qa_overlap + 0.10 * season_score + 0.05 * cloud_score

    return {
        "donor_kind": kind.name,
        "donor_sensor": kind.sensor,
        "donor_image": img,
        "donor_product_id": product_id,
        "donor_date": date,
        "cloud_cover": cloud,
        "doy_diff": doy_diff,
        "year_diff": year_diff,
        "abs_date_diff": abs_date_diff,
        "simple_gap_coverage": simple_gap,
        "qa_clear_gap_coverage": qa_gap,
        "simple_overlap_coverage": simple_overlap,
        "qa_clear_overlap_coverage": qa_overlap,
        "simple_balanced": simple_balanced,
        "qa_balanced": qa_balanced,
        "score": score,
    }


def strip_image(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != "donor_image"}


def target_row(target: TargetScene) -> dict[str, Any]:
    return {
        "target_pr": target.pr,
        "target_scene": target.desc,
        "target_date": f"{target.date:%Y-%m-%d}",
        "target_path": target.path,
        "target_row": target.row,
    }


def plan_target(
    target: TargetScene,
    fishnet: ee.FeatureCollection,
    max_candidates: int,
    cloud_max: int,
    allow_cloud_fallback: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tgt = target_image(target)
    roi = hkh_roi_for_target(tgt, fishnet)
    selected = []
    all_candidates = []

    for kind in DONOR_KINDS.values():
        images, fallback = get_candidate_images(
            kind,
            target,
            max_candidates,
            cloud_max,
            allow_cloud_fallback,
        )
        scored = []
        for img in images:
            metrics = score_candidate(img, kind, tgt, roi)
            metrics.update(target_row(target))
            metrics["fallback_filter"] = fallback
            scored.append(metrics)
            all_candidates.append(strip_image(metrics))
        if scored:
            best = max(scored, key=lambda x: x["score"])
            best["selected"] = True
            selected.append(strip_image(best))
        else:
            selected.append(
                {
                    **target_row(target),
                    "donor_kind": kind.name,
                    "donor_sensor": kind.sensor,
                    "donor_product_id": "",
                    "donor_date": "",
                    "fallback_filter": fallback,
                    "selected": False,
                    "score": 0.0,
                }
            )
    return selected, all_candidates


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plan_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_candidates": args.max_candidates,
        "cloud_max": args.cloud_max,
        "allow_cloud_fallback": args.allow_cloud_fallback,
        "subset": args.subset or "",
    }


def target_cache_path(out_dir: Path, target: TargetScene) -> Path:
    return out_dir / "targets" / f"{target.desc}.json"


def write_target_cache(
    out_dir: Path,
    target: TargetScene,
    params: dict[str, Any],
    selected: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> None:
    path = target_cache_path(out_dir, target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "target_scene": target.desc,
                "target_pr": target.pr,
                "params": params,
                "selected": selected,
                "candidates": candidates,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def read_target_cache(path: Path, params: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("params") != params:
        return None
    return data.get("selected", []), data.get("candidates", [])


def aggregate_cached(out_dir: Path, params: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for path in sorted((out_dir / "targets").glob("*.json")):
        cached = read_target_cache(path, params)
        if cached is None:
            continue
        selected, candidates = cached
        selected_rows.extend(selected)
        candidate_rows.extend(candidates)
    write_csv(out_dir / "selected_raw_donors.csv", selected_rows)
    write_csv(out_dir / "candidate_raw_donors.csv", candidate_rows)
    (out_dir / "selected_raw_donors.json").write_text(
        json.dumps(selected_rows, indent=2) + "\n", encoding="utf-8"
    )
    return selected_rows, candidate_rows


def stack_for_export(img: ee.Image, kind: DonorKind) -> ee.Image:
    bands = kind.spectral_bands + QA_BANDS
    # Float32 avoids mixed dtype export errors. QA uint16 values remain exact.
    return img.select(bands).toFloat()


def export_aligned_stack(
    img: ee.Image,
    kind: DonorKind,
    desc: str,
    folder: str,
    region: ee.Geometry,
    target_proj: ee.Projection,
) -> ee.batch.Task:
    proj_info = target_proj.getInfo()
    crs = proj_info["crs"]
    transform = proj_info["transform"]
    task = ee.batch.Export.image.toDrive(
        image=stack_for_export(img, kind),
        description=desc,
        folder=folder,
        region=region,
        crs=crs,
        crsTransform=transform,
        maxPixels=1e9,
    )
    task.start()
    return task


def export_selected(rows: list[dict[str, Any]], folder: str) -> None:
    fishnet = ee_feature_collection_from_geojson(FISHNET_PATH)
    target_lookup = {
        export_description(p, r, y, m, d): TargetScene(p, r, y, m, d)
        for p, r, y, m, d in export_scenes()
    }
    for row in rows:
        if not row.get("donor_product_id"):
            continue
        target = target_lookup[row["target_scene"]]
        tgt = target_image(target)
        region = hkh_roi_for_target(tgt, fishnet)
        target_proj = tgt.select("B1").projection()
        kind = DONOR_KINDS[row["donor_kind"]]
        donor = ee.Image(f"{kind.collection}/{row['donor_product_id']}")
        desc = f"rawdonor_{target.desc}_{row['donor_kind']}_{row['donor_date'].replace('-', '')}"
        export_aligned_stack(donor, kind, desc, folder, region, target_proj)
        print(f"queued {desc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan raw LT05/LE07 donors for local HKH gap filling")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--subset", type=str, default=None, help="Comma-separated path/rows, e.g. 144-039,148-035")
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument(
        "--cloud-max",
        type=int,
        default=15,
        help="Default CLOUD_COVER upper bound before optional fallback (default: 15)",
    )
    parser.add_argument(
        "--allow-cloud-fallback",
        action="store_true",
        help="If no low-cloud candidates exist, allow wider cloud/date fallback",
    )
    parser.add_argument("--export-selected", action="store_true", help="Queue raw donor exports after planning")
    parser.add_argument("--folder", default=EXPORT_FOLDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initialize_ee()
    subset = {x.strip() for x in args.subset.split(",")} if args.subset else set()
    fishnet = ee_feature_collection_from_geojson(FISHNET_PATH)
    params = plan_params(args)

    selected_rows: list[dict[str, Any]] = []
    for path, row, year, month, day in export_scenes():
        target = TargetScene(path, row, year, month, day)
        if subset and target.pr not in subset:
            continue
        if not is_slc_off(year, month, day):
            continue

        cache_path = target_cache_path(args.out_dir, target)
        cached = read_target_cache(cache_path, params)
        if cached is not None:
            print(f"skip cached {target.desc} ({target.pr})")
            continue

        print(f"planning {target.desc} ({target.pr})")
        try:
            selected, candidates = plan_target(
                target,
                fishnet,
                args.max_candidates,
                args.cloud_max,
                args.allow_cloud_fallback,
            )
        except Exception as exc:
            print(f"ERROR planning {target.desc}: {exc}", file=sys.stderr)
            aggregate_cached(args.out_dir, params)
            raise

        write_target_cache(args.out_dir, target, params, selected, candidates)
        selected_rows.extend(selected)
        aggregate_cached(args.out_dir, params)
        for row_out in selected:
            print(
                "  {kind:12s} {date:10s} score={score:.3f} "
                "simple_gap={sg:.2f} qa_gap={qg:.2f} simple_ov={so:.2f} qa_ov={qo:.2f}".format(
                    kind=row_out.get("donor_kind", ""),
                    date=row_out.get("donor_date", ""),
                    score=float(row_out.get("score", 0.0)),
                    sg=float(row_out.get("simple_gap_coverage", 0.0)),
                    qg=float(row_out.get("qa_clear_gap_coverage", 0.0)),
                    so=float(row_out.get("simple_overlap_coverage", 0.0)),
                    qo=float(row_out.get("qa_clear_overlap_coverage", 0.0)),
                )
            )

    selected_rows, _candidate_rows = aggregate_cached(args.out_dir, params)
    print(f"wrote {args.out_dir / 'selected_raw_donors.csv'}")
    print(f"wrote {args.out_dir / 'candidate_raw_donors.csv'}")

    if args.export_selected:
        export_selected(selected_rows, args.folder)


if __name__ == "__main__":
    main()
