#!/usr/bin/env python3
"""Export HKH Landsat 7 C02 TOA scenes with SLC-off gap filling.

Uses the 41 ICIMOD report-corrected WRS path/row scene targets. Report LT05
rows are replaced by the selected LE07 proxy scenes so every exported scene is
Landsat 7 ETM+.

Exports full WRS scene extents by default. Use --tile-index only for debug.

Usage:
  uv run python google_earth_engine/export_hkh_gapfill.py --dry-run
  uv run python google_earth_engine/export_hkh_gapfill.py --subset 133-040,133-041
  uv run python google_earth_engine/export_hkh_gapfill.py --tile-index 189 --subset 148-035
  uv run python google_earth_engine/export_hkh_gapfill.py --tile-index 189 --subset 148-035 --donors 3
  uv run python google_earth_engine/export_hkh_gapfill.py
  uv run python google_earth_engine/export_hkh_gapfill.py --metadata-only
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import ee

PROJECT = "hkh-glacier-mapping"
COLLECTION = "LANDSAT/LE07/C02/T1_TOA"
EXPORT_FOLDER = "HKH rebuild gapfill"
DEFAULT_METADATA_DIR = Path("output/hkh_metadata")
FISHNET_PATH = Path("google_earth_engine/hkh_fishnet.geojson")
SLC_FAILURE = datetime(2003, 5, 31)
MIN_GAP_COVERAGE = 0.10
MIN_OVERLAP_COVERAGE = 0.05
MIN_MARGINAL_COVERAGE = 0.03
MAX_DOY_DIFF = 120
PLANNING_SCALE = 300
SOFTBLEND_TEMPERATURE = 0.05

# Export all LE07 spectral bands. Preprocessing chooses final training bands.
# C02 LE07 has dual thermal bands: B6_VCID_1 (low gain), B6_VCID_2 (high gain).
LE07_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6_VCID_1", "B6_VCID_2", "B7"]
OPTICAL_MASK_BANDS = ["B1", "B2", "B3", "B4", "B5", "B7"]

# ---------------------------------------------------------------------------
# Scene definitions: (path, row, year, month, day, report_sensor)
# Base = Bibek's ids.js. Six dates corrected to match ICIMOD report.
# ---------------------------------------------------------------------------
REPORT_SCENES = [
    (153, 36, 2006, 9, 24, "LE07"),
    (152, 35, 2006, 9, 17, "LE07"),
    (152, 34, 2006, 7, 31, "LE07"),
    (152, 33, 2006, 7, 31, "LE07"),
    (151, 35, 2006, 7, 8, "LE07"),
    (151, 34, 2005, 8, 22, "LE07"),
    (150, 36, 2005, 9, 16, "LE07"),
    (150, 35, 2007, 11, 9, "LE07"),
    (150, 34, 2005, 9, 16, "LE07"),
    (149, 37, 2004, 10, 24, "LE07"),
    (149, 36, 2007, 11, 2, "LE07"),
    (149, 35, 2007, 9, 15, "LE07"),
    (149, 34, 2006, 7, 26, "LE07"),
    (148, 37, 2007, 11, 27, "LE07"),
    (148, 36, 2005, 9, 2, "LE07"),
    (148, 35, 2006, 11, 8, "LE07"),
    (147, 38, 2004, 9, 8, "LE07"),
    (147, 37, 2006, 9, 30, "LE07"),
    (147, 36, 2006, 9, 30, "LE07"),
    (147, 35, 2005, 8, 26, "LE07"),
    (146, 39, 2005, 11, 23, "LE07"),
    (146, 38, 2006, 9, 23, "LE07"),
    (146, 37, 2007, 12, 31, "LE07"),
    (146, 36, 2009, 8, 14, "LE07"),
    (145, 39, 2001, 10, 20, "LE07"),
    (144, 39, 2005, 12, 11, "LE07"),
    (143, 39, 2008, 12, 12, "LE07"),
    (143, 40, 2005, 10, 17, "LE07"),
    (142, 40, 2008, 11, 3, "LE07"),
    (141, 40, 2005, 11, 12, "LT05"),
    (141, 41, 2005, 11, 12, "LT05"),
    (140, 41, 2007, 12, 21, "LE07"),
    (139, 41, 2007, 12, 14, "LE07"),
    (138, 41, 2007, 12, 23, "LE07"),
    (137, 41, 2006, 1, 27, "LE07"),
    (136, 41, 2006, 7, 31, "LE07"),
    (136, 40, 2008, 11, 9, "LE07"),
    (135, 40, 2008, 12, 4, "LE07"),
    (134, 40, 2009, 9, 27, "LE07"),
    (133, 40, 2004, 12, 11, "LE07"),
    (133, 41, 2009, 11, 7, "LE07"),
]

# LE07 proxies for report LT05 rows.
LE07_PROXIES = {
    (141, 40): (2005, 11, 4),
    (141, 41): (2005, 11, 4),
}


def initialize_ee() -> None:
    ee.Authenticate(auth_mode="localhost")
    ee.Initialize(project=PROJECT)


def export_scenes() -> list[tuple[int, int, int, int, int]]:
    """Return ICIMOD-corrected LE07 scene targets."""
    scenes = []
    for path, row, year, month, day, _report_sensor in REPORT_SCENES:
        if (path, row) in LE07_PROXIES:
            year, month, day = LE07_PROXIES[(path, row)]
        scenes.append((path, row, year, month, day))
    return scenes


def parse_subset(value: str | None) -> set[str]:
    return set(x.strip() for x in value.split(",")) if value else set()


def pr_string(path: int, row: int) -> str:
    return f"{path:03d}-{row:03d}"


def image_id(path: int, row: int, year: int, month: int, day: int) -> str:
    scene_id = f"LE07_{path:03d}{row:03d}_{year:04d}{month:02d}{day:02d}"
    return f"{COLLECTION}/{scene_id}"


def export_description(path: int, row: int, year: int, month: int, day: int) -> str:
    return f"{path:03d}{row:03d}_{year:04d}{month:02d}{day:02d}"


def is_slc_off(year: int, month: int, day: int) -> bool:
    return datetime(year, month, day) > SLC_FAILURE


# ---------------------------------------------------------------------------
# SLC-off fill algorithm (USGS, Gorelick, Donchyts)
# ---------------------------------------------------------------------------
def fill_slc_off_gaps(
    src: ee.Image,
    fill: ee.Image,
    min_neighbors: int = 64,
    kernel_size: int = 5,
    upscale: bool = True,
    fallback_to_donor: bool = False,
) -> ee.Image:
    """Regress one selected pre-SLC LE07 donor scene onto an SLC-off scene.

    Donor selection happens outside this function via choose_donors(), so
    --donors 1 and --donors N consume the same ranked donor list.

    Credits
    -------
    USGS published the original LS7 SLC-off gap-filling algorithm.
    Noel Gorelick recreated it for Google Earth Engine:
    https://code.earthengine.google.com/d20cba5268ccbe117e2fc1c5fefc33f3
    Genadii Donchyts modified it for faster performance:
    https://code.earthengine.google.com/2ead14966758793579dfb31b94855275
    This Python implementation is based on Genadii Donchyts' code.
    """
    return _apply_donor_fill(
        src,
        fill,
        min_neighbors=min_neighbors,
        kernel_size=kernel_size,
        upscale=upscale,
        fallback_to_donor=fallback_to_donor,
    )


def _apply_donor_fill(
    src: ee.Image,
    fill: ee.Image,
    min_neighbors: int = 64,
    kernel_size: int = 5,
    upscale: bool = True,
    fallback_to_donor: bool = False,
) -> ee.Image:
    """Core single-donor regression fill used by single and multi-donor paths."""
    min_scale = 1 / 3
    max_scale = 3

    common = src.mask().And(fill.mask())
    donor_common = fill.updateMask(common)
    src_common = src.updateMask(common)

    regress = donor_common.addBands(src_common)
    regress = regress.select(ee.List(regress.bandNames()).sort())

    kernel = ee.Kernel.square(kernel_size * 30, "meters", False)
    ratio = 5

    if upscale:
        fit = (
            regress.reduceResolution(ee.Reducer.median(), False, 500)
            .reproject(regress.select(0).projection().scale(ratio, ratio))
            .reduceNeighborhood(
                ee.Reducer.linearFit().forEach(src.bandNames()), kernel, None, False
            )
            .unmask()
            .reproject(regress.select(0).projection().scale(ratio, ratio))
        )
    else:
        fit = regress.reduceNeighborhood(
            ee.Reducer.linearFit().forEach(src.bandNames()), kernel, None, False
        )

    offset = fit.select(".*_offset")
    scale = fit.select(".*_scale")

    reducer = ee.Reducer.mean().combine(ee.Reducer.stdDev(), None, True)

    if upscale:
        src_stats = (
            src.reduceResolution(ee.Reducer.median(), False, 500)
            .reproject(src.select(0).projection().scale(ratio, ratio))
            .reduceNeighborhood(reducer, kernel, None, False)
            .reproject(src.select(0).projection().scale(ratio, ratio))
        )
        fill_stats = (
            fill.reduceResolution(ee.Reducer.median(), False, 500)
            .reproject(fill.select(0).projection().scale(ratio, ratio))
            .reduceNeighborhood(reducer, kernel, None, False)
            .reproject(fill.select(0).projection().scale(ratio, ratio))
        )
    else:
        src_stats = src.reduceNeighborhood(reducer, kernel, None, False)
        fill_stats = fill.reduceNeighborhood(reducer, kernel, None, False)

    scale2 = src_stats.select(".*stdDev").divide(fill_stats.select(".*stdDev"))
    offset2 = src_stats.select(".*mean").subtract(
        fill_stats.select(".*mean").multiply(scale2)
    )

    invalid = scale.lt(min_scale).Or(scale.gt(max_scale))
    scale = scale.where(invalid, scale2)
    offset = offset.where(invalid, offset2)

    invalid2 = scale.lt(min_scale).Or(scale.gt(max_scale))
    scale = scale.where(invalid2, 1)
    offset = offset.where(
        invalid2, src_stats.select(".*mean").subtract(fill_stats.select(".*mean"))
    )

    count = common.reduceNeighborhood(ee.Reducer.count(), kernel, None, True, "boxcar")
    scaled = fill.multiply(scale).add(offset).updateMask(count.gte(min_neighbors))

    out = src.unmask(scaled, True)
    if fallback_to_donor:
        residual = src.mask().reduce(ee.Reducer.min()).Not().And(
            out.mask().reduce(ee.Reducer.min()).Not()
        )
        out = out.unmask(fill.updateMask(residual), True)
    return out


def target_gap_mask(target: ee.Image) -> ee.Image:
    """Target SLC/no-data gaps from optical common mask only."""
    return target.select(OPTICAL_MASK_BANDS).mask().reduce(ee.Reducer.min()).Not()


def clear_valid_mask(img: ee.Image) -> ee.Image:
    """Clear-valid mask for donor planning only.

    Reject fill, dilated cloud, cloud, cloud shadow, optical saturation, and
    dropped pixels. Keep snow/ice valid.
    """
    optical = img.select(OPTICAL_MASK_BANDS).mask().reduce(ee.Reducer.min())
    qa_pixel = img.select("QA_PIXEL")
    qa_radsat = img.select("QA_RADSAT")
    bad_pixel_bits = (1 << 0) | (1 << 1) | (1 << 3) | (1 << 4)
    bad_radsat_bits = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 6) | (1 << 9)
    qa_clear = qa_pixel.bitwiseAnd(bad_pixel_bits).eq(0)
    unsaturated = qa_radsat.bitwiseAnd(bad_radsat_bits).eq(0)
    return optical.And(qa_clear).And(unsaturated)


def mask_fraction(numer: ee.Image, denom: ee.Image, roi: ee.Geometry) -> float:
    img = ee.Image.cat(
        numer.And(denom).rename("num").unmask(0).toFloat(),
        denom.rename("den").unmask(0).toFloat(),
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


def build_candidate_pool(
    path: int,
    row: int,
    target_img: ee.Image,
    cloud_threshold: int,
) -> ee.ImageCollection:
    target_date = ee.Date(target_img.get("system:time_start"))
    target_doy = ee.Number(target_date.getRelative("day", "year"))
    target_year = ee.Number(target_date.get("year"))

    def add_rank_props(img: ee.Image) -> ee.Image:
        img_date = ee.Date(img.get("system:time_start"))
        img_doy = ee.Number(img_date.getRelative("day", "year"))
        img_year = ee.Number(img_date.get("year"))
        doy_diff = img_doy.subtract(target_doy).abs()
        doy_wrap = ee.Number(365).subtract(doy_diff)
        return img.set("_year_diff", target_year.subtract(img_year).abs()).set(
            "_doy_diff", doy_diff.min(doy_wrap)
        )

    return (
        ee.ImageCollection(COLLECTION)
        .filter(ee.Filter.eq("WRS_PATH", path))
        .filter(ee.Filter.eq("WRS_ROW", row))
        .filterDate("1999-04-15", "2003-05-31")
        .filter(ee.Filter.lt("CLOUD_COVER", cloud_threshold))
        .map(add_rank_props)
        .filter(ee.Filter.lte("_doy_diff", MAX_DOY_DIFF))
    )


def score_donor(
    donor: ee.Image,
    target: ee.Image,
    roi: ee.Geometry,
    target_gap: ee.Image,
    target_clear: ee.Image,
) -> dict:
    donor_clear = clear_valid_mask(donor)
    return {
        "image": donor,
        "date": donor.get("DATE_ACQUIRED").getInfo(),
        "scene_id": donor.get("LANDSAT_PRODUCT_ID").getInfo(),
        "cloud_cover": float(donor.get("CLOUD_COVER").getInfo()),
        "doy_diff": float(donor.get("_doy_diff").getInfo()),
        "year_diff": float(donor.get("_year_diff").getInfo()),
        "gap_coverage": mask_fraction(donor_clear, target_gap, roi),
        "overlap_coverage": mask_fraction(donor_clear, target_clear, roi),
        "clear_mask": donor_clear,
    }


def choose_forced_donor(
    path: int,
    row: int,
    target: ee.Image,
    roi: ee.Geometry,
    donor_date: str,
) -> tuple[list[dict], int | None]:
    """Return one exact-date donor for debugging, scored with normal QA metrics."""
    target_gap = target_gap_mask(target)
    target_clear = clear_valid_mask(target)
    for cloud_threshold in (10, 20, 100):
        pool = build_candidate_pool(path, row, target, cloud_threshold).filter(
            ee.Filter.eq("DATE_ACQUIRED", donor_date)
        )
        count = int(pool.size().getInfo())
        if count == 0:
            continue
        donor = ee.Image(pool.first())
        metrics = score_donor(donor, target, roi, target_gap, target_clear)
        metrics["marginal_gap_coverage"] = metrics["gap_coverage"]
        return [metrics], cloud_threshold
    return [], None


def choose_donors(
    path: int,
    row: int,
    target: ee.Image,
    roi: ee.Geometry,
    max_donors: int,
    donor_ranking: str,
) -> tuple[list[dict], int | None]:
    """Choose donors by requested score, then greedy marginal gap coverage."""
    target_gap = target_gap_mask(target)
    target_clear = clear_valid_mask(target)

    for cloud_threshold in (10, 20):
        pool = build_candidate_pool(path, row, target, cloud_threshold)
        count = int(pool.size().getInfo())
        if count == 0:
            continue

        scored = []
        for idx in range(count):
            donor = ee.Image(pool.toList(count).get(idx))
            metrics = score_donor(donor, target, roi, target_gap, target_clear)
            if (
                metrics["gap_coverage"] >= MIN_GAP_COVERAGE
                and metrics["overlap_coverage"] >= MIN_OVERLAP_COVERAGE
            ):
                scored.append(metrics)

        if donor_ranking == "coverage":
            scored.sort(
                key=lambda x: (
                    -x["gap_coverage"],
                    -x["overlap_coverage"],
                    x["doy_diff"],
                    x["year_diff"],
                    x["cloud_cover"],
                )
            )
        elif donor_ranking == "overlap":
            scored.sort(
                key=lambda x: (
                    -x["overlap_coverage"],
                    -x["gap_coverage"],
                    x["doy_diff"],
                    x["year_diff"],
                    x["cloud_cover"],
                )
            )
        else:
            # balanced: prefer donors that both cover target gaps and agree
            # with target clear pixels. Tile 161 showed this avoids a high-gap
            # but low-overlap donor that looked worse visually.
            scored.sort(
                key=lambda x: (
                    -(x["gap_coverage"] * x["overlap_coverage"]),
                    -x["gap_coverage"],
                    -x["overlap_coverage"],
                    x["doy_diff"],
                    x["year_diff"],
                    x["cloud_cover"],
                )
            )

        selected = []
        remaining_gap = target_gap
        for cand in scored:
            marginal = mask_fraction(cand["clear_mask"], remaining_gap, roi)
            if selected and marginal < MIN_MARGINAL_COVERAGE:
                continue
            cand["marginal_gap_coverage"] = marginal
            selected.append(cand)
            remaining_gap = remaining_gap.And(cand["clear_mask"].Not())
            if len(selected) >= max_donors:
                break

        if selected:
            return selected, cloud_threshold

    return [], None


def _donor_fill_mask(donor: dict, fill_mask_mode: str) -> ee.Image:
    """Compute fill-time mask for a donor according to mode.

    - simple: band validity only (Bibek paper)
    - cloud-only: QA_PIXEL (cloud/shadow/snow), no RADSAT
    - qa-clear: QA_PIXEL + QA_RADSAT (strict)
    """
    if fill_mask_mode == "simple":
        return ee.Image.constant(1)
    img = donor["image"]
    optical = img.select(OPTICAL_MASK_BANDS).mask().reduce(ee.Reducer.min())
    qa_pixel = img.select("QA_PIXEL")
    bad_pixel_bits = (1 << 0) | (1 << 1) | (1 << 3) | (1 << 4)
    qa_clear = qa_pixel.bitwiseAnd(bad_pixel_bits).eq(0)
    if fill_mask_mode == "cloud-only":
        return optical.And(qa_clear)
    # qa-clear: also mask RADSAT
    qa_radsat = img.select("QA_RADSAT")
    bad_radsat_bits = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 6) | (1 << 9)
    unsaturated = qa_radsat.bitwiseAnd(bad_radsat_bits).eq(0)
    return optical.And(qa_clear).And(unsaturated)


def _prep_donors(donors: list[dict], fill_mask_mode: str, bands: list[str]):
    """Return list of (donor_image, donor_valid_mask).

    Donor image keeps only bands with valid data (simple mask).
    The donor_valid mask applies QA gating on the *output* — regression kernel
    sees all valid pixels and only fill output is pruned by QA.
    """
    result = []
    for d in donors:
        img = d["image"].select(bands)
        # Simple band-validity mask for regression input
        simple_mask = img.mask().reduce(ee.Reducer.min())
        valid = _donor_fill_mask(d, fill_mask_mode)
        result.append((img.updateMask(simple_mask), valid))
    return result


def apply_ordered_donor_fill(
    target: ee.Image,
    donors: list[dict],
    kernel_size: int = 5,
    min_neighbors: int = 64,
    upscale: bool = True,
    fallback_to_donor: bool = False,
    fill_mask_mode: str = "simple",
) -> ee.Image:
    target_bands = target.select(LE07_BANDS)
    out = target_bands
    remaining_gap = target_gap_mask(target)
    donor_imgs = _prep_donors(donors, fill_mask_mode, LE07_BANDS)

    for idx, (donor, donor_valid) in enumerate(donor_imgs):
        candidate = _apply_donor_fill(
            target_bands,
            donor,
            min_neighbors=min_neighbors,
            kernel_size=kernel_size,
            upscale=upscale,
            fallback_to_donor=fallback_to_donor and idx == 0,
        )
        candidate_valid = candidate.select(OPTICAL_MASK_BANDS).mask().reduce(ee.Reducer.min())
        use_pixels = remaining_gap.And(candidate_valid).And(donor_valid)
        out = out.unmask(candidate.updateMask(use_pixels), True)
        remaining_gap = remaining_gap.And(use_pixels.Not())

    return out


def apply_median_donor_fill(
    target: ee.Image,
    donors: list[dict],
    kernel_size: int = 5,
    min_neighbors: int = 64,
    upscale: bool = True,
    fallback_to_donor: bool = False,
    fill_mask_mode: str = "simple",
) -> ee.Image:
    target_bands = target.select(LE07_BANDS)
    target_gap = target_gap_mask(target)
    donor_imgs = _prep_donors(donors, fill_mask_mode, LE07_BANDS)
    candidates = []

    for idx, (donor, donor_valid) in enumerate(donor_imgs):
        candidate = _apply_donor_fill(
            target_bands,
            donor,
            min_neighbors=min_neighbors,
            kernel_size=kernel_size,
            upscale=upscale,
            fallback_to_donor=fallback_to_donor and idx == 0,
        )
        candidate_valid = candidate.select(OPTICAL_MASK_BANDS).mask().reduce(ee.Reducer.min())
        use_pixels = target_gap.And(candidate_valid).And(donor_valid)
        candidates.append(candidate.updateMask(use_pixels))

    gap_fill = ee.ImageCollection(candidates).median()
    return target_bands.unmask(gap_fill, True)


def add_ndsi_brightness(img: ee.Image) -> ee.Image:
    ndsi = img.normalizedDifference(["B2", "B5"]).rename("NDSI")
    brightness = img.select(OPTICAL_MASK_BANDS).reduce(ee.Reducer.mean()).rename("BRIGHTNESS")
    return img.addBands([ndsi, brightness])


def donor_score_surface(src: ee.Image, donor: ee.Image, kernel_size: int = 5) -> ee.Image:
    """Old-style local raw-difference score. Low score = better match."""
    srcx = add_ndsi_brightness(src.select(OPTICAL_MASK_BANDS))
    donx = add_ndsi_brightness(donor.select(OPTICAL_MASK_BANDS))
    common = srcx.select(OPTICAL_MASK_BANDS).mask().reduce(ee.Reducer.min()).And(
        donx.select(OPTICAL_MASK_BANDS).mask().reduce(ee.Reducer.min())
    )
    score_bands = OPTICAL_MASK_BANDS + ["NDSI", "BRIGHTNESS"]
    diffs = srcx.select(score_bands).subtract(
        donx.select(score_bands)
    ).abs().updateMask(common)
    kernel = ee.Kernel.square(kernel_size * 30, "meters", False)
    local = diffs.reduceNeighborhood(ee.Reducer.mean(), kernel, None, False)
    score = local.reduce(ee.Reducer.mean()).rename("score")
    min_s = score.reduceNeighborhood(ee.Reducer.min(), kernel, None, False)
    max_s = score.reduceNeighborhood(ee.Reducer.max(), kernel, None, False)
    return score.subtract(min_s).divide(max_s.subtract(min_s).add(1e-12)).unmask(1)


def apply_blended_donor_fill(
    target: ee.Image,
    donors: list[dict],
    kernel_size: int = 5,
    min_neighbors: int = 64,
    upscale: bool = True,
    fallback_to_donor: bool = False,
    fallback_best: bool = False,
    fill_mask_mode: str = "simple",
) -> ee.Image:
    target_bands = target.select(LE07_BANDS)
    target_gap = target_gap_mask(target)
    donor_imgs = _prep_donors(donors, fill_mask_mode, LE07_BANDS)
    num = ee.Image.constant([0] * len(LE07_BANDS)).rename(LE07_BANDS).toFloat()
    den = ee.Image.constant(0).rename("weight").toFloat()

    for idx, (donor, donor_valid) in enumerate(donor_imgs):
        candidate = _apply_donor_fill(
            target_bands,
            donor,
            min_neighbors=min_neighbors,
            kernel_size=kernel_size,
            upscale=upscale,
            fallback_to_donor=fallback_to_donor and idx == 0,
        )
        candidate_valid = candidate.select(OPTICAL_MASK_BANDS).mask().reduce(ee.Reducer.min())
        use_pixels = target_gap.And(candidate_valid).And(donor_valid)
        weight_value = donors[idx]["gap_coverage"] * donors[idx]["overlap_coverage"]
        weight = ee.Image.constant(weight_value).updateMask(use_pixels).rename("weight")
        num = num.add(candidate.toFloat().multiply(weight))
        den = den.add(weight)

    gap_fill = num.divide(den).updateMask(den.gt(0))
    out = target_bands.unmask(gap_fill, True)

    if fallback_best and donors:
        best_donor = _prep_donors(donors[:1], fill_mask_mode, LE07_BANDS)[0][0]
        best = _apply_donor_fill(
            target_bands,
            best_donor,
            min_neighbors=min_neighbors,
            kernel_size=kernel_size,
            upscale=upscale,
            fallback_to_donor=fallback_to_donor,
        )
        best_valid = best.select(OPTICAL_MASK_BANDS).mask().reduce(ee.Reducer.min())
        out = out.unmask(best.updateMask(target_gap.And(best_valid)), True)

    return out


def apply_soft_blended_donor_fill(
    target: ee.Image,
    donors: list[dict],
    kernel_size: int = 5,
    min_neighbors: int = 64,
    upscale: bool = True,
    fallback_to_donor: bool = False,
    fill_mask_mode: str = "simple",
) -> ee.Image:
    target_bands = target.select(LE07_BANDS)
    target_gap = target_gap_mask(target)
    donor_imgs = _prep_donors(donors, fill_mask_mode, LE07_BANDS)
    num = ee.Image.constant([0] * len(LE07_BANDS)).rename(LE07_BANDS).toFloat()
    den = ee.Image.constant(0).rename("weight").toFloat()

    for donor, donor_valid in donor_imgs:
        candidate = _apply_donor_fill(
            target_bands,
            donor,
            min_neighbors=min_neighbors,
            kernel_size=kernel_size,
            upscale=upscale,
            fallback_to_donor=fallback_to_donor,
        )
        candidate = apply_common_optical_mask(candidate.select(LE07_BANDS)).toFloat()
        fill_valid = candidate.mask().reduce(ee.Reducer.min())
        score = donor_score_surface(target_bands, donor, kernel_size=kernel_size)
        weight = score.divide(SOFTBLEND_TEMPERATURE).multiply(-1).exp()
        weight = weight.updateMask(target_gap).updateMask(fill_valid).updateMask(donor_valid).rename("weight")
        num = num.add(candidate.multiply(weight))
        den = den.add(weight)

    gap_fill = num.divide(den).updateMask(den.gt(0))
    return target_bands.unmask(gap_fill, True)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------
def load_fishnet_tile(tile_index: int) -> ee.Geometry:
    with FISHNET_PATH.open() as handle:
        data = json.load(handle)
    for feat in data["features"]:
        props = feat.get("properties", {})
        if int(props.get("_export_index", -1)) == tile_index:
            return ee.Geometry(feat["geometry"])
    raise ValueError(f"Fishnet tile not found: {tile_index}")


def apply_common_optical_mask(img: ee.Image) -> ee.Image:
    """Keep pixels valid in all optical bands.

    This prevents false-color boundary artifacts where some bands have data and
    others are masked. Thermal masks are ignored so valid optical pixels are not
    erased by thermal-edge behavior.
    """
    mask = img.select(OPTICAL_MASK_BANDS).mask().reduce(ee.Reducer.min())
    return img.updateMask(mask)


def prepare_export_image(img: ee.Image) -> ee.Image:
    """Preserve TOA reflectance dynamic range."""
    return img.toFloat()


def start_export(
    img: ee.Image,
    desc: str,
    folder: str,
    region: ee.Geometry | None = None,
) -> ee.batch.Task:
    task = ee.batch.Export.image.toDrive(
        image=img,
        description=desc,
        folder=folder,
        region=region,
        scale=30,
        maxPixels=1e9,
    )
    task.start()
    return task


def build_scene_metadata(
    path: int,
    row: int,
    year: int,
    month: int,
    day: int,
    tile_index: int | None,
    tile_suffix: str,
    export_folder: str,
    fill_disabled: bool,
    donors_requested: int,
    donor_rank: int,
    fill_mask_mode: str,
    multi_fill: str,
    forced_donor_date: str | None,
    donor_ranking: str,
) -> dict:
    desc = export_description(path, row, year, month, day)
    target_date = f"{year:04d}-{month:02d}-{day:02d}"
    slc_off = is_slc_off(year, month, day)

    return {
        "pr": pr_string(path, row),
        "target_scene": desc,
        "target_image_id": image_id(path, row, year, month, day),
        "target_date": target_date,
        "target_sensor": "LE07",
        "slc_off": slc_off,
        "fill_enabled": not fill_disabled,
        "donors_requested": donors_requested,
        "donor_rank_requested": donor_rank,
        "forced_donor_date": forced_donor_date or "",
        "multi_fill": multi_fill,
        "donor_ranking": donor_ranking,
        "fill_mask_mode_requested": fill_mask_mode,
        "fill_mask_mode_effective": "disabled" if fill_disabled else "none",
        "planner_donors_selected": 0,
        "donors_used": 0,
        "cloud_threshold_used": None,
        "tile_index": tile_index,
        "tile_suffix": tile_suffix,
        "export_folder": export_folder,
        "planner_donor_scene_ids": "",
        "planner_donor_dates": "",
        "planner_donor_gap_coverages": "",
        "planner_donor_overlap_coverages": "",
        "planner_donor_marginal_coverages": "",
        "used_donor_scene_ids": "",
        "used_donor_dates": "",
        "used_donor_gap_coverages": "",
        "used_donor_overlap_coverages": "",
        "used_donor_marginal_coverages": "",
    }


def write_metadata(out_dir: Path, metadata_rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in metadata_rows:
        path = out_dir / f"{row['target_scene']}.json"
        path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)


def common_neighbor_mask(
    target_bands: ee.Image,
    donor_bands: ee.Image,
    kernel_size: int,
    min_neighbors: int,
) -> ee.Image:
    """Pixels with enough target/donor common neighbors for regression."""
    common = target_bands.mask().And(donor_bands.mask())
    kernel = ee.Kernel.square(kernel_size * 30, "meters", False)
    count = common.reduceNeighborhood(ee.Reducer.count(), kernel, None, True, "boxcar")
    return count.gte(min_neighbors).reduce(ee.Reducer.min()).rename("common_count_ok")


def diagnostic_mask_image(
    target: ee.Image,
    donor_info: dict,
    filled: ee.Image,
    fill_mask_mode: str,
    kernel_size: int,
    min_neighbors: int,
) -> ee.Image:
    """Mask diagnostic bands for one target/donor fill."""
    target_bands = target.select(LE07_BANDS)
    donor_bands = donor_info["image"].select(LE07_BANDS)
    donor_simple = donor_bands.mask().reduce(ee.Reducer.min())
    donor_qa_clear = donor_info["clear_mask"]
    donor_fill_valid = _donor_fill_mask(donor_info, fill_mask_mode)
    filled_valid = filled.select(OPTICAL_MASK_BANDS).mask().reduce(ee.Reducer.min())
    final_common = apply_common_optical_mask(filled).select(OPTICAL_MASK_BANDS).mask().reduce(ee.Reducer.min())

    bands = [
        target_gap_mask(target).rename("target_gap"),
        clear_valid_mask(target).rename("target_clear"),
        donor_simple.rename("donor_simple_valid"),
        donor_qa_clear.rename("donor_qa_clear"),
        donor_fill_valid.rename("donor_fill_valid"),
        common_neighbor_mask(target_bands, donor_bands, kernel_size, min_neighbors),
        filled_valid.rename("filled_valid_before_common"),
        final_common.rename("filled_valid_after_common"),
    ]
    return ee.Image.cat([b.unmask(0).toUint8() for b in bands])


def export_all(args: argparse.Namespace) -> None:
    subset = parse_subset(args.subset)
    tile_geom = load_fishnet_tile(args.tile_index) if args.tile_index is not None else None
    tile_suffix = f"_tile{args.tile_index}" if args.tile_index is not None else ""
    tile_suffix += args.name_suffix
    metadata_rows: list[dict] = []

    print(f"=== HKH LE07 SLC-OFF FILL → {args.folder} ===")
    for path, row, year, month, day in export_scenes():
        pr = pr_string(path, row)
        if subset and pr not in subset:
            continue

        desc = export_description(path, row, year, month, day)
        print(f"  {desc}{tile_suffix}  ({pr})")

        metadata = build_scene_metadata(
            path=path,
            row=row,
            year=year,
            month=month,
            day=day,
            tile_index=args.tile_index,
            tile_suffix=tile_suffix,
            export_folder=args.folder,
            fill_disabled=args.disable_fill,
            donors_requested=args.donors,
            donor_rank=args.donor_rank,
            fill_mask_mode=args.fill_mask_mode,
            multi_fill=args.multi_fill,
            forced_donor_date=args.forced_donor_date,
            donor_ranking=args.donor_ranking,
        )

        if args.dry_run:
            metadata_rows.append(metadata)
            continue

        img = ee.Image(image_id(path, row, year, month, day))
        target_img = img
        roi = tile_geom if tile_geom is not None else img.geometry()
        donors = []
        cloud_threshold_used = None
        if metadata["slc_off"] and not args.disable_fill:
            if args.forced_donor_date:
                donors, cloud_threshold_used = choose_forced_donor(
                    path, row, img, roi, args.forced_donor_date
                )
            else:
                need = max(args.donors, args.donor_rank)
                donors, cloud_threshold_used = choose_donors(
                    path, row, img, roi, need, args.donor_ranking
                )
            metadata["planner_donors_selected"] = len(donors)
            metadata["cloud_threshold_used"] = cloud_threshold_used
            metadata["planner_donor_scene_ids"] = ";".join(d["scene_id"] for d in donors)
            metadata["planner_donor_dates"] = ";".join(d["date"] for d in donors)
            metadata["planner_donor_gap_coverages"] = ";".join(
                f"{d['gap_coverage']:.4f}" for d in donors
            )
            metadata["planner_donor_overlap_coverages"] = ";".join(
                f"{d['overlap_coverage']:.4f}" for d in donors
            )
            metadata["planner_donor_marginal_coverages"] = ";".join(
                f"{d['marginal_gap_coverage']:.4f}" for d in donors
            )

            if not donors:
                print("    no coverage-qualified pre-SLC donors — exporting with stripes")
            else:
                if args.donors == 1 and args.forced_donor_date:
                    use_donors = donors[:1]
                    metadata["fill_mask_mode_effective"] = "single_donor_simple_regression"
                elif args.donors == 1:
                    if args.donor_rank > len(donors):
                        raise ValueError(
                            f"--donor-rank {args.donor_rank} unavailable; "
                            f"planner returned {len(donors)} donors"
                        )
                    use_donors = [donors[args.donor_rank - 1]]
                    metadata["fill_mask_mode_effective"] = "single_donor_simple_regression"
                else:
                    use_donors = donors[:args.donors]
                    metadata["fill_mask_mode_effective"] = args.fill_mask_mode

                metadata["donors_used"] = len(use_donors)
                metadata["used_donor_scene_ids"] = ";".join(d["scene_id"] for d in use_donors)
                metadata["used_donor_dates"] = ";".join(d["date"] for d in use_donors)
                metadata["used_donor_gap_coverages"] = ";".join(
                    f"{d['gap_coverage']:.4f}" for d in use_donors
                )
                metadata["used_donor_overlap_coverages"] = ";".join(
                    f"{d['overlap_coverage']:.4f}" for d in use_donors
                )
                metadata["used_donor_marginal_coverages"] = ";".join(
                    f"{d['marginal_gap_coverage']:.4f}" for d in use_donors
                )

                print(
                    "    donors used: "
                    + ", ".join(
                        f"{d['date']} gap={d['gap_coverage']:.2f} "
                        f"overlap={d['overlap_coverage']:.2f}"
                        for d in use_donors
                    )
                )
                donor_img, _ = _prep_donors(use_donors[:1], args.fill_mask_mode, LE07_BANDS)[0]
                if args.donors == 1:
                    img = fill_slc_off_gaps(
                        img.select(LE07_BANDS),
                        donor_img,
                        kernel_size=args.kernel_size,
                        min_neighbors=args.min_neighbors,
                        upscale=args.upscale,
                        fallback_to_donor=args.fallback_to_donor,
                    )
                elif args.multi_fill == "median":
                    img = apply_median_donor_fill(
                        img,
                        use_donors,
                        kernel_size=args.kernel_size,
                        min_neighbors=args.min_neighbors,
                        upscale=args.upscale,
                        fallback_to_donor=args.fallback_to_donor,
                        fill_mask_mode=args.fill_mask_mode,
                    )
                elif args.multi_fill == "softblend":
                    img = apply_soft_blended_donor_fill(
                        img,
                        use_donors,
                        kernel_size=args.kernel_size,
                        min_neighbors=args.min_neighbors,
                        upscale=args.upscale,
                        fallback_to_donor=args.fallback_to_donor,
                        fill_mask_mode=args.fill_mask_mode,
                    )
                elif args.multi_fill in {"blend", "blendfb"}:
                    img = apply_blended_donor_fill(
                        img,
                        use_donors,
                        kernel_size=args.kernel_size,
                        min_neighbors=args.min_neighbors,
                        upscale=args.upscale,
                        fallback_to_donor=args.fallback_to_donor,
                        fallback_best=args.multi_fill == "blendfb",
                        fill_mask_mode=args.fill_mask_mode,
                    )
                else:
                    img = apply_ordered_donor_fill(
                        img,
                        use_donors,
                        kernel_size=args.kernel_size,
                        min_neighbors=args.min_neighbors,
                        upscale=args.upscale,
                        fallback_to_donor=args.fallback_to_donor,
                        fill_mask_mode=args.fill_mask_mode,
                    )

                if args.diagnostic_masks:
                    img = diagnostic_mask_image(
                        target_img,
                        use_donors[0],
                        img,
                        args.fill_mask_mode,
                        args.kernel_size,
                        args.min_neighbors,
                    )

        metadata_rows.append(metadata)
        if args.metadata_only:
            continue

        if args.diagnostic_masks:
            export_img = img
        else:
            export_img = img.select(LE07_BANDS)
            if not args.disable_common_mask:
                export_img = apply_common_optical_mask(export_img)
            export_img = prepare_export_image(export_img)
        if tile_geom is not None:
            export_img = export_img.clip(tile_geom)

        ranking_tag = "" if args.donor_ranking == "balanced" else f"_{args.donor_ranking}"
        forced_tag = f"_donor{args.forced_donor_date.replace('-', '')}" if args.forced_donor_date else ""
        rank_tag = f"_r{args.donor_rank}" if args.donor_rank > 1 and not forced_tag else ""
        if args.diagnostic_masks:
            fill_suffix = f"{ranking_tag}{forced_tag}{rank_tag}_diagnostic_masks"
        elif args.donors > 1:
            fill_suffix = f"{ranking_tag}{rank_tag}_d{args.donors}_{args.multi_fill}_{args.fill_mask_mode}"
        else:
            fill_suffix = f"{ranking_tag}{forced_tag or rank_tag}"
        start_export(export_img, f"hkh_{desc}{tile_suffix}{fill_suffix}", args.folder, tile_geom)

    if metadata_rows:
        write_metadata(args.metadata_out_dir, metadata_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export HKH LE07 scenes with USGS/Gorelick/Donchyts SLC-off filling."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Generate metadata only, no Drive exports",
    )
    parser.add_argument(
        "--subset",
        type=str,
        help="Comma-separated path/rows, e.g. '133-040,133-041'",
    )
    parser.add_argument(
        "--tile-index",
        type=int,
        default=None,
        help="Export only one fishnet tile for faster debugging",
    )
    parser.add_argument(
        "--name-suffix",
        type=str,
        default="",
        help="Extra suffix appended to export description, e.g. '_debug'",
    )
    parser.add_argument(
        "--folder",
        default=EXPORT_FOLDER,
        help=f"Google Drive output folder (default: {EXPORT_FOLDER})",
    )
    parser.add_argument(
        "--metadata-out-dir",
        type=Path,
        default=DEFAULT_METADATA_DIR,
        help=f"Local metadata output dir (default: {DEFAULT_METADATA_DIR})",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=5,
        help="Local regression kernel size in pixels (default: 5)",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=64,
        help="Minimum common neighbors required to fill a pixel (default: 64)",
    )
    parser.add_argument(
        "--upscale",
        dest="upscale",
        action="store_true",
        help="Use reduced-resolution fast path (default)",
    )
    parser.add_argument(
        "--no-upscale",
        dest="upscale",
        action="store_false",
        help="Use full-resolution local fit",
    )
    parser.add_argument(
        "--fallback-to-donor",
        action="store_true",
        help="Fill residual holes with raw donor pixels after regression fill",
    )
    parser.add_argument(
        "--fill-mask-mode",
        choices=("simple", "cloud-only", "qa-clear"),
        default="simple",
        help="Multi-donor fill-time mask. Ignored by --donors 1, which uses single-donor simple regression.",
    )
    parser.add_argument(
        "--donor-ranking",
        choices=("coverage", "overlap", "balanced"),
        default="balanced",
        help="Donor sort before greedy marginal selection (default: balanced=gap*overlap)",
    )
    parser.add_argument(
        "--donors",
        type=int,
        default=1,
        help="Number of ranked pre-SLC donors to use (default: 1)",
    )
    parser.add_argument(
        "--donor-rank",
        type=int,
        default=1,
        help="When --donors 1, use the Nth greedy-planned donor instead of the top (default: 1)",
    )
    parser.add_argument(
        "--forced-donor-date",
        type=str,
        default=None,
        help="Debug only: force exact donor DATE_ACQUIRED (YYYY-MM-DD); requires --donors 1",
    )
    parser.add_argument(
        "--multi-fill",
        choices=("median", "blend", "blendfb", "softblend", "ordered"),
        default="ordered",
        help="How to combine donors when --donors > 1 (default: ordered)",
    )
    parser.add_argument(
        "--disable-fill",
        action="store_true",
        help="Export target scene without filling for debugging",
    )
    parser.add_argument(
        "--disable-common-mask",
        action="store_true",
        help="Skip optical common-mask reduction for debugging",
    )
    parser.add_argument(
        "--diagnostic-masks",
        action="store_true",
        help="Export diagnostic mask bands instead of spectral bands",
    )
    parser.set_defaults(upscale=True)
    args = parser.parse_args()
    if args.donors < 1:
        parser.error("--donors must be >= 1")
    if args.donor_rank < 1:
        parser.error("--donor-rank must be >= 1")
    if args.forced_donor_date and args.donors != 1:
        parser.error("--forced-donor-date requires --donors 1")
    return args


def main() -> None:
    args = parse_args()
    initialize_ee()
    export_all(args)


if __name__ == "__main__":
    main()
