#!/usr/bin/env python3
"""Audit Step 2 metrics and select a narrow donor export slate.

Local-only policy script. No GEE calls.

This replaces the earlier split audit/select scripts. It treats Step 2 metrics as
frozen expensive evidence and chooses a deliberately narrow initial slate:

- up to 1 LT05 donor per target
- up to 1 LE07 pre-SLC donor per target
- up to 1 LE07 post-SLC donor per target

Rationale: old baseline-style workflows used one donor and were not terrible.
Start narrow, export/test a few scenes, then widen only when local validation says
we need more candidates.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TARGETS_JSON = Path("dataset/outputs/1_targets.json")
METRICS_CSV = Path("dataset/outputs/2_donor_metrics.csv")
STEP2_SUMMARY_CSV = Path("dataset/outputs/2_target_summary.csv")
OUTPUT_DIR = Path("dataset/outputs")

POLICY_VERSION = "narrow_one_per_family_v1"

AUDIT_SUMMARY_JSON = OUTPUT_DIR / "3_audit_summary.json"
FLAGS_CSV = OUTPUT_DIR / "3_metric_flags.csv"
SCORES_CSV = OUTPUT_DIR / "3_donor_scores.csv"
NARROW_SLATE_CSV = OUTPUT_DIR / "3_donor_slate_narrow.csv"
TARGET_SUMMARY_CSV = OUTPUT_DIR / "3_target_slate_summary.csv"
STORAGE_ESTIMATE_JSON = OUTPUT_DIR / "3_storage_estimate.json"

DONOR_KINDS = ["lt05", "le07_slc_on", "le07_slc_off"]
MAX_SELECTED_PER_TARGET_KIND = 1
RAW_STACK_BANDS = 11
RAW_DTYPE_BYTES = 4  # Float32 export: optical + QA + helper masks.
GIB = 1024**3

# Hard gates. If no candidate passes strict gate for a family, the best relaxed
# candidate is still chosen but marked caution=true with relaxation reason.
MIN_SPECTRAL_OVERLAP_FRACTION = 0.01
MIN_QA_OVERLAP = 0.20
MIN_SIMPLE_OVERLAP = 0.60

LT05_MIN_QA_GAP = 0.20
LT05_MAX_SPECTRAL_RESIDUAL = 0.15
LT05_MAX_DOY = 45

LE07_ON_MIN_QA_GAP = 0.20
LE07_ON_MAX_SPECTRAL_RESIDUAL = 0.15
LE07_ON_MAX_DOY = 60

LE07_OFF_MIN_SIMPLE_GAP = 0.50
LE07_OFF_MIN_QA_GAP = 0.20
LE07_OFF_MAX_SPECTRAL_RESIDUAL = 0.18
LE07_OFF_MAX_DOY = 60

# Score scale constants. These are policy knobs, not GEE-derived truth.
SPECTRAL_SIGMA = 0.08
NDSI_SIGMA = 0.08
NDVI_SIGMA = 0.10
BRIGHTNESS_SIGMA = 0.06
SEASON_SIGMA_DAYS = 30.0
DATE_SIGMA_DAYS = 365.0
CLOUD_MAX = 10.0

OUTPUT_SCORE_FIELDS = [
    "policy_version",
    "selected_narrow",
    "selection_tier",
    "caution",
    "caution_reasons",
    "family_score",
    "coverage_score",
    "similarity_score",
    "temporal_score",
    "gap_score",
    "spectral_score",
    "ndsi_score",
    "brightness_score",
    "season_score",
    "date_score",
    "cloud_score",
]

SLATE_FIELDS = [
    "target_id",
    "target_scene",
    "target_pr",
    "target_date",
    "target_filename",
    "donor_kind",
    "donor_sensor",
    "donor_collection",
    "donor_product_id",
    "donor_date",
    "cloud_cover",
    "family_score",
    "selection_tier",
    "caution",
    "caution_reasons",
    "simple_gap_coverage",
    "qa_gap_coverage",
    "simple_overlap_coverage",
    "qa_overlap_coverage",
    "spectral_overlap_fraction",
    "mean_median_norm_residual",
    "ndsi_residual",
    "abs_date_diff_days",
    "doy_diff",
    "estimated_uncompressed_gib",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_targets() -> dict[int, dict[str, Any]]:
    return {int(row["id"]): row for row in json.loads(TARGETS_JSON.read_text(encoding="utf-8"))}


def f(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def clamp01(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, value))


def exp_score(value: float, sigma: float) -> float:
    if math.isnan(value):
        return 0.0
    return math.exp(-max(0.0, value) / sigma)


def target_raw_gib(target: dict[str, Any]) -> float:
    width = int(target["export_width_estimate"])
    height = int(target["export_height_estimate"])
    return width * height * RAW_STACK_BANDS * RAW_DTYPE_BYTES / GIB


def ratio_quantiles(values: list[float]) -> dict[str, float | None]:
    vals = sorted(v for v in values if not math.isnan(v))
    if not vals:
        return {"min": None, "median": None, "p95": None, "max": None}

    def pct(p: float) -> float:
        idx = (len(vals) - 1) * p
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return vals[lo]
        return vals[lo] * (hi - idx) + vals[hi] * (idx - lo)

    return {
        "min": vals[0],
        "median": statistics.median(vals),
        "p95": pct(0.95),
        "max": vals[-1],
    }


def score_row(row: dict[str, str], target: dict[str, Any]) -> dict[str, Any]:
    qa_gap = clamp01(f(row, "qa_gap_coverage"))
    simple_gap = clamp01(f(row, "simple_gap_coverage"))
    qa_overlap = clamp01(f(row, "qa_overlap_coverage"))
    simple_overlap = clamp01(f(row, "simple_overlap_coverage"))
    qa_balanced = clamp01(f(row, "qa_balanced"))

    coverage_score = (
        0.35 * qa_gap
        + 0.20 * simple_gap
        + 0.25 * qa_overlap
        + 0.20 * simple_overlap
    )
    gap_score = 0.55 * qa_gap + 0.30 * simple_gap + 0.15 * qa_balanced

    spectral_score = exp_score(f(row, "mean_median_norm_residual"), SPECTRAL_SIGMA)
    ndsi_score = exp_score(f(row, "ndsi_residual"), NDSI_SIGMA)
    ndvi_score = exp_score(f(row, "ndvi_residual"), NDVI_SIGMA)
    brightness_score = exp_score(f(row, "brightness_residual"), BRIGHTNESS_SIGMA)
    similarity_score = (
        0.50 * spectral_score
        + 0.25 * ndsi_score
        + 0.15 * brightness_score
        + 0.10 * ndvi_score
    )

    season_score = exp_score(f(row, "doy_diff"), SEASON_SIGMA_DAYS)
    date_score = exp_score(f(row, "abs_date_diff_days"), DATE_SIGMA_DAYS)
    cloud_score = clamp01(1.0 - f(row, "cloud_cover", CLOUD_MAX) / CLOUD_MAX)
    temporal_score = 0.65 * season_score + 0.35 * date_score

    kind = row["donor_kind"]
    if kind == "lt05":
        # LT05: no SLC stripes, so spectral/temporal match should matter strongly.
        family_score = (
            0.30 * coverage_score
            + 0.40 * similarity_score
            + 0.20 * temporal_score
            + 0.10 * cloud_score
        )
    elif kind == "le07_slc_on":
        # Pre-SLC LE07: often older, so spectral/season dominates absolute date.
        family_score = (
            0.30 * coverage_score
            + 0.45 * similarity_score
            + 0.15 * season_score
            + 0.10 * cloud_score
        )
    elif kind == "le07_slc_off":
        # Post-SLC donors must cover target gaps; otherwise they waste export time.
        family_score = (
            0.55 * gap_score
            + 0.25 * similarity_score
            + 0.15 * temporal_score
            + 0.05 * cloud_score
        )
    else:
        raise ValueError(kind)

    caution_reasons = hard_gate_failures(row)
    return {
        **row,
        "policy_version": POLICY_VERSION,
        "family_score": family_score,
        "coverage_score": coverage_score,
        "similarity_score": similarity_score,
        "temporal_score": temporal_score,
        "gap_score": gap_score,
        "spectral_score": spectral_score,
        "ndsi_score": ndsi_score,
        "brightness_score": brightness_score,
        "season_score": season_score,
        "date_score": date_score,
        "cloud_score": cloud_score,
        "selected_narrow": False,
        "selection_tier": "not_selected",
        "caution": bool(caution_reasons),
        "caution_reasons": ";".join(caution_reasons),
        "estimated_uncompressed_gib": target_raw_gib(target),
    }


def hard_gate_failures(row: dict[str, Any]) -> list[str]:
    kind = row["donor_kind"]
    reasons = []
    if f(row, "spectral_overlap_fraction") < MIN_SPECTRAL_OVERLAP_FRACTION:
        reasons.append("low_spectral_overlap")
    if f(row, "qa_overlap_coverage") < MIN_QA_OVERLAP:
        reasons.append("low_qa_overlap")
    if f(row, "simple_overlap_coverage") < MIN_SIMPLE_OVERLAP:
        reasons.append("low_simple_overlap")

    spectral = f(row, "mean_median_norm_residual")
    doy = f(row, "doy_diff")
    if kind == "lt05":
        if f(row, "qa_gap_coverage") < LT05_MIN_QA_GAP:
            reasons.append("lt05_low_qa_gap")
        if spectral > LT05_MAX_SPECTRAL_RESIDUAL:
            reasons.append("lt05_high_spectral_residual")
        if doy > LT05_MAX_DOY:
            reasons.append("lt05_high_doy")
    elif kind == "le07_slc_on":
        if f(row, "qa_gap_coverage") < LE07_ON_MIN_QA_GAP:
            reasons.append("slc_on_low_qa_gap")
        if spectral > LE07_ON_MAX_SPECTRAL_RESIDUAL:
            reasons.append("slc_on_high_spectral_residual")
        if doy > LE07_ON_MAX_DOY:
            reasons.append("slc_on_high_doy")
    elif kind == "le07_slc_off":
        if f(row, "simple_gap_coverage") < LE07_OFF_MIN_SIMPLE_GAP:
            reasons.append("slc_off_low_simple_gap")
        if f(row, "qa_gap_coverage") < LE07_OFF_MIN_QA_GAP:
            reasons.append("slc_off_low_qa_gap")
        if spectral > LE07_OFF_MAX_SPECTRAL_RESIDUAL:
            reasons.append("slc_off_high_spectral_residual")
        if doy > LE07_OFF_MAX_DOY:
            reasons.append("slc_off_high_doy")
    return reasons


def select_narrow(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        grouped[(row["target_id"], row["donor_kind"])].append(row)

    selected = []
    for (_target_id, _kind), rows in sorted(grouped.items(), key=lambda item: (int(item[0][0]), item[0][1])):
        strict = [row for row in rows if not row["caution"]]
        pool = strict if strict else rows
        chosen = sorted(pool, key=lambda row: f(row, "family_score"), reverse=True)[:MAX_SELECTED_PER_TARGET_KIND]
        for row in chosen:
            row["selected_narrow"] = True
            row["selection_tier"] = "strict" if row in strict else "relaxed_best_available"
            selected.append(row)
    return selected


def flag_rows(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flags = []
    for row in scored:
        reasons = []
        for key in [
            "simple_gap_coverage",
            "qa_gap_coverage",
            "simple_overlap_coverage",
            "qa_overlap_coverage",
            "simple_balanced",
            "qa_balanced",
            "gap_collision",
            "qa_gap_collision",
            "spectral_overlap_fraction",
        ]:
            value = f(row, key)
            if not math.isnan(value) and not (0.0 <= value <= 1.000001):
                reasons.append(f"ratio_out_of_range:{key}={value}")
        reasons.extend(hard_gate_failures(row))
        if reasons:
            flags.append(
                {
                    "target_id": row["target_id"],
                    "target_scene": row["target_scene"],
                    "donor_kind": row["donor_kind"],
                    "donor_date": row["donor_date"],
                    "donor_product_id": row["donor_product_id"],
                    "selected_narrow": row.get("selected_narrow", False),
                    "reasons": ";".join(reasons),
                }
            )
    return flags


def build_target_summary(
    targets: dict[int, dict[str, Any]],
    scored: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    step2_summary: list[dict[str, str]],
) -> list[dict[str, Any]]:
    by_target = defaultdict(list)
    by_selected = defaultdict(list)
    for row in scored:
        by_target[int(row["target_id"])].append(row)
    for row in selected:
        by_selected[int(row["target_id"])].append(row)
    step2_by_id = {int(row["target_id"]): row for row in step2_summary}

    rows = []
    for target_id in sorted(targets):
        target_rows = by_target[target_id]
        selected_rows = by_selected[target_id]
        step2 = step2_by_id.get(target_id, {})
        out = {
            "target_id": target_id,
            "target_scene": targets[target_id]["scene"],
            "target_pr": targets[target_id]["path_row"],
            "candidate_count": len(target_rows),
            "selected_count": len(selected_rows),
            "estimated_selected_uncompressed_gib": sum(f(row, "estimated_uncompressed_gib") for row in selected_rows),
            "step2_complete": step2.get("complete", ""),
        }
        for kind in DONOR_KINDS:
            kind_rows = [row for row in target_rows if row["donor_kind"] == kind]
            kind_selected = [row for row in selected_rows if row["donor_kind"] == kind]
            out[f"{kind}_candidate_count"] = len(kind_rows)
            out[f"{kind}_selected_count"] = len(kind_selected)
            out[f"{kind}_pool_status"] = step2.get(f"{kind}_pool_status", "")
            out[f"{kind}_best_score"] = max((f(row, "family_score") for row in kind_rows), default=math.nan)
            out[f"{kind}_selected_date"] = kind_selected[0]["donor_date"] if kind_selected else ""
            out[f"{kind}_selected_caution"] = kind_selected[0]["caution"] if kind_selected else ""
        rows.append(out)
    return rows


def storage_estimate(
    targets: dict[int, dict[str, Any]],
    metrics: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    all_donor_gib = sum(target_raw_gib(targets[int(row["target_id"])]) for row in metrics)
    selected_gib = sum(f(row, "estimated_uncompressed_gib") for row in selected)
    target_gib = sum(target_raw_gib(target) for target in targets.values())
    return {
        "assumptions": {
            "raw_stack_bands": RAW_STACK_BANDS,
            "dtype": "float32",
            "bytes_per_pixel_per_band": RAW_DTYPE_BYTES,
            "compression": "not assumed; GeoTIFF compression may reduce actual size",
        },
        "all_strict_donors": {
            "donor_count": len(metrics),
            "uncompressed_gib": all_donor_gib,
        },
        "narrow_slate_donors": {
            "donor_count": len(selected),
            "uncompressed_gib": selected_gib,
        },
        "all_targets_once": {
            "target_count": len(targets),
            "uncompressed_gib": target_gib,
        },
        "narrow_slate_plus_all_targets": {
            "file_count": len(selected) + len(targets),
            "uncompressed_gib": selected_gib + target_gib,
        },
        "notes": [
            "These are upper-bound uncompressed Float32 stack estimates.",
            "Actual GeoTIFFs may be smaller with compression but should not be assumed small.",
            "All strict donors are likely too large to export casually.",
        ],
    }


def audit_summary(
    targets: dict[int, dict[str, Any]],
    metrics: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    flags: list[dict[str, Any]],
    step2_summary: list[dict[str, str]],
) -> dict[str, Any]:
    duplicate_keys = [key for key, count in Counter(row["candidate_key"] for row in metrics).items() if count > 1]
    selected_by_kind = Counter(row["donor_kind"] for row in selected)
    metrics_by_kind = Counter(row["donor_kind"] for row in metrics)
    selected_cautions = [row for row in selected if row["caution"]]
    weak_pools = {
        kind: [row["target_id"] for row in step2_summary if row.get(f"{kind}_pool_status") != "ok"]
        for kind in DONOR_KINDS
    }
    return {
        "policy_version": POLICY_VERSION,
        "target_count": len(targets),
        "metric_rows": len(metrics),
        "duplicate_candidate_key_count": len(duplicate_keys),
        "duplicate_candidate_keys": duplicate_keys[:20],
        "candidate_count_by_kind": dict(metrics_by_kind),
        "selected_narrow_count": len(selected),
        "selected_narrow_count_by_kind": dict(selected_by_kind),
        "selected_caution_count": len(selected_cautions),
        "flagged_row_count": len(flags),
        "weak_strict_pools": weak_pools,
        "selected_score_quantiles_by_kind": {
            kind: ratio_quantiles([f(row, "family_score") for row in selected if row["donor_kind"] == kind])
            for kind in DONOR_KINDS
        },
        "hard_gates": {
            "min_spectral_overlap_fraction": MIN_SPECTRAL_OVERLAP_FRACTION,
            "min_qa_overlap": MIN_QA_OVERLAP,
            "min_simple_overlap": MIN_SIMPLE_OVERLAP,
            "lt05_min_qa_gap": LT05_MIN_QA_GAP,
            "lt05_max_spectral_residual": LT05_MAX_SPECTRAL_RESIDUAL,
            "lt05_max_doy": LT05_MAX_DOY,
            "le07_on_min_qa_gap": LE07_ON_MIN_QA_GAP,
            "le07_on_max_spectral_residual": LE07_ON_MAX_SPECTRAL_RESIDUAL,
            "le07_on_max_doy": LE07_ON_MAX_DOY,
            "le07_off_min_simple_gap": LE07_OFF_MIN_SIMPLE_GAP,
            "le07_off_min_qa_gap": LE07_OFF_MIN_QA_GAP,
            "le07_off_max_spectral_residual": LE07_OFF_MAX_SPECTRAL_RESIDUAL,
            "le07_off_max_doy": LE07_OFF_MAX_DOY,
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]], preferred_fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(preferred_fields or [])
    for key in sorted({key for row in rows for key in row}):
        if key not in fields:
            fields.append(key)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fields})
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        json.dump(data, tmp, indent=2)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def csv_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return value


def main() -> None:
    targets = read_targets()
    metrics = read_csv(METRICS_CSV)
    step2_summary = read_csv(STEP2_SUMMARY_CSV)
    scored = [score_row(row, targets[int(row["target_id"])]) for row in metrics]
    selected = select_narrow(scored)
    flags = flag_rows(scored)
    target_summary = build_target_summary(targets, scored, selected, step2_summary)
    storage = storage_estimate(targets, scored, selected)
    audit = audit_summary(targets, scored, selected, flags, step2_summary)

    scored.sort(key=lambda row: (int(row["target_id"]), row["donor_kind"], -f(row, "family_score")))
    selected.sort(key=lambda row: (int(row["target_id"]), row["donor_kind"], -f(row, "family_score")))

    write_csv(SCORES_CSV, scored, OUTPUT_SCORE_FIELDS)
    write_csv(NARROW_SLATE_CSV, selected, SLATE_FIELDS)
    write_csv(FLAGS_CSV, flags)
    write_csv(TARGET_SUMMARY_CSV, target_summary)
    write_json(STORAGE_ESTIMATE_JSON, storage)
    write_json(AUDIT_SUMMARY_JSON, audit)

    print(json.dumps({"audit": audit, "storage_estimate": storage}, indent=2))
    print(f"wrote {AUDIT_SUMMARY_JSON}")
    print(f"wrote {FLAGS_CSV}")
    print(f"wrote {SCORES_CSV}")
    print(f"wrote {NARROW_SLATE_CSV}")
    print(f"wrote {TARGET_SUMMARY_CSV}")
    print(f"wrote {STORAGE_ESTIMATE_JSON}")


if __name__ == "__main__":
    main()
