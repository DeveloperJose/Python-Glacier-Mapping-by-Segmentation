#!/usr/bin/env python3
"""Forensics for relaxed-valid full8 vs legacy all-channel runs.

No training. Uses existing processed datasets and test evaluation CSVs.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATA_ROOT = Path("/home/devj/local-arch/data/HKH")
OUT_ROOT = Path("output")
OUT_DIR = Path("dataset/outputs")

DATASETS = {
    "legacy": DATA_ROOT / "comprehensive_v3",
    "agreement": DATA_ROOT / "comprehensive_v3_hkh_full8_agreement_quality_step3_relaxed_valid",
    "raw": DATA_ROOT / "comprehensive_v3_hkh_full8_raw_target_relaxed_valid",
}

RUN_PATTERNS = [
    "*relaxed_valid*_allch_bs8_seed42_desktop_20260706_*",
    "*relaxed_valid*_allch_bs8_seed4[34]_desktop_20260706_*",
    "legacy_comprehensive_v3_*_allch_bs8_seed42_desktop_20260705_*",
    "legacy_comprehensive_v3_*_allch_bs8_seed4[34]_desktop_20260706_*",
]

BANDS = ["B1", "B2", "B3", "B4", "B5", "B6_VCID1", "B6_VCID2", "B7"]
LABELS = {0: "background", 1: "ci", 2: "dci", 255: "ignore"}


def load_manifest(dataset: str, split: str = "test") -> dict[str, Any]:
    return json.loads((DATASETS[dataset] / split / "manifest.json").read_text())


def record_key(record: dict[str, Any]) -> str:
    return str(record["source_tiff_file"]).replace("tiff_", "").replace(".npy", "")


def get_indices(dataset: str, split: str = "test") -> dict[str, int]:
    manifest = load_manifest(dataset, split)
    return {record_key(r): int(r["index"]) for r in manifest["records"]}


def variant_from_name(name: str) -> str:
    if "legacy_comprehensive" in name:
        return "legacy"
    if "agreement_quality_step3" in name:
        return "agreement"
    if "raw_target" in name:
        return "raw"
    if "nspi_timeseries" in name:
        return "nspi"
    return "unknown"


def task_from_name(name: str) -> str:
    return "dci" if "_dci_" in name else "ci"


def seed_from_name(name: str) -> int:
    match = re.search(r"seed(\d+)", name)
    if match is None:
        raise ValueError(f"No seed in {name}")
    return int(match.group(1))


def csv_metrics_path(run_dir: Path) -> Path | None:
    paths = sorted((run_dir / "test_evaluations" / "csv_metrics").glob("epoch*.csv"))
    return paths[-1] if paths else None


def collect_runs() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for pattern in RUN_PATTERNS:
        for run_dir in OUT_ROOT.glob(pattern):
            if run_dir in seen:
                continue
            seen.add(run_dir)
            metrics_path = run_dir / "test_evaluations" / "test_metrics.json"
            csv_path = csv_metrics_path(run_dir)
            if not metrics_path.exists() or csv_path is None:
                continue
            name = run_dir.name
            task = task_from_name(name)
            metrics = json.loads(metrics_path.read_text())
            iou_key = "full_test_dci_iou" if task == "dci" else "full_test_ci_iou"
            rows.append(
                {
                    "task": task,
                    "seed": seed_from_name(name),
                    "variant": variant_from_name(name),
                    "iou": float(metrics[iou_key]),
                    "run_dir": str(run_dir),
                    "csv_path": str(csv_path),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["variant"].isin(["legacy", "agreement", "raw"])].copy()
    return df.sort_values("run_dir").drop_duplicates(
        ["task", "seed", "variant"], keep="last"
    )


def paired_slice_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    index_maps = {name: get_indices(name) for name in DATASETS}
    manifest_keys = {name: set(m.keys()) for name, m in index_maps.items()}
    common_all = set.intersection(*manifest_keys.values())
    rows: list[dict[str, Any]] = []
    for run in runs.itertuples(index=False):
        dataset = run.variant
        idx_to_key = {idx: key for key, idx in index_maps[dataset].items()}
        metric_df = pd.read_csv(run.csv_path)
        iou_col = "dci_iou" if run.task == "dci" else "ci_iou"
        for row in metric_df.itertuples(index=False):
            sample = getattr(row, "tile")
            idx = int(str(sample).replace("sample_", ""))
            key = idx_to_key.get(idx)
            if key is None or key not in common_all:
                continue
            rows.append(
                {
                    "task": run.task,
                    "seed": int(run.seed),
                    "variant": run.variant,
                    "slice_key": key,
                    "iou": float(getattr(row, iou_col)),
                    "precision": float(getattr(row, f"{run.task}_precision")),
                    "recall": float(getattr(row, f"{run.task}_recall")),
                }
            )
    return pd.DataFrame(rows)


def label_retention_common() -> pd.DataFrame:
    keys = {name: get_indices(name) for name in DATASETS}
    common = sorted(set.intersection(*(set(v) for v in keys.values())))
    rows: list[dict[str, Any]] = []
    ys = {
        name: np.load(DATASETS[name] / "test" / "y.npy", mmap_mode="r")
        for name in DATASETS
    }
    for variant, idx_map in keys.items():
        for key in common:
            y = ys[variant][idx_map[key]]
            counts = {LABELS[k]: int((y == k).sum()) for k in LABELS}
            rows.append({"variant": variant, "slice_key": key, **counts})
    return pd.DataFrame(rows)


def sample_band_values(
    dataset: str,
    class_value: int | None,
    common_keys: list[str],
    max_per_slice: int,
    rng: np.random.Generator,
) -> dict[str, list[np.ndarray]]:
    idx_map = get_indices(dataset)
    x = np.load(DATASETS[dataset] / "test" / "X.npy", mmap_mode="r")
    y = np.load(DATASETS[dataset] / "test" / "y.npy", mmap_mode="r")
    norm = np.load(DATASETS[dataset] / "normalize_train.npy")
    mean = norm[0, : len(BANDS)].astype(np.float32)
    std = norm[1, : len(BANDS)].astype(np.float32)
    out: dict[str, list[np.ndarray]] = {band: [] for band in BANDS}
    for key in common_keys:
        idx = idx_map[key]
        yy = y[idx]
        if class_value is None:
            mask = yy != 255
        else:
            mask = yy == class_value
        flat = np.flatnonzero(mask.ravel())
        if flat.size == 0:
            continue
        take_n = min(max_per_slice, flat.size)
        take = rng.choice(flat, size=take_n, replace=False)
        rows = take // yy.shape[1]
        cols = take % yy.shape[1]
        for band_idx, band in enumerate(BANDS):
            vals = x[idx, band_idx, rows, cols].astype(np.float32) * std[band_idx] + mean[band_idx]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                out[band].append(vals)
    return out


def summarize_values(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {k: math.nan for k in ["n", "mean", "std", "p1", "p5", "p50", "p95", "p99", "min", "max"]}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p1": float(np.quantile(values, 0.01)),
        "p5": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def band_distribution_audit() -> pd.DataFrame:
    common = sorted(set.intersection(*(set(get_indices(v)) for v in DATASETS)))
    rng = np.random.default_rng(42)
    rows: list[dict[str, Any]] = []
    for class_name, class_value in [("valid", None), ("ci", 1), ("dci", 2)]:
        for variant in ["legacy", "agreement", "raw"]:
            sampled = sample_band_values(
                variant,
                class_value=class_value,
                common_keys=common,
                max_per_slice=2000 if class_value is None else 4000,
                rng=rng,
            )
            for band, chunks in sampled.items():
                vals = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
                rows.append({"variant": variant, "class": class_name, "band": band, **summarize_values(vals)})
    return pd.DataFrame(rows)


def write_markdown(
    runs: pd.DataFrame,
    slice_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    band_df: pd.DataFrame,
) -> None:
    lines: list[str] = ["# Relaxed-valid vs legacy forensics", ""]
    lines.append("## Run-level all-channel IoU")
    summary = runs.groupby(["task", "variant"]).agg(
        n=("iou", "count"), mean_iou=("iou", "mean"), std_iou=("iou", "std")
    ).reset_index()
    lines.append("| task | variant | n | mean IoU | std |")
    lines.append("|---|---|---:|---:|---:|")
    for row in summary.sort_values(["task", "mean_iou"], ascending=[True, False]).itertuples(index=False):
        lines.append(f"| {row.task} | {row.variant} | {row.n} | {row.mean_iou:.6f} | {row.std_iou:.6f} |")
    lines.append("")

    lines.append("## Common-slice per-sample metric means")
    sm = slice_df.groupby(["task", "variant"]).agg(
        n=("iou", "count"), mean_iou=("iou", "mean"), std_iou=("iou", "std"), mean_precision=("precision", "mean"), mean_recall=("recall", "mean")
    ).reset_index()
    lines.append("| task | variant | n | mean slice IoU | std | precision | recall |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in sm.sort_values(["task", "mean_iou"], ascending=[True, False]).itertuples(index=False):
        lines.append(f"| {row.task} | {row.variant} | {row.n} | {row.mean_iou:.6f} | {row.std_iou:.6f} | {row.mean_precision:.6f} | {row.mean_recall:.6f} |")
    lines.append("")

    lines.append("## Paired common-slice deltas")
    deltas: list[dict[str, Any]] = []
    for (task, seed, slice_key), group in slice_df.groupby(["task", "seed", "slice_key"]):
        s = group.set_index("variant")
        for a, b in [("agreement", "legacy"), ("raw", "legacy"), ("agreement", "raw")]:
            if a in s.index and b in s.index:
                deltas.append({"task": task, "seed": seed, "comparison": f"{a}-{b}", "delta": float(s.loc[a, "iou"] - s.loc[b, "iou"])})
    ddf = pd.DataFrame(deltas)
    if not ddf.empty:
        ddf.to_csv(OUT_DIR / "relaxed_legacy_common_slice_paired_deltas.csv", index=False)
        dm = ddf.groupby(["task", "comparison"]).agg(
            n=("delta", "count"), mean_delta=("delta", "mean"), std_delta=("delta", "std"), win_rate=("delta", lambda x: float((x > 0).mean()))
        ).reset_index()
        lines.append("| task | comparison | n | mean delta | std | win rate |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for row in dm.itertuples(index=False):
            lines.append(f"| {row.task} | {row.comparison} | {row.n} | {row.mean_delta:+.6f} | {row.std_delta:.6f} | {row.win_rate:.3f} |")
    lines.append("")

    lines.append("## Label counts on common test slices")
    lm = labels_df.groupby("variant")[["background", "ci", "dci", "ignore"]].sum().reset_index()
    total = lm[["background", "ci", "dci", "ignore"]].sum(axis=1)
    lm["ignore_pct"] = lm["ignore"] / total * 100.0
    lines.append("| variant | background | CI | DCI | ignore | ignore % |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in lm.itertuples(index=False):
        lines.append(f"| {row.variant} | {row.background} | {row.ci} | {row.dci} | {row.ignore} | {row.ignore_pct:.2f} |")
    lines.append("")

    lines.append("## Landsat band distribution red flags")
    lines.append("Sampled common test slices, de-normalized with each dataset train stats.")
    for cls in ["valid", "ci", "dci"]:
        lines.append(f"### Class: {cls}")
        lines.append("| band | legacy p50 | agreement p50 | raw p50 | legacy p99 | agreement p99 | raw p99 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        piv = band_df[band_df["class"] == cls].set_index(["band", "variant"])
        for band in BANDS:
            def get(v: str, stat: str) -> float:
                return float(piv.loc[(band, v), stat])
            lines.append(
                f"| {band} | {get('legacy','p50'):.4g} | {get('agreement','p50'):.4g} | {get('raw','p50'):.4g} | "
                f"{get('legacy','p99'):.4g} | {get('agreement','p99'):.4g} | {get('raw','p99'):.4g} |"
            )
        lines.append("")

    (OUT_DIR / "relaxed_legacy_forensics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = collect_runs()
    runs.to_csv(OUT_DIR / "relaxed_legacy_runs_used.csv", index=False)
    slice_df = paired_slice_metrics(runs)
    slice_df.to_csv(OUT_DIR / "relaxed_legacy_common_slice_metrics.csv", index=False)
    labels_df = label_retention_common()
    labels_df.to_csv(OUT_DIR / "relaxed_legacy_common_slice_label_counts.csv", index=False)
    band_df = band_distribution_audit()
    band_df.to_csv(OUT_DIR / "relaxed_legacy_band_distribution_audit.csv", index=False)
    write_markdown(runs, slice_df, labels_df, band_df)
    print(f"runs: {len(runs)}")
    print(f"common-slice metric rows: {len(slice_df)}")
    print(f"wrote {OUT_DIR / 'relaxed_legacy_forensics.md'}")


if __name__ == "__main__":
    main()
