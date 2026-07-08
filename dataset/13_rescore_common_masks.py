#!/usr/bin/env python3
"""Re-score trained models on identical common-slice masks.

Purpose: test whether legacy advantage remains when predictions are evaluated on the
same pixels/labels.

For each run:
- Feed model its own dataset X for common slice keys.
- Compare prediction against canonical legacy labels and relaxed-valid agreement labels.
- Report metrics under three policies:
  common_valid: pixels valid in both legacy and relaxed labels, y=relaxed (same class as legacy here)
  legacy_valid: pixels valid in legacy labels, y=legacy
  relaxed_valid: pixels valid in relaxed labels, y=relaxed

No training.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from glacier_mapping.model.evaluation import IoU, precision, predict_slice, tp_fp_fn
from glacier_mapping.model.evaluation import load_lightning_module, resolve_prediction_device
from glacier_mapping.utils.gpu import cleanup_gpu_memory

DATA_ROOT = Path("/home/devj/local-arch/data/HKH")
OUT_ROOT = Path("output")
OUT_DIR = Path("dataset/outputs/common_mask_rescore")

DATASETS = {
    "legacy": DATA_ROOT / "comprehensive_v3",
    "agreement": DATA_ROOT / "comprehensive_v3_hkh_full8_agreement_quality_step3_relaxed_valid",
    "raw": DATA_ROOT / "comprehensive_v3_hkh_full8_raw_target_relaxed_valid",
    "c01dn": DATA_ROOT / "comprehensive_v3_hkh_full8_agreement_quality_step3_relaxed_valid_c01dn_matched",
    "hybrid": DATA_ROOT / "comprehensive_v3_legacy_x_agreement_relaxed_labels",
}

RUN_PATTERNS = {
    "legacy": [
        "legacy_comprehensive_v3_*_allch_bs8_seed42_desktop_20260705_*",
        "legacy_comprehensive_v3_*_allch_bs8_seed4[34]_desktop_20260706_*",
    ],
    "agreement": ["hkh_full8_agreement_quality_step3_relaxed_valid_*_allch_bs8_seed4[234]_desktop_20260706_*"],
    "raw": ["hkh_full8_raw_target_relaxed_valid_*_allch_bs8_seed4[234]_desktop_20260706_*"],
    "c01dn": ["hkh_full8_agreement_quality_step3_relaxed_valid_c01dn_matched_*_allch_bs8_seed4[234]_desktop_20260707_*"],
    "hybrid": ["legacy_x_agreement_relaxed_labels_*_allch_bs8_seed4[234]_desktop_20260707_*"],
}

POLICIES = ["common_valid", "legacy_valid", "relaxed_valid"]
TARGET_CLASS = {"ci": 1, "dci": 2}


@dataclass(frozen=True)
class RunSpec:
    variant: str
    task: str
    seed: int
    run_dir: Path
    checkpoint: Path
    dataset_dir: Path


def key_from_record(record: dict[str, Any]) -> str:
    return str(record["source_tiff_file"]).replace("tiff_", "").replace(".npy", "")


def index_map(dataset_dir: Path, split: str = "test") -> dict[str, int]:
    manifest = json.loads((dataset_dir / split / "manifest.json").read_text())
    return {key_from_record(record): int(record["index"]) for record in manifest["records"]}


def task_from_name(name: str) -> str:
    return "dci" if "_dci_" in name else "ci"


def seed_from_name(name: str) -> int:
    match = re.search(r"seed(\d+)", name)
    if match is None:
        raise ValueError(f"No seed in {name}")
    return int(match.group(1))


def best_checkpoint(run_dir: Path) -> Path:
    log_path = run_dir / "training.log"
    if log_path.exists():
        text = log_path.read_text(errors="ignore")
        matches = re.findall(r"Loaded best checkpoint for final test eval: (.*\.ckpt)", text)
        for match in reversed(matches):
            path = Path(match.strip())
            if path.exists():
                return path
    ckpts = sorted((run_dir / "checkpoints").glob("*val_loss=*.ckpt"))
    if ckpts:
        def loss(path: Path) -> float:
            m = re.search(r"val_loss=([-0-9.]+)", path.name)
            return float(m.group(1)) if m else float("inf")
        return min(ckpts, key=loss)
    last = run_dir / "checkpoints" / "last.ckpt"
    if last.exists():
        return last
    raise FileNotFoundError(f"No checkpoint found for {run_dir}")


def collect_runs(variants: list[str]) -> list[RunSpec]:
    specs: list[RunSpec] = []
    seen: set[tuple[str, str, int]] = set()
    for variant in variants:
        candidates: list[Path] = []
        for pattern in RUN_PATTERNS[variant]:
            candidates.extend(OUT_ROOT.glob(pattern))
        # latest path for duplicate variant/task/seed
        for run_dir in sorted(candidates):
            if not (run_dir / "test_evaluations" / "test_metrics.json").exists():
                continue
            task = task_from_name(run_dir.name)
            seed = seed_from_name(run_dir.name)
            key = (variant, task, seed)
            if key in seen:
                continue
            seen.add(key)
            specs.append(
                RunSpec(
                    variant=variant,
                    task=task,
                    seed=seed,
                    run_dir=run_dir,
                    checkpoint=best_checkpoint(run_dir),
                    dataset_dir=DATASETS[variant],
                )
            )
    return sorted(specs, key=lambda s: (s.task, s.variant, s.seed))


def metric_counts(y_pred: np.ndarray, y_true: np.ndarray, valid: np.ndarray, target_class: int) -> tuple[int, int, int]:
    pred_bin = (y_pred[valid] == 1).astype(np.uint8)
    true_bin = (y_true[valid] == target_class).astype(np.uint8)
    tp, fp, fn = tp_fp_fn(torch.from_numpy(pred_bin), torch.from_numpy(true_bin))
    return int(tp), int(fp), int(fn)


def score_run(spec: RunSpec, common_keys: list[str], gpu: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    device = resolve_prediction_device(gpu)
    module = load_lightning_module(spec.checkpoint, device, processed_data_path=spec.dataset_dir)
    module.eval()

    idx_model = index_map(spec.dataset_dir)
    idx_legacy = index_map(DATASETS["legacy"])
    idx_relaxed = index_map(DATASETS["agreement"])

    x_model = np.load(spec.dataset_dir / "test" / "X.npy", mmap_mode="r")
    y_legacy = np.load(DATASETS["legacy"] / "test" / "y.npy", mmap_mode="r")
    y_relaxed = np.load(DATASETS["agreement"] / "test" / "y.npy", mmap_mode="r")

    target_class = TARGET_CLASS[spec.task]
    sums = {policy: {"tp": 0, "fp": 0, "fn": 0, "pixels": 0, "positives": 0} for policy in POLICIES}
    rows: list[dict[str, Any]] = []

    for key in tqdm(common_keys, desc=f"{spec.variant}-{spec.task}-s{spec.seed}"):
        pred, invalid_mask = predict_slice(
            module,
            x_model[idx_model[key]],
            fill_holes=True,
            preprocessed_chw=True,
        )
        yl = np.asarray(y_legacy[idx_legacy[key]])
        yr = np.asarray(y_relaxed[idx_relaxed[key]])

        masks = {
            "common_valid": (yl != 255) & (yr != 255),
            "legacy_valid": yl != 255,
            "relaxed_valid": yr != 255,
        }
        truths = {
            "common_valid": yr,
            "legacy_valid": yl,
            "relaxed_valid": yr,
        }
        if invalid_mask is not None:
            for policy in POLICIES:
                masks[policy] &= ~invalid_mask

        for policy in POLICIES:
            valid = masks[policy]
            y_true = truths[policy]
            tp, fp, fn = metric_counts(pred, y_true, valid, target_class)
            sums[policy]["tp"] += tp
            sums[policy]["fp"] += fp
            sums[policy]["fn"] += fn
            sums[policy]["pixels"] += int(valid.sum())
            sums[policy]["positives"] += int((y_true[valid] == target_class).sum())
            rows.append(
                {
                    "variant": spec.variant,
                    "task": spec.task,
                    "seed": spec.seed,
                    "slice_key": key,
                    "policy": policy,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision(tp, fp, fn),
                    "recall": precision(tp, fn, fp),  # placeholder overwritten below
                    "iou": IoU(tp, fp, fn),
                    "valid_pixels": int(valid.sum()),
                    "positive_pixels": int((y_true[valid] == target_class).sum()),
                }
            )
            rows[-1]["recall"] = tp / (tp + fn + 1e-10)

    metrics: list[dict[str, Any]] = []
    for policy, vals in sums.items():
        tp, fp, fn = vals["tp"], vals["fp"], vals["fn"]
        metrics.append(
            {
                "variant": spec.variant,
                "task": spec.task,
                "seed": spec.seed,
                "policy": policy,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision(tp, fp, fn),
                "recall": tp / (tp + fn + 1e-10),
                "iou": IoU(tp, fp, fn),
                "valid_pixels": vals["pixels"],
                "positive_pixels": vals["positives"],
                "run_dir": str(spec.run_dir),
                "checkpoint": str(spec.checkpoint),
            }
        )
    cleanup_gpu_memory()
    del module
    return metrics, rows


def write_markdown(metrics: pd.DataFrame, summary: pd.DataFrame, paired: pd.DataFrame) -> None:
    lines: list[str] = ["# Common-mask prediction re-score", ""]
    lines.append("Policies:")
    lines.append("- `common_valid`: pixels valid in both legacy and relaxed labels; classes agree there.")
    lines.append("- `legacy_valid`: original legacy evaluation mask and labels.")
    lines.append("- `relaxed_valid`: relaxed-valid evaluation mask and labels.")
    lines.append("")
    for task in ["dci", "ci"]:
        for policy in POLICIES:
            lines.append(f"## {task.upper()} {policy}")
            lines.append("| variant | n | mean IoU | std | precision | recall |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            sub = summary[(summary.task == task) & (summary.policy == policy)].sort_values("mean_iou", ascending=False)
            for row in sub.itertuples(index=False):
                lines.append(
                    f"| {row.variant} | {int(row.n)} | {row.mean_iou:.6f} | {row.std_iou:.6f} | {row.mean_precision:.6f} | {row.mean_recall:.6f} |"
                )
            lines.append("")
    lines.append("## Paired deltas vs legacy")
    lines.append("| task | policy | variant | n | mean delta | std | min | max |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for row in paired.itertuples(index=False):
        lines.append(
            f"| {row.task} | {row.policy} | {row.variant} | {int(row.n)} | {row.mean_delta:+.6f} | {row.std_delta:.6f} | {row.min_delta:+.6f} | {row.max_delta:+.6f} |"
        )
    (OUT_DIR / "common_mask_rescore_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--variants", default="legacy,agreement,raw,c01dn,hybrid")
    parser.add_argument("--tasks", default="dci,ci")
    parser.add_argument("--seeds", default="42,43,44")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    tasks = {v.strip() for v in args.tasks.split(",") if v.strip()}
    seeds = {int(v.strip()) for v in args.seeds.split(",") if v.strip()}

    # Always require canonical label sources even if not included as model variants.
    required_sources = sorted(set(variants) | {"legacy", "agreement"})
    idx_maps = {variant: index_map(DATASETS[variant]) for variant in required_sources}
    common_keys = sorted(set.intersection(*(set(m) for m in idx_maps.values())))
    (OUT_DIR / "common_keys.json").write_text(json.dumps(common_keys, indent=2), encoding="utf-8")
    print(f"common_keys={len(common_keys)}")

    specs = [s for s in collect_runs(variants) if s.task in tasks and s.seed in seeds]
    pd.DataFrame([s.__dict__ for s in specs]).to_csv(OUT_DIR / "runs_used.csv", index=False)
    print(f"runs={len(specs)}")

    all_metrics: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for spec in specs:
        metrics, rows = score_run(spec, common_keys, args.gpu)
        all_metrics.extend(metrics)
        all_rows.extend(rows)
        pd.DataFrame(all_metrics).to_csv(OUT_DIR / "aggregate_metrics_partial.csv", index=False)

    metrics_df = pd.DataFrame(all_metrics)
    rows_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(OUT_DIR / "aggregate_metrics.csv", index=False)
    rows_df.to_csv(OUT_DIR / "per_slice_metrics.csv", index=False)

    summary = metrics_df.groupby(["task", "policy", "variant"]).agg(
        n=("iou", "count"),
        mean_iou=("iou", "mean"),
        std_iou=("iou", "std"),
        min_iou=("iou", "min"),
        max_iou=("iou", "max"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
    ).reset_index()
    summary.to_csv(OUT_DIR / "summary_by_variant.csv", index=False)

    deltas: list[dict[str, Any]] = []
    for (task, policy, seed), group in metrics_df.groupby(["task", "policy", "seed"]):
        g = group.set_index("variant")
        if "legacy" not in g.index:
            continue
        for variant in sorted(set(g.index) - {"legacy"}):
            deltas.append(
                {
                    "task": task,
                    "policy": policy,
                    "seed": seed,
                    "variant": variant,
                    "delta": float(g.loc[variant, "iou"] - g.loc["legacy", "iou"]),
                }
            )
    deltas_df = pd.DataFrame(deltas)
    deltas_df.to_csv(OUT_DIR / "paired_deltas_vs_legacy.csv", index=False)
    if deltas_df.empty:
        paired = pd.DataFrame(
            columns=["task", "policy", "variant", "n", "mean_delta", "std_delta", "min_delta", "max_delta"]
        )
    else:
        paired = deltas_df.groupby(["task", "policy", "variant"]).agg(
            n=("delta", "count"),
            mean_delta=("delta", "mean"),
            std_delta=("delta", "std"),
            min_delta=("delta", "min"),
            max_delta=("delta", "max"),
        ).reset_index()
    paired.to_csv(OUT_DIR / "paired_delta_summary_vs_legacy.csv", index=False)
    write_markdown(metrics_df, summary, paired)
    print(summary.sort_values(["task", "policy", "mean_iou"], ascending=[True, True, False]).to_string(index=False))
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
