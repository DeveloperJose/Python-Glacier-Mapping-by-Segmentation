#!/usr/bin/env python3
"""Create hybrid dataset: canonical legacy C01/uint8 X + relaxed-valid labels.

Purpose: evidence-based isolation after C01 rebuild was blocked by EE deprecation.
- X/image domain: existing canonical legacy `/comprehensive_v3` packed arrays.
- y/labels: relaxed-valid agreement labels.
- Samples: intersection of source slice keys per split.

This tests whether exact legacy image domain plus corrected relaxed-valid mask/labels
closes the remaining gap better than the C02-derived C01DN approximation.
No training is run here.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

DATA_ROOT = Path("/home/devj/local-arch/data/HKH")
LEGACY_DIR = DATA_ROOT / "comprehensive_v3"
RELAXED_DIR = DATA_ROOT / "comprehensive_v3_hkh_full8_agreement_quality_step3_relaxed_valid"
TARGET_DIR = DATA_ROOT / "comprehensive_v3_legacy_x_agreement_relaxed_labels"
OUT_DIR = Path("dataset/outputs")
IGNORE_LABEL = 255


def load_records(dataset_dir: Path, split: str) -> dict[str, dict[str, Any]]:
    manifest = json.loads((dataset_dir / split / "manifest.json").read_text())
    records: dict[str, dict[str, Any]] = {}
    for record in manifest["records"]:
        key = str(record["source_tiff_file"]).replace("tiff_", "").replace(".npy", "")
        records[key] = record
    return records


def build_split(split: str) -> dict[str, Any]:
    legacy_records = load_records(LEGACY_DIR, split)
    relaxed_records = load_records(RELAXED_DIR, split)
    common_keys = sorted(set(legacy_records) & set(relaxed_records))
    if not common_keys:
        raise ValueError(f"No common records for split {split}")

    legacy_x = np.load(LEGACY_DIR / split / "X.npy", mmap_mode="r")
    relaxed_y = np.load(RELAXED_DIR / split / "y.npy", mmap_mode="r")

    target_split = TARGET_DIR / split
    target_split.mkdir(parents=True, exist_ok=True)
    out_x = np.lib.format.open_memmap(
        target_split / "X.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(common_keys), legacy_x.shape[1], legacy_x.shape[2], legacy_x.shape[3]),
    )
    out_y = np.lib.format.open_memmap(
        target_split / "y.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(len(common_keys), relaxed_y.shape[1], relaxed_y.shape[2]),
    )

    packed_records: list[dict[str, Any]] = []
    for out_idx, key in enumerate(tqdm(common_keys, desc=f"Build {split}")):
        legacy_idx = int(legacy_records[key]["index"])
        relaxed_idx = int(relaxed_records[key]["index"])
        out_x[out_idx] = legacy_x[legacy_idx]
        out_y[out_idx] = relaxed_y[relaxed_idx]
        packed_records.append(
            {
                "index": out_idx,
                "source_tiff_file": f"tiff_{key}.npy",
                "source_mask_file": f"mask_{key}.npy",
                "legacy_x_index": legacy_idx,
                "relaxed_y_index": relaxed_idx,
            }
        )
    out_x.flush()
    out_y.flush()

    manifest = {
        "format": "comprehensive_v3_hybrid",
        "layout": "NCHW",
        "normalized": True,
        "normalization": "mean-std",
        "x": "X.npy",
        "y": "y.npy",
        "num_samples": len(common_keys),
        "shape": [len(common_keys), legacy_x.shape[1], legacy_x.shape[2], legacy_x.shape[3]],
        "label_shape": [len(common_keys), relaxed_y.shape[1], relaxed_y.shape[2]],
        "dtype": {"x": "float32", "y": "uint8"},
        "records": packed_records,
        "x_source": str(LEGACY_DIR / split / "X.npy"),
        "y_source": str(RELAXED_DIR / split / "y.npy"),
        "join_key": "source_tiff_file without tiff_/npy prefix",
    }
    (target_split / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_stats() -> None:
    stats: dict[str, Any] = {}
    for split in ["train", "val", "test"]:
        y = np.load(TARGET_DIR / split / "y.npy", mmap_mode="r")
        total = int(y.size)
        counts = {
            "background": int((y == 0).sum()),
            "clean_ice": int((y == 1).sum()),
            "debris_ice": int((y == 2).sum()),
            "masked_invalid": int((y == IGNORE_LABEL).sum()),
        }
        valid = total - counts["masked_invalid"]
        stats[split] = {
            "slices": int(y.shape[0]),
            "total_pixels": total,
            "pixels": counts,
            "percentages_all_pixels": {k: v / total * 100 for k, v in counts.items()},
            "percentages_valid_pixels": {
                "background": counts["background"] / valid * 100 if valid else 0.0,
                "clean_ice": counts["clean_ice"] / valid * 100 if valid else 0.0,
                "debris_ice": counts["debris_ice"] / valid * 100 if valid else 0.0,
            },
        }
    (TARGET_DIR / "dataset_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def audit_dataset(manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    audit: dict[str, Any] = {"target_dir": str(TARGET_DIR), "splits": {}}
    for split, manifest in manifests.items():
        x = np.load(TARGET_DIR / split / "X.npy", mmap_mode="r")
        y = np.load(TARGET_DIR / split / "y.npy", mmap_mode="r")
        # Spot-check first/middle/last against sources.
        checks = []
        for out_idx in sorted({0, len(manifest["records"]) // 2, len(manifest["records"]) - 1}):
            record = manifest["records"][out_idx]
            legacy_x = np.load(LEGACY_DIR / split / "X.npy", mmap_mode="r")
            relaxed_y = np.load(RELAXED_DIR / split / "y.npy", mmap_mode="r")
            checks.append(
                {
                    "index": out_idx,
                    "x_equal": bool(np.array_equal(x[out_idx], legacy_x[record["legacy_x_index"]])),
                    "y_equal": bool(np.array_equal(y[out_idx], relaxed_y[record["relaxed_y_index"]])),
                }
            )
        audit["splits"][split] = {
            "x_shape": list(x.shape),
            "y_shape": list(y.shape),
            "records": len(manifest["records"]),
            "checks": checks,
        }
    return audit


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if TARGET_DIR.exists():
        if not args.force:
            raise FileExistsError(f"Target exists: {TARGET_DIR}. Use --force.")
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifests = {split: build_split(split) for split in ["train", "val", "test"]}
    shutil.copy2(LEGACY_DIR / "band_metadata.json", TARGET_DIR / "band_metadata.json")
    shutil.copy2(LEGACY_DIR / "normalize_train.npy", TARGET_DIR / "normalize_train.npy")
    for split in ["train", "val", "test"]:
        shutil.copy2(LEGACY_DIR / "normalize_train.npy", TARGET_DIR / f"normalize_{split}.npy")
    write_stats()
    audit = audit_dataset(manifests)
    (OUT_DIR / "legacy_x_relaxed_labels_hybrid_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(f"Created {TARGET_DIR}")
    print(json.dumps(audit["splits"], indent=2))


if __name__ == "__main__":
    main()
