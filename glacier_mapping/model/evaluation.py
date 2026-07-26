from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm

from glacier_mapping.lightning.glacier_module import GlacierSegmentationModule
from glacier_mapping.utils.gpu import cleanup_gpu_memory


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def precision(tp, fp, fn):
    return tp / (tp + fp + 1e-10)


def tp_fp_fn(pred, true, label=1):
    tp = ((pred == label) & (true == label)).sum().item()
    fp = ((pred == label) & (true != label)).sum().item()
    fn = ((pred != label) & (true == label)).sum().item()
    return tp, fp, fn


def recall(tp, fp, fn):
    return tp / (tp + fn + 1e-10)


def dice(tp, fp, fn):
    return (2 * tp) / (2 * tp + fp + fn + 1e-10)


def IoU(tp, fp, fn):
    return tp / (tp + fp + fn + 1e-10)


CLASS_TO_INDEX = {"bg": 0, "ci": 1, "dci": 2}


@dataclass
class FullSplitEvaluationResult:
    metrics: dict[str, float]
    rows: list[list[Any]]
    columns: list[str]
    output_dir: Path | None = None
    status: str = "SUCCESS"


def metric_name(scope: str, split: str, target: str, metric: str) -> str:
    """Build canonical metric names such as full_val_dci_iou."""
    return f"{scope}_{split}_{target}_{metric}"


def full_metric_name(split: str, target: str, metric: str) -> str:
    return metric_name("full", split, target, metric)


def get_probabilities(module, x_full, *, preprocessed_chw: bool = False):
    """Get a probability cube from a Lightning module using softmax."""
    if preprocessed_chw:
        x_chw = np.array(x_full, dtype=np.float32, copy=True)
        inp = torch.from_numpy(np.expand_dims(x_chw, 0)).to(module.device)
    else:
        use_ch = module.use_channels
        x = x_full[:, :, use_ch]
        x_norm = module.normalize(x)
        inp = (
            torch.from_numpy(np.expand_dims(x_norm, 0))
            .permute(0, 3, 1, 2)
            .float()
            .to(module.device)
        )
    logits = module.forward(inp)
    probs = (
        torch.nn.functional.softmax(logits, dim=1)[0]
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
    )
    return probs


def resolve_prediction_threshold(module, threshold=None) -> float | list[float] | None:
    """Resolve explicit or config-driven prediction thresholds."""
    if threshold is not None:
        return threshold

    config_threshold = getattr(module, "metrics_opts", {}).get("threshold", [0.5])
    if len(module.output_classes) == 1:
        return (
            config_threshold[0]
            if isinstance(config_threshold, list)
            else config_threshold
        )
    return config_threshold


def predict_from_probs(probs, module, threshold=None, *, fill_holes=True):
    """Convert probabilities to hard predictions using module configuration."""
    threshold = resolve_prediction_threshold(module, threshold)
    if len(module.output_classes) == 1:
        y_pred = (probs[:, :, 1] >= threshold).astype(np.uint8)
        if fill_holes:
            y_pred = binary_fill_holes(y_pred).astype(np.uint8)
        return y_pred

    return np.argmax(probs, axis=2).astype(np.uint8)


def predict_whole(module, whole_arr, window_size, threshold=None):
    """Predict on a whole image using sliding windows, stitching results."""
    y_pred = np.zeros((whole_arr.shape[0], whole_arr.shape[1]), dtype=np.uint8)

    for row in range(0, whole_arr.shape[0], window_size[0]):
        for column in range(0, whole_arr.shape[1], window_size[1]):
            current_slice = whole_arr[
                row : row + window_size[0], column : column + window_size[1], :
            ]

            if current_slice.shape[:2] != window_size:
                temp = np.zeros(
                    (window_size[0], window_size[1], whole_arr.shape[2]),
                    dtype=whole_arr.dtype,
                )
                temp[: current_slice.shape[0], : current_slice.shape[1], :] = (
                    current_slice
                )
                current_slice = temp

            pred, _ = predict_slice(module, current_slice, threshold)

            endrow_dest = min(row + window_size[0], y_pred.shape[0])
            endrow_source = min(window_size[0], y_pred.shape[0] - row)
            endcolumn_dest = min(column + window_size[1], y_pred.shape[1])
            endcolumn_source = min(window_size[1], y_pred.shape[1] - column)

            y_pred[row:endrow_dest, column:endcolumn_dest] = pred[
                0:endrow_source, 0:endcolumn_source
            ]

    return y_pred


def get_pr_iou(pred, true):
    """Calculate precision, recall, IoU, and confusion components."""
    pred_t = torch.from_numpy(pred.astype(np.uint8))
    true_t = torch.from_numpy(true.astype(np.uint8))
    tp, fp, fn = tp_fp_fn(pred_t, true_t)
    return (precision(tp, fp, fn), recall(tp, fp, fn), IoU(tp, fp, fn), tp, fp, fn)


def calculate_binary_metrics(y_pred, y_true, target_class, mask=None):
    """Calculate binary precision/recall/IoU for a target class."""
    if mask is not None:
        valid = ~mask
        y_pred_valid = y_pred[valid]
        y_true_valid = y_true[valid]
    else:
        y_pred_valid = y_pred
        y_true_valid = y_true

    t_bin = (y_true_valid == target_class).astype(np.uint8)
    p_bin = (y_pred_valid == 1).astype(np.uint8)

    tp_, fp_, fn_ = tp_fp_fn(torch.from_numpy(p_bin), torch.from_numpy(t_bin))
    return (
        precision(tp_, fp_, fn_),
        recall(tp_, fp_, fn_),
        IoU(tp_, fp_, fn_),
        tp_,
        fp_,
        fn_,
    )


def create_invalid_mask(x_full, y_true):
    """Create invalid mask for legacy HWC imagery evaluation.

    Ignore-label pixels are always invalid. Empty-imagery detection only applies to
    legacy HWC inputs where the last axis is the channel dimension.
    """
    invalid = y_true == 255
    if x_full.ndim != 3:
        return invalid
    if x_full.shape[:2] != y_true.shape:
        return invalid
    return (np.all(x_full == 0, axis=2)) | invalid


def merge_ci_debris(
    prob_ci: np.ndarray, prob_dci: np.ndarray, thr_ci: float, thr_dci: float
) -> tuple[np.ndarray, np.ndarray]:
    """Combine two binary CI/DCI models into one 3-class map."""
    if prob_ci is None or prob_dci is None:
        raise ValueError("Both prob_ci and prob_dci must be provided")

    ci_mask = binary_fill_holes(prob_ci[:, :, 1] >= thr_ci)
    dci_mask = binary_fill_holes(prob_dci[:, :, 1] >= thr_dci)

    if ci_mask is None:
        ci_mask = prob_ci[:, :, 1] >= thr_ci
    if dci_mask is None:
        dci_mask = prob_dci[:, :, 1] >= thr_dci

    height, width = ci_mask.shape
    merged = np.zeros((height, width), dtype=np.uint8)
    merged[ci_mask] = CLASS_TO_INDEX["ci"]
    merged[dci_mask] = CLASS_TO_INDEX["dci"]

    probs = np.zeros((height, width, 3), dtype=np.float32)
    probs[:, :, CLASS_TO_INDEX["ci"]] = prob_ci[:, :, 1]
    probs[:, :, CLASS_TO_INDEX["dci"]] = prob_dci[:, :, 1]
    probs[:, :, CLASS_TO_INDEX["bg"]] = np.minimum(prob_ci[:, :, 0], prob_dci[:, :, 0])

    return merged, probs


def resolve_prediction_device(gpu_id: int | None = 0) -> torch.device:
    if gpu_id is None or gpu_id < 0:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU")
        return torch.device("cpu")
    return torch.device(f"cuda:{gpu_id}")


def load_lightning_module(
    checkpoint_path: str | Path,
    device: torch.device,
    processed_data_path: str | Path | None = None,
) -> GlacierSegmentationModule:
    checkpoint_path = Path(checkpoint_path)
    if processed_data_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "loader_opts" in checkpoint["hyper_parameters"]:
            old_processed_dir = checkpoint["hyper_parameters"]["loader_opts"][
                "processed_dir"
            ]
            checkpoint["hyper_parameters"]["loader_opts"]["processed_dir"] = str(
                processed_data_path
            )
            print(
                f"Overriding processed_dir: {old_processed_dir} -> {processed_data_path}"
            )

        temp_ckpt = checkpoint_path.parent / "temp_modified.ckpt"
        torch.save(checkpoint, temp_ckpt)

        try:
            module = GlacierSegmentationModule.load_from_checkpoint(temp_ckpt)
        finally:
            if temp_ckpt.exists():
                temp_ckpt.unlink()
    else:
        module = GlacierSegmentationModule.load_from_checkpoint(checkpoint_path)

    module.to(device)
    if getattr(module, "aryal_2023_behavior", False):
        for child in module.model.modules():
            if isinstance(child, (torch.nn.Dropout, torch.nn.Dropout2d)):
                child.p = 1e-8
        module.train()
    else:
        module.eval()
    return module


def get_split_tiles(module: GlacierSegmentationModule, split: str) -> list[Path]:
    data_dir = Path(module.processed_dir) / split
    return sorted(data_dir.glob("tiff*"))


def _iter_split_samples(
    module: GlacierSegmentationModule, split: str
) -> tuple[Iterator[tuple[str, np.ndarray, np.ndarray, bool]], int]:
    """Yield legacy HWC or packed normalized CHW split samples."""
    data_dir = Path(module.processed_dir) / split
    tiles = sorted(data_dir.glob("tiff*"))
    if tiles:

        def iter_tiles() -> Iterator[tuple[str, np.ndarray, np.ndarray, bool]]:
            for tile in tiles:
                y_path = tile.with_name(tile.name.replace("tiff", "mask"))
                yield (
                    tile.name,
                    np.load(tile, mmap_mode="r"),
                    np.load(y_path, mmap_mode="r").astype(np.uint8),
                    False,
                )

        return iter_tiles(), len(tiles)

    x_path = data_dir / "X.npy"
    y_path = data_dir / "y.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"No legacy tiles or packed X.npy/y.npy found under {data_dir}"
        )

    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    if x.ndim != 4 or y.ndim != 3 or x.shape[0] != y.shape[0]:
        raise ValueError(f"Invalid packed split shapes: X={x.shape}, y={y.shape}")

    def iter_packed() -> Iterator[tuple[str, np.ndarray, np.ndarray, bool]]:
        for index in range(len(x)):
            yield f"sample_{index:06d}", x[index], y[index], True

    return iter_packed(), len(x)


def evaluate_full_split_modules(
    *,
    frame_ci: GlacierSegmentationModule | None = None,
    frame_dci: GlacierSegmentationModule | None = None,
    split: str = "test",
    ci_threshold: float = 0.5,
    dci_threshold: float = 0.5,
    output_dir: str | Path | None = None,
    save_probabilities: bool = True,
    progress: bool = True,
) -> FullSplitEvaluationResult:
    """Evaluate one CI/DCI model or a paired CI+DCI model on a full split."""
    has_ci = frame_ci is not None
    has_dci = frame_dci is not None
    if not (has_ci or has_dci):
        raise ValueError("Must provide at least one model")

    module = frame_ci if frame_ci is not None else frame_dci
    if module is None:
        raise RuntimeError("No valid model loaded for full split evaluation")

    samples, sample_count = _iter_split_samples(module, split)

    out_path = Path(output_dir) if output_dir is not None else None
    preds_dir = None
    if out_path is not None:
        preds_dir = out_path / "preds"
        preds_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running full {split} evaluation on {sample_count} samples...")

    rows, acc = _run_full_split_prediction(
        frame_ci=frame_ci,
        frame_dci=frame_dci,
        ci_threshold=ci_threshold,
        dci_threshold=dci_threshold,
        samples=samples,
        sample_count=sample_count,
        preds_dir=preds_dir,
        save_probabilities=save_probabilities,
        progress=progress,
    )

    metrics, total_row = _aggregate_metrics(
        acc=acc,
        split=split,
        has_ci=has_ci,
        has_dci=has_dci,
    )
    rows.append(total_row)

    rows_with_checkpoint = [["prediction"] + row for row in rows]
    columns = _csv_columns(has_ci=has_ci, has_dci=has_dci)

    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_with_checkpoint, columns=columns).to_csv(
            out_path / "metrics.csv", index=False
        )
        import json

        with open(out_path / "test_metrics.json", "w") as file:
            json.dump(metrics, file, indent=2)

    return FullSplitEvaluationResult(
        metrics=metrics,
        rows=rows_with_checkpoint,
        columns=columns,
        output_dir=out_path,
    )


def evaluate_single_checkpoint(
    checkpoint_path: str | Path,
    *,
    split: str = "val",
    target: str = "dci",
    threshold: float = 0.5,
    processed_data_path: str | Path | None = None,
    gpu_id: int | None = 0,
    output_dir: str | Path | None = None,
    save_probabilities: bool = True,
) -> FullSplitEvaluationResult:
    target = _normalize_target(target)
    device = resolve_prediction_device(gpu_id)
    module = load_lightning_module(checkpoint_path, device, processed_data_path)
    try:
        return evaluate_full_split_modules(
            frame_ci=module if target == "ci" else None,
            frame_dci=module if target == "dci" else None,
            split=split,
            ci_threshold=threshold,
            dci_threshold=threshold,
            output_dir=output_dir,
            save_probabilities=save_probabilities,
        )
    finally:
        del module
        cleanup_gpu_memory(synchronize=False)


def evaluate_paired_checkpoints(
    ci_checkpoint_path: str | Path,
    dci_checkpoint_path: str | Path,
    *,
    split: str = "val",
    ci_threshold: float = 0.5,
    dci_threshold: float = 0.5,
    processed_data_path: str | Path | None = None,
    gpu_id: int | None = 0,
    output_dir: str | Path | None = None,
    save_probabilities: bool = True,
) -> FullSplitEvaluationResult:
    device = resolve_prediction_device(gpu_id)
    frame_ci = load_lightning_module(ci_checkpoint_path, device, processed_data_path)
    frame_dci = load_lightning_module(dci_checkpoint_path, device, processed_data_path)
    try:
        return evaluate_full_split_modules(
            frame_ci=frame_ci,
            frame_dci=frame_dci,
            split=split,
            ci_threshold=ci_threshold,
            dci_threshold=dci_threshold,
            output_dir=output_dir,
            save_probabilities=save_probabilities,
        )
    finally:
        del frame_ci, frame_dci
        cleanup_gpu_memory(synchronize=False)


def _run_full_split_prediction(
    *,
    frame_ci: GlacierSegmentationModule | None,
    frame_dci: GlacierSegmentationModule | None,
    ci_threshold: float,
    dci_threshold: float,
    samples: Iterator[tuple[str, np.ndarray, np.ndarray, bool]],
    sample_count: int,
    preds_dir: Path | None,
    save_probabilities: bool,
    progress: bool,
) -> tuple[list[list[Any]], dict[str, float]]:
    rows: list[list[Any]] = []
    acc = {
        "ci_tp": 0.0,
        "ci_fp": 0.0,
        "ci_fn": 0.0,
        "dci_tp": 0.0,
        "dci_fp": 0.0,
        "dci_fn": 0.0,
    }

    has_ci = frame_ci is not None
    has_dci = frame_dci is not None
    iterator = (
        tqdm(samples, total=sample_count, desc="Predicting") if progress else samples
    )

    with torch.no_grad():
        for name, x_full, y_full, preprocessed_chw in iterator:
            base = Path(name).stem
            invalid = (
                y_full == 255
                if preprocessed_chw
                else create_invalid_mask(x_full, y_full)
            )
            valid = ~invalid

            if has_ci ^ has_dci:
                frame = frame_ci if frame_ci is not None else frame_dci
                if frame is None:
                    raise RuntimeError("Missing single-model frame")
                target = "ci" if has_ci else "dci"
                model_class = CLASS_TO_INDEX[target]
                threshold = ci_threshold if has_ci else dci_threshold

                y_pred, mask = predict_slice(
                    frame,
                    x_full,
                    threshold,
                    fill_holes=True,
                    preprocessed_chw=preprocessed_chw,
                )
                if save_probabilities and preds_dir is not None:
                    probs = get_probabilities(
                        frame, x_full, preprocessed_chw=preprocessed_chw
                    )
                    np.save(preds_dir / f"{base}_probs.npy", probs)

                p_, r_, iou_, tp_, fp_, fn_ = calculate_binary_metrics(
                    y_pred, y_full, model_class, invalid
                )

                acc[f"{target}_tp"] += tp_
                acc[f"{target}_fp"] += fp_
                acc[f"{target}_fn"] += fn_
                rows.append([name, p_, r_, iou_])

            else:
                if frame_ci is None or frame_dci is None:
                    raise RuntimeError("Both CI and DCI frames are required")

                prob_ci = get_probabilities(
                    frame_ci, x_full, preprocessed_chw=preprocessed_chw
                )
                prob_dci = get_probabilities(
                    frame_dci, x_full, preprocessed_chw=preprocessed_chw
                )
                merged, probs = merge_ci_debris(
                    prob_ci, prob_dci, ci_threshold, dci_threshold
                )
                if save_probabilities and preds_dir is not None:
                    np.save(preds_dir / f"{base}_probs.npy", probs)

                p_ci, r_ci, i_ci, tp_, fp_, fn_ = get_pr_iou(
                    (merged[valid] == CLASS_TO_INDEX["ci"]).astype(np.uint8),
                    (y_full[valid] == CLASS_TO_INDEX["ci"]).astype(np.uint8),
                )
                acc["ci_tp"] += tp_
                acc["ci_fp"] += fp_
                acc["ci_fn"] += fn_

                p_dci, r_dci, i_dci, tp_, fp_, fn_ = get_pr_iou(
                    (merged[valid] == CLASS_TO_INDEX["dci"]).astype(np.uint8),
                    (y_full[valid] == CLASS_TO_INDEX["dci"]).astype(np.uint8),
                )
                acc["dci_tp"] += tp_
                acc["dci_fp"] += fp_
                acc["dci_fn"] += fn_

                rows.append([name, p_ci, r_ci, i_ci, p_dci, r_dci, i_dci])

    return rows, acc


def _aggregate_metrics(
    *,
    acc: dict[str, float],
    split: str,
    has_ci: bool,
    has_dci: bool,
) -> tuple[dict[str, float], list[Any]]:
    metrics: dict[str, float] = {}

    if has_ci:
        p_ci = precision(acc["ci_tp"], acc["ci_fp"], acc["ci_fn"])
        r_ci = recall(acc["ci_tp"], acc["ci_fp"], acc["ci_fn"])
        i_ci = IoU(acc["ci_tp"], acc["ci_fp"], acc["ci_fn"])
        metrics.update(
            {
                full_metric_name(split, "ci", "precision"): float(p_ci),
                full_metric_name(split, "ci", "recall"): float(r_ci),
                full_metric_name(split, "ci", "iou"): float(i_ci),
            }
        )

    if has_dci:
        p_dci = precision(acc["dci_tp"], acc["dci_fp"], acc["dci_fn"])
        r_dci = recall(acc["dci_tp"], acc["dci_fp"], acc["dci_fn"])
        i_dci = IoU(acc["dci_tp"], acc["dci_fp"], acc["dci_fn"])
        metrics.update(
            {
                full_metric_name(split, "dci", "precision"): float(p_dci),
                full_metric_name(split, "dci", "recall"): float(r_dci),
                full_metric_name(split, "dci", "iou"): float(i_dci),
            }
        )

    if has_ci and has_dci:
        total_row = [
            "TOTAL",
            metrics[full_metric_name(split, "ci", "precision")],
            metrics[full_metric_name(split, "ci", "recall")],
            metrics[full_metric_name(split, "ci", "iou")],
            metrics[full_metric_name(split, "dci", "precision")],
            metrics[full_metric_name(split, "dci", "recall")],
            metrics[full_metric_name(split, "dci", "iou")],
        ]
    elif has_ci:
        total_row = [
            "TOTAL",
            metrics[full_metric_name(split, "ci", "precision")],
            metrics[full_metric_name(split, "ci", "recall")],
            metrics[full_metric_name(split, "ci", "iou")],
        ]
    else:
        total_row = [
            "TOTAL",
            metrics[full_metric_name(split, "dci", "precision")],
            metrics[full_metric_name(split, "dci", "recall")],
            metrics[full_metric_name(split, "dci", "iou")],
        ]

    return metrics, total_row


def _csv_columns(*, has_ci: bool, has_dci: bool) -> list[str]:
    if has_ci and has_dci:
        return [
            "checkpoint",
            "tile",
            "ci_precision",
            "ci_recall",
            "ci_iou",
            "dci_precision",
            "dci_recall",
            "dci_iou",
        ]

    target = "ci" if has_ci else "dci"
    return [
        "checkpoint",
        "tile",
        f"{target}_precision",
        f"{target}_recall",
        f"{target}_iou",
    ]


def predict_slice(
    module,
    slice_arr: np.ndarray,
    threshold: float | None = None,
    *,
    use_mask: bool = True,
    fill_holes: bool = True,
    preprocessed_chw: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Unified slice prediction with optional binary_fill_holes.

    All evaluation code should use this instead of module.predict_slice() to
    ensure consistent post-processing (fill_holes, masking).
    """
    if preprocessed_chw:
        _mask = None
        chw = np.array(slice_arr, dtype=np.float32, copy=True)
        _x = torch.from_numpy(np.expand_dims(chw, axis=0)).to(module.device)
    else:
        slice_arr = slice_arr[:, :, module.use_channels]
        slice_arr = module.normalize(slice_arr)
        _mask = np.sum(slice_arr, axis=2) == 0 if use_mask else None
        _x = (
            torch.from_numpy(np.expand_dims(slice_arr, axis=0))
            .permute(0, 3, 1, 2)
            .float()
            .to(module.device)
        )
    with torch.no_grad():
        _y = module.forward(_x)

    if len(module.output_classes) == 1:
        threshold = resolve_prediction_threshold(module, threshold)
        if isinstance(threshold, (int, float)):
            threshold = [threshold]

        _y = torch.nn.functional.softmax(_y, dim=1)
        _y = _y.cpu().numpy()
        if _y.shape[0] == 1:
            _y = _y[0]

        y_pred = (_y[1] >= threshold[0]).astype(np.uint8)
        if fill_holes:
            y_pred = binary_fill_holes(y_pred).astype(np.uint8)
    else:
        _y = torch.nn.functional.softmax(_y, dim=1)
        _y = _y.cpu().numpy()
        if _y.shape[0] == 1:
            _y = _y[0]
        y_pred = np.argmax(_y, axis=0).astype(np.uint8)

    if use_mask and _mask is not None:
        y_pred[_mask] = 0

    return y_pred, _mask


def _normalize_target(target: str) -> str:
    normalized = target.lower().replace("debris", "dci").replace("cleanice", "ci")
    if normalized not in {"ci", "dci"}:
        raise ValueError(f"target must be 'ci' or 'dci', got {target!r}")
    return normalized
