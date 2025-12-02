#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified predictor for:
 - Single CleanIce model
 - Single Debris model
 - Merged CleanIce + Debris binary models → 3-class output

Outputs for each test tile:
 - *_probs.npy (probability cube)
 - *_viz.png   (8-panel visualization)

Also produces:
 - metrics.csv (per-tile metrics)
 - summary printout
"""

import argparse
import pathlib
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from addict import Dict
from tqdm import tqdm

import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib

from glacier_mapping.lightning.glacier_module import GlacierSegmentationModule
from glacier_mapping.model.metrics import tp_fp_fn, precision, recall, IoU
from glacier_mapping.model.visualize import (
    build_cmap,
    make_rgb_preview,
    label_to_color,
    make_confidence_map,
    make_entropy_map,
    make_tp_fp_fn_masks,
    make_eight_panel,
)
from glacier_mapping.data.data import BAND_NAMES
from glacier_mapping.utils.prediction import (
    get_probabilities, merge_ci_debris, create_invalid_mask
)

def load_lightning_module(checkpoint_path, device):
    """Load Lightning module from checkpoint."""
    module = GlacierSegmentationModule.load_from_checkpoint(checkpoint_path)
    module.eval()
    module.to(device)
    return module

# Use non-interactive backend for plotting
matplotlib.use("Agg")


# Helpers
# Removed duplicate functions - now using shared utilities from glacier_mapping.utils.prediction


def get_pr_iou(pred, true):
    pred_t = torch.from_numpy(pred.astype(np.uint8))
    true_t = torch.from_numpy(true.astype(np.uint8))
    tp, fp, fn = tp_fp_fn(pred_t, true_t)
    return (precision(tp, fp, fn), recall(tp, fp, fn), IoU(tp, fp, fn), tp, fp, fn)


# ========================================================================
# FEATURE IMPORTANCE (GRADIENT-BASED SALIENCY)
# ========================================================================


def compute_feature_importance(
    module,
    test_tiles,
    target_class_idx,
    num_samples=None,
    save_spatial=False,
    max_spatial=10,
    output_dir=None,
):
    """
    Compute gradient-based feature importance for input channels.

    Uses backpropagation to measure how much each input channel contributes
    to the prediction of a target class. Higher gradient magnitude indicates
    higher importance.

    Args:
        module: Lightning module with loaded model
        test_tiles: List of test tile paths
        target_class_idx: Class index to compute gradients for (0-indexed in model output)
        num_samples: Number of samples to use (None = all)
        save_spatial: Whether to save spatial saliency heatmaps
        max_spatial: Max number of spatial maps to save
        output_dir: Directory to save spatial maps (if save_spatial=True)

    Returns:
        channel_gradients: np.array of shape (num_channels,) with importance scores
        spatial_maps: List of (tile_name, saliency_map) tuples
    """
    module.eval()
    device = module.device
    use_channels = module.use_channels
    num_channels = len(use_channels)

    # Select subset of tiles if requested
    if num_samples is not None and num_samples < len(test_tiles):
        tiles_to_use = test_tiles[:num_samples]
    else:
        tiles_to_use = test_tiles

    print(f"Computing feature importance using {len(tiles_to_use)} test samples...")
    print(f"Target class index: {target_class_idx}")

    # Accumulate gradients across all samples
    channel_gradients = np.zeros(num_channels, dtype=np.float64)
    spatial_maps = []

    for idx, tile_path in enumerate(tqdm(tiles_to_use, desc="Computing saliency")):
        # Load tile
        x_full = np.load(tile_path)
        x = x_full[:, :, use_channels]

        # Normalize
        x_norm = module.normalize(x)

        # Convert to tensor with gradient tracking
        x_tensor = torch.from_numpy(x_norm).float().to(device).unsqueeze(0)
        x_tensor = x_tensor.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x_tensor.requires_grad_(True)

        # Forward pass
        logits = module(x_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)

        # Get mean activation for target class
        target_prob = probs[0, target_class_idx, :, :].mean()

        # Backward pass to compute gradients
        target_prob.backward()

        # Extract per-channel gradient magnitude
        grads = None
        if x_tensor.grad is not None:
            grads = x_tensor.grad.data.abs()  # (1, C, H, W)
            channel_grad = (
                grads.sum(dim=(2, 3)).cpu().numpy()[0]
            )  # Sum over spatial dims -> (C,)
        else:
            channel_grad = np.zeros(len(use_channels))

        # Accumulate
        channel_gradients += channel_grad

        # Save spatial map if requested
        if save_spatial and idx < max_spatial and output_dir is not None and grads is not None:
            spatial_maps.append((tile_path.name, grads[0].cpu().numpy()))  # (C, H, W)

        # Clear gradients to free memory
        x_tensor.grad = None
        del x_tensor, logits, probs, grads

    # Average across samples
    channel_gradients /= len(tiles_to_use)

    # Save spatial maps if requested
    if save_spatial and spatial_maps and output_dir is not None:
        spatial_dir = output_dir / "spatial_maps"
        spatial_dir.mkdir(parents=True, exist_ok=True)
        save_spatial_saliency_maps(
            spatial_maps, BAND_NAMES[use_channels], spatial_dir, top_k=6
        )

    return channel_gradients, spatial_maps


def save_feature_importance_results(importance_scores, channel_names, output_dir):
    """
    Save feature importance scores as CSV and bar plot visualization.

    Args:
        importance_scores: np.array of shape (num_channels,)
        channel_names: List or array of channel name strings
        output_dir: pathlib.Path to output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize scores to sum to 1
    total = importance_scores.sum()
    normalized = importance_scores / total if total > 0 else importance_scores

    # Create DataFrame
    df = pd.DataFrame(
        {
            "channel_idx": range(len(channel_names)),
            "channel_name": channel_names,
            "importance_score": importance_scores,
            "normalized_score": normalized,
        }
    )

    # Sort by importance (descending)
    df = df.sort_values("importance_score", ascending=False).reset_index(drop=True)

    # Save CSV
    csv_path = output_dir / "channel_importance.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved feature importance CSV: {csv_path}")

    # Print ALL channels (not just top 10)
    print(f"\n{'=' * 70}")
    print("CHANNEL IMPORTANCE RANKING (ALL CHANNELS)")
    print(f"{'=' * 70}")
    print(
        f"{'Rank':<6}{'Idx':<6}{'Channel':<20}{'Raw Score':<15}{'Norm. Score':<15}{'Bar'}"
    )
    print("-" * 70)

    for rank, (i, row) in enumerate(df.iterrows()):
        bar_length = int(row["normalized_score"] * 50)
        bar = "█" * bar_length
        print(
            f"{rank + 1:<6}{row['channel_idx']:<6}{row['channel_name']:<20}"
            f"{row['importance_score']:<15.2f}{row['normalized_score']:<15.4f}{bar}"
        )

    # Create bar plot
    plot_path = output_dir / "channel_importance_barplot.png"
    plot_channel_importance(
        df["channel_name"].values,
        df["channel_idx"].values,
        df["normalized_score"].values,
        plot_path,
    )
    print(f"\nSaved feature importance plot: {plot_path}")


def plot_channel_importance(channel_names, channel_indices, scores, save_path):
    """
    Create horizontal bar plot of channel importance.

    Args:
        channel_names: Array of channel name strings
        channel_indices: Array of channel indices
        scores: Array of normalized importance scores
        save_path: pathlib.Path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, max(6, len(channel_names) * 0.35)))

    y_pos = np.arange(len(channel_names))
    colors = plt.colormaps['viridis'](scores / scores.max() if scores.max() > 0 else scores)

    ax.barh(y_pos, scores, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)

    # Create labels with both index and name (e.g., "[12] NDSI")
    labels = [f"[{idx}] {name}" for idx, name in zip(channel_indices, channel_names)]
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()  # Highest importance at top
    ax.set_xlabel("Normalized Importance Score", fontsize=11, fontweight="bold")
    ax.set_title(
        "Channel Importance (Gradient-Based Saliency)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_xlim(0, scores.max() * 1.1 if scores.max() > 0 else 1)

    # Add value labels on bars
    for i, (score, color) in enumerate(zip(scores, colors)):
        ax.text(
            score + scores.max() * 0.01,
            i,
            f"{score:.4f}",
            va="center",
            fontsize=8,
            color="black",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_spatial_saliency_maps(spatial_maps, channel_names, output_dir, top_k=6):
    """
    Save spatial saliency heatmaps for top-K most important channels.

    Args:
        spatial_maps: List of (tile_name, saliency_array) tuples
        channel_names: Array of channel names
        output_dir: Directory to save maps
        top_k: Number of top channels to visualize per sample
    """
    for tile_name, saliency in spatial_maps:
        # saliency shape: (C, H, W)
        # Compute per-channel total importance
        channel_totals = saliency.sum(axis=(1, 2))
        top_indices = np.argsort(channel_totals)[::-1][:top_k]

        # Create grid visualization
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        for i, ch_idx in enumerate(top_indices):
            ax = axes[i]
            heatmap = saliency[ch_idx]
            im = ax.imshow(heatmap, cmap="hot", interpolation="nearest")
            ax.set_title(
                f"{channel_names[ch_idx]}\n(Score: {channel_totals[ch_idx]:.1f})",
                fontsize=10,
            )
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)

        # Hide unused subplots
        for i in range(len(top_indices), len(axes)):
            axes[i].axis("off")

        fig.suptitle(
            f"Spatial Saliency Maps: {tile_name}", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        save_path = output_dir / f"{pathlib.Path(tile_name).stem}_saliency.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


def get_checkpoint_paths(runs_dir, run_name, model_type):
    """
    Get list of checkpoint paths based on model_type.

    Args:
        model_type: "all" for all checkpoints, or specific name like "best", "final"

    Returns:
        List of tuples: [(checkpoint_path, checkpoint_name), ...]
    """
    models_dir = runs_dir / run_name / "models"

    if model_type == "all":
        # All model_*.pt files, but exclude "best" since it's a duplicate
        ckpts = sorted(models_dir.glob("model_*.pt"))
        ckpt_pairs = [(ckpt, ckpt.stem.replace("model_", "")) for ckpt in ckpts]
        # Filter out "best" checkpoint
        return [(path, name) for path, name in ckpt_pairs if name != "best"]
    else:
        # Single checkpoint
        ckpt = models_dir / f"model_{model_type}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return [(ckpt, model_type)]


def load_existing_checkpoint_results(comparison_csv_path):
    """
    Load existing checkpoint results from checkpoints_comparison.csv.

    Returns:
        dict: Mapping from checkpoint name to metrics dict, or empty dict if file doesn't exist
    """
    if not comparison_csv_path.exists():
        return {}

    df = pd.read_csv(comparison_csv_path)
    results = {}

    for _, row in df.iterrows():
        ckpt_name = row["checkpoint"]
        metrics = row.to_dict()
        del metrics["checkpoint"]
        results[ckpt_name] = metrics

    return results


def identify_best_checkpoint(results_df, has_ci, has_deb, metric_strategy="IoU"):
    """
    Identify the best checkpoint based on specified metric strategy.

    Args:
        results_df: DataFrame with checkpoint results
        has_ci: bool, whether CleanIce model is included
        has_deb: bool, whether Debris model is included
        metric_strategy: str, metric to use for determining best checkpoint

    Returns:
        tuple: (best_checkpoint_name, best_metric_value)
    """
    if has_ci and has_deb:
        # Merged CI + Debris case
        if metric_strategy == "average_IoU":
            # Use average of CI_IoU and Deb_IoU
            best_idx = (results_df["CI_IoU"] + results_df["Deb_IoU"]).idxmax()
            best_metric = (
                results_df.loc[best_idx, "CI_IoU"] + results_df.loc[best_idx, "Deb_IoU"]
            ) / 2
        elif metric_strategy == "CI_IoU":
            best_idx = results_df["CI_IoU"].idxmax()
            best_metric = results_df.loc[best_idx, "CI_IoU"]
        elif metric_strategy == "Deb_IoU":
            best_idx = results_df["Deb_IoU"].idxmax()
            best_metric = results_df.loc[best_idx, "Deb_IoU"]
        elif metric_strategy == "CI_Precision":
            best_idx = results_df["CI_P"].idxmax()
            best_metric = results_df.loc[best_idx, "CI_P"]
        elif metric_strategy == "Deb_Precision":
            best_idx = results_df["Deb_P"].idxmax()
            best_metric = results_df.loc[best_idx, "Deb_P"]
        else:
            # Default to average IoU
            best_idx = (results_df["CI_IoU"] + results_df["Deb_IoU"]).idxmax()
            best_metric = (
                results_df.loc[best_idx, "CI_IoU"] + results_df.loc[best_idx, "Deb_IoU"]
            ) / 2
    else:
        # Single model case
        if metric_strategy == "IoU":
            best_idx = results_df["IoU"].idxmax()
            best_metric = results_df.loc[best_idx, "IoU"]
        elif metric_strategy == "Precision":
            best_idx = results_df["P"].idxmax()
            best_metric = results_df.loc[best_idx, "P"]
        elif metric_strategy == "Recall":
            best_idx = results_df["R"].idxmax()
            best_metric = results_df.loc[best_idx, "R"]
        else:
            # Default to IoU
            best_idx = results_df["IoU"].idxmax()
            best_metric = results_df.loc[best_idx, "IoU"]

    best_checkpoint = results_df.loc[best_idx, "checkpoint"]
    return best_checkpoint, best_metric


# Prediction runner for a single checkpoint combination


def run_prediction(
    frame_ci,
    frame_deb,
    thr_ci,
    thr_deb,
    test_tiles,
    cmap,
    vis_mode,
    vis_maxw,
    preds_dir,
    has_ci,
    has_deb,
):
    """
    Run predictions on all test tiles for a single checkpoint combination.

    Returns:
        df_rows: List of per-tile metric rows
        acc: Dict of accumulated TP/FP/FN counts
    """
    df_rows = []
    acc = dict(
        ci_tp=0.0,
        ci_fp=0.0,
        ci_fn=0.0,
        db_tp=0.0,
        db_fp=0.0,
        db_fn=0.0,
    )

    for tile in tqdm(test_tiles, desc="Predicting"):
        name = tile.name
        base = tile.stem

        x_full = np.load(tile)
        y_full = np.load(tile.parent / name.replace("tiff", "mask")).astype(np.uint8)
        invalid = create_invalid_mask(x_full, y_full)
        valid = ~invalid

        x_rgb = make_rgb_preview(x_full)
        prob_path = preds_dir / f"{base}_probs.npy"
        viz_path = preds_dir / f"{base}.png"

        # ===============================================================
        # SINGLE MODEL
        # ===============================================================
        if has_ci ^ has_deb:
            frame = frame_ci if has_ci else frame_deb
            model_class = 1 if has_ci else 2
            model_name = "CleanIce" if has_ci else "Debris"
            thr = thr_ci if has_ci else thr_deb

            probs = get_probabilities(frame, x_full)  # (H,W,2)
            np.save(prob_path, probs)

            pred_bin = (probs[:, :, 1] >= thr).astype(np.uint8)

            # GT comparison
            tv = y_full[valid]
            pv = pred_bin[valid]
            t_bin = (tv == model_class).astype(np.uint8)

            P, R, iou, tp, fp, fn = get_pr_iou(pv, t_bin)

            # accumulate
            if has_ci:
                acc["ci_tp"] += tp
                acc["ci_fp"] += fp
                acc["ci_fn"] += fn
            else:
                acc["db_tp"] += tp
                acc["db_fp"] += fp
                acc["db_fn"] += fn

            df_rows.append([name, P, R, iou])

            # Visualization label maps - use binary labeling for consistency
            y_gt = y_full.copy()
            y_gt[invalid] = 255

            # For binary visualization: convert to 0=NOT~class, 1=class, 255=mask
            y_gt_vis = np.zeros_like(y_full)
            y_gt_vis[valid & (y_full == model_class)] = 1
            y_gt_vis[valid & (y_full != model_class)] = 0
            y_gt_vis[invalid] = 255

            y_pred = np.zeros_like(y_full)
            y_pred[valid & (pred_bin == 1)] = 1  # class
            y_pred[valid & (pred_bin == 0)] = 0  # NOT~class
            y_pred[invalid] = 255

            # TP/FP/FN masks - use binary visualization labels
            gt_pos = (y_gt_vis == 1) & valid  # class pixels in GT
            pred_pos = (y_pred == 1) & valid  # class pixels in prediction
            tp_mask = gt_pos & pred_pos
            fp_mask = (~gt_pos) & pred_pos
            fn_mask = gt_pos & (~pred_pos)
            tp_rgb, fp_rgb, fn_rgb = make_tp_fp_fn_masks(tp_mask, fp_mask, fn_mask)

            # Confidence & entropy
            conf_rgb = make_confidence_map(probs[:, :, 1], invalid_mask=invalid)
            entropy_rgb = make_entropy_map(probs, invalid_mask=invalid)

            metrics_text = f"{model_name}: P={P:.3f} R={R:.3f} IoU={iou:.3f}"

            composite = make_eight_panel(
                x_rgb=x_rgb,
                gt_rgb=label_to_color(y_gt_vis, cmap),
                pr_rgb=label_to_color(y_pred, cmap),
                conf_rgb=conf_rgb,
                tp_rgb=tp_rgb,
                fp_rgb=fp_rgb,
                fn_rgb=fn_rgb,
                entropy_rgb=entropy_rgb,
                metrics_text=metrics_text,
            )

        # ===============================================================
        # MERGED CI + DEBRIS BINARY MODELS
        # ===============================================================
        else:
            prob_ci = get_probabilities(frame_ci, x_full)
            prob_db = get_probabilities(frame_deb, x_full)

            merged, probs = merge_ci_debris(prob_ci, prob_db, thr_ci, thr_deb)
            np.save(prob_path, probs)

            y_gt = y_full.copy()
            y_gt[invalid] = 255

            merged_vis = merged.copy()
            merged_vis[invalid] = 255

            # CleanIce metrics
            Pci, Rci, Ici, tp, fp, fn = get_pr_iou(
                (merged[valid] == 1).astype(np.uint8),
                (y_full[valid] == 1).astype(np.uint8),
            )
            acc["ci_tp"] += tp
            acc["ci_fp"] += fp
            acc["ci_fn"] += fn

            # Debris metrics
            Pdb, Rdb, Idb, tp, fp, fn = get_pr_iou(
                (merged[valid] == 2).astype(np.uint8),
                (y_full[valid] == 2).astype(np.uint8),
            )
            acc["db_tp"] += tp
            acc["db_fp"] += fp
            acc["db_fn"] += fn

            df_rows.append([name, Pci, Rci, Ici, Pdb, Rdb, Idb])

            # TP/FP/FN full 3-class
            tp_mask = (merged == y_full) & (~invalid) & (y_full != 0)
            fp_mask = (merged != y_full) & (~invalid) & (merged != 0)
            fn_mask = (merged != y_full) & (~invalid) & (y_full != 0)
            tp_rgb, fp_rgb, fn_rgb = make_tp_fp_fn_masks(tp_mask, fp_mask, fn_mask)

            # Confidence = prob of predicted class
            conf_map = probs[
                np.arange(probs.shape[0])[:, None],
                np.arange(probs.shape[1])[None, :],
                merged,
            ]
            conf_map[invalid] = 0
            conf_rgb = make_confidence_map(conf_map, invalid_mask=invalid)
            entropy_rgb = make_entropy_map(probs, invalid_mask=invalid)

            metrics_text = (
                f"CI: P={Pci:.3f} R={Rci:.3f} IoU={Ici:.3f} | "
                f"Deb: P={Pdb:.3f} R={Rdb:.3f} IoU={Idb:.3f}"
            )

            composite = make_eight_panel(
                x_rgb=x_rgb,
                gt_rgb=label_to_color(y_gt, cmap),
                pr_rgb=label_to_color(merged_vis, cmap),
                conf_rgb=conf_rgb,
                tp_rgb=tp_rgb,
                fp_rgb=fp_rgb,
                fn_rgb=fn_rgb,
                entropy_rgb=entropy_rgb,
                metrics_text=metrics_text,
            )

        # ===============================================================
        # Resize if needed
        # ===============================================================
        if vis_mode == "scaled" and composite.shape[1] > vis_maxw:
            scale = vis_maxw / composite.shape[1]
            composite = cv2.resize(
                composite,
                (vis_maxw, int(composite.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )

        cv2.imwrite(str(viz_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

    return df_rows, acc


# Main
def load_config_with_server_paths(config_path, server_name="desktop"):
    """Load prediction config and construct paths from servers.yaml"""
    config = Dict(yaml.safe_load(open(config_path)))

    # Load server paths from configs directory
    servers_cfg = Dict(yaml.safe_load((Path("configs") / "servers.yaml").read_text()))
    server = servers_cfg[server_name]

    # Auto-generate prediction name from model runs (handle missing sections)
    run_names = []
    if "cleanice" in config:
        run_names.append(config.cleanice.run_name)
    if "debris" in config:
        run_names.append(config.debris.run_name)
    prediction_name = "_".join(run_names)

    # Construct paths using output_path instead of code_path
    config.runs_dir = f"{server.output_path}/runs"
    config.output_dir = f"{server.output_path}/predictions/{prediction_name}"

    return config, server


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="conf/unet_predict.yaml", help="Path to prediction config file")
    parser.add_argument(
        "--server", required=True, choices=["desktop", "bilbo", "frodo"]
    )
    args = parser.parse_args()

    # Load config with server paths
    config_path = Path(args.config)
    conf, server = load_config_with_server_paths(config_path, args.server)

    # Get GPU device
    gpu = conf.get("gpu_rank", 0)
    if isinstance(gpu, str) and gpu.lower() == "cpu":
        gpu = "cpu"
    else:
        gpu = int(gpu)

    runs_dir = pathlib.Path(conf.runs_dir)
    out_root = pathlib.Path(conf.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Visualization settings
    vis_conf = conf.get("visualization", {})
    vis_mode = vis_conf.get("mode", "scaled")  # scaled or fullres
    vis_maxw = int(vis_conf.get("max_width", 2400))

    has_ci = "cleanice" in conf
    has_deb = "debris" in conf

    if not (has_ci or has_deb):
        raise RuntimeError("predict YAML must specify cleanice: or debris:")

    # ------------------------------------------------------------------
    # Get checkpoint lists
    # ------------------------------------------------------------------
    if has_ci:
        ck = conf.cleanice
        ci_checkpoints = get_checkpoint_paths(runs_dir, ck.run_name, ck.model_type)
        thr_ci = float(ck.threshold)
    else:
        ci_checkpoints = [(None, None)]
        thr_ci = None

    if has_deb:
        ck = conf.debris
        deb_checkpoints = get_checkpoint_paths(runs_dir, ck.run_name, ck.model_type)
        thr_deb = float(ck.threshold)
    else:
        deb_checkpoints = [(None, None)]
        thr_deb = None

    # Build appropriate colormap based on prediction mode
    if has_ci and has_deb:
        cmap = build_cmap(3, is_binary=False)
    elif has_ci:
        cmap = build_cmap(2, is_binary=True, classname="CleanIce")
    else:
        cmap = build_cmap(2, is_binary=True, classname="Debris")

    # CSV columns
    if has_ci and has_deb:
        columns = [
            "checkpoint",
            "tile",
            "CleanIce_precision",
            "CleanIce_recall",
            "CleanIce_IoU",
            "Debris_precision",
            "Debris_recall",
            "Debris_IoU",
        ]
    else:
        cname = "CleanIce" if has_ci else "Debris"
        columns = [
            "checkpoint",
            "tile",
            f"{cname}_precision",
            f"{cname}_recall",
            f"{cname}_IoU",
        ]

    # Summary for all checkpoints
    all_checkpoint_results = []

    # ------------------------------------------------------------------
    # Load existing checkpoint results if available
    # ------------------------------------------------------------------
    if has_ci and not has_deb:
        run_base = conf.cleanice.run_name
    elif has_deb and not has_ci:
        run_base = conf.debris.run_name
    else:
        run_base = f"ci_{conf.cleanice.run_name}__db_{conf.debris.run_name}"

    assert run_base is not None, "run_base must be set"
    print("Run", run_base)

    comparison_csv_path = out_root / run_base / "checkpoints_comparison.csv"
    existing_results = load_existing_checkpoint_results(comparison_csv_path)

    # if existing_results:
    #     print(
    #         f"\nFound existing results for {len(existing_results)} checkpoints in {comparison_csv_path}"
    #     )
    #     print(f"Will skip computation for already-processed checkpoints.\n")
    # else:
    #     print("\nNo existing results found. Will compute all checkpoints.\n")

    # ------------------------------------------------------------------
    # LOOP OVER ALL CHECKPOINT COMBINATIONS
    # ------------------------------------------------------------------
    if has_ci and has_deb:
        print(
            f"Running predictions for {len(ci_checkpoints)} CI × {len(deb_checkpoints)} Debris checkpoint combinations\n"
        )
    elif has_ci:
        print(
            f"Running predictions for {len(ci_checkpoints)} CleanIce checkpoint combinations\n"
        )
    else:
        print(
            f"Running predictions for {len(deb_checkpoints)} Debris checkpoint combinations\n"
        )

    for ci_ckpt_path, ci_ckpt_name in ci_checkpoints:
        for deb_ckpt_path, deb_ckpt_name in deb_checkpoints:
            # Determine checkpoint name
            if has_ci and not has_deb:
                ckpt_name = ci_ckpt_name or "unknown"
            elif has_deb and not has_ci:
                ckpt_name = deb_ckpt_name or "unknown"
            else:
                ckpt_name = f"ci_{ci_ckpt_name or 'unknown'}__db_{deb_ckpt_name or 'unknown'}"

            # Check if this checkpoint was already processed
            if ckpt_name in existing_results:
                # print(f"\n{'=' * 60}")
                # print(f"SKIPPING {ckpt_name} (already processed)")
                # print(f"{'=' * 60}")

                # Load cached metrics
                cached = existing_results[ckpt_name]
                cached["checkpoint"] = ckpt_name
                all_checkpoint_results.append(cached)

                # Print cached results
                # if has_ci and has_deb:
                #     print(
                #         f"CleanIce: P={cached['CI_P']:.4f} R={cached['CI_R']:.4f} IoU={cached['CI_IoU']:.4f}"
                #     )
                #     print(
                #         f"Debris:   P={cached['Deb_P']:.4f} R={cached['Deb_R']:.4f} IoU={cached['Deb_IoU']:.4f}"
                #     )
                # else:
                #     print(
                #         f"P={cached['P']:.4f} R={cached['R']:.4f} IoU={cached['IoU']:.4f}"
                #     )

                continue

            # Load models
            # print(f"\n{'=' * 60}")
            # print(f"PROCESSING {ckpt_name}")
            # print(f"{'=' * 60}")

            frame_ci = None
            frame_deb = None

            if has_ci:
                # print(f"Loading CleanIce: {ci_ckpt_path}")
                frame_ci = load_lightning_module(ci_ckpt_path, gpu)

            if has_deb:
                # print(f"Loading Debris: {deb_ckpt_path}")
                frame_deb = load_lightning_module(deb_ckpt_path, gpu)

            # Get test tiles (from whichever model was loaded)
            if has_ci and frame_ci is not None:
                data_dir = pathlib.Path(frame_ci.processed_dir)
            elif has_deb and frame_deb is not None:
                data_dir = pathlib.Path(frame_deb.processed_dir)
            else:
                raise RuntimeError("No valid model loaded for test tiles")
            test_tiles = sorted(pathlib.Path(data_dir, "test").glob("tiff*"))

            # Create output directory for this checkpoint
            out_dir = out_root / run_base / ckpt_name
            preds_dir = out_dir / "preds"
            preds_dir.mkdir(parents=True, exist_ok=True)

            print(f"Output dir: {out_dir}")
            print(f"Running on {len(test_tiles)} test tiles...")

            # Run predictions

            df_rows, acc = run_prediction(
                frame_ci,
                frame_deb,
                thr_ci,
                thr_deb,
                test_tiles,
                cmap,
                vis_mode,
                vis_maxw,
                preds_dir,
                has_ci,
                has_deb,
            )

            # Compute summary metrics

            if has_ci and has_deb:
                Pci = precision(acc["ci_tp"], acc["ci_fp"], acc["ci_fn"])
                Rci = recall(acc["ci_tp"], acc["ci_fp"], acc["ci_fn"])
                Ici = IoU(acc["ci_tp"], acc["ci_fp"], acc["ci_fn"])

                Pdb = precision(acc["db_tp"], acc["db_fp"], acc["db_fn"])
                Rdb = recall(acc["db_tp"], acc["db_fp"], acc["db_fn"])
                Idb = IoU(acc["db_tp"], acc["db_fp"], acc["db_fn"])

                print(f"\n===== {ckpt_name} SUMMARY =====")
                print(f"CleanIce: P={Pci:.4f} R={Rci:.4f} IoU={Ici:.4f}")
                print(f"Debris:   P={Pdb:.4f} R={Rdb:.4f} IoU={Idb:.4f}")

                # df_rows.append([ckpt_name, "TOTAL", Pci, Rci, Ici, Pdb, Rdb, Idb])
                df_rows.append(["TOTAL", Pci, Rci, Ici, Pdb, Rdb, Idb])

                all_checkpoint_results.append(
                    {
                        "checkpoint": ckpt_name,
                        "CI_P": Pci,
                        "CI_R": Rci,
                        "CI_IoU": Ici,
                        "Deb_P": Pdb,
                        "Deb_R": Rdb,
                        "Deb_IoU": Idb,
                    }
                )

            else:
                if has_ci:
                    tp = acc["ci_tp"]
                    fp = acc["ci_fp"]
                    fn = acc["ci_fn"]
                    cname = "CleanIce"
                else:
                    tp = acc["db_tp"]
                    fp = acc["db_fp"]
                    fn = acc["db_fn"]
                    cname = "Debris"

                P = precision(tp, fp, fn)
                R = recall(tp, fp, fn)
                iou = IoU(tp, fp, fn)

                print(f"\n===== {ckpt_name} SUMMARY =====")
                print(f"{cname}: P={P:.4f} R={R:.4f} IoU={iou:.4f}")

                # df_rows.append([ckpt_name, "TOTAL", P, R, iou])
                df_rows.append(["TOTAL", P, R, iou])
                all_checkpoint_results.append(
                    {"checkpoint": ckpt_name, "P": P, "R": R, "IoU": iou}
                )

            # Add checkpoint column to all rows
            df_rows_with_ckpt = [[ckpt_name] + row for row in df_rows]

            # Save per-tile metrics for this checkpoint
            df = pd.DataFrame(df_rows_with_ckpt, columns=pd.Index(columns))
            df.to_csv(out_dir / "metrics.csv", index=False)
            print(f"Saved metrics: {out_dir / 'metrics.csv'}")

            # Free GPU memory
            del frame_ci, frame_deb
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # FEATURE IMPORTANCE ANALYSIS (if enabled)
    # ------------------------------------------------------------------
    if conf.get("feature_importance", {}).get("enabled", False):
        fi_conf = conf.feature_importance

        print(f"\n{'=' * 80}")
        print("FEATURE IMPORTANCE ANALYSIS")
        print(f"{'=' * 80}")

        # Determine which checkpoint to use for saliency
        # Use the first checkpoint from the appropriate model
        saliency_frame = None
        saliency_label = ""

        if has_ci and not has_deb:
            ci_ckpt_info = ci_checkpoints[0]
            ci_ckpt_path = ci_ckpt_info[0] if ci_ckpt_info else None
            if ci_ckpt_path is None:
                raise RuntimeError("No CleanIce checkpoint found for saliency")
            print(f"Loading CleanIce checkpoint for saliency: {ci_ckpt_path.name}")
            saliency_frame = load_lightning_module(ci_ckpt_path, gpu)
            saliency_label = "CleanIce"
            target_class = 1  # Binary: foreground class
        elif has_deb and not has_ci:
            deb_ckpt_info = deb_checkpoints[0]
            deb_ckpt_path = deb_ckpt_info[0] if deb_ckpt_info else None
            if deb_ckpt_path is None:
                raise RuntimeError("No Debris checkpoint found for saliency")
            print(f"Loading Debris checkpoint for saliency: {deb_ckpt_path.name}")
            saliency_frame = load_lightning_module(deb_ckpt_path, gpu)
            saliency_label = "Debris"
            target_class = 1  # Binary: foreground class
        else:
            # For merged models, use cleanice frame
            ci_ckpt_info = ci_checkpoints[0]
            ci_ckpt_path = ci_ckpt_info[0] if ci_ckpt_info else None
            if ci_ckpt_path is None:
                raise RuntimeError("No CleanIce checkpoint found for saliency")
            print(f"Loading CleanIce checkpoint for saliency: {ci_ckpt_path.name}")
            saliency_frame = load_lightning_module(ci_ckpt_path, gpu)
            target_class = fi_conf.get("target_class", 1)
            saliency_label = f"Class_{target_class}"

        # Get test tiles
        data_dir = pathlib.Path(saliency_frame.processed_dir)
        test_tiles_for_saliency = sorted(data_dir.glob("test/tiff*"))

        print(f"Model: {saliency_label}")
        print(f"Total test tiles available: {len(test_tiles_for_saliency)}")

        # Compute feature importance
        fi_output_dir = out_root / run_base / "feature_importance"
        fi_output_dir.mkdir(parents=True, exist_ok=True)

        importance_scores, spatial_maps = compute_feature_importance(
            saliency_frame,
            test_tiles_for_saliency,
            target_class_idx=target_class,
            num_samples=fi_conf.get("num_samples"),
            save_spatial=fi_conf.get("save_spatial_maps", False),
            max_spatial=fi_conf.get("max_spatial_samples", 10),
            output_dir=fi_output_dir,
        )

        # Get channel names
        use_channels = saliency_frame.use_channels
        channel_names = BAND_NAMES[use_channels]

        # Save results
        save_feature_importance_results(importance_scores, channel_names, fi_output_dir)

        print("\n✓ Feature importance analysis complete.")
        print(f"  Results saved to: {fi_output_dir}")

        # Free GPU memory
        del saliency_frame
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # FINAL SUMMARY ACROSS ALL CHECKPOINTS
    # ------------------------------------------------------------------
    if len(all_checkpoint_results) > 1:
        if has_ci and has_deb:
            mode = "CleanIce+DebrisIce"
        elif has_ci:
            mode = "CleanIce"
        elif has_deb:
            mode = "DebrisIce"
        else:
            mode = "???"

        print(f"SUMMARY ACROSS ALL {mode} CHECKPOINTS")
        print("=" * 80)
        summary_df = pd.DataFrame(all_checkpoint_results)

        # Identify best checkpoint based on IoU
        if has_ci and has_deb:
            # Use average of CI_IoU and Deb_IoU for merged models
            best_idx = (summary_df["CI_IoU"] + summary_df["Deb_IoU"]).idxmax()
        else:
            # Use IoU for single models
            best_idx = summary_df["IoU"].idxmax()

        best_checkpoint = summary_df.loc[best_idx, "checkpoint"]

        # Sort by IoU in descending order (best checkpoint will naturally be first)
        if has_ci and has_deb:
            # Sort by average of CI_IoU and Deb_IoU
            summary_df["_avg_iou"] = (summary_df["CI_IoU"] + summary_df["Deb_IoU"]) / 2
            summary_df = summary_df.sort_values("_avg_iou", ascending=False).drop(
                "_avg_iou", axis=1
            )
        else:
            # Sort by IoU
            summary_df = summary_df.sort_values("IoU", ascending=False)

        # Reorder columns for better readability
        if has_ci and has_deb:
            column_order = [
                "checkpoint",
                "CI_P",
                "CI_R",
                "CI_IoU",
                "Deb_P",
                "Deb_R",
                "Deb_IoU",
            ]
        else:
            column_order = ["checkpoint", "P", "R", "IoU"]

        summary_df = summary_df[column_order]

        summary_path = out_root / run_base / "checkpoints_comparison.csv"
        summary_df.to_csv(summary_path, index=False)

        print(summary_df.to_string(index=False))
        print(f"\nSaved checkpoint comparison: {summary_path}")

    print("\n✓ Prediction complete for all checkpoints.")
