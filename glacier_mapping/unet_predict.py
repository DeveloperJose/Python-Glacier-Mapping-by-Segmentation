#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified predictor with:
 - thresholds inside each model block
 - 3-panel visualizations (Input | GT | Prediction)
 - dynamic colormaps for binary or multi-class
 - printed TOTAL test-set metrics
 - everything saved under preds/
"""

import pathlib
import yaml
from addict import Dict
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import cv2
from scipy.ndimage import binary_fill_holes

from model.frame import Framework
import model.metrics as metrics

# -----------------------------------------------------------------------------
# Colors
# -----------------------------------------------------------------------------
COLOR_BG     = np.array([181, 101, 29], dtype=np.uint8)   # Brown
COLOR_CI     = np.array([135, 206, 250], dtype=np.uint8)  # Light Blue
COLOR_DEB    = np.array([255,   0,   0], dtype=np.uint8)  # Red
COLOR_IGNORE = np.array([0, 0, 0], dtype=np.uint8)        # Black

GLOBAL_CMAP_MULTI = {
    0: COLOR_BG,
    1: COLOR_CI,
    2: COLOR_DEB,
    255: COLOR_IGNORE,
}

# -----------------------------------------------------------------------------
# Dynamic colormap for binary models
# -----------------------------------------------------------------------------
def build_dynamic_cmap(output_classes):
    """
    Build consistent color maps for GT + prediction.
    """
    # Multi-class BG/CI/DEB
    if output_classes == [0, 1, 2]:
        return GLOBAL_CMAP_MULTI

    # Binary CleanIce
    if output_classes == [1]:
        # 0 = NOT~CI (BG+Debris), 1 = CI
        return {
            0: COLOR_BG,
            1: COLOR_CI,
            255: COLOR_IGNORE,
        }

    # Binary Debris
    if output_classes == [2]:
        # 0 = NOT~Debris (BG+CI), 1 = Debris
        return {
            0: COLOR_BG,
            1: COLOR_DEB,
            255: COLOR_IGNORE,
        }

    raise ValueError(f"Unsupported class combination for colormap: {output_classes}")


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def save_comparison_image(x_full, y_true, y_pred, save_path, output_classes):
    """
    Save 3-column side-by-side visualization:
        [ Input | Ground Truth | Prediction ]
    """

    H, W, _ = x_full.shape
    cmap = build_dynamic_cmap(output_classes)

    # --- Input (RGB) ---
    x_rgb = x_full[..., [2, 1, 0]].astype(np.float32)
    x_rgb = (255 * (x_rgb - x_rgb.min()) / (x_rgb.max() - x_rgb.min() + 1e-6)).astype(
        np.uint8
    )

    # --- Ground truth (0/1/2/255) ---
    gt_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cls, col in cmap.items():
        gt_rgb[y_true == cls] = col

    # --- Prediction (same label scheme as y_true for the given model) ---
    pred_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cls, col in cmap.items():
        pred_rgb[y_pred == cls] = col

    # --- Title helper ---
    def add_title(img, txt):
        bar = np.full((40, img.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(
            bar,
            txt,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        return np.vstack([bar, img])

    x_disp = add_title(x_rgb, "INPUT")
    gt_disp = add_title(gt_rgb, "GROUND TRUTH")
    pr_disp = add_title(pred_rgb, "PREDICTION")

    combined = np.hstack([x_disp, gt_disp, pr_disp])

    save_path = save_path.with_suffix(".png")
    cv2.imwrite(str(save_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------
def get_tp_fp_fn(pred, true):
    pred_t = torch.from_numpy(pred.astype(np.uint8))
    true_t = torch.from_numpy(true.astype(np.uint8))
    tp, fp, fn = metrics.tp_fp_fn(pred_t, true_t)
    return float(tp), float(fp), float(fn)


def get_PR_IoU(tp, fp, fn):
    return (
        float(metrics.precision(tp, fp, fn)),
        float(metrics.recall(tp, fp, fn)),
        float(metrics.IoU(tp, fp, fn)),
    )


# -----------------------------------------------------------------------------
# Model inference
# -----------------------------------------------------------------------------
def run_single_model(frame, x_full):
    """
    Returns probability cube (H,W,C) for a single model.
    """
    use_ch = frame.loader_opts.use_channels
    x = x_full[:, :, use_ch]
    x_norm = frame.normalize(x)

    inp = torch.from_numpy(np.expand_dims(x_norm, 0)).float()
    logits = frame.infer(inp)
    probs = torch.nn.functional.softmax(logits, dim=3).squeeze(0).cpu().numpy()
    return probs


def merge_ci_debris(prob_ci, prob_deb, thr_ci, thr_deb):
    """
    Merge CleanIce & Debris binary models into 3-class labels:
        0 = BG, 1 = CI, 2 = Debris
    """
    ci_mask = binary_fill_holes(prob_ci[:, :, 1] >= thr_ci)
    deb_mask = binary_fill_holes(prob_deb[:, :, 1] >= thr_deb)

    H, W = ci_mask.shape
    merged = np.zeros((H, W), dtype=np.uint8)
    merged[ci_mask] = 1
    merged[deb_mask] = 2

    # probability cube (author-style)
    probs = np.zeros((H, W, 3), dtype=np.float32)
    probs[:, :, 1] = prob_ci[:, :, 1]
    probs[:, :, 2] = prob_deb[:, :, 1]
    probs[:, :, 0] = np.minimum(prob_ci[:, :, 0], prob_deb[:, :, 0])

    return merged, probs


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("\nLoading config...")
    conf = Dict(yaml.safe_load(open("./conf/unet_predict.yaml")))

    gpu = int(conf.get("gpu_rank", 0))
    runs_dir = pathlib.Path(conf.runs_dir)
    out_root = pathlib.Path(conf.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    has_ci = "cleanice" in conf
    has_deb = "debris" in conf

    if not (has_ci or has_deb):
        raise RuntimeError("YAML must contain either cleanice: or debris:")

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    if has_ci:
        ci = conf.cleanice
        ci_ckpt = runs_dir / ci.run_name / "models" / f"model_{ci.model_type}.pt"
        print("Loading CleanIce model: ", ci_ckpt)
        frame_ci = Framework.from_checkpoint(ci_ckpt, device=gpu, testing=True)
        thr_ci = float(ci.get("threshold", 0.5))

    if has_deb:
        db = conf.debris
        db_ckpt = runs_dir / db.run_name / "models" / f"model_{db.model_type}.pt"
        print("Loading Debris model:  ", db_ckpt)
        frame_deb = Framework.from_checkpoint(db_ckpt, device=gpu, testing=True)
        thr_deb = float(db.get("threshold", 0.5))

    # Dataset location (take it from whichever model we have)
    data_dir = (
        frame_ci.loader_opts.processed_dir
        if has_ci
        else frame_deb.loader_opts.processed_dir
    )
    test_dir = pathlib.Path(data_dir) / "test"
    tiles = sorted(test_dir.glob("tiff*"))

    print(f"Found {len(tiles)} test tiles.\n")

    # ------------------------------------------------------------------
    # Output folder naming
    # ------------------------------------------------------------------
    if has_ci and not has_deb:
        run_name = conf.cleanice.run_name
    elif has_deb and not has_ci:
        run_name = conf.debris.run_name
    else:
        run_name = f"ci_{conf.cleanice.run_name}__db_{conf.debris.run_name}"

    out_dir = out_root / run_name
    preds_dir = out_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # CSV header
    # ------------------------------------------------------------------
    if has_ci and has_deb:
        columns = [
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
        columns = ["tile", f"{cname}_precision", f"{cname}_recall", f"{cname}_IoU"]

    df = []

    # Global accumulators
    tot_tp = 0.0
    tot_fp = 0.0
    tot_fn = 0.0
    acc = {"ci_tp": 0.0, "ci_fp": 0.0, "ci_fn": 0.0,
           "deb_tp": 0.0, "deb_fp": 0.0, "deb_fn": 0.0}

    # ------------------------------------------------------------------
    # LOOP
    # ------------------------------------------------------------------
    for tile in tqdm(tiles, desc="Predicting"):
        name = tile.name
        x_full = np.load(tile)
        y_full = np.load(tile.parent / name.replace("tiff", "mask")).astype(np.uint8)

        # invalid / ignore mask
        invalid = (np.sum(x_full, axis=2) == 0) | (y_full == 255)
        valid = ~invalid

        # SINGLE MODEL CASE --------------------------------------------
        if has_ci ^ has_deb:
            frame = frame_ci if has_ci else frame_deb
            probs = run_single_model(frame, x_full)

            # Choose threshold & class_id
            if has_ci:
                thr = thr_ci
                class_id = 1   # CleanIce in ground truth
            else:
                thr = thr_deb
                class_id = 2   # Debris in ground truth

            # Positive-class probability in channel 1
            pred_bin = (probs[:, :, 1] >= thr).astype(np.uint8)  # 0/1

            # Metrics: compare predicted CI/DEB vs GT class
            tv = y_full[valid]
            pv_bin = pred_bin[valid]
            t_bin = (tv == class_id).astype(np.uint8)

            tp, fp, fn = get_tp_fp_fn(pv_bin, t_bin)
            P, R, I = get_PR_IoU(tp, fp, fn)

            tot_tp += tp
            tot_fp += fp
            tot_fn += fn

            df.append([name, P, R, I])

            # Visualization label image:
            #   0 = NOT~class, 1 = class, 255 = ignore
            y_pred_vis = np.zeros_like(y_full, dtype=np.uint8)
            y_pred_vis[valid] = pred_bin[valid]
            y_pred_vis[invalid] = 255

            # For GT visualization in binary case:
            #   0 = NOT~class, 1 = class, 255 = ignore
            y_gt_vis = np.zeros_like(y_full, dtype=np.uint8)
            y_gt_vis[valid] = (tv == class_id).astype(np.uint8)
            y_gt_vis[invalid] = 255

            save_comparison_image(
                x_full,
                y_gt_vis,
                y_pred_vis,
                preds_dir / f"{pathlib.Path(name).stem}.png",
                frame.loader_opts.output_classes,  # [1] or [2]
            )

        # TWO MODEL MERGE ----------------------------------------------
        else:
            prob_ci = run_single_model(frame_ci, x_full)
            prob_deb = run_single_model(frame_deb, x_full)

            merged, probs = merge_ci_debris(prob_ci, prob_deb, thr_ci, thr_deb)
            # merged: 0=BG, 1=CI, 2=Debris
            merged_vis = merged.copy()
            merged_vis[invalid] = 255

            tv = y_full[valid]
            pv = merged[valid]

            # CleanIce metrics
            p_ci = (pv == 1).astype(np.uint8)
            t_ci = (tv == 1).astype(np.uint8)
            tp, fp, fn = get_tp_fp_fn(p_ci, t_ci)
            acc["ci_tp"] += tp
            acc["ci_fp"] += fp
            acc["ci_fn"] += fn
            Pci, Rci, Ici = get_PR_IoU(tp, fp, fn)

            # Debris metrics
            p_db = (pv == 2).astype(np.uint8)
            t_db = (tv == 2).astype(np.uint8)
            tp, fp, fn = get_tp_fp_fn(p_db, t_db)
            acc["deb_tp"] += tp
            acc["deb_fp"] += fp
            acc["deb_fn"] += fn
            Pdb, Rdb, Idb = get_PR_IoU(tp, fp, fn)

            df.append([name, Pci, Rci, Ici, Pdb, Rdb, Idb])

            # Save probability cube
            np.save(preds_dir / name.replace("tiff", "probs.npy"), probs)

            # Visualization uses full multi-class map
            y_gt_vis = y_full.copy()
            y_gt_vis[invalid] = 255

            save_comparison_image(
                x_full,
                y_gt_vis,
                merged_vis,
                preds_dir / f"{pathlib.Path(name).stem}.png",
                [0, 1, 2],
            )

    # ------------------------------------------------------------------
    # TOTAL METRICS
    # ------------------------------------------------------------------
    print("\n===== TEST SET SUMMARY =====")

    if has_ci and has_deb:
        Pci, Rci, Ici = get_PR_IoU(acc["ci_tp"], acc["ci_fp"], acc["ci_fn"])
        Pdb, Rdb, Idb = get_PR_IoU(acc["deb_tp"], acc["deb_fp"], acc["deb_fn"])

        print(f"CleanIce: Precision={Pci:.4f}, Recall={Rci:.4f}, IoU={Ici:.4f}")
        print(f"Debris:   Precision={Pdb:.4f}, Recall={Rdb:.4f}, IoU={Idb:.4f}")

        df.append(["TOTAL", Pci, Rci, Ici, Pdb, Rdb, Idb])

    else:
        P_tot, R_tot, I_tot = get_PR_IoU(tot_tp, tot_fp, tot_fn)
        cname = "CleanIce" if has_ci else "Debris"
        print(f"{cname}: Precision={P_tot:.4f}, Recall={R_tot:.4f}, IoU={I_tot:.4f}")
        df.append(["TOTAL", P_tot, R_tot, I_tot])

    df = pd.DataFrame(df, columns=columns)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "metrics.csv", index=False)

    print("\n✓ Prediction complete.")
    print("Metrics CSV:", out_dir / "metrics.csv")
    print("Predictions + visuals:", preds_dir)

