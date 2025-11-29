#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified predictor with:
 - thresholds inside each model block
 - 8-panel visualizations (TIFF | GT | Pred | Confidence / TP | FP | FN | Entropy)
 - VIRIDIS for confidence / entropy
 - categorical colors unified with training (BG=Brown, CI=Light Blue, Debris=Red, Mask=Black)
 - printed TOTAL test-set metrics
 - everything saved under preds/

Supports:
  • Single CleanIce model
  • Single Debris model
  • CleanIce + Debris merged (author-style)
"""

import pathlib
import yaml
from addict import Dict
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_fill_holes
import cv2

from model.frame import Framework
import model.metrics as metrics

from model.visualize import (
    build_cmap,
    make_rgb_preview,
    label_to_color,
    make_confidence_map,
    make_entropy_map,
    make_tp_fp_fn_masks,
    make_eight_panel,
)


# -------------------------------------------------------------
# Metric helpers
# -------------------------------------------------------------
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


# -------------------------------------------------------------
# Model inference
# -------------------------------------------------------------
def run_single_model(frame, x_full):
    """
    Returns probability cube (H,W,C) for a single model (after softmax).
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

    # probability cube (3-class)
    probs = np.zeros((H, W, 3), dtype=np.float32)
    probs[:, :, 1] = prob_ci[:, :, 1]
    probs[:, :, 2] = prob_deb[:, :, 1]
    probs[:, :, 0] = np.minimum(prob_ci[:, :, 0], prob_deb[:, :, 0])

    return merged, probs


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
if __name__ == "__main__":

    print("\nLoading config...")
    conf = Dict(yaml.safe_load(open("./conf/unet_predict.yaml")))

    gpu      = int(conf.get("gpu_rank", 0))
    runs_dir = pathlib.Path(conf.runs_dir)
    out_root = pathlib.Path(conf.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Visualization config
    vis_conf = conf.get("visualization", {})
    vis_mode = vis_conf.get("mode", "scaled")        # "scaled" or "fullres"
    vis_maxw = int(vis_conf.get("max_width", 2400))

    # Check for models
    has_ci  = "cleanice" in conf
    has_deb = "debris"  in conf

    if not (has_ci or has_deb):
        raise RuntimeError("YAML must specify cleanice: or debris:")

    # ---------------------------------------------------------
    # Load CI model
    # ---------------------------------------------------------
    if has_ci:
        ci = conf.cleanice
        ckpt_ci = runs_dir / ci.run_name / "models" / f"model_{ci.model_type}.pt"
        print("Loading CleanIce model:", ckpt_ci)
        frame_ci = Framework.from_checkpoint(ckpt_ci, device=gpu, testing=True)
        thr_ci = float(ci.threshold)

    # ---------------------------------------------------------
    # Load Debris model
    # ---------------------------------------------------------
    if has_deb:
        db = conf.debris
        ckpt_db = runs_dir / db.run_name / "models" / f"model_{db.model_type}.pt"
        print("Loading Debris model:", ckpt_db)
        frame_deb = Framework.from_checkpoint(ckpt_db, device=gpu, testing=True)
        thr_deb = float(db.threshold)

    # Dataset path
    data_dir = (
        frame_ci.loader_opts.processed_dir
        if has_ci else frame_deb.loader_opts.processed_dir
    )
    test_dir = pathlib.Path(data_dir) / "test"
    tiles    = sorted(test_dir.glob("tiff*"))

    print(f"Found {len(tiles)} test tiles.\n")

    # ---------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------
    if has_ci and not has_deb:
        run_name = conf.cleanice.run_name
    elif has_deb and not has_ci:
        run_name = conf.debris.run_name
    else:
        run_name = f"ci_{conf.cleanice.run_name}__db_{conf.debris.run_name}"

    out_dir   = out_root / run_name
    preds_dir = out_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # CSV header
    # ---------------------------------------------------------
    if has_ci and has_deb:
        columns = [
            "tile",
            "CleanIce_precision", "CleanIce_recall", "CleanIce_IoU",
            "Debris_precision",   "Debris_recall",   "Debris_IoU",
        ]
    else:
        cname = "CleanIce" if has_ci else "Debris"
        columns = ["tile", f"{cname}_precision", f"{cname}_recall", f"{cname}_IoU"]

    df_rows = []

    # Accumulators
    tot_tp = tot_fp = tot_fn = 0.0
    acc = dict(ci_tp=0.0, ci_fp=0.0, ci_fn=0.0,
               deb_tp=0.0, deb_fp=0.0, deb_fn=0.0)

    # Unified categorical cmap for GT/Pred
    cmap_dataset = build_cmap(num_classes=3, is_binary=False, classname=None)

    # ---------------------------------------------------------
    # MAIN LOOP
    # ---------------------------------------------------------
    for tile in tqdm(tiles, desc="Predicting"):

        name = tile.name
        base = pathlib.Path(name).stem

        x_full = np.load(tile)
        y_full = np.load(tile.parent / name.replace("tiff", "mask")).astype(np.uint8)

        invalid = (np.sum(x_full, axis=2) == 0) | (y_full == 255)
        valid   = ~invalid

        x_rgb = make_rgb_preview(x_full)

        save_img_path  = preds_dir / f"{base}.png"
        save_prob_path = preds_dir / f"{base}_probs.npy"

        # -----------------------------------------------------
        # SINGLE-MODEL PREDICTION
        # -----------------------------------------------------
        if has_ci ^ has_deb:

            frame = frame_ci if has_ci else frame_deb
            probs = run_single_model(frame, x_full)  # (H,W,2)

            if has_ci:
                thr = thr_ci
                class_id = 1
                model_name = "CleanIce"
            else:
                thr = thr_deb
                class_id = 2
                model_name = "Debris"

            pred_bin = (probs[:, :, 1] >= thr).astype(np.uint8)

            # Metrics (binary, class-specific)
            tv = y_full[valid]
            pv = pred_bin[valid]
            t_bin = (tv == class_id).astype(np.uint8)

            tp, fp, fn = get_tp_fp_fn(pv, t_bin)
            P, R, I = get_PR_IoU(tp, fp, fn)

            tot_tp += tp
            tot_fp += fp
            tot_fn += fn

            df_rows.append([name, P, R, I])

            np.save(save_prob_path, probs)

            # GT & Prediction label maps (0/BG, 1=CI, 2=Deb, 255=mask)
            y_gt_vis = y_full.copy()
            y_gt_vis[invalid] = 255

            y_pred_vis = np.zeros_like(y_full, dtype=np.uint8)
            y_pred_vis[valid & (pred_bin == 1)] = class_id
            y_pred_vis[valid & (pred_bin == 0)] = 0
            y_pred_vis[invalid] = 255

            # TP/FP/FN relative to "class_id"
            gt_pos   = (y_full == class_id) & valid
            pred_pos = (pred_bin == 1) & valid

            tp_mask = gt_pos & pred_pos
            fp_mask = (~gt_pos) & pred_pos
            fn_mask = gt_pos & (~pred_pos)

            tp_rgb, fp_rgb, fn_rgb = make_tp_fp_fn_masks(tp_mask, fp_mask, fn_mask)

            # Confidence & Entropy
            conf_map = probs[:, :, 1].copy()
            conf_map[invalid] = 0.0
            conf_rgb = make_confidence_map(conf_map, invalid_mask=invalid)

            entropy_rgb = make_entropy_map(probs, invalid_mask=invalid)

            metrics_text = f"{model_name}: P={P:.3f}  R={R:.3f}  IoU={I:.3f}"

            composite = make_eight_panel(
                x_rgb=x_rgb,
                gt_rgb=label_to_color(y_gt_vis, cmap_dataset),
                pr_rgb=label_to_color(y_pred_vis, cmap_dataset),
                conf_rgb=conf_rgb,
                tp_rgb=tp_rgb,
                fp_rgb=fp_rgb,
                fn_rgb=fn_rgb,
                entropy_rgb=entropy_rgb,
                metrics_text=metrics_text,
            )

            # Optional scaling for big images
            if vis_mode == "scaled" and composite.shape[1] > vis_maxw:
                scale = vis_maxw / composite.shape[1]
                new_h = int(composite.shape[0] * scale)
                composite = cv2.resize(
                    composite, (vis_maxw, new_h), interpolation=cv2.INTER_AREA
                )

            cv2.imwrite(
                str(save_img_path),
                cv2.cvtColor(composite, cv2.COLOR_RGB2BGR),
            )

        # -----------------------------------------------------
        # MERGED CI + DEBRIS MODEL
        # -----------------------------------------------------
        else:
            prob_ci  = run_single_model(frame_ci,  x_full)
            prob_deb = run_single_model(frame_deb, x_full)

            merged, probs = merge_ci_debris(prob_ci, prob_deb, thr_ci, thr_deb)
            np.save(save_prob_path, probs)

            merged_vis = merged.copy()
            merged_vis[invalid] = 255

            tv = y_full[valid]
            pv = merged[valid]

            # CI metrics
            p_ci = (pv == 1).astype(np.uint8)
            t_ci = (tv == 1).astype(np.uint8)
            tp_ci, fp_ci, fn_ci = get_tp_fp_fn(p_ci, t_ci)
            acc["ci_tp"] += tp_ci
            acc["ci_fp"] += fp_ci
            acc["ci_fn"] += fn_ci
            Pci, Rci, Ici = get_PR_IoU(tp_ci, fp_ci, fn_ci)

            # Debris metrics
            p_db = (pv == 2).astype(np.uint8)
            t_db = (tv == 2).astype(np.uint8)
            tp_db, fp_db, fn_db = get_tp_fp_fn(p_db, t_db)
            acc["deb_tp"] += tp_db
            acc["deb_fp"] += fp_db
            acc["deb_fn"] += fn_db
            Pdb, Rdb, Idb = get_PR_IoU(tp_db, fp_db, fn_db)

            df_rows.append([name, Pci, Rci, Ici, Pdb, Rdb, Idb])

            # GT for visualization
            y_gt_vis = y_full.copy()
            y_gt_vis[invalid] = 255

            # TP/FP/FN on full 3-class labels
            gt_pos   = (y_full != 0) & (~invalid)
            pred_pos = (merged != 0) & (~invalid)

            tp_mask = (merged == y_full) & gt_pos
            fp_mask = (merged != y_full) & pred_pos
            fn_mask = (merged != y_full) & gt_pos

            tp_rgb, fp_rgb, fn_rgb = make_tp_fp_fn_masks(tp_mask, fp_mask, fn_mask)

            # Confidence = P(predicted class)
            H, W = merged.shape
            conf_map = np.zeros((H, W), dtype=np.float32)
            for cls in [0, 1, 2]:
                idx = (merged == cls)
                conf_map[idx] = probs[idx, cls]
            conf_map[invalid] = 0.0
            conf_rgb = make_confidence_map(conf_map, invalid_mask=invalid)

            entropy_rgb = make_entropy_map(probs, invalid_mask=invalid)

            metrics_text = (
                f"CI: P={Pci:.3f} R={Rci:.3f} IoU={Ici:.3f} | "
                f"Deb: P={Pdb:.3f} R={Rdb:.3f} IoU={Idb:.3f}"
            )

            composite = make_eight_panel(
                x_rgb=x_rgb,
                gt_rgb=label_to_color(y_gt_vis, cmap_dataset),
                pr_rgb=label_to_color(merged_vis, cmap_dataset),
                conf_rgb=conf_rgb,
                tp_rgb=tp_rgb,
                fp_rgb=fp_rgb,
                fn_rgb=fn_rgb,
                entropy_rgb=entropy_rgb,
                metrics_text=metrics_text,
            )

            if vis_mode == "scaled" and composite.shape[1] > vis_maxw:
                scale = vis_maxw / composite.shape[1]
                new_h = int(composite.shape[0] * scale)
                composite = cv2.resize(
                    composite, (vis_maxw, new_h), interpolation=cv2.INTER_AREA
                )

            cv2.imwrite(
                str(save_img_path),
                cv2.cvtColor(composite, cv2.COLOR_RGB2BGR),
            )

    # -----------------------------------------------------
    # TOTAL METRICS
    # -----------------------------------------------------
    print("\n===== TEST SET SUMMARY =====")

    if has_ci and has_deb:
        Pci, Rci, Ici = get_PR_IoU(acc["ci_tp"], acc["ci_fp"], acc["ci_fn"])
        Pdb, Rdb, Idb = get_PR_IoU(acc["deb_tp"], acc["deb_fp"], acc["deb_fn"])

        print(f"CleanIce: Precision={Pci:.4f}, Recall={Rci:.4f}, IoU={Ici:.4f}")
        print(f"Debris:   Precision={Pdb:.4f}, Recall={Rdb:.4f}, IoU={Idb:.4f}")

        df_rows.append(["TOTAL", Pci, Rci, Ici, Pdb, Rdb, Idb])

    else:
        Ptot, Rtot, Itot = get_PR_IoU(tot_tp, tot_fp, tot_fn)
        cname = "CleanIce" if has_ci else "Debris"
        print(f"{cname}: Precision={Ptot:.4f}, Recall={Rtot:.4f}, IoU={Itot:.4f}")
        df_rows.append(["TOTAL", Ptot, Rtot, Itot])

    df = pd.DataFrame(df_rows, columns=columns)
    df.to_csv(out_dir / "metrics.csv", index=False)

    print("\n✓ Prediction complete.")
    print("Metrics CSV:", out_dir / "metrics.csv")
    print("Predictions + visuals:", preds_dir)

