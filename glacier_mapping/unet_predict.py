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

import yaml
import pathlib
import numpy as np
import pandas as pd
from addict import Dict
from tqdm import tqdm

import torch
import cv2
from scipy.ndimage import binary_fill_holes

from glacier_mapping.model.frame import Framework
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


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def softmax_probs(frame, x_full):
    """
    Returns probability cube (H, W, C) for a single model.
    """
    use_ch = frame.loader_opts.use_channels
    x = x_full[:, :, use_ch]
    x_norm = frame.normalize(x)

    inp = torch.from_numpy(np.expand_dims(x_norm, 0)).float()
    logits = frame.infer(inp)
    probs = torch.nn.functional.softmax(logits, dim=3)[0].cpu().numpy()
    return probs


def merge_ci_debris(prob_ci, prob_deb, thr_ci, thr_deb):
    """
    Combine two binary models into a 3-class map:
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


def get_pr_iou(pred, true):
    pred_t = torch.from_numpy(pred.astype(np.uint8))
    true_t = torch.from_numpy(true.astype(np.uint8))
    tp, fp, fn = tp_fp_fn(pred_t, true_t)
    return (
        precision(tp, fp, fn),
        recall(tp, fp, fn),
        IoU(tp, fp, fn),
        tp, fp, fn
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":

    conf = Dict(yaml.safe_load(open("./conf/unet_predict.yaml")))
    gpu = int(conf.get("gpu_rank", 0))

    runs_dir = pathlib.Path(conf.runs_dir)
    out_root = pathlib.Path(conf.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Visualization settings
    vis_conf = conf.get("visualization", {})
    vis_mode = vis_conf.get("mode", "scaled")      # scaled or fullres
    vis_maxw = int(vis_conf.get("max_width", 2400))

    has_ci  = "cleanice" in conf
    has_deb = "debris"  in conf

    if not (has_ci or has_deb):
        raise RuntimeError("predict YAML must specify cleanice: or debris:")

    # ------------------------------------------------------------------
    # Load CI model
    # ------------------------------------------------------------------
    if has_ci:
        ck = conf.cleanice
        ckpt_ci = runs_dir / ck.run_name / "models" / f"model_{ck.model_type}.pt"
        print("Loading CleanIce:", ckpt_ci)
        frame_ci = Framework.from_checkpoint(ckpt_ci, device=gpu, testing=True)
        thr_ci = float(ck.threshold)

    # ------------------------------------------------------------------
    # Load Debris model
    # ------------------------------------------------------------------
    if has_deb:
        ck = conf.debris
        ckpt_db = runs_dir / ck.run_name / "models" / f"model_{ck.model_type}.pt"
        print("Loading Debris:", ckpt_db)
        frame_deb = Framework.from_checkpoint(ckpt_db, device=gpu, testing=True)
        thr_deb = float(ck.threshold)

    # Determine dataset path
    data_dir = (
        frame_ci.loader_opts.processed_dir if has_ci else
        frame_deb.loader_opts.processed_dir
    )

    test_tiles = sorted(pathlib.Path(data_dir, "test").glob("tiff*"))
    print(f"Found {len(test_tiles)} test tiles.\n")

    # ------------------------------------------------------------------
    # Output dirs
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

    # CSV
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

    # accumulators
    acc = dict(
        ci_tp=0.0, ci_fp=0.0, ci_fn=0.0,
        db_tp=0.0, db_fp=0.0, db_fn=0.0,
    )

    # Build appropriate colormap based on prediction mode
    if has_ci and has_deb:
        # Merged multi-class: BG, CI, Debris, mask
        cmap = build_cmap(3, is_binary=False)
    elif has_ci:
        # Binary CleanIce: NOT~CI, CI, mask
        cmap = build_cmap(2, is_binary=True, classname="CleanIce")
    else:
        # Binary Debris: NOT~Debris, Debris, mask
        cmap = build_cmap(2, is_binary=True, classname="Debris")

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------
    for tile in tqdm(test_tiles, desc="Predicting"):

        name = tile.name
        base = tile.stem

        x_full = np.load(tile)
        y_full = np.load(tile.parent / name.replace("tiff", "mask")).astype(np.uint8)
        invalid = (np.sum(x_full, axis=2) == 0) | (y_full == 255)
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
            model_name  = "CleanIce" if has_ci else "Debris"
            thr = thr_ci if has_ci else thr_deb

            probs = softmax_probs(frame, x_full)  # (H,W,2)
            np.save(prob_path, probs)

            pred_bin = (probs[:, :, 1] >= thr).astype(np.uint8)

            # GT comparison
            tv = y_full[valid]
            pv = pred_bin[valid]
            t_bin = (tv == model_class).astype(np.uint8)

            P, R, I, tp, fp, fn = get_pr_iou(pv, t_bin)

            # accumulate
            if has_ci:
                acc["ci_tp"] += tp; acc["ci_fp"] += fp; acc["ci_fn"] += fn
            else:
                acc["db_tp"] += tp; acc["db_fp"] += fp; acc["db_fn"] += fn

            df_rows.append([name, P, R, I])

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
            gt_pos   = (y_gt_vis == 1) & valid  # class pixels in GT
            pred_pos = (y_pred == 1) & valid     # class pixels in prediction
            tp_mask = gt_pos & pred_pos
            fp_mask = (~gt_pos) & pred_pos
            fn_mask = gt_pos & (~pred_pos)
            tp_rgb, fp_rgb, fn_rgb = make_tp_fp_fn_masks(tp_mask, fp_mask, fn_mask)

            # Confidence & entropy
            conf_rgb = make_confidence_map(probs[:, :, 1], invalid_mask=invalid)
            entropy_rgb = make_entropy_map(probs, invalid_mask=invalid)

            metrics_text = f"{model_name}: P={P:.3f} R={R:.3f} IoU={I:.3f}"

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
            prob_ci  = softmax_probs(frame_ci,  x_full)
            prob_db  = softmax_probs(frame_deb, x_full)

            merged, probs = merge_ci_debris(prob_ci, prob_db, thr_ci, thr_deb)
            np.save(prob_path, probs)

            y_gt = y_full.copy()
            y_gt[invalid] = 255

            merged_vis = merged.copy()
            merged_vis[invalid] = 255

            # CleanIce metrics
            Pci, Rci, Ici, tp, fp, fn = get_pr_iou(
                (merged[valid] == 1).astype(np.uint8),
                (y_full[valid] == 1).astype(np.uint8)
            )
            acc["ci_tp"] += tp; acc["ci_fp"] += fp; acc["ci_fn"] += fn

            # Debris metrics
            Pdb, Rdb, Idb, tp, fp, fn = get_pr_iou(
                (merged[valid] == 2).astype(np.uint8),
                (y_full[valid] == 2).astype(np.uint8)
            )
            acc["db_tp"] += tp; acc["db_fp"] += fp; acc["db_fn"] += fn

            df_rows.append([name, Pci, Rci, Ici, Pdb, Rdb, Idb])

            # TP/FP/FN full 3-class
            tp_mask = (merged == y_full) & (~invalid) & (y_full != 0)
            fp_mask = (merged != y_full) & (~invalid) & (merged != 0)
            fn_mask = (merged != y_full) & (~invalid) & (y_full != 0)
            tp_rgb, fp_rgb, fn_rgb = make_tp_fp_fn_masks(tp_mask, fp_mask, fn_mask)

            # Confidence = prob of predicted class
            conf_map = probs[np.arange(probs.shape[0])[:,None],
                             np.arange(probs.shape[1])[None,:],
                             merged]
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

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------
    print("\n===== TEST SET SUMMARY =====")

    if has_ci and has_deb:
        Pci = precision(acc["ci_tp"], acc["ci_fp"], acc["ci_fn"])
        Rci = recall(acc["ci_tp"], acc["ci_fp"], acc["ci_fn"])
        Ici = IoU(acc["ci_tp"], acc["ci_fp"], acc["ci_fn"])

        Pdb = precision(acc["db_tp"], acc["db_fp"], acc["db_fn"])
        Rdb = recall(acc["db_tp"], acc["db_fp"], acc["db_fn"])
        Idb = IoU(acc["db_tp"], acc["db_fp"], acc["db_fn"])

        print(f"CleanIce TOTAL: P={Pci:.4f} R={Rci:.4f} IoU={Ici:.4f}")
        print(f"Debris   TOTAL: P={Pdb:.4f} R={Rdb:.4f} IoU={Idb:.4f}")

        df_rows.append(["TOTAL", Pci, Rci, Ici, Pdb, Rdb, Idb])

    else:
        if has_ci:
            tp = acc["ci_tp"]; fp = acc["ci_fp"]; fn = acc["ci_fn"]
            cname = "CleanIce"
        else:
            tp = acc["db_tp"]; fp = acc["db_fp"]; fn = acc["db_fn"]
            cname = "Debris"

        P = precision(tp, fp, fn); R = recall(tp, fp, fn); I = IoU(tp, fp, fn)
        print(f"{cname} TOTAL: P={P:.4f} R={R:.4f} IoU={I:.4f}")
        df_rows.append(["TOTAL", P, R, I])

    df = pd.DataFrame(df_rows, columns=columns)
    df.to_csv(out_dir / "metrics.csv", index=False)

    print("\n✓ Prediction complete.")
    print("Metrics CSV:", out_dir / "metrics.csv")
    print("Visualization PNGs:", preds_dir)

