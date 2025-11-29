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
    return (precision(tp, fp, fn), recall(tp, fp, fn), IoU(tp, fp, fn), tp, fp, fn)


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
        # All model_*.pt files
        ckpts = sorted(models_dir.glob("model_*.pt"))
        return [(ckpt, ckpt.stem.replace("model_", "")) for ckpt in ckpts]
    else:
        # Single checkpoint
        ckpt = models_dir / f"model_{model_type}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return [(ckpt, model_type)]


# ---------------------------------------------------------------------
# Prediction runner for a single checkpoint combination
# ---------------------------------------------------------------------
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
            model_name = "CleanIce" if has_ci else "Debris"
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
                acc["ci_tp"] += tp
                acc["ci_fp"] += fp
                acc["ci_fn"] += fn
            else:
                acc["db_tp"] += tp
                acc["db_fp"] += fp
                acc["db_fn"] += fn

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
            gt_pos = (y_gt_vis == 1) & valid  # class pixels in GT
            pred_pos = (y_pred == 1) & valid  # class pixels in prediction
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
            prob_ci = softmax_probs(frame_ci, x_full)
            prob_db = softmax_probs(frame_deb, x_full)

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


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    conf = Dict(yaml.safe_load(open("./conf/unet_predict.yaml")))
    # gpu = int(conf.get("gpu_rank", 0))
    gpu = conf.get("gpu_rank")

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
    # LOOP OVER ALL CHECKPOINT COMBINATIONS
    # ------------------------------------------------------------------
    print(
        f"\nRunning predictions for {len(ci_checkpoints)} CI × {len(deb_checkpoints)} Debris checkpoint combinations\n"
    )

    for ci_ckpt_path, ci_ckpt_name in ci_checkpoints:
        for deb_ckpt_path, deb_ckpt_name in deb_checkpoints:
            # ----------------------------------------------------------
            # Load models
            # ----------------------------------------------------------
            frame_ci = None
            frame_deb = None

            if has_ci:
                print(f"\nLoading CleanIce: {ci_ckpt_path}")
                frame_ci = Framework.from_checkpoint(
                    ci_ckpt_path, device=gpu, testing=True
                )

            if has_deb:
                print(f"Loading Debris: {deb_ckpt_path}")
                frame_deb = Framework.from_checkpoint(
                    deb_ckpt_path, device=gpu, testing=True
                )

            # Get test tiles (from whichever model was loaded)
            data_dir = (
                frame_ci.loader_opts.processed_dir
                if has_ci
                else frame_deb.loader_opts.processed_dir
            )
            test_tiles = sorted(pathlib.Path(data_dir, "test").glob("tiff*"))

            # ----------------------------------------------------------
            # Create output directory for this checkpoint
            # ----------------------------------------------------------
            if has_ci and not has_deb:
                ckpt_name = ci_ckpt_name
                run_base = conf.cleanice.run_name
            elif has_deb and not has_ci:
                ckpt_name = deb_ckpt_name
                run_base = conf.debris.run_name
            else:
                ckpt_name = f"ci_{ci_ckpt_name}__db_{deb_ckpt_name}"
                run_base = f"ci_{conf.cleanice.run_name}__db_{conf.debris.run_name}"

            out_dir = out_root / run_base / ckpt_name
            preds_dir = out_dir / "preds"
            preds_dir.mkdir(parents=True, exist_ok=True)

            print(f"Output dir: {out_dir}")
            print(f"Running on {len(test_tiles)} test tiles...")

            # ----------------------------------------------------------
            # Run predictions
            # ----------------------------------------------------------
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

            # ----------------------------------------------------------
            # Compute summary metrics
            # ----------------------------------------------------------
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
                I = IoU(tp, fp, fn)

                print(f"\n===== {ckpt_name} SUMMARY =====")
                print(f"{cname}: P={P:.4f} R={R:.4f} IoU={I:.4f}")

                # df_rows.append([ckpt_name, "TOTAL", P, R, I])
                df_rows.append(["TOTAL", P, R, I])
                all_checkpoint_results.append(
                    {"checkpoint": ckpt_name, "P": P, "R": R, "IoU": I}
                )

            # Add checkpoint column to all rows
            df_rows_with_ckpt = [[ckpt_name] + row for row in df_rows]

            # Save per-tile metrics for this checkpoint
            df = pd.DataFrame(df_rows_with_ckpt, columns=columns)
            df.to_csv(out_dir / "metrics.csv", index=False)
            print(f"Saved metrics: {out_dir / 'metrics.csv'}")

            # Free GPU memory
            del frame_ci, frame_deb
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # FINAL SUMMARY ACROSS ALL CHECKPOINTS
    # ------------------------------------------------------------------
    if len(all_checkpoint_results) > 1:
        print("\n" + "=" * 80)
        print("SUMMARY ACROSS ALL CHECKPOINTS")
        print("=" * 80)

        summary_df = pd.DataFrame(all_checkpoint_results)
        summary_path = out_root / run_base / "checkpoints_comparison.csv"
        summary_df.to_csv(summary_path, index=False)

        print(summary_df.to_string(index=False))
        print(f"\nSaved checkpoint comparison: {summary_path}")

    print("\n✓ Prediction complete for all checkpoints.")
