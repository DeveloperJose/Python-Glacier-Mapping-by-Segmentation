#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training, validation, logging, and full-tile evaluation helpers.
Unified visualization pipeline matching unet_predict + visualize.py
"""

import datetime
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import glacier_mapping.model.losses as model_losses
import glacier_mapping.model.metrics as model_metrics

from glacier_mapping.model.visualize import (
    build_cmap,
    make_rgb_preview,
    label_to_color,
    make_confidence_map,
    make_entropy_map,
    make_tp_fp_fn_masks,
    make_eight_panel,
)

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)


# ============================================================
# Utility logging
# ============================================================
def log(level, message):
    message = "{}\t{}\\t{}".format(
        datetime.datetime.now().strftime("%d-%m-%Y, %H:%M:%S"),
        logging._levelToName[level],
        message,
    )
    message = "SystemLog: " + message
    logging.log(level, message)


# ============================================================
# TRAINING
# ============================================================
def train_epoch(epoch, loader, frame):
    metric_names = frame.metrics_opts.metrics
    n_classes = frame.num_classes
    threshold = frame.metrics_opts.threshold

    loss_sum = 0.0
    tp = torch.zeros(n_classes)
    fp = torch.zeros(n_classes)
    fn = torch.zeros(n_classes)

    iterator = tqdm(loader, desc="Train Iter")

    for i, (x, y_onehot, y_int) in enumerate(iterator):
        frame.zero_grad()
        y_hat, batch_loss = frame.optimize(x, y_onehot, y_int.squeeze(-1))
        frame.step()

        batch_loss_f = float(batch_loss.detach())
        loss_sum += batch_loss_f

        y_hat = frame.act(y_hat)
        ignore = (y_int.squeeze(-1) == 255).cpu().numpy()
        _tp, _fp, _fn = frame.metrics(y_hat, y_onehot, ignore, threshold)

        tp += _tp
        fp += _fp
        fn += _fn

        iterator.set_description(
            f"Train, Epoch={epoch}, Step={i}, "
            f"Loss={batch_loss_f:.3f}, Avg={loss_sum/(i+1):.3f}"
        )

    avg_loss = loss_sum / (i + 1)
    return avg_loss, get_metrics(tp, fp, fn, metric_names), frame.get_loss_alpha()


# ============================================================
# VALIDATION
# ============================================================
def validate(epoch, loader, frame, test=False):
    n_classes = frame.num_classes
    metric_names = frame.metrics_opts.metrics

    total_loss = 0.0
    count_batches = 0

    tp = torch.zeros(n_classes)
    fp = torch.zeros(n_classes)
    fn = torch.zeros(n_classes)

    desc = "Test Iter" if test else "Val Iter"
    iterator = tqdm(loader, desc=desc)

    def channel_first(x):
        # NHWC -> NCHW
        return x.permute(0, 3, 1, 2)

    for i, (x, y_onehot, y_int) in enumerate(iterator):
        y_hat = frame.infer(x)
        batch_loss = frame.calc_loss(
            channel_first(y_hat),
            channel_first(y_onehot),
            y_int.squeeze(-1),
        )
        batch_loss_f = float(batch_loss.detach())

        y_hat = frame.act(y_hat)

        ignore = (y_int.squeeze(-1) == 255).cpu()  # (B,H,W) bool
        if ignore.all():
            continue

        y_true_cls = torch.argmax(y_onehot, dim=-1).cpu()   # (B,H,W)
        y_pred_cls = torch.argmax(y_hat.cpu(), dim=-1)      # (B,H,W)

        valid = ~ignore
        y_true_valid = y_true_cls[valid]
        y_pred_valid = y_pred_cls[valid]

        for c in range(n_classes):
            pred_c = (y_pred_valid == c).long()
            true_c = (y_true_valid == c).long()
            tp_c, fp_c, fn_c = model_metrics.tp_fp_fn(pred_c, true_c)
            tp[c] += tp_c
            fp[c] += fp_c
            fn[c] += fn_c

        total_loss += batch_loss_f
        count_batches += 1

        iterator.set_description(
            f"{desc}, Epoch={epoch}, Step={i}, Loss={batch_loss_f:.3f}, "
            f"Avg={total_loss/max(count_batches,1):.3f}"
        )

    if count_batches == 0:
        print(f"[WARN] {desc}: all patches ignored.")
        return 0.0, get_metrics(tp, fp, fn, metric_names)

    avg_loss = total_loss / count_batches

    if not test:
        frame.val_operations(avg_loss)

    return avg_loss, get_metrics(tp, fp, fn, metric_names)


# ============================================================
# METRIC LOGGING
# ============================================================
def log_metrics(writer, frame, metrics, epoch, stage):
    for key, vals in metrics.items():
        for name, metric in zip(frame.mask_names, vals):
            writer.add_scalar(f"{stage}_{key}/{name}", metric, epoch)


# ============================================================
# TENSORBOARD IMAGE LOGGING (8-PANEL)
# ============================================================
def log_images(writer, frame, batch, epoch, stage, normalize):
    """
    Logs the SAME 8-panel composite used in predictor:

        TIFF | GT | PRED | CONF
        TP    | FP | FN   | ENTROPY

    Now includes a metrics header (precision/recall/IoU).
    """

    # ------------------------------
    # 1. Fetch a batch sample
    # ------------------------------
    x, y_onehot, y_int = next(iter(batch))

    x_np = x.cpu().numpy()[0]
    y_hat = frame.act(frame.infer(x))
    yhat_np = y_hat.cpu().numpy()[0]

    y_gt = torch.argmax(y_onehot, dim=-1).cpu().numpy()[0]
    y_pred = torch.argmax(y_hat.cpu(), dim=-1).numpy()[0]

    ignore_mask = (y_int.cpu().numpy()[0] == 255)
    y_gt[ignore_mask] = 255
    y_pred[ignore_mask] = 255

    # ------------------------------
    # 2. Categorical styling
    # ------------------------------
    num_classes = frame.num_classes
    is_binary = frame.is_binary
    classname = frame.mask_names[-1] if is_binary else None
    cmap = build_cmap(num_classes, is_binary, classname)

    x_rgb = make_rgb_preview(x_np)
    gt_rgb = label_to_color(y_gt, cmap)
    pr_rgb = label_to_color(y_pred, cmap)

    # ------------------------------
    # 3. Confidence & Entropy maps
    # ------------------------------
    if is_binary:
        conf = yhat_np[..., 1]
    else:
        conf = np.max(yhat_np, axis=-1)

    conf_rgb = make_confidence_map(conf, invalid_mask=ignore_mask)
    entropy_rgb = make_entropy_map(yhat_np, invalid_mask=ignore_mask)

    # ------------------------------
    # 4. TP / FP / FN (non-background)
    # ------------------------------
    tp_mask = (y_pred == y_gt) & (~ignore_mask) & (y_gt != 0)
    fp_mask = (y_pred != y_gt) & (~ignore_mask) & (y_pred != 0)
    fn_mask = (y_pred != y_gt) & (~ignore_mask) & (y_gt != 0)

    tp_rgb, fp_rgb, fn_rgb = make_tp_fp_fn_masks(tp_mask, fp_mask, fn_mask)

    # ------------------------------
    # 5. Compute metrics for header
    # ------------------------------
    metric_names = frame.metrics_opts.metrics
    metric_string_parts = []

    # Compute per-class stats
    for ci, cname in enumerate(frame.mask_names):
        pred_c = (y_pred == ci).astype(np.uint8)
        true_c = (y_gt == ci).astype(np.uint8)

        tp, fp, fn = model_metrics.tp_fp_fn(
            torch.from_numpy(pred_c),
            torch.from_numpy(true_c)
        )

        P = model_metrics.precision(tp, fp, fn)
        R = model_metrics.recall(tp, fp, fn)
        I = model_metrics.IoU(tp, fp, fn)

        metric_string_parts.append(f"{cname}: P={P:.3f} R={R:.3f} IoU={I:.3f}")

    metrics_text = " | ".join(metric_string_parts)

    # ------------------------------
    # 6. Assemble 8-panel composite
    # ------------------------------
    composite = make_eight_panel(
        x_rgb=x_rgb,
        gt_rgb=gt_rgb,
        pr_rgb=pr_rgb,
        conf_rgb=conf_rgb,
        tp_rgb=tp_rgb,
        fp_rgb=fp_rgb,
        fn_rgb=fn_rgb,
        entropy_rgb=entropy_rgb,
        metrics_text=metrics_text,
    )

    # ------------------------------
    # 7. Log to TensorBoard
    # ------------------------------
    img_tensor = torch.tensor(composite).permute(2, 0, 1).float() / 255.0
    writer.add_image(f"{stage}/visualization", img_tensor, epoch)

# ============================================================
# FULL-TILE EVALUATION (consistent with predictor)
# ============================================================
def evaluate_full_test_tiles(frame, writer, epoch, output_dir):
    """
    Produces PNGs and TensorBoard logs using the SAME unified 8-panel layout.
    """

    import cv2

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(frame.loader_opts.processed_dir)
    test_tiles = sorted((data_dir / "test").glob("tiff*"))

    n_classes = frame.num_classes
    is_binary = frame.is_binary
    classname = frame.mask_names[-1] if is_binary else None
    threshold = frame.metrics_opts.threshold

    rows = []
    tp_sum = np.zeros(n_classes)
    fp_sum = np.zeros(n_classes)
    fn_sum = np.zeros(n_classes)

    # ------------------------------
    # LOOP THROUGH TEST TILES
    # ------------------------------
    for x_path in tqdm(test_tiles):
        x = np.load(x_path)
        y_pred, invalid_mask = frame.predict_slice(x, threshold)

        y_true_raw = np.load(
            x_path.with_name(x_path.name.replace("tiff", "mask"))
        ).astype(np.uint8)

        ignore = (y_true_raw == 255)
        if invalid_mask is not None:
            ignore |= invalid_mask

        # Prepare GT for metrics (shift by +1 for classes)
        y_true = y_true_raw.copy()
        y_true[ignore] = 0
        valid = ~ignore
        y_true[valid] += 1

        y_pred_valid = y_pred[valid]
        y_true_valid = y_true[valid]

        # ------------------------------
        # Per-class metrics
        # ------------------------------
        row = [x_path.name]
        for ci in range(n_classes):
            label = ci + 1
            p = (y_pred_valid == label).astype(np.uint8)
            t = (y_true_valid == label).astype(np.uint8)

            tp, fp, fn = model_metrics.tp_fp_fn(
                torch.from_numpy(p),
                torch.from_numpy(t),
            )
            tp_sum[ci] += tp
            fp_sum[ci] += fp
            fn_sum[ci] += fn

            row += [
                model_metrics.precision(tp, fp, fn),
                model_metrics.recall(tp, fp, fn),
                model_metrics.IoU(tp, fp, fn),
            ]
        rows.append(row)

    # ------------------------------
    # Write CSV
    # ------------------------------
    cols = ["tile"]
    for cname in frame.mask_names:
        cols += [f"{cname}_precision", f"{cname}_recall", f"{cname}_IoU"]

    pd.DataFrame(rows, columns=cols).to_csv(
        output_dir / f"full_eval_epoch{epoch}.csv",
        index=False,
    )

    # ------------------------------
    # Log summary metrics
    # ------------------------------
    totals = []
    for ci in range(n_classes):
        tp, fp, fn = tp_sum[ci], fp_sum[ci], fn_sum[ci]
        prec = model_metrics.precision(tp, fp, fn)
        rec = model_metrics.recall(tp, fp, fn)
        iou = model_metrics.IoU(tp, fp, fn)
        totals.append((prec, rec, iou))

    for (prec, rec, iou), cname in zip(totals, frame.mask_names):
        writer.add_scalar(f"fulltest_precision/{cname}", prec, epoch)
        writer.add_scalar(f"fulltest_recall/{cname}", rec, epoch)
        writer.add_scalar(f"fulltest_iou/{cname}", iou, epoch)

    # ------------------------------
    # Full-tile visualization (first 4 tiles)
    # ------------------------------
    cmap = build_cmap(n_classes, is_binary, classname)
    NUM_SAMPLES = min(4, len(test_tiles))

    for idx, x_path in enumerate(test_tiles[:NUM_SAMPLES]):
        x_full = np.load(x_path)
        H, W, _ = x_full.shape

        y_pred, invalid_mask = frame.predict_slice(x_full, threshold)
        y_true_raw = np.load(
            x_path.with_name(x_path.name.replace("tiff", "mask"))
        ).astype(np.uint8)

        ignore = (y_true_raw == 255)
        if invalid_mask is not None:
            ignore |= invalid_mask

        # Visualization label maps (0/1/2/255)
        y_gt_vis = y_true_raw.copy()
        y_gt_vis[ignore] = 255

        y_pred_vis = y_pred.copy()
        y_pred_vis[ignore] = 255

        # Convert to RGB
        x_rgb = make_rgb_preview(x_full)
        gt_rgb = label_to_color(y_gt_vis, cmap)
        pr_rgb = label_to_color(y_pred_vis, cmap)

        # Confidence & entropy
        yhat_full = frame.act(
            frame.infer(
                torch.from_numpy(
                    np.expand_dims(
                        x_full[:, :, frame.loader_opts.use_channels], 0
                    )
                ).float()
            )
        ).cpu().numpy()[0]

        if is_binary:
            conf = yhat_full[..., 1]
        else:
            conf = np.max(yhat_full, axis=-1)

        conf_rgb = make_confidence_map(conf, invalid_mask=ignore)
        entropy_rgb = make_entropy_map(yhat_full, invalid_mask=ignore)

        # TP / FP / FN (non-background)
        tp_mask = (y_pred == y_true_raw) & (~ignore) & (y_true_raw != 0)
        fp_mask = (y_pred != y_true_raw) & (~ignore) & (y_pred != 0)
        fn_mask = (y_pred != y_true_raw) & (~ignore) & (y_true_raw != 0)

        tp_rgb, fp_rgb, fn_rgb = make_tp_fp_fn_masks(tp_mask, fp_mask, fn_mask)

        # ------ Compute metrics per tile ------
        metric_string_parts = []
        for ci, cname in enumerate(frame.mask_names):
            pred_c = (y_pred == ci).astype(np.uint8)
            true_c = (y_true_raw == ci).astype(np.uint8)

            tp, fp, fn = model_metrics.tp_fp_fn(
                torch.from_numpy(pred_c),
                torch.from_numpy(true_c),
            )

            P = model_metrics.precision(tp, fp, fn)
            R = model_metrics.recall(tp, fp, fn)
            I = model_metrics.IoU(tp, fp, fn)

            metric_string_parts.append(f"{cname}: P={P:.3f} R={R:.3f} IoU={I:.3f}")

        metrics_text = " | ".join(metric_string_parts)

        # ------ Build panel ------
        composite = make_eight_panel(
            x_rgb=x_rgb,
            gt_rgb=gt_rgb,
            pr_rgb=pr_rgb,
            conf_rgb=conf_rgb,
            tp_rgb=tp_rgb,
            fp_rgb=fp_rgb,
            fn_rgb=fn_rgb,
            entropy_rgb=entropy_rgb,
            metrics_text=metrics_text,
        )

        out_path = output_dir / f"fulltile_{idx}_epoch{epoch}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

        writer.add_image(
            f"fulltest/fulltile_{idx}",
            torch.tensor(composite).permute(2, 0, 1).float() / 255.0,
            epoch,
        )


# ============================================================
# Loss, metrics, utilities
# ============================================================
def get_loss(outchannels, opts=None):
    """
    Build loss function.
    For opts.name == "custom":
      - multi-class compatible
      - for binary (2 channels) only uses foreground channel 1.
    """
    if opts is None:
        return model_losses.customloss()

    ls = 0 if opts.label_smoothing == "None" else opts.label_smoothing
    name = opts.name

    if name == "custom":
        fg_classes = [1] if outchannels == 2 else list(range(1, outchannels))
        return model_losses.customloss(
            act=torch.nn.Softmax(dim=1),
            smooth=1.0,
            label_smoothing=ls,
            foreground_classes=fg_classes,
        )

    raise ValueError(f"Loss not recognized: {name}")


def get_metrics(tp, fp, fn, metric_names):
    metrics = {}
    for name in metric_names:
        metrics[name] = getattr(model_metrics, name)(tp, fp, fn)
    return metrics


def print_conf(conf):
    for k, v in conf.items():
        log(logging.INFO, f"{k} = {v}")


def print_metrics(frame, train_metric, val_metric, test_metric):
    def clean(metric_dict):
        return {
            cname: {m: float(metric_dict[m][i]) for m in frame.metrics_opts.metrics}
            for i, cname in enumerate(frame.mask_names)
        }

    log(logging.INFO, f"Train | {clean(train_metric)}")
    log(logging.INFO, f"Val   | {clean(val_metric)}")
    log(logging.INFO, f"Test  | {clean(test_metric)}\n")


def get_current_lr(frame):
    return float(frame.get_current_lr())


def find_lr(frame, train_loader, init_value, final_value):
    logs, losses = frame.find_lr(train_loader, init_value, final_value)
    plt.plot(logs, losses)
    plt.xlabel("LR (log scale)")
    plt.ylabel("loss")
    plt.savefig("lr_curve.png")
    print("LR curve saved")


def compute_dataset_stats(name, loader):
    total_counts = defaultdict(int)
    total_pixels = 0
    num_images = 0

    for x, y_onehot, y_int in loader:
        y = y_int.squeeze()
        unique, counts = np.unique(y.cpu().numpy(), return_counts=True)
        for cls, cnt in zip(unique.tolist(), counts.tolist()):
            total_counts[int(cls)] += int(cnt)
        total_pixels += y.numel()
        num_images += y.shape[0]

    stats = {}
    for cls in [0, 1, 2, 255]:
        cls_count = total_counts.get(cls, 0)
        stats[cls] = {
            "count": cls_count,
            "percent": (cls_count / total_pixels) * 100 if total_pixels > 0 else 0.0,
        }

    return {
        "dataset": name,
        "num_images": num_images,
        "total_pixels": total_pixels,
        "stats": {
            "BG (0)": stats[0],
            "CleanIce (1)": stats[1],
            "Debris (2)": stats[2],
            "Mask (255)": stats[255],
        },
    }


def print_stats_table(results):
    print("\n================ DATASET STATISTICS ================\n")

    for res in results:
        print(f"Dataset: {res['dataset']}")
        print(f"Images:  {res['num_images']}")
        print(f"Pixels:  {res['total_pixels']:,}\n")

        print(f"{'Class':<14}{'Count':>14}{'Percent':>12}")
        print("-" * 42)
        for cls, info in res["stats"].items():
            print(f"{cls:<14}{info['count']:>14}{info['percent']:>11.2f}%")
        print("")


def log_stats_tensorboard(writer: SummaryWriter, results):
    for res in results:
        prefix = f"dataset_stats/{res['dataset']}"
        for cls, info in res["stats"].items():
            cname = cls.replace(" ", "_").replace("(", "").replace(")", "")
            writer.add_scalar(f"{prefix}/{cname}_percent", info["percent"], 0)
            writer.add_scalar(f"{prefix}/{cname}_count", info["count"], 0)


def print_epoch_summary(epoch, train_metric, val_metric, test_metric, mask_names):
    def fmt(v):
        if isinstance(v, torch.Tensor):
            v = v.item()
        if isinstance(v, (float, np.floating)):
            return f"{v:.4f}"
        return str(v)

    print(f"\n===== Epoch {epoch} Summary =====")
    print("{:<8} {:<12} {:<10} {:<10} {:<10}".format(
        "Split", "Class", "Precision", "Recall", "IoU"
    ))
    print("-" * 54)

    for split, metrics in [
        ("Train", train_metric),
        ("Val",   val_metric),
        ("Test",  test_metric),
    ]:
        for i, cname in enumerate(mask_names):
            print("{:<8} {:<12} {:<10} {:<10} {:<10}".format(
                split,
                cname,
                fmt(metrics["precision"][i]),
                fmt(metrics["recall"][i]),
                fmt(metrics["IoU"][i]),
            ))
        print("-" * 25)
    print("")

