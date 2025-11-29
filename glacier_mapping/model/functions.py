#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training, validation, logging, and full-tile evaluation helpers.
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
from torchvision.utils import make_grid
from tqdm import tqdm

import glacier_mapping.model.losses as model_losses
import glacier_mapping.model.metrics as model_metrics

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)


# ============================================================
# ==================== GLOBAL COLOR CONSTANTS =================
# ============================================================

COLOR_BG = np.array([181, 101, 29], dtype=np.uint8)   # Brown
COLOR_CI = np.array([135, 206, 250], dtype=np.uint8)  # Light Blue
COLOR_DEB = np.array([255, 0, 0], dtype=np.uint8)     # Red
COLOR_IGNORE = np.array([0, 0, 0], dtype=np.uint8)    # Black

GLOBAL_CMAP = {
    0: COLOR_BG,
    1: COLOR_CI,
    2: COLOR_DEB,
    255: COLOR_IGNORE,
}


# -----------------------------------------
# Logging helper with timestamp
# -----------------------------------------
def log(level, message):
    message = "{}\t{}\t{}".format(
        datetime.datetime.now().strftime("%d-%m-%Y, %H:%M:%S"),
        logging._levelToName[level],
        message,
    )
    message = "SystemLog: " + message
    logging.log(level, message)


# ============================================================
# =============== TRAINING + VALIDATION =======================
# ============================================================

def train_epoch(epoch, loader, frame):
    metrics = frame.metrics_opts.metrics
    n_classes = frame.num_classes
    threshold = frame.metrics_opts.threshold

    loss = 0.0
    tp = torch.zeros(n_classes)
    fp = torch.zeros(n_classes)
    fn = torch.zeros(n_classes)

    iterator = tqdm(loader, desc="Train Iter")

    for i, (x, y_onehot, y_int) in enumerate(iterator):
        frame.zero_grad()

        y_hat, batch_loss = frame.optimize(x, y_onehot, y_int.squeeze(-1))
        frame.step()

        batch_loss_f = float(batch_loss.detach())
        loss += batch_loss_f

        y_hat = frame.act(y_hat)

        ignore = (y_int.squeeze(-1) == 255).cpu().numpy()
        _tp, _fp, _fn = frame.metrics(y_hat, y_onehot, ignore, threshold)
        tp += _tp
        fp += _fp
        fn += _fn

        iterator.set_description(
            f"Train, Epoch={epoch}, Step={i}, "
            f"Loss={batch_loss_f:.3f}, Avg={loss/(i+1):.3f}"
        )

    metrics = get_metrics(tp, fp, fn, metrics)
    return loss / (i + 1), metrics, frame.get_loss_alpha()


def validate(epoch, loader, frame, test=False):
    """
    Validation / test pass with IGNORE-AWARE metrics.

    - Skips fully ignored patches (all pixels 255)
    - Computes metrics only on valid (non-255) pixels
    """
    metrics = frame.metrics_opts.metrics
    n_classes = frame.num_classes

    total_loss = 0.0
    count_batches = 0

    tp = torch.zeros(n_classes)
    fp = torch.zeros(n_classes)
    fn = torch.zeros(n_classes)

    desc = "Test Iter" if test else "Val Iter"
    iterator = tqdm(loader, desc=desc)

    def channel_first(x):  # NHWC → NCHW
        return x.permute(0, 3, 1, 2)

    for i, (x, y_onehot, y_int) in enumerate(iterator):
        # Forward
        y_hat = frame.infer(x)
        batch_loss = frame.calc_loss(
            channel_first(y_hat),
            channel_first(y_onehot),
            y_int.squeeze(-1),
        )
        batch_loss_f = float(batch_loss.detach())

        # Apply activation
        y_hat = frame.act(y_hat)

        ignore = (y_int.squeeze(-1) == 255).cpu()  # (B,H,W) bool

        # Skip if everything is ignored
        if ignore.all():
            continue

        # From one-hot to class index
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
            f"{desc}, Epoch={epoch}, Step={i}, "
            f"Loss={batch_loss_f:.3f}, Avg={total_loss/max(count_batches,1):.3f}"
        )

    if count_batches == 0:
        print(f"[WARN] {desc}: All patches were ignored (255). Returning zeros.")
        return 0.0, get_metrics(tp, fp, fn, metrics)

    avg_loss = total_loss / count_batches

    if not test:
        frame.val_operations(avg_loss)

    return avg_loss, get_metrics(tp, fp, fn, metrics)


# ============================================================
# =====================  LOG METRICS  =========================
# ============================================================

def log_metrics(writer, frame, metrics, epoch, stage):
    for key, vals in metrics.items():
        for name, metric in zip(frame.mask_names, vals):
            writer.add_scalar(f"{stage}_{key}/{name}", metric, epoch)


# ============================================================
# =====================  LOG IMAGES  ==========================
# ============================================================

def log_images(writer, frame, batch, epoch, stage, normalize):
    """
    Logs slice-level visualization:
      - Input x (once)
      - GT mask
      - Predicted mask
      - Side-by-side GT vs Pred
      - Overlay (input + pred)
    """
    log_once = (epoch == 1)
    normalize_name = frame.loader_opts.normalize

    x, y_onehot, y_int = next(iter(batch))
    y_hat = frame.act(frame.infer(x))

    x_np = x.cpu().numpy()
    y_np = y_onehot.cpu().numpy()
    yhat_np = y_hat.cpu().numpy()
    yint_np = y_int.cpu().numpy()

    y_cls = np.argmax(y_np, axis=3)
    yhat_cls = np.argmax(yhat_np, axis=3)

    ignore_mask = (yint_np == 255).squeeze(-1)
    y_cls[ignore_mask] = 255
    yhat_cls[ignore_mask] = 255

    cmap = GLOBAL_CMAP

    B, H, W = y_cls.shape
    y_rgb = np.zeros((B, H, W, 3), dtype=np.uint8)
    yhat_rgb = np.zeros((B, H, W, 3), dtype=np.uint8)

    for cls, col in cmap.items():
        y_rgb[y_cls == cls] = col
        yhat_rgb[yhat_cls == cls] = col

    if normalize_name == "mean-std":
        mean, std = normalize
        x_np = (x_np * std) + mean
    else:
        x_np = np.clip(x_np, 0, 1)

    x_np = x_np[..., [2, 1, 0]]  # BGR→RGB convention

    x_t = torch.tensor(x_np).float()
    y_t = torch.tensor(y_rgb).float() / 255.0
    yhat_t = torch.tensor(yhat_rgb).float() / 255.0

    # Ensure NHWC before concat
    if y_t.dim() == 4 and y_t.shape[1] == 3:
        y_t = y_t.permute(0, 2, 3, 1)
        yhat_t = yhat_t.permute(0, 2, 3, 1)

    side_by_side = torch.cat([y_t, yhat_t], dim=2)
    overlay = (0.6 * (x_t / 255.0) + 0.4 * yhat_t).clamp(0, 1)

    pm = lambda t: t.permute(0, 3, 1, 2)

    if log_once:
        writer.add_image(f"{stage}/x", make_grid(pm(x_t / 255.0)), epoch)
        writer.add_image(f"{stage}/y_gt", make_grid(pm(y_t)), epoch)

    writer.add_image(f"{stage}/y_pred", make_grid(pm(yhat_t)), epoch)
    writer.add_image(f"{stage}/gt_vs_pred", make_grid(pm(side_by_side)), epoch)
    writer.add_image(f"{stage}/overlay", make_grid(pm(overlay)), epoch)


# ============================================================
# ====================  FULL TILE EVAL  =======================
# ============================================================

def evaluate_full_test_tiles(frame, writer, epoch, output_dir):
    """
    Full-image evaluation with TensorBoard visualization.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(frame.loader_opts.processed_dir)
    test_tiles = sorted((data_dir / "test").glob("tiff*"))

    n_classes = frame.num_classes
    mask_names = frame.mask_names
    threshold = frame.metrics_opts.threshold

    tp_sum = np.zeros(n_classes)
    fp_sum = np.zeros(n_classes)
    fn_sum = np.zeros(n_classes)

    rows = []
    print(f"[Full Eval] Processing {len(test_tiles)} full tiles...")

    for x_path in tqdm(test_tiles):
        x = np.load(x_path)
        y_pred, invalid_mask = frame.predict_slice(x, threshold)

        y_true_raw = np.load(
            x_path.parent / x_path.name.replace("tiff", "mask")
        ).astype(np.uint8)

        ignore_mask = (y_true_raw == 255)
        if invalid_mask is not None:
            ignore_mask |= invalid_mask

        y_true = y_true_raw.copy()
        y_true[ignore_mask] = 0
        valid = ~ignore_mask

        y_true[valid] += 1

        y_pred_valid = y_pred[valid]
        y_true_valid = y_true[valid]

        row = [str(x_path.name)]

        for ci in range(n_classes):
            label = ci + 1
            p = (y_pred_valid == label).astype(np.uint8)
            t = (y_true_valid == label).astype(np.uint8)

            tp, fp, fn = model_metrics.tp_fp_fn(
                torch.from_numpy(p), torch.from_numpy(t)
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

    # ---- TOTAL ROW ----
    total = ["Total"]
    totals_per_class = []

    for ci in range(n_classes):
        tp, fp, fn = tp_sum[ci], fp_sum[ci], fn_sum[ci]
        prec = model_metrics.precision(tp, fp, fn)
        rec = model_metrics.recall(tp, fp, fn)
        iou = model_metrics.IoU(tp, fp, fn)

        total += [prec, rec, iou]
        totals_per_class.append((prec, rec, iou))

    rows.append(total)

    print(f"\n===== Full-Tile Test Metrics (Epoch {epoch}) =====")
    print("{:<12} {:<10} {:<10} {:<10}".format("Class", "Precision", "Recall", "IoU"))
    print("-" * 48)
    for cname, (prec, rec, iou) in zip(mask_names, totals_per_class):
        print("{:<12} {:<10.4f} {:<10.4f} {:<10.4f}".format(cname, prec, rec, iou))
    print("-" * 48 + "\n")

    cols = ["tile"]
    for cname in mask_names:
        cols += [f"{cname}_precision", f"{cname}_recall", f"{cname}_IoU"]

    pd.DataFrame(rows, columns=cols).to_csv(
        output_dir / f"full_eval_epoch{epoch}.csv", index=False
    )

    for (prec, rec, iou), cname in zip(totals_per_class, mask_names):
        writer.add_scalar(f"fulltest_precision/{cname}", prec, epoch)
        writer.add_scalar(f"fulltest_recall/{cname}", rec, epoch)
        writer.add_scalar(f"fulltest_iou/{cname}", iou, epoch)

    # Visualization: a few samples
    try:
        NUM_SAMPLES = min(5, len(test_tiles))
        for idx, sample_path in enumerate(test_tiles[:NUM_SAMPLES]):
            x_full = np.load(sample_path)
            x_rgb = x_full[..., [2, 1, 0]].astype(np.float32) / 255.0

            y_pred, _ = frame.predict_slice(x_full, threshold)
            y_gt = np.load(
                sample_path.parent / sample_path.name.replace("tiff", "mask")
            ).astype(np.uint8)

            pred_rgb = np.zeros((*y_pred.shape, 3), dtype=np.uint8)
            gt_rgb = np.zeros((*y_gt.shape, 3), dtype=np.uint8)

            for k, col in GLOBAL_CMAP.items():
                pred_rgb[y_pred == k] = col
                gt_rgb[y_gt == k] = col

            x_t = torch.tensor(x_rgb).permute(2, 0, 1).float()
            ygt_t = torch.tensor(gt_rgb).permute(2, 0, 1).float() / 255.0
            ypred_t = torch.tensor(pred_rgb).permute(2, 0, 1).float() / 255.0

            side_by_side = torch.cat([ygt_t, ypred_t], dim=2)
            overlay = (0.6 * x_t + 0.4 * ypred_t).clamp(0, 1)

            base = f"fulltest/sample_{idx}"
            writer.add_image(f"{base}_input", x_t, epoch)
            writer.add_image(f"{base}_gt", ygt_t, epoch)
            writer.add_image(f"{base}_pred", ypred_t, epoch)
            writer.add_image(f"{base}_gt_vs_pred", side_by_side, epoch)
            writer.add_image(f"{base}_overlay", overlay, epoch)

    except Exception as e:
        print(f"[Full Eval] Visualization error: {e}")


# ============================================================
# ==================== MISC HELPERS ==========================
# ============================================================

def get_loss(outchannels, opts=None):
    """
    Build loss function.

    For opts.name == "custom":
      - multi-class compatible
      - but for binary (2 channels) it only uses foreground channel (index 1),
        mimicking original CleanIce training.
    """
    if opts is None:
        return model_losses.customloss()

    ls = 0 if opts.label_smoothing == "None" else opts.label_smoothing
    name = opts.name

    if name == "custom":
        # For binary output (BG vs FG), only channel 1 is foreground
        if outchannels == 2:
            fg_classes = [1]
        else:
            # For multi-class, treat everything except background (0) as foreground
            fg_classes = list(range(1, outchannels))

        return model_losses.customloss(
            act=torch.nn.Softmax(dim=1),
            smooth=1.0,
            label_smoothing=ls,
            foreground_classes=fg_classes,
        )

    raise ValueError(f"Loss function not recognized: {name}")


def get_metrics(tp, fp, fn, metric_names):
    metrics = {}
    for name in metric_names:
        fun = getattr(model_metrics, name)
        metrics[name] = fun(tp, fp, fn)
    return metrics


def print_conf(conf):
    for k, v in conf.items():
        log(logging.INFO, f"{k} = {v}")


def print_metrics(frame, train_metric, val_metric, test_metric):
    def clean(mdict):
        return {
            cname: {m: float(mdict[m][i]) for m in frame.metrics_opts.metrics}
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
    def fmt(val):
        if isinstance(val, torch.Tensor):
            val = val.item()
        if isinstance(val, (float, np.floating)):
            return f"{val:.4f}"
        return str(val)

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

