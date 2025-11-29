#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight utility module.

Everything related to model training, validation, loss computation,
and full-tile evaluation has been moved into Framework.

This file now provides:
 - logging helpers
 - TensorBoard helpers
 - LR finder wrapper
 - dataset statistics utilities
"""

import datetime
import logging
from collections import defaultdict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)


# ============================================================
# BASIC LOGGING UTILITIES
# ============================================================
def log(level, message):
    """
    Timestamped logger used throughout the project.
    """
    message = "{}\t{}   {}".format(
        datetime.datetime.now().strftime("%d-%m-%Y, %H:%M:%S"),
        logging._levelToName[level],
        message,
    )
    logging.log(level, "SystemLog: " + message)


def print_conf(conf):
    """
    Pretty-print config dictionary.
    """
    for k, v in conf.items():
        log(logging.INFO, f"{k} = {v}")


# ============================================================
# LR FINDER WRAPPER
# ============================================================
def find_lr(frame, train_loader, init_value=1e-9, final_value=1.0):
    """
    Thin wrapper around Framework.find_lr().
    Saves a learning-rate curve plot.
    """
    logs, losses = frame.find_lr(train_loader, init_value, final_value)

    plt.plot(logs, losses)
    plt.xlabel("Learning Rate (log10)")
    plt.ylabel("Loss")
    plt.title("LR Finder")
    plt.grid(True)
    plt.savefig("lr_curve.png")
    print("LR curve saved to lr_curve.png")

    return logs, losses


# ============================================================
# DATASET STATISTICS (OPTIONAL UTILITIES)
# ============================================================
def compute_dataset_stats(name, loader):
    """
    Compute per-class pixel statistics for a dataset.
    """
    total_counts = defaultdict(int)
    total_pixels = 0
    num_images = 0

    for _, _, y_int in loader:
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
            "percent": (cls_count / total_pixels) * 100 if total_pixels else 0.0,
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
    """
    Pretty-print dataset statistics to console.
    """
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
    """
    Write dataset statistics to TensorBoard.
    """
    for res in results:
        prefix = f"dataset_stats/{res['dataset']}"
        for cls, info in res["stats"].items():
            cname = cls.replace(" ", "_").replace("(", "").replace(")", "")
            writer.add_scalar(f"{prefix}/{cname}_percent", info["percent"], 0)
            writer.add_scalar(f"{prefix}/{cname}_count", info["count"], 0)


# ============================================================
# TRAINING SUMMARY (END-OF-EPOCH PRINTER)
# ============================================================
def print_epoch_summary(epoch, train_metric, val_metric, test_metric, mask_names):
    """
    End-of-epoch pretty table.
    """
    def fmt(v):
        if isinstance(v, torch.Tensor):
            v = v.item()
        return f"{float(v):.4f}"

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

