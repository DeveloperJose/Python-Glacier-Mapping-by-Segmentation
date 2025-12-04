"""Analysis utilities for glacier mapping."""

import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl


def find_lr(
    module: pl.LightningModule, train_loader, init_value: float, final_value: float
) -> Tuple[List[float], List[float]]:
    """Learning rate finder.

    Args:
        module: Lightning module
        train_loader: Training data loader
        init_value: Initial learning rate
        final_value: Final learning rate

    Returns:
        Tuple of (log_learning_rates, losses)
    """
    number_in_epoch = len(train_loader) - 1
    update_step = (final_value / init_value) ** (1 / number_in_epoch)
    lr = init_value

    # Use Lightning module's optimizer
    optimizer = module.optimizers()
    optimizer.param_groups[0]["lr"] = lr

    best_loss = 0.0
    batch_num = 0
    losses = []
    log_lrs = []

    for i, data in enumerate(tqdm(train_loader, desc="LR Finder")):
        batch_num += 1
        inputs, labels = data
        optimizer.zero_grad()

        # Use Lightning module forward
        outputs = module(inputs.permute(0, 3, 1, 2))
        loss = module.compute_loss(
            outputs, labels.permute(0, 3, 1, 2), labels.squeeze(-1)
        )

        if batch_num > 1 and loss > 4 * best_loss:
            return log_lrs[10:-5], losses[10:-5]
        if loss < best_loss or batch_num == 1:
            best_loss = loss
            best_lr = lr

        loss.backward()
        optimizer.step()

        losses.append(loss.detach())
        log_lrs.append(math.log10(lr))
        lr = lr * update_step
        optimizer.param_groups[0]["lr"] = lr

    return log_lrs[10:-5], losses[10:-5]


def lr_finder(
    module: pl.LightningModule,
    train_loader,
    init_value: float = 1e-9,
    final_value: float = 1.0,
) -> Tuple[List[float], List[float]]:
    """Thin wrapper around find_lr(). Produces an LR curve plot.

    Args:
        module: Lightning module
        train_loader: Training data loader
        init_value: Initial learning rate
        final_value: Final learning rate

    Returns:
        Tuple of (log_learning_rates, losses)
    """
    logs, losses = find_lr(module, train_loader, init_value, final_value)

    plt.figure(figsize=(8, 5))
    plt.plot(logs, losses)
    plt.xlabel("Learning Rate (log10)")
    plt.ylabel("Loss")
    plt.title("LR Finder")
    plt.grid(True)
    plt.savefig("lr_curve.png")
    print("LR curve saved to lr_curve.png")

    return logs, losses


def compute_dataset_stats(dataloader, name: str) -> Dict[str, Any]:
    """Compute dataset statistics.

    Args:
        dataloader: Data loader
        name: Dataset name

    Returns:
        Statistics dictionary
    """
    total_counts = defaultdict(int)
    total_pixels = 0
    num_images = 0

    for _, _, y_int in dataloader:
        y = y_int.squeeze()  # (H,W)
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


def print_stats_table(results: List[Dict[str, Any]]) -> None:
    """Print dataset statistics table.

    Args:
        results: List of statistics dictionaries
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


def log_stats_tensorboard(writer, results: List[Dict[str, Any]]) -> None:
    """Write dataset statistics to TensorBoard.

    Args:
        writer: TensorBoard writer
        results: List of statistics dictionaries
    """
    for res in results:
        prefix = f"dataset_stats/{res['dataset']}"
        for cls, info in res["stats"].items():
            cname = cls.replace(" ", "_").replace("(", "").replace(")", "")
            writer.add_scalar(f"{prefix}/{cname}_percent", info["percent"], 0)
            writer.add_scalar(f"{prefix}/{cname}_count", info["count"], 0)
