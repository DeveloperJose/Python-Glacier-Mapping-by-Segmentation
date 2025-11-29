#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a U-Net model on the prepared glacier dataset.

Includes:
 - AMP + sigma-weighted custom loss
 - OneCycleLR or ReduceLROnPlateau (configured in YAML)
 - Dataset statistics (optional)
 - Per-epoch metrics
 - Full-tile evaluation with unified 8-panel visuals
 - Best-epoch metrics printed at end
 - notify-send desktop notification (Linux)
"""

import gc
import json
import logging
import pathlib
import random
import subprocess
import warnings
from timeit import default_timer as timer

import numpy as np
import torch
import yaml
from addict import Dict
from torch.utils.tensorboard import SummaryWriter

import glacier_mapping.model.functions as fn
from glacier_mapping.model.frame import Framework
from glacier_mapping.data.data import fetch_loaders

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
random.seed(41)
np.random.seed(41)
torch.manual_seed(41)
torch.cuda.manual_seed(41)
torch.cuda.manual_seed_all(41)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":

    # ------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------
    conf = Dict(yaml.safe_load(open("./conf/unet_train.yaml")))

    run_name: str = conf.training_opts.run_name
    output_dir = pathlib.Path(conf.training_opts.output_dir) / run_name
    model_output_dir = output_dir / "models"
    early_stopping: int = conf.training_opts.early_stopping

    full_eval_every: int = int(getattr(conf.training_opts, "full_eval_every", 5))

    # Sanity check: physics model
    if (
        "phys" in run_name
        and conf.loader_opts.physics_channel not in conf.loader_opts.use_channels
    ):
        raise ValueError("Training a phys model but physics channel is missing.")

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------
    train_loader, val_loader, test_loader = fetch_loaders(**conf.loader_opts)

    # ------------------------------------------------------------
    # Create training framework
    # ------------------------------------------------------------
    frame = Framework.from_config("./conf/unet_train.yaml")

    # Optional: fine-tuning
    if conf.training_opts.fine_tune:
        fn.log(logging.INFO, "Finetuning from previous final model…")
        final_model_path = output_dir / "models" / "model_final.pt"

        frame = Framework.from_checkpoint(
            final_model_path,
            testing=False,
        )
        frame.freeze_layers()

    # Optional: LR Finder
    if conf.training_opts.find_lr:
        frame.lr_finder(train_loader, init_value=1e-9, final_value=1.0)
        raise SystemExit

    # ------------------------------------------------------------
    # Prepare directories
    # ------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # TensorBoard
    # ------------------------------------------------------------
    writer = SummaryWriter(output_dir / "logs")
    writer.add_text("Configuration", json.dumps(conf, indent=4))

    fn.print_conf(conf)
    fn.log(
        logging.INFO,
        f"#Train={len(train_loader)}, #Val={len(val_loader)}, #Test={len(test_loader)}",
    )

    # Save conf
    with open(output_dir / "conf.json", "w") as f:
        json.dump(conf, f, indent=4, sort_keys=True)

    # Load normalization info for logging thumbnails (still passed into log_images)
    norm_path = pathlib.Path(conf.loader_opts.processed_dir) / "normalize_train.npy"
    _normalize = np.load(norm_path)
    if conf.loader_opts.normalize == "min-max":
        _normalize = (
            _normalize[2][conf.loader_opts.use_channels],
            _normalize[3][conf.loader_opts.use_channels],
        )
    else:
        _normalize = (
            np.append(_normalize[0], 0)[conf.loader_opts.use_channels],
            np.append(_normalize[1], 1)[conf.loader_opts.use_channels],
        )

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    best_val_loss = np.inf
    best_epoch = None
    best_train_metric = None
    best_val_metric = None
    epochs_without_improvement = 0

    # Track all improvements
    improvement_checkpoints = []  # List of dicts: {epoch, val_loss, train_metric, val_metric}

    start_time = timer()
    final_epoch = 0

    for epoch in range(1, conf.training_opts.epochs + 1):

        # ------------------------ TRAIN ------------------------
        loss_train, train_metric, loss_alpha = frame.train_one_epoch(
            epoch, train_loader
        )
        frame.log_metrics(writer, train_metric, epoch, "train")

        # ------------------------ VALIDATE ---------------------
        loss_val, val_metric = frame.validate_one_epoch(
            epoch, val_loader, test=False
        )
        frame.log_metrics(writer, val_metric, epoch, "val")

        # ------------------------ LOG IMAGES -------------------
        if (epoch - 1) % 5 == 0:
            frame.log_images(writer, train_loader, epoch, "train", _normalize)
            frame.log_images(writer, val_loader, epoch, "val", _normalize)

        # ------------------------ FULL TILE EVAL ---------------
        if epoch % full_eval_every == 0:
            fn.log(logging.INFO, f"Full-tile eval at epoch {epoch}…")
            frame.evaluate_full_test_tiles(
                writer=writer,
                epoch=epoch,
                output_dir=output_dir / "full_eval",
            )

        # ------------------------ LOG SCALARS -------------------
        writer.add_scalar("loss_train", loss_train, epoch)
        writer.add_scalar("loss_val", loss_val, epoch)
        writer.add_scalar("lr", frame.get_current_lr(), epoch)

        for idx, sigma in enumerate(loss_alpha):
            writer.add_scalar(f"sigma/{idx + 1}", sigma, epoch)

        # Print epoch summary (now using Framework method)
        frame.print_epoch_summary(
            epoch,
            train_metric=train_metric,
            val_metric=val_metric,
            # test_metric=val_metric,  # formatting only
        )

        # ------------------------ CHECKPOINT MANAGEMENT ----------------
        # Save checkpoint if validation loss improved
        if loss_val < best_val_loss:
            best_val_loss = float(loss_val)
            best_epoch = epoch
            best_train_metric = train_metric
            best_val_metric = val_metric
            epochs_without_improvement = 0

            # Save two checkpoints:
            # 1. Always save as "best" (overwrites previous best)
            frame.save(model_output_dir, "best")

            # 2. Save with epoch and val_loss for historical tracking
            frame.save_improvement(model_output_dir, epoch, float(loss_val))

            # Track this improvement
            checkpoint_data = {
                "epoch": epoch,
                "val_loss": float(loss_val),
                "train_metric": train_metric,
                "val_metric": val_metric
            }
            improvement_checkpoints.append(checkpoint_data)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping:
            fn.log(
                logging.INFO,
                f"Early stopping at epoch {epoch} (best={best_val_loss:.4f})",
            )
            break

        # housekeeping
        torch.cuda.empty_cache()
        gc.collect()
        writer.flush()
        final_epoch = epoch

    # ------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------
    frame.save(model_output_dir, "final")
    writer.close()
    elapsed = timer() - start_time

    fn.log(logging.INFO, f"Finished training {run_name} in {elapsed:.2f}s")

    # ------------------------------------------------------------
    # Print IMPROVEMENTS SUMMARY
    # ------------------------------------------------------------
    print(f"\n================ CHECKPOINT IMPROVEMENTS ================\n")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Total improvements saved: {len(improvement_checkpoints)}\n")
    print("{:<8} {:<12} {:<10} {:<10} {:<10}".format(
    "Epoch", "Val Loss", "Precision", "Recall", "IoU"))
    print("-" * 58)

    def to_float(x):
        if isinstance(x, torch.Tensor):
            return float(x.detach().cpu().item())
        return float(x)

    for checkpoint in improvement_checkpoints:
        epoch = checkpoint["epoch"]
        val_loss = checkpoint["val_loss"]
        val_metric = checkpoint["val_metric"]

        if isinstance(val_metric["precision"], list):
            p = to_float(val_metric["precision"][0])
            r = to_float(val_metric["recall"][0])
            iou = to_float(val_metric["IoU"][0])
        else:
            p = to_float(val_metric["precision"])
            r = to_float(val_metric["recall"])
            iou = to_float(val_metric["IoU"])

        print(f"{epoch:<8} {val_loss:<12.6f} {p:<10.4f} {r:<10.4f} {iou:<10.4f}")
    print("-" * 58 + "\n")

    # ------------------------------------------------------------
    # Print BEST-EPOCH SUMMARY (detailed)
    # ------------------------------------------------------------
    if best_val_metric is not None:
        print("\n================ BEST EPOCH DETAILED ================\n")
        print("{:<12} {:<10} {:<10} {:<10}".format("Class", "Precision", "Recall", "IoU"))
        print("-" * 48)
        for cname, (p, r, i) in zip(
            frame.mask_names,
            zip(
                best_val_metric["precision"],
                best_val_metric["recall"],
                best_val_metric["IoU"],
            ),
        ):
            print(f"{cname:<12} {p:.4f}     {r:.4f}     {i:.4f}")
        print("-" * 48 + "\n")

    # ------------------------------------------------------------
    # Save checkpoints summary
    # ------------------------------------------------------------
    checkpoints_summary = {
        "total_epochs_trained": final_epoch,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_improvements": len(improvement_checkpoints),
        "improvement_checkpoints": improvement_checkpoints
    }
    with open(output_dir / "checkpoints_summary.json", "w") as f:
        json.dump(checkpoints_summary, f, indent=4)

    # ------------------------------------------------------------
    # Desktop Notification (Linux)
    # ------------------------------------------------------------
    try:
        import shutil

        if shutil.which("notify-send") is not None:
            subprocess.run(
                ["notify-send", "Training Completed", f"Run {run_name} finished."],
                check=False,
            )
    except Exception:
        pass

    print(f"Training run complete: {run_name}")

