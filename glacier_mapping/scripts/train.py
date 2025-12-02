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

import argparse
import gc
import json
import logging
import pathlib
import random
import re
import subprocess
import warnings
from timeit import default_timer as timer

import numpy as np
import torch
import yaml
from addict import Dict
from torch.utils.tensorboard import SummaryWriter

import mlflow
import glacier_mapping.utils.logging as fn
from glacier_mapping.core.frame import Framework
from glacier_mapping.data.data import fetch_loaders

random.seed(41)
np.random.seed(41)
torch.manual_seed(41)
torch.cuda.manual_seed(41)
torch.cuda.manual_seed_all(41)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Path to experiment config YAML (e.g., conf/experiments/exp_001_baseline_ci_v2.yaml)",
    )
    parser.add_argument(
        "--config-dict",
        help="Config dict as JSON string (alternative to --config)",
    )
    args = parser.parse_args()

    # Load experiment config from file or dict
    if args.config_dict:
        conf = Dict(json.loads(args.config_dict))
    elif args.config:
        config_path = args.config
        conf = Dict(yaml.safe_load(open(config_path)))
    else:
        parser.error("Either --config or --config-dict must be provided")

    run_name: str = conf.training_opts.run_name
    output_dir = pathlib.Path(conf.training_opts.output_dir)
    model_output_dir = output_dir / "models"
    early_stopping: int = conf.training_opts.early_stopping

    full_eval_every: int = int(getattr(conf.training_opts, "full_eval_every", 5))
    num_viz_samples: int = int(getattr(conf.training_opts, "num_viz_samples", 4))

    train_loader, val_loader, test_loader = fetch_loaders(**conf.loader_opts)

    # Check for existing checkpoints to resume from
    start_epoch = 1
    resume_checkpoint = None

    # Look for existing checkpoints in output directory
    if output_dir.exists():
        models_dir = output_dir / "models"
        if models_dir.exists():
            # Priority 1: Check for best model
            best_model_path = models_dir / "model_best.pt"
            if best_model_path.exists():
                resume_checkpoint = best_model_path
                fn.log(logging.INFO, f"Found best checkpoint: {best_model_path}")
            else:
                # Priority 2: Find the latest epoch checkpoint
                epoch_checkpoints = list(models_dir.glob("model_*.pt"))
                # Filter out improvement checkpoints (model_epochXXXX_valXXXXXX.pt)
                epoch_checkpoints = [
                    cp
                    for cp in epoch_checkpoints
                    if not re.search(r"model_epoch\d+_val", cp.name)
                ]

                if epoch_checkpoints:
                    # Sort by epoch number
                    def get_epoch_from_path(path):
                        match = re.search(r"model_(\d+)\.pt", path.name)
                        return int(match.group(1)) if match else 0

                    epoch_checkpoints.sort(key=get_epoch_from_path)
                    resume_checkpoint = epoch_checkpoints[-1]
                    fn.log(
                        logging.INFO,
                        f"Found latest epoch checkpoint: {resume_checkpoint}",
                    )

    # Initialize framework from experiment config
    if resume_checkpoint and not conf.training_opts.get("disable_resume", False):
        fn.log(logging.INFO, f"Resuming training from checkpoint: {resume_checkpoint}")
        frame = Framework.from_checkpoint(resume_checkpoint, testing=False)

        # Extract the epoch from checkpoint
        if resume_checkpoint.name == "model_best.pt":
            # For best model, we need to check the checkpoints summary to get the epoch
            summary_path = output_dir / "checkpoints_summary.json"
            if summary_path.exists():
                with open(summary_path, "r") as f:
                    summary = json.load(f)
                    start_epoch = summary.get("best_epoch", 1) + 1
                    fn.log(
                        logging.INFO,
                        f"Resuming from epoch {start_epoch} (best model was epoch {summary.get('best_epoch', '?')})",
                    )
            else:
                start_epoch = 1
                fn.log(
                    logging.WARNING,
                    "Could not determine epoch from best model, starting from epoch 1",
                )
        else:
            # Extract epoch from filename like "model_15.pt"
            epoch_match = re.search(r"model_(\d+)\.pt", resume_checkpoint.name)
            if epoch_match:
                start_epoch = int(epoch_match.group(1)) + 1
                fn.log(
                    logging.INFO,
                    f"Resuming from epoch {start_epoch} (checkpoint was epoch {epoch_match.group(1)})",
                )
            else:
                start_epoch = 1
                fn.log(
                    logging.WARNING,
                    "Could not determine epoch from checkpoint, starting from epoch 1",
                )
    else:
        # Fresh start
        if args.config_dict:
            frame = Framework.from_dict(conf)
        else:
            frame = Framework.from_config(args.config)

        if resume_checkpoint and conf.training_opts.get("disable_resume", False):
            fn.log(logging.INFO, "Resume disabled by config, starting fresh training")

    if conf.training_opts.fine_tune:
        fn.log(logging.INFO, "Finetuning from previous final model…")
        final_model_path = output_dir / "models" / "model_final.pt"

        frame = Framework.from_checkpoint(
            final_model_path,
            testing=False,
        )
        frame.freeze_layers()

    if conf.training_opts.find_lr:
        frame.lr_finder(train_loader, init_value=1e-9, final_value=1.0)
        raise SystemExit

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
    # Load existing metrics if resuming
    if resume_checkpoint and start_epoch > 1:
        summary_path = output_dir / "checkpoints_summary.json"
        if summary_path.exists():
            with open(summary_path, "r") as f:
                summary = json.load(f)
                best_val_loss = summary.get("best_val_loss", np.inf)
                best_epoch = summary.get("best_epoch", None)
                # Note: We don't restore the detailed metrics to avoid complexity
                best_train_metric = None
                best_val_metric = None
                epochs_without_improvement = 0
                improvement_checkpoints = summary.get("improvement_checkpoints", [])
                fn.log(
                    logging.INFO,
                    f"Loaded training history: best_val_loss={best_val_loss:.6f}, best_epoch={best_epoch}",
                )
        else:
            # Reset metrics if no summary found
            best_val_loss = np.inf
            best_epoch = None
            best_train_metric = None
            best_val_metric = None
            epochs_without_improvement = 0
            improvement_checkpoints = []
            fn.log(logging.WARNING, "No training history found, starting fresh metrics")
    else:
        # Fresh start
        best_val_loss = np.inf
        best_epoch = None
        best_train_metric = None
        best_val_metric = None
        epochs_without_improvement = 0
        improvement_checkpoints = []  # List of dicts: {epoch, val_loss, train_metric, val_metric}

    start_time = timer()
    final_epoch = 0

    fn.log(
        logging.INFO,
        f"Starting training from epoch {start_epoch} to {conf.training_opts.epochs}",
    )
    for epoch in range(start_epoch, conf.training_opts.epochs + 1):
        # ------------------------ TRAIN ------------------------
        loss_train, train_metric, loss_alpha = frame.train_one_epoch(
            epoch, train_loader
        )
        frame.log_metrics(writer, train_metric, epoch, "train")

        # ------------------------ VALIDATE ---------------------
        loss_val, val_metric = frame.validate_one_epoch(epoch, val_loader, test=False)
        frame.log_metrics(writer, val_metric, epoch, "val")

        # ------------------------ LOG IMAGES -------------------
        if (epoch - 1) % 5 == 0:
            frame.log_images(
                writer, train_loader, epoch, "train", _normalize, num_viz_samples
            )
            frame.log_images(
                writer, val_loader, epoch, "val", _normalize, num_viz_samples
            )

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
                "val_metric": val_metric,
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
    print("\n================ CHECKPOINT IMPROVEMENTS ================\n")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Total improvements saved: {len(improvement_checkpoints)}\n")
    print(
        "{:<8} {:<12} {:<10} {:<10} {:<10}".format(
            "Epoch", "Val Loss", "Precision", "Recall", "IoU"
        )
    )
    print("-" * 58)

    def to_float(x):
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return float(x.detach().cpu().item())
            else:
                return x.detach().cpu().tolist()
        return float(x)

    for checkpoint in improvement_checkpoints:
        epoch = checkpoint["epoch"]
        val_loss = checkpoint["val_loss"]
        val_metric = checkpoint["val_metric"]

        precision_vals = to_float(val_metric["precision"])
        recall_vals = to_float(val_metric["recall"])
        iou_vals = to_float(val_metric["IoU"])

        # Use first class values for summary display
        # Extract first value for display (works for scalars, lists, and tensors)
        def get_first(val):  # type: ignore
            if isinstance(val, (int, float)):
                return float(val)
            elif isinstance(val, (list, tuple)) and len(val) > 0:
                return float(val[0])  # type: ignore
            else:
                # Assume it's indexable (tensor, array, etc.)
                try:
                    return float(val[0])  # type: ignore
                except (TypeError, IndexError):
                    return float(val)  # type: ignore

        p = get_first(precision_vals)
        r = get_first(recall_vals)
        iou = get_first(iou_vals)

        print(f"{epoch:<8} {val_loss:<12.6f} {p:<10.4f} {r:<10.4f} {iou:<10.4f}")
    print("-" * 58 + "\n")

    # ------------------------------------------------------------
    # Print BEST-EPOCH SUMMARY (detailed)
    # ------------------------------------------------------------
    if best_val_metric is not None:
        print("\n================ BEST EPOCH DETAILED ================\n")
        print(
            "{:<12} {:<10} {:<10} {:<10}".format("Class", "Precision", "Recall", "IoU")
        )
        print("-" * 48)
        for cname, (p, r, i) in zip(
            frame.mask_names,
            zip(
                best_val_metric["precision"],
                best_val_metric["recall"],
                best_val_metric["IoU"],
            ),
        ):
            p_float = float(p.detach().cpu().item())
            r_float = float(r.detach().cpu().item())
            i_float = float(i.detach().cpu().item())
            print(f"{cname:<12} {p_float:.4f}     {r_float:.4f}     {i_float:.4f}")
        print("-" * 48 + "\n")

    # ------------------------------------------------------------
    # Save checkpoints summary
    # ------------------------------------------------------------
    def convert_tensors_to_python(obj):
        """Recursively convert tensors in nested dicts/lists to Python types."""
        if isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return float(obj.detach().cpu().item())
            else:
                return obj.detach().cpu().tolist()
        elif isinstance(obj, dict):
            return {k: convert_tensors_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensors_to_python(item) for item in obj]
        else:
            return obj

    # Convert all tensors in improvement_checkpoints to Python types
    serializable_checkpoints = [
        convert_tensors_to_python(checkpoint) for checkpoint in improvement_checkpoints
    ]

    checkpoints_summary = {
        "total_epochs_trained": final_epoch,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss)
        if isinstance(best_val_loss, torch.Tensor)
        else best_val_loss,
        "total_improvements": len(improvement_checkpoints),
        "improvement_checkpoints": serializable_checkpoints,
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
