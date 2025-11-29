#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a U-Net model on the prepared glacier dataset.

Includes:
 - dataset statistics (optional)
 - clean per-epoch metric tables
 - optional full-test-tile evaluation every N epochs
 - AMP + OneCycleLR + sigma-weighted custom loss
"""

import gc
import json
import logging
import pathlib
import random
import warnings
import subprocess
from timeit import default_timer as timer

import numpy as np
import torch
import yaml
from addict import Dict
from torch.utils.tensorboard import SummaryWriter

import glacier_mapping.model.functions as fn
from glacier_mapping.data.data import fetch_loaders
from glacier_mapping.model.frame import Framework

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


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------
    conf = Dict(yaml.safe_load(open("./conf/unet_train.yaml")))

    run_name: str = conf.training_opts.run_name
    output_dir = pathlib.Path(conf.training_opts.output_dir) / run_name
    model_output_dir = output_dir / "models"
    early_stopping: int = conf.training_opts.early_stopping

    # How often to run full test-tile evaluation
    full_eval_every: int = int(
        getattr(conf.training_opts, "full_eval_every", 5)
    )

    # Physics sanity check
    if (
        "phys" in run_name
        and conf.loader_opts.physics_channel not in conf.loader_opts.use_channels
    ):
        raise ValueError("Training phys model without phys channel")

    # ------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------
    train_loader, val_loader, test_loader = fetch_loaders(**conf.loader_opts)

    # ------------------------------------------------------------
    # Framework
    # ------------------------------------------------------------
    frame = Framework(
        loss_opts=conf.loss_opts,
        loader_opts=conf.loader_opts,
        model_opts=conf.model_opts,
        optimizer_opts=conf.optim_opts,
        reg_opts=conf.reg_opts,
        metrics_opts=conf.metrics_opts,
        training_opts=conf.training_opts,
        device=int(conf.training_opts.gpu_rank),
    )

    # Optional fine-tuning from final model of previous run
    if conf.training_opts.fine_tune:
        fn.log(logging.INFO, "Finetuning the model")
        run_name += "_finetuned"
        final_model_path = output_dir / "models" / "model_final.pt"

        frame = Framework.from_checkpoint(
            final_model_path, device=int(conf.training_opts.gpu_rank)
        )
        frame.freeze_layers()

    # Optional LR finder
    if conf.training_opts.find_lr:
        fn.find_lr(frame, train_loader, init_value=1e-9, final_value=1)
        raise SystemExit

    # ------------------------------------------------------------
    # Prepare directories
    # ------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # TensorBoard + logging
    # ------------------------------------------------------------
    writer = SummaryWriter(output_dir / "logs")
    writer.add_text("Configuration Parameters", json.dumps(conf))
    fn.print_conf(conf)
    fn.log(
        logging.INFO,
        (
            f"# Training Instances = {len(train_loader)}, "
            f"# Validation Instances = {len(val_loader)}"
        ),
    )

    # Save conf to file
    with open(output_dir / "conf.json", "w") as f:
        json.dump(conf, f, sort_keys=True, indent=4)

    # ------------------------------------------------------------
    # Load normalization array for image logging
    # ------------------------------------------------------------
    norm_path = pathlib.Path(conf.loader_opts.processed_dir) / "normalize_train.npy"
    _normalize = np.load(norm_path)

    if conf.loader_opts.normalize == "min-max":
        _normalize = (
            _normalize[2][conf.loader_opts.use_channels],
            _normalize[3][conf.loader_opts.use_channels],
        )
    elif conf.loader_opts.normalize == "mean-std":
        _normalize = (
            np.append(_normalize[0], 0)[conf.loader_opts.use_channels],
            np.append(_normalize[1], 1)[conf.loader_opts.use_channels],
        )
    else:
        raise ValueError("Normalize must be min-max or mean-std")

    # ------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------
    best_val_loss = np.inf
    epochs_without_improvement = 0
    start_time = timer()

    for epoch in range(1, conf.training_opts.epochs + 1):
        # --------------------- Train ---------------------
        loss_train, train_metric, loss_alpha = fn.train_epoch(
            epoch, train_loader, frame
        )
        fn.log_metrics(writer, frame, train_metric, epoch, "train")

        # --------------------- Validation ----------------
        loss_val, val_metric = fn.validate(epoch, val_loader, frame, test=False)
        fn.log_metrics(writer, frame, val_metric, epoch, "val")

        # --------------------- (Optional) Test Patch Metrics ----------------
        loss_test = 0.0
        test_metric = {
            m: [0.0] * frame.num_classes for m in frame.metrics_opts.metrics
        }

        # --------------------- Log images ----------------
        if (epoch - 1) % 5 == 0:
            fn.log_images(writer, frame, train_loader, epoch, "train", _normalize)
            fn.log_images(writer, frame, val_loader, epoch, "val", _normalize)

        # --------------------- Full test-tile evaluation -------------
        if full_eval_every > 0 and epoch % full_eval_every == 0:
            fn.log(
                logging.INFO,
                f"Running full test-tile evaluation at epoch {epoch}...",
            )
            fn.evaluate_full_test_tiles(
                frame=frame,
                writer=writer,
                epoch=epoch,
                output_dir=output_dir / "full_eval",
            )

        # --------------------- Scalars to TensorBoard ----------------------
        writer.add_scalar("loss_train", loss_train, epoch)
        writer.add_scalar("loss_val", loss_val, epoch)
        writer.add_scalar("loss_test", loss_test, epoch)
        writer.add_scalar("lr", fn.get_current_lr(frame), epoch)

        for idx, sigma in enumerate(loss_alpha):
            writer.add_scalar(f"sigma/{idx + 1}", sigma, epoch)

        # --------------------- Console summary ----------------------------
        fn.print_epoch_summary(
            epoch,
            train_metric=train_metric,
            val_metric=val_metric,
            test_metric=test_metric,
            mask_names=frame.mask_names,
        )

        # --------------------- Early stopping ------------------------------
        if loss_val < best_val_loss:
            frame.save(model_output_dir, "best")
            best_val_loss = float(loss_val)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping:
            fn.log(
                logging.INFO,
                f"Early stopping at epoch {epoch} "
                f"| Best Val Loss = {best_val_loss:.4f}",
            )
            break

        torch.cuda.empty_cache()
        writer.flush()
        gc.collect()

    # ------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------
    frame.save(model_output_dir, "final")
    writer.close()

    fn.log(
        logging.INFO,
        f"Finished training {run_name} | Training took {timer() - start_time:.2f} sec",
    )

    # Desktop notification (Debian)
    try:
        subprocess.run(
            ["notify-send", "Training Done", f"Run {run_name} has finished."],
            check=False,
        )
    except Exception:
        pass

