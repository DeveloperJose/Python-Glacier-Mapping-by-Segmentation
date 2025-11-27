#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program trains a U-Net model on the provided dataset.
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

import model.functions as fn
from data.data import fetch_loaders
from model.frame import Framework

random.seed(41)
np.random.seed(41)
torch.manual_seed(41)
torch.cuda.manual_seed(41)
torch.cuda.manual_seed_all(41)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    conf = Dict(yaml.safe_load(open("./conf/unet_train.yaml")))

    # % Get some training opts
    run_name: str = conf.training_opts.run_name
    output_dir = pathlib.Path(conf.training_opts.output_dir) / run_name
    model_output_dir = output_dir / "models"
    early_stopping: int = conf.training_opts.early_stopping

    if (
        "phys" in run_name
        and conf.loader_opts.physics_channel not in conf.loader_opts.use_channels
    ):
        raise ValueError("Training phys model without phys channel")

    # % Loaders
    train_loader, val_loader, test_folder = fetch_loaders(**conf.loader_opts)

    # % Framework
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

    if conf.training_opts.fine_tune:
        fn.log(logging.INFO, "Finetuning the model")
        run_name += "_finetuned"
        final_model_path = output_dir / "models" / "model_final.pt"
        frame = Framework.from_checkpoint(
            final_model_path, device=int(conf.training_opts.gpu_rank)
        )
        frame.freeze_layers()

    if conf.find_lr:
        fn.find_lr(frame, train_loader, init_value=1e-9, final_value=1)
        exit()

    # % Setup logging
    writer = SummaryWriter(output_dir / "logs")
    writer.add_text("Configuration Parameters", json.dumps(conf))
    fn.print_conf(conf)
    fn.log(
        logging.INFO,
        f"# Training Instances = {len(train_loader)}, # Validation Instances = {len(val_loader)}",
    )

    # % Save conf
    with open(output_dir / "conf.json", "w") as f:
        j = json.dumps(conf, sort_keys=True)
        f.write(j)

    # % Load normalization arrays for image logging
    _normalize = np.load(
        pathlib.Path(conf.loader_opts.processed_dir) / "normalize_train.npy"
    )
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

    # % Training Body
    loss_val = np.inf
    epochs_without_improvement = 0
    start_time = timer()
    for epoch in range(1, conf.training_opts.epochs + 1):
        # train loop
        loss_train, train_metric, loss_alpha = fn.train_epoch(
            epoch, train_loader, frame
        )
        fn.log_metrics(writer, frame, train_metric, epoch, "train")

        # validation loop
        new_loss_val, val_metric = fn.validate(epoch, val_loader, frame)
        fn.log_metrics(writer, frame, val_metric, epoch, "val")

        # test loop
        loss_test, test_metric = fn.validate(epoch, val_loader, frame, test=True)
        fn.log_metrics(writer, frame, test_metric, epoch, "test")

        if (epoch - 1) % 5 == 0:
            fn.log_images(writer, frame, train_loader, epoch, "train", _normalize)
            fn.log_images(writer, frame, val_loader, epoch, "val", _normalize)

        # Save best model
        if new_loss_val < loss_val:
            frame.save(model_output_dir, "best")
            loss_val = float(new_loss_val)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        lr = fn.get_current_lr(frame)
        writer.add_scalar("loss_train", loss_train, epoch)
        writer.add_scalar("loss_val", new_loss_val, epoch)
        writer.add_scalar("loss_test", loss_test, epoch)

        writer.add_scalar("lr", lr, epoch)
        for idx, sigma in enumerate(loss_alpha):
            writer.add_scalar(f"sigma/{idx + 1}", sigma, epoch)

        fn.print_metrics(frame, train_metric, val_metric, test_metric)
        torch.cuda.empty_cache()
        writer.flush()
        gc.collect()

        if epochs_without_improvement >= early_stopping:
            fn.log(
                logging.INFO,
                f"Early stopping at epoch {epoch} | Best Val Loss = {loss_val}",
            )
            break

    frame.save(model_output_dir, "final")
    writer.close()

    fn.log(
        logging.INFO,
        f"Finished training {run_name} | Training took {timer() - start_time:.2f}sec",
    )

    subprocess.run(["notify-send", "Training Done", "The model has finished training."])
