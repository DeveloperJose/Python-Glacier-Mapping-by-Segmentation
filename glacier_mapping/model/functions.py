#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 23:18:43 2020

@author: mibook

Training/Validation Functions
"""

import datetime
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
from tqdm import tqdm

import glacier_mapping.model.losses as model_losses
import glacier_mapping.model.metrics as model_metrics

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)
# HANDLER = logging.StreamHandler()
# HANDLER.setLevel(logging.INFO)
# LOGGER.addHandler(HANDLER)
FORMATTER = logging.Formatter("%(message)s")


def log(level, message):
    """Log the message at a given level (from the standard logging package levels: ERROR, INFO, DEBUG etc).
    Add a datetime prefix to the log message, and a SystemLog: prefix provided it is public data.

    Args:
        level (int): logging level, best set by using logging.(INFO|DEBUG|WARNING) etc
        message (str): mesage to log
    """
    message = "{}\t{}\t{}".format(
        datetime.datetime.now().strftime("%d-%m-%Y, %H:%M:%S"),
        logging._levelToName[level],
        message,
    )
    message = "SystemLog: " + message
    logging.log(level, message)


def train_epoch(epoch, loader, frame):
    """Train model for one epoch

    This makes one pass through a dataloader and updates the model in the
    associated frame.

    :param loader: A pytorch DataLoader containing x,y pairs
      with which to train the model.
    :type loader: torch.data.utils.DataLoader
    :param frame: A Framework object wrapping both the model and the
      optimization setup.
    :type frame: Framework
    :param metrics_opts: A dictionary whose keys specify which metrics to
      compute on the predictions from the model.
    :type metrics_opts: dict
    :return (train_loss, metrics): A tuple containing the average epoch loss
      and the metrics on the training set.
    """
    metrics = frame.metrics_opts.metrics
    n_classes: int = frame.num_classes
    threshold = frame.metrics_opts.threshold

    loss, batch_loss, tp, fp, fn = (
        0,
        0,
        torch.zeros(n_classes),
        torch.zeros(n_classes),
        torch.zeros(n_classes),
    )
    train_iterator = tqdm(
        loader, desc="Train Iter (Epoch=X Steps=X loss=X.XXX lr=X.XXXXXXX)"
    )
    for i, (x, y) in enumerate(train_iterator):
        frame.zero_grad()
        y_hat, batch_loss = frame.optimize(x, y)
        frame.step()
        batch_loss = float(batch_loss.detach())
        loss += batch_loss
        y_hat = frame.act(y_hat)
        mask = y.sum(axis=3) == 0
        _tp, _fp, _fn = frame.metrics(y_hat, y, mask, threshold)
        tp += _tp
        fp += _fp
        fn += _fn
        train_iterator.set_description(
            "Train, Epoch=%d Steps=%d Loss=%5.3f Avg_Loss=%5.3f "
            % (epoch, i, batch_loss, loss / (i + 1))
        )
    metrics = get_metrics(tp, fp, fn, metrics)
    loss_alpha = frame.get_loss_alpha()

    return loss / (i + 1), metrics, loss_alpha


def validate(epoch, loader, frame, test=False):
    """Compute Metrics on a Validation Loader

    To honestly evaluate a model, we should compute its metrics on a validation
    dataset. This runs the model in frame over the data in loader, compute all
    the metrics specified in metrics_opts.

    :param loader: A DataLoader containing x,y pairs with which to validate the
      model.
    :type loader: torch.utils.data.DataLoader
    :param frame: A Framework object wrapping both the model and the
      optimization setup.
    :type frame: Framework
    :param metrics_opts: A dictionary whose keys specify which metrics to
      compute on the predictions from the model.
    :type metrics_opts: dict
    :return (val_loss, metrics): A tuple containing the average validation loss
      and the metrics on the validation set.
    """
    metrics = frame.metrics_opts.metrics
    n_classes: int = frame.num_classes
    threshold = frame.metrics_opts.threshold
    loss, batch_loss, tp, fp, fn = (
        0,
        0,
        torch.zeros(n_classes),
        torch.zeros(n_classes),
        torch.zeros(n_classes),
    )
    if test:
        iterator = tqdm(
            loader, desc="Test Iter (Epoch=X Steps=X loss=X.XXX lr=X.XXXXXXX)"
        )
    else:
        iterator = tqdm(
            loader, desc="Val Iter (Epoch=X Steps=X loss=X.XXX lr=X.XXXXXXX)"
        )

    def channel_first(x):
        return x.permute(0, 3, 1, 2)

    for i, (x, y) in enumerate(iterator):
        y_hat = frame.infer(x)
        batch_loss = frame.calc_loss(channel_first(y_hat), channel_first(y))
        batch_loss = float(batch_loss.detach())
        loss += batch_loss
        y_hat = frame.act(y_hat)
        mask = y.sum(axis=3) == 0
        _tp, _fp, _fn = frame.metrics(y_hat, y, mask, threshold)
        tp += _tp
        fp += _fp
        fn += _fn
        if test:
            iterator.set_description(
                "Test,   Epoch=%d Steps=%d Loss=%5.3f Avg_Loss=%5.3f "
                % (epoch, i, batch_loss, loss / (i + 1))
            )
        else:
            iterator.set_description(
                "Val,   Epoch=%d Steps=%d Loss=%5.3f Avg_Loss=%5.3f "
                % (epoch, i, batch_loss, loss / (i + 1))
            )
    if not test:
        frame.val_operations(loss / len(loader.dataset))
    metrics = get_metrics(tp, fp, fn, metrics)

    return loss / (i + 1), metrics


def log_metrics(writer, frame, metrics, epoch, stage):
    """Log metrics for tensorboard
    A function that logs metrics from training and testing to tensorboard
    Args:
        writer(SummaryWriter): The tensorboard summary object
        metrics(Dict): Dictionary of metrics to record
        avg_loss(float): The average loss across all epochs
        epoch(int): Total number of training cycles
        stage(String): Train/Val
        mask_names(List): Names of the mask(prediction) to log mmetrics for
    """
    for k, v in metrics.items():
        for name, metric in zip(frame.mask_names, v):
            writer.add_scalar(f"{stage}_{str(k)}/{name}", metric, epoch)


def log_images(writer, frame, batch, epoch, stage, normalize):
    """Log images for tensorboard

    Args:
        writer (SummaryWriter): The tensorboard summary object
        frame (Framework): The model to use for inference
        batch (tensor): The batch of samples on which to make predictions
        epoch (int): Current epoch number
        stage (string): specified pipeline stage

    Return:
        Images Logged onto tensorboard
    """
    # threshold = frame.loader_opts.threshold
    normalize_name = frame.loader_opts.normalize
    # use_physics = frame.use_physics

    batch = next(iter(batch))
    colors = {
        0: np.array((255, 0, 0)),
        1: np.array((222, 184, 135)),
        2: np.array((95, 158, 160)),
        # 3: np.array((165, 42, 42)),
    }

    def pm(x):
        return x.permute(0, 3, 1, 2)

    def squash(x):
        return (x - x.min()) / (x.max() - x.min())

    x, y = batch
    y_mask = np.sum(y.cpu().numpy(), axis=3) == 0
    y_hat = frame.act(frame.infer(x))
    y = np.argmax(y.cpu().numpy(), axis=3) + 1

    # _y_hat = np.zeros((y_hat.shape[0], y_hat.shape[1], y_hat.shape[2]))
    # y_hat = y_hat.cpu().numpy()
    # for i in range(1, 3):
    #    _y_hat[y_hat[:, :, :, i] >= threshold[i - 1]] = i + 1
    # _y_hat[_y_hat == 0] = 1
    # _y_hat[y_mask] = 0
    # y_hat = _y_hat

    y_hat = np.argmax(y_hat.cpu().numpy(), axis=3) + 1

    y[y_mask] = 0
    y_hat[y_mask] = 0
    _y = np.zeros((y.shape[0], y.shape[1], y.shape[2], 3))
    _y_hat = np.zeros((y_hat.shape[0], y_hat.shape[1], y_hat.shape[2], 3))

    for i in range(len(colors)):
        _y[y == i] = colors[i]
        _y_hat[y_hat == i] = colors[i]
    y = _y
    y_hat = _y_hat
    if normalize_name == "mean-std":
        # if use_physics:
        #     # print(x.shape, type(normalize), type(normalize[0]), normalize[0].shape, normalize[1].shape)
        #     # x[batch, rows, cols, channels]
        #     x[:, :, :, :-1] = (x[:, :, :, :-1] * normalize[1][:-1]) + normalize[0][:-1]
        #     #     # Revert mean-std for all physics channels
        #     #     for phys_idx in range(len(use_physics)):
        #     p_mu = x[:, :, -1].mean()
        #     p_std = x[:, :, -1].std()
        #     x[:, :, -1] = (x[:, :, -1] * p_std) + p_mu
        # else:
        x = (x * normalize[1]) + normalize[0]
    else:
        x = torch.clamp(x, 0, 1)
    try:
        writer.add_image(
            f"{stage}/x", make_grid(pm(squash(x[:, :, :, [4, 3, 1]]))), epoch
        )
    except Exception:
        try:
            writer.add_image(
                f"{stage}/x", make_grid(pm(squash(x[:, :, :, [0, 1, 2]]))), epoch
            )
        except Exception:
            writer.add_image(
                f"{stage}/x", make_grid(pm(squash(x[:, :, :, [0]]))), epoch
            )
    writer.add_image(f"{stage}/y", make_grid(pm(squash(torch.tensor(y)))), epoch)
    writer.add_image(
        f"{stage}/y_hat", make_grid(pm(squash(torch.tensor(y_hat)))), epoch
    )


def get_loss(outchannels, opts=None):
    if opts is None:
        return model_losses.diceloss()
    if opts.label_smoothing == "None":
        label_smoothing = 0
    else:
        label_smoothing = opts.label_smoothing

    if opts.name == "dice":
        loss_fn = model_losses.diceloss(
            act=torch.nn.Softmax(dim=1),
            outchannels=outchannels,
            label_smoothing=label_smoothing,
            masked=opts.masked,
            gaussian_blur_sigma=opts.gaussian_blur_sigma,
        )
    elif opts.name == "boundary":
        loss_fn = model_losses.boundaryloss()
    elif opts.name == "iou":
        loss_fn = model_losses.iouloss(
            act=torch.nn.Softmax(dim=1), outchannels=outchannels, masked=opts.masked
        )
    elif opts.name == "ce":
        loss_fn = model_losses.celoss(
            act=torch.nn.Softmax(dim=1), outchannels=outchannels, masked=opts.masked
        )
    elif opts.name == "nll":
        loss_fn = model_losses.nllloss(
            act=torch.nn.Softmax(dim=1), outchannels=outchannels, masked=opts.masked
        )
    elif opts.name == "focal":
        loss_fn = model_losses.focalloss(
            act=torch.nn.Softmax(dim=1), outchannels=outchannels, masked=opts.masked
        )
    elif opts.name == "custom":
        loss_fn = model_losses.customloss(
            act=torch.nn.Softmax(dim=1), outchannels=outchannels, masked=opts.masked
        )
    else:
        raise ValueError("Loss must be defined!")
    return loss_fn


def get_metrics(tp, fp, fn, metric_names):
    """
    Aggregate --inplace-- the mean of a matrix of tensor (across rows)
    Args:
        metrics (dict(troch.Tensor)): the matrix to get mean of
    """
    metrics = dict.fromkeys(metric_names, 0)
    for metric_name, arr in metrics.items():
        metric_fun = getattr(model_metrics, metric_name)
        metrics[metric_name] = metric_fun(tp, fp, fn)
    return metrics


def print_conf(conf):
    for key, value in conf.items():
        log(logging.INFO, "{} = {}".format(key, value))


def print_metrics(frame, train_metric, val_metric, test_metric, round=2):
    train_classes, val_classes, test_classes = dict(), dict(), dict()
    for i, c in enumerate(frame.mask_names):
        train_metric_log, val_metric_log, test_metric_log = dict(), dict(), dict()
        for metric in frame.metrics_opts.metrics:
            train_metric_log[metric] = np.round(train_metric[metric][i].item(), 2)
            val_metric_log[metric] = np.round(val_metric[metric][i].item(), 2)
            test_metric_log[metric] = np.round(test_metric[metric][i].item(), 2)
        train_classes[c] = train_metric_log
        val_classes[c] = val_metric_log
        test_classes[c] = test_metric_log
    log(logging.INFO, "Train | {}".format(train_classes))
    log(logging.INFO, "Val | {}".format(val_classes))
    log(logging.INFO, "Test | {}\n".format(test_classes))


def get_current_lr(frame):
    lr = frame.get_current_lr()
    return np.float32(lr)


def find_lr(frame, train_loader, init_value, final_value):
    logs, losses = frame.find_lr(train_loader, init_value, final_value)
    plt.plot(logs, losses)
    plt.xlabel("learning rate (log scale)")
    plt.ylabel("loss")
    plt.savefig("Optimal lr curve.png")
    print("plot saved")
