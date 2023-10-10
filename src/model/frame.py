#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 22:59:16 2020

@author: mibook

Frame to Combine Model with Optimizer

This wraps the model and optimizer objects needed in training, so that each
training step can be concisely called with a single method.
"""
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage.morphology import binary_fill_holes
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import model.functions as fn

from .metrics import *
from .unet import Unet


class Framework:
    """
    Class to Wrap all the Training Steps

    """

    def __init__(
        self,
        loss_opts=None,
        loader_opts=None,
        model_opts=None,
        optimizer_opts=None,
        reg_opts=None,
        metrics_opts=None,
        device=None,
    ):
        """
        Set Class Attrributes
        """
        # % Device
        if isinstance(device, int):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.device.type == "cuda":
                torch.cuda.set_device(device)
        else:
            self.device = torch.device("cpu")

        # % Data Loader
        self.loader_opts = loader_opts
        self.use_physics = loader_opts.physics_channel in loader_opts.use_channels
        self.use_channels = loader_opts.use_channels
        output_classes = loader_opts.output_classes

        # Dataframe
        self.df = pd.read_csv(Path(loader_opts.processed_dir) / "slice_meta.csv")

        # Binary Model or Multi-Class Model?
        if len(output_classes) == 1:
            cl_name = loader_opts.class_names[output_classes[0]]
            self.mask_names = [f"NOT~{cl_name}", cl_name]
        else:
            self.mask_names = [loader_opts.class_names[i] for i in output_classes]

        self.num_classes = len(self.mask_names)
        self.multi_class = self.num_classes > 1
        self.is_binary = len(output_classes) == 1
        if self.is_binary:
            self.binary_class_idx = loader_opts.output_classes[0]

        # Normalization
        self.normalization = loader_opts.normalize
        self.norm_arr = np.load(Path(loader_opts.processed_dir) / "normalize_train.npy")
        assert (
            self.normalization == "mean-std" or self.normalization == "min-max"
        ), "Invalid normalization"

        # % Model
        self.model_opts = model_opts
        self.model_opts.args.inchannels = len(loader_opts.use_channels)
        self.model_opts.args.outchannels = self.num_classes
        self.model = Unet(**model_opts.args).to(self.device)

        # % Loss
        self.loss_opts = loss_opts
        self.loss_fn = fn.get_loss(self.num_classes, loss_opts).to(self.device)
        if loss_opts is not None:
            self.loss_alpha = torch.tensor([loss_opts.alpha]).to(self.device)
        else:
            self.loss_alpha = torch.tensor([0.0]).to(self.device)
        self.sigma1, self.sigma2 = torch.tensor([1.0]).to(self.device), torch.tensor(
            [1.0]
        ).to(self.device)
        self.sigma1, self.sigma2 = (
            self.sigma1.requires_grad_(),
            self.sigma2.requires_grad_(),
        )
        if loss_opts is None:
            self.loss_weights = torch.tensor([1.0, 1.0, 1.0]).to(self.device)
        else:
            self.loss_weights = torch.tensor(loss_opts.weights).to(self.device)

        # % Optimizer
        self.optimizer_opts = optimizer_opts
        if optimizer_opts is None:
            optimizer_opts = {"name": "Adam", "args": {"lr": 0.001}}
        _optimizer_params = [
            {"params": self.model.parameters(), **optimizer_opts["args"]},
            {"params": self.sigma1, **optimizer_opts["args"]},
            {"params": self.sigma2, **optimizer_opts["args"]},
        ]
        optimizer_def = getattr(torch.optim, optimizer_opts["name"])
        self.optimizer = optimizer_def(_optimizer_params)
        self.lrscheduler = ReduceLROnPlateau(
            self.optimizer, "min", verbose=True, patience=15, factor=0.1, min_lr=1e-9
        )

        # % Regularization
        self.reg_opts = reg_opts
        self.metrics_opts = metrics_opts

    def optimize(self, x, y):
        """
        Take a single gradient step

        Args:
            X: raw training data
            y: labels
        Return:
            optimization
        """
        x = x.permute(0, 3, 1, 2).to(self.device)
        y = y.permute(0, 3, 1, 2).to(self.device)
        y_hat = self.model(x)
        loss = self.calc_loss(y_hat, y)
        loss.backward()
        return y_hat.permute(0, 2, 3, 1), loss

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_current_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def step(self):
        self.optimizer.step()

    def val_operations(self, val_loss):
        """
        Update the LR Scheduler
        """
        self.lrscheduler.step(val_loss)

    def save(self, out_dir, epoch):
        """Save frame as a checkpoint file"""
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "sigma1": self.sigma1,
            "sigma2": self.sigma2,
            "loss_opts": self.loss_opts,
            "loader_opts": self.loader_opts,
            "model_opts": self.model_opts,
            "optimizer_opts": self.optimizer_opts,
            "reg_opts": self.reg_opts,
            "metrics_opts": self.metrics_opts,
        }
        model_path = Path(out_dir, f"model_{epoch}.pt")
        torch.save(state, model_path)
        print(f"Saved model {epoch}")

    @staticmethod
    def from_checkpoint(checkpoint_path: Path, device=None, testing=False):
        """Load a frame from a checkpoint file"""
        assert checkpoint_path.exists(), "checkpoint_path does not exist"
        if torch.cuda.is_available() and device != "cpu":
            state = torch.load(checkpoint_path)
        else:
            state = torch.load(checkpoint_path, map_location="cpu")

        if testing:
            state["model_opts"].args.dropout = 0.00000001

        frame = Framework(
            loss_opts=state["loss_opts"],
            loader_opts=state["loader_opts"],
            model_opts=state["model_opts"],
            optimizer_opts=state["optimizer_opts"],
            reg_opts=state["reg_opts"],
            metrics_opts=state["metrics_opts"],
            device=device,
        )
        frame.model.load_state_dict(state["state_dict"])
        frame.optimizer.load_state_dict(state["optimizer_state_dict"])
        frame.sigma1 = state["sigma1"]
        frame.sigma2 = state["sigma2"]
        return frame

    def infer(self, x):
        """Make a prediction for a given x

        Args:
            x: input x

        Return:
            Prediction

        """
        x = x.permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            y = self.model(x)
        return y.permute(0, 2, 3, 1)

    def calc_loss(self, y_hat, y):
        """Compute loss given a prediction

        Args:
            y_hat: Prediction
            y: Label

        Return:
            Loss values

        """
        y_hat = y_hat.to(self.device)
        y = y.to(self.device)
        # loss = self.loss_fn(y_hat, y)
        diceloss, boundaryloss = self.loss_fn(y_hat, y)
        diceloss = diceloss.sum()
        # loss = torch.add(torch.mul(diceloss.clone(), self.loss_alpha), torch.mul(boundaryloss.clone(), (1-self.loss_alpha)))
        loss = torch.add(
            torch.add(
                torch.mul(
                    torch.div(1, torch.mul(2, torch.square(self.sigma1))),
                    diceloss.clone(),
                ),
                torch.mul(
                    torch.div(1, torch.mul(2, torch.square(self.sigma2))),
                    boundaryloss.clone(),
                ),
            ),
            torch.abs(torch.log(torch.mul(self.sigma1, self.sigma2))),
        )
        if self.reg_opts:
            for reg_type in self.reg_opts.keys():
                reg_fun = globals()[reg_type]
                penalty = reg_fun(
                    self.model.parameters(), self.reg_opts[reg_type], self.device
                )
                loss += penalty
        return loss.abs()

    def get_loss_alpha(self):
        # return self.loss_alpha.item()
        return (self.sigma1.item(), self.sigma2.item())

    def metrics(self, y_hat, y, mask, threshold):
        """Loop over metrics in train.yaml

        Args:
            y_hat: Predictions
            y: Labels

        Return:
            results

        """
        n_classes = y.shape[3]
        _y_hat = np.zeros((y_hat.shape[0], y_hat.shape[1], y_hat.shape[2]))
        y_hat = y_hat.detach().cpu().numpy()
        for i in range(1, n_classes):
            _y_hat[y_hat[:, :, :, i] >= threshold[i - 1]] = i + 1
        _y_hat[_y_hat == 0] = 1
        _y_hat[mask] = -1
        y_hat = _y_hat

        y = np.argmax(y.cpu().numpy(), axis=3) + 1
        y[mask] = -1

        tp, fp, fn = (
            torch.zeros(n_classes),
            torch.zeros(n_classes),
            torch.zeros(n_classes),
        )
        for i in range(0, n_classes):
            _y_hat = (y_hat == i + 1).astype(np.uint8)
            _y = (y == i + 1).astype(np.uint8)
            _tp, _fp, _fn = tp_fp_fn(_y_hat, _y)
            tp[i] = _tp
            fp[i] = _fp
            fn[i] = _fn

        return tp, fp, fn

    def segment(self, y_hat):
        """Predict a class given logits
        Args:
            y_hat: logits output
        Return:
            Probability of class in case of binary classification
            or one-hot tensor in case of multi class"""
        if self.multi_class:
            y_hat = torch.argmax(y_hat, axis=3)
            y_hat = torch.nn.functional.one_hot(y_hat, num_classes=self.num_classes)
        else:
            y_hat = torch.sigmoid(y_hat)

        return y_hat

    def act(self, logits):
        """Applies activation function based on the model
        Args:
            y_hat: logits output
        Returns:
            logits after applying activation function"""

        if self.multi_class:
            y_hat = torch.nn.Softmax(3)(logits)
        else:
            y_hat = torch.sigmoid(logits)
        return y_hat

    def freeze_layers(self, layers=None):
        for i, layer in enumerate(self.model.parameters()):
            if layers is None:
                layer.requires_grad = False
            elif i < layers:  # Freeze 60 out of 75 layers, retrain on last 15 only
                layer.requires_grad = False
            else:
                pass

    def find_lr(self, train_loader, init_value, final_value):
        number_in_epoch = len(train_loader) - 1
        update_step = (final_value / init_value) ** (1 / number_in_epoch)
        lr = init_value
        self.optimizer.param_groups[0]["lr"] = lr
        best_loss = 0.0
        batch_num = 0
        losses = []
        log_lrs = []
        iterator = tqdm(
            train_loader, desc="Current lr=XX.XX Steps=XX Loss=XX.XX Best lr=XX.XX "
        )
        for i, data in enumerate(iterator):
            batch_num += 1
            inputs, labels = data
            self.optimizer.zero_grad()
            inputs = inputs.permute(0, 3, 1, 2).to(self.device)
            labels = labels.permute(0, 3, 1, 2).to(self.device)
            outputs = self.model(inputs)
            loss = self.calc_loss(outputs, labels)
            # Crash out if loss explodes
            if batch_num > 1 and loss > 4 * best_loss:
                return log_lrs[10:-5], losses[10:-5]
            # Record the best loss
            if loss < best_loss or batch_num == 1:
                best_loss = loss
                best_lr = lr
            # Do the backward pass and optimize
            loss.backward()
            self.optimizer.step()
            iterator.set_description(
                "Current lr=%5.9f Steps=%d Loss=%5.3f Best lr=%5.9f "
                % (lr, i, loss, best_lr)
            )
            # Store the values
            losses.append(loss.detach())
            log_lrs.append(math.log10(lr))
            # Update the lr for the next step and store
            lr = lr * update_step
            self.optimizer.param_groups[0]["lr"] = lr
        return log_lrs[10:-5], losses[10:-5]

    def get_model_device(self):
        return self.model, self.device

    def normalize(self, x):
        if self.normalization == "mean-std":
            _mean, _std = self.norm_arr[0], self.norm_arr[1]
            return (x - _mean) / _std
        elif self.normalization == "min-max":
            _min, _max = self.norm_arr[2], self.norm_arr[3]
            return (np.clip(x, _min, _max) - _min) / (_max - _min)
        else:
            raise Exception("Invalid normalization")

    def predict_whole(self, whole_arr, window_size, threshold=None):
        # Reduce to only needed channels to speed up further computations, normalize, and get mask
        whole_arr = whole_arr[:, :, self.use_channels]
        whole_arr = self.normalize(whole_arr)
        mask = np.sum(whole_arr[:, :, :3], axis=2) < 0.001

        # Perform sliding window prediction
        y_pred = np.zeros((whole_arr.shape[0], whole_arr.shape[1]), dtype=np.uint8)
        for row in range(0, whole_arr.shape[0], window_size[0]):
            for column in range(0, whole_arr.shape[1], window_size[1]):
                # Get slice from input array, pad with zeros if needed
                current_slice = whole_arr[
                    row : row + window_size[0], column : column + window_size[1], :
                ]
                if (
                    current_slice.shape[0] != window_size[0]
                    or current_slice.shape[1] != window_size[1]
                ):
                    temp = np.zeros(
                        (window_size[0], window_size[1], whole_arr.shape[2])
                    )
                    temp[
                        : current_slice.shape[0], : current_slice.shape[1], :
                    ] = current_slice
                    current_slice = temp

                # Slice prediction then place slice prediction into whole image prediction
                pred = self.predict_slice(
                    current_slice, threshold, preprocess=False, use_mask=False
                )

                endrow_dest = row + window_size[0]
                endrow_source = window_size[0]
                endcolumn_dest = column + window_size[0]
                endcolumn_source = window_size[1]
                if endrow_dest > y_pred.shape[0]:
                    endrow_source = y_pred.shape[0] - row
                    endrow_dest = y_pred.shape[0]
                if endcolumn_dest > y_pred.shape[1]:
                    endcolumn_source = y_pred.shape[1] - column
                    endcolumn_dest = y_pred.shape[1]

                y_pred[row:endrow_dest, column:endcolumn_dest] = pred[
                    0:endrow_source, 0:endcolumn_source
                ]

        y_pred[mask] = 0
        return y_pred, mask

    def predict_slice(self, slice_arr, threshold=None, preprocess=True, use_mask=True):
        # Process threshold parameter
        if threshold is None:
            threshold = [0.5] * self.num_classes
        elif isinstance(threshold, (int, float)):
            threshold = [threshold] * self.num_classes
        assert isinstance(threshold, list) and len(threshold) == self.num_classes

        # Reduce to only needed channels to speed up further computations, normalize, and get mask
        if preprocess:
            slice_arr = slice_arr[:, :, self.use_channels]
            slice_arr = self.normalize(slice_arr)

        # Send array to torch and then get prediction
        _x = torch.from_numpy(np.expand_dims(slice_arr, axis=0)).float()
        _y = self.infer(_x)
        _y = torch.nn.Softmax(3)(_y)
        _y = np.squeeze(_y.cpu())
        assert _y.shape[2] == self.num_classes

        # Threshold + fill holes + add mask to prediction
        y_pred = np.zeros((_y.shape[0], _y.shape[1]), dtype=np.uint8)
        for i in range(self.num_classes):
            _class = _y[:, :, i] >= threshold[i]
            _class = binary_fill_holes(_class)
            y_pred[_class] = i

        if use_mask:
            mask = np.sum(slice_arr[:, :, :3], axis=2) < 0.001
            y_pred[mask] = 0
            return y_pred, mask
        return y_pred

    def get_y_true(self, label_mask: np.ndarray, mask=None):
        y_true = np.zeros((label_mask.shape[0], label_mask.shape[1]), dtype=np.uint8)
        if self.is_binary:
            assert (
                self.binary_class_idx != 0
            ), "You are trying to predict BG instead of CI or DCG"
            y_true[label_mask[:, :, self.binary_class_idx - 1] != 1] = 0
            y_true[label_mask[:, :, self.binary_class_idx - 1] == 1] = 1
        else:
            # Label mask is always just CleanIce and Debris so do +1 to match the prediction labels
            for i in range(label_mask.shape[2]):
                y_true[label_mask[:, :, i] == 1] = i + 1

        if mask is not None:
            y_true[mask] = 0
        return y_true
