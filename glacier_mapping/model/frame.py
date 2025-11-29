#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from addict import Dict
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from tqdm import tqdm

from glacier_mapping.model.losses import customloss
from glacier_mapping.model.metrics import (
    tp_fp_fn,
    precision,
    recall,
    IoU,
    l1_reg,
    l2_reg,
)
from glacier_mapping.model.unet import Unet
from glacier_mapping.model.visualize import (
    build_cmap_from_mask_names,
    make_rgb_preview,
    label_to_color,
    make_confidence_map,
    make_entropy_map,
    make_tp_fp_fn_masks,
    make_eight_panel,
)


class Framework:
    """
    Wraps model, loss, optimizer, scheduler, and all training / inference utilities.
    """

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, conf_path, device=None):
        """
        Build a Framework directly from a YAML config file.

        conf_path: path to e.g. ./conf/unet_train.yaml
        """
        conf_path = Path(conf_path)
        conf = Dict(yaml.safe_load(open(conf_path)))

        # Prefer top-level gpu_rank, fallback to training_opts.gpu_rank, default 0
        gpu_rank = int(conf.get("gpu_rank", conf.get("training_opts", {}).get("gpu_rank", 0)))
        if device is None:
            device = gpu_rank

        return cls(
            loss_opts=conf.loss_opts,
            loader_opts=conf.loader_opts,
            model_opts=conf.model_opts,
            optimizer_opts=conf.optim_opts,
            reg_opts=getattr(conf, "reg_opts", None),
            metrics_opts=conf.metrics_opts,
            training_opts=conf.training_opts,
            scheduler_opts=getattr(conf, "scheduler_opts", None),
            device=device,
        )

    @staticmethod
    def from_checkpoint(
        checkpoint_path: Path,
        device=None,
        new_data_path=None,
        testing=False,
        override=None,
    ):
        """
        Load a Framework from a checkpoint file.
        """
        if override is None:
            override = {}

        checkpoint_path = Path(checkpoint_path)
        assert checkpoint_path.exists(), "checkpoint_path does not exist"

        with torch.serialization.safe_globals([Dict]):
            if torch.cuda.is_available() and device != "cpu":
                state = torch.load(checkpoint_path)
            else:
                state = torch.load(checkpoint_path, map_location="cpu")

        # Dropout tweak for test-time
        if testing and "model_opts" in state:
            state["model_opts"].args.dropout = 1e-8

        # Allow simple overrides
        state.update(override)

        # Optionally change dataset location
        if new_data_path is not None:
            new_data_path = Path(new_data_path)
            state["loader_opts"].processed_dir = new_data_path

        frame = Framework(
            loss_opts=state["loss_opts"],
            loader_opts=state["loader_opts"],
            model_opts=state["model_opts"],
            optimizer_opts=state["optimizer_opts"],
            reg_opts=state["reg_opts"],
            metrics_opts=state["metrics_opts"],
            training_opts=state["training_opts"],
            scheduler_opts=state.get("scheduler_opts", None),
            device=device,
        )
        frame.model.load_state_dict(state["state_dict"])
        frame.optimizer.load_state_dict(state["optimizer_state_dict"])

        # Backwards compatibility with old sigma storage
        if "sigma1" in state:
            frame.sigma_list = [state["sigma1"], state["sigma2"]]
        else:
            frame.sigma_list = state["sigma_list"]

        return frame

    # ------------------------------------------------------------------
    # Core init
    # ------------------------------------------------------------------
    def __init__(
        self,
        loss_opts=None,
        loader_opts=None,
        model_opts=None,
        optimizer_opts=None,
        reg_opts=None,
        metrics_opts=None,
        training_opts=None,
        scheduler_opts=None,
        device=None,
    ):
        self.scaler = GradScaler()

        # ------------------- Device -----------------------------------
        if isinstance(device, int):
            self.device = torch.device(
                f"cuda:{device}" if torch.cuda.is_available() else "cpu"
            )
            if self.device.type == "cuda":
                torch.cuda.set_device(self.device)
                print(f"Using GPU {self.device}")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        # ------------------- Data-related opts -------------------------
        self.loader_opts = loader_opts
        self.use_physics = loader_opts.physics_channel in loader_opts.use_channels
        self.use_channels = loader_opts.use_channels
        output_classes = loader_opts.output_classes

        # Slice metadata (for steps_per_epoch estimation etc.)
        self.df = pd.read_csv(Path(loader_opts.processed_dir) / "slice_meta.csv")

        # Binary vs multi-class
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

        # Normalization parameters (subset to use_channels)
        self.normalization = loader_opts.normalize
        self.norm_arr_full = np.load(
            Path(loader_opts.processed_dir) / "normalize_train.npy"
        )
        assert self.normalization in ["mean-std", "min-max"], "Invalid normalization"
        self.norm_arr = self.norm_arr_full[:, self.use_channels]

        # ------------------- Model -------------------------------------
        self.model_opts = model_opts
        self.model_opts.args.inchannels = len(loader_opts.use_channels)
        self.model_opts.args.outchannels = self.num_classes
        self.model = Unet(**model_opts.args).to(self.device)

        # ------------------- Loss --------------------------------------
        self.loss_opts = loss_opts
        self.loss_fn = self._build_loss(self.num_classes, loss_opts).to(self.device)

        # ------------------- Optimizer + sigma -------------------------
        self.optimizer_opts = optimizer_opts or Dict(
            {"name": "Adam", "args": {"lr": 0.001}}
        )
        opt_args = dict(self.optimizer_opts["args"])
        if "lr" in opt_args:
            opt_args["lr"] = float(opt_args["lr"])
        if "weight_decay" in opt_args:
            opt_args["weight_decay"] = float(opt_args["weight_decay"])

        # main model params
        _optimizer_params = [{"params": self.model.parameters(), **opt_args}]

        # CustomLoss sigma list (uncertainty weighting; 2 terms: dice + boundary)
        self.sigma_list = []
        for _ in range(self.loss_fn.n_sigma):
            sigma = torch.tensor([1.0], requires_grad=True, device=self.device)
            self.sigma_list.append(sigma)
            _optimizer_params.append({"params": sigma, **opt_args})

        optimizer_def = getattr(torch.optim, self.optimizer_opts["name"])
        self.optimizer = optimizer_def(_optimizer_params)

        # ------------------- Scheduler ---------------------------------
        self.training_opts = training_opts
        self.scheduler_opts = scheduler_opts
        self.scheduler_type = None
        self.lrscheduler = None

        if self.scheduler_opts is not None and self.scheduler_opts.get("name"):
            self._init_scheduler(opt_args)

        # ------------------- Regularization & Metrics ------------------
        self.reg_opts = reg_opts
        self.metrics_opts = metrics_opts

    # ------------------------------------------------------------------
    # Loss builder
    # ------------------------------------------------------------------
    def _build_loss(self, outchannels, opts=None):
        """
        Only customloss is supported.
        For binary (2 channels) we only count foreground channel 1 in the dice term.
        For multi-class, we use classes 1..C-1 (ignore background).
        """
        if opts is None:
            return customloss()

        ls = 0 if opts.label_smoothing == "None" else opts.label_smoothing

        fg_classes = [1] if outchannels == 2 else list(range(1, outchannels))
        return customloss(
            act=torch.nn.Softmax(dim=1),
            smooth=1.0,
            label_smoothing=ls,
            foreground_classes=fg_classes,
        )

    # ------------------------------------------------------------------
    # Scheduler init
    # ------------------------------------------------------------------
    def _init_scheduler(self, opt_args):
        name = self.scheduler_opts.name
        self.scheduler_type = name

        if name == "OneCycleLR":
            num_samples = len(self.df)
            batch_size = self.loader_opts.batch_size
            steps_per_epoch = math.ceil(num_samples / batch_size)

            kwargs = dict(self.scheduler_opts.args)
            # max_lr often expressed relative to base lr; if missing, use 1.5x
            if "max_lr" not in kwargs:
                kwargs["max_lr"] = float(opt_args["lr"]) * 1.5

            self.lrscheduler = OneCycleLR(
                self.optimizer,
                steps_per_epoch=steps_per_epoch,
                epochs=int(self.training_opts.epochs),
                **kwargs,
            )

        elif name == "ReduceLROnPlateau":
            kwargs = dict(self.scheduler_opts.args)

            # Robust type casting in case YAML gave us strings
            if "factor" in kwargs:
                kwargs["factor"] = float(kwargs["factor"])
            if "min_lr" in kwargs:
                # Can be scalar or list-like; convert all to float
                mlr = kwargs["min_lr"]
                if isinstance(mlr, (list, tuple)):
                    kwargs["min_lr"] = [float(v) for v in mlr]
                else:
                    kwargs["min_lr"] = float(mlr)

            # mode, patience, etc. are fine as-is (int/str)
            self.lrscheduler = ReduceLROnPlateau(
                self.optimizer,
                **kwargs,
            )

        elif name in ["None", None]:
            self.scheduler_type = None
            self.lrscheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {name}")

        if self.scheduler_type:
            print(f"Using scheduler: {self.scheduler_type}")

    # ================================================================
    # TRAINING OPS
    # ================================================================
    def optimize(self, x, y_onehot, y_int):
        """
        Forward + backward pass (no optimizer / scheduler step).

        x        : (N,H,W,C)
        y_onehot: (N,H,W,C_out)
        y_int   : (N,H,W)
        """
        self.optimizer.zero_grad(set_to_none=True)

        x = x.permute(0, 3, 1, 2).to(self.device, non_blocking=True)
        y_onehot = y_onehot.permute(0, 3, 1, 2).to(self.device, non_blocking=True)
        y_int = y_int.to(self.device, non_blocking=True)

        with autocast(enabled=True):
            y_hat = self.model(x)
            loss = self.calc_loss(y_hat, y_onehot, y_int)

        self.scaler.scale(loss).backward()

        # clip gradients after unscale
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        return y_hat.permute(0, 2, 3, 1), loss

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_current_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def step(self):
        """
        Optimizer update + per-batch scheduler update (for OneCycleLR).
        """
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.scheduler_type == "OneCycleLR" and self.lrscheduler is not None:
            self.lrscheduler.step()

    def val_operations(self, val_loss):
        """
        Per-epoch scheduler updates (e.g., ReduceLROnPlateau).
        """
        if self.scheduler_type == "ReduceLROnPlateau" and self.lrscheduler is not None:
            self.lrscheduler.step(val_loss)

    # ------------------------------------------------------------------
    # Per-epoch training & validation
    # ------------------------------------------------------------------
    def _compute_metrics_dict(self, tp, fp, fn, metric_names):
        func_map = {
            "IoU": IoU,
            "precision": precision,
            "recall": recall,
        }
        metrics = {}
        for name in metric_names:
            if name not in func_map:
                raise ValueError(f"Unknown metric: {name}")
            metrics[name] = func_map[name](tp, fp, fn)
        return metrics

    def train_one_epoch(self, epoch, loader):
        """
        Single training epoch on a dataloader.

        Returns:
            avg_loss (float),
            metrics_dict (dict of tensors),
            loss_alpha (list of sigma scalars)
        """
        metric_names = self.metrics_opts.metrics
        n_classes = self.num_classes
        threshold = self.metrics_opts.threshold

        loss_sum = 0.0
        tp = torch.zeros(n_classes)
        fp = torch.zeros(n_classes)
        fn = torch.zeros(n_classes)

        iterator = tqdm(loader, desc="Train", leave=False)

        for i, (x, y_onehot, y_int) in enumerate(iterator):
            self.zero_grad()
            y_hat, batch_loss = self.optimize(x, y_onehot, y_int.squeeze(-1))
            self.step()

            batch_loss_f = float(batch_loss.detach())
            loss_sum += batch_loss_f

            # metrics() expects probabilities, NHWC
            y_hat_act = self.act(y_hat)
            ignore = (y_int.squeeze(-1) == 255).cpu().numpy()
            _tp, _fp, _fn = self.metrics(y_hat_act, y_onehot, ignore, threshold)

            tp += _tp
            fp += _fp
            fn += _fn

            iterator.set_description(
                f"Train Ep={epoch} Step={i} "
                f"Loss={batch_loss_f:.3f} Avg={loss_sum/(i+1):.3f}"
            )

        avg_loss = loss_sum / (i + 1)
        metrics_dict = self._compute_metrics_dict(tp, fp, fn, metric_names)
        loss_alpha = self.get_loss_alpha()
        return avg_loss, metrics_dict, loss_alpha

    def validate_one_epoch(self, epoch, loader, test=False):
        """
        Validation or test pass.

        Uses argmax over classes (no thresholds) for metrics, matching original val logic.
        """
        n_classes = self.num_classes
        metric_names = self.metrics_opts.metrics

        total_loss = 0.0
        count_batches = 0

        tp_tot = torch.zeros(n_classes)
        fp_tot = torch.zeros(n_classes)
        fn_tot = torch.zeros(n_classes)

        desc = "Test" if test else "Val"
        iterator = tqdm(loader, desc=desc, leave=False)

        for i, (x, y_onehot, y_int) in enumerate(iterator):
            # logits, NHWC
            y_hat_logits = self.infer(x)

            # loss expects NCHW
            y_hat_ch = y_hat_logits.permute(0, 3, 1, 2)
            y_onehot_ch = y_onehot.permute(0, 3, 1, 2)
            batch_loss = self.calc_loss(
                y_hat_ch,
                y_onehot_ch,
                y_int.squeeze(-1),
            )
            batch_loss_f = float(batch_loss.detach())

            y_hat_act = self.act(y_hat_logits)

            ignore = (y_int.squeeze(-1) == 255).cpu()  # (B,H,W) bool
            if ignore.all():
                continue

            y_true_cls = torch.argmax(y_onehot, dim=-1).cpu()   # (B,H,W)
            y_pred_cls = torch.argmax(y_hat_act.cpu(), dim=-1)  # (B,H,W)

            valid = ~ignore
            y_true_valid = y_true_cls[valid]
            y_pred_valid = y_pred_cls[valid]

            for c in range(n_classes):
                pred_c = (y_pred_valid == c).long()
                true_c = (y_true_valid == c).long()
                tp_c, fp_c, fn_c = tp_fp_fn(pred_c, true_c)
                tp_tot[c] += tp_c
                fp_tot[c] += fp_c
                fn_tot[c] += fn_c

            total_loss += batch_loss_f
            count_batches += 1

            iterator.set_description(
                f"{desc} Ep={epoch} Step={i} "
                f"Loss={batch_loss_f:.3f} Avg={total_loss/max(count_batches,1):.3f}"
            )

        if count_batches == 0:
            print(f"[WARN] {desc}: all patches ignored.")
            metrics_dict = self._compute_metrics_dict(tp_tot, fp_tot, fn_tot, metric_names)
            return 0.0, metrics_dict

        avg_loss = total_loss / count_batches

        if not test:
            self.val_operations(avg_loss)

        metrics_dict = self._compute_metrics_dict(tp_tot, fp_tot, fn_tot, metric_names)
        return avg_loss, metrics_dict

    # ================================================================
    # CHECKPOINT I/O
    # ================================================================
    def save(self, out_dir, epoch):
        """
        Save checkpoint (state dicts + configs).
        """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "sigma_list": self.sigma_list,
            "loss_opts": self.loss_opts,
            "loader_opts": self.loader_opts,
            "model_opts": self.model_opts,
            "optimizer_opts": self.optimizer_opts,
            "reg_opts": self.reg_opts,
            "metrics_opts": self.metrics_opts,
            "training_opts": self.training_opts,
            "scheduler_opts": self.scheduler_opts,
        }
        model_path = Path(out_dir, f"model_{epoch}.pt")
        torch.save(state, model_path)
        print(f"Saved model {epoch}")

    # ================================================================
    # INFERENCE + LOSS
    # ================================================================
    def infer(self, x):
        """
        Inference with no grad.

        x: (N,H,W,C)
        Returns logits (N,H,W,C)
        """
        training = self.model.training
        x = x.permute(0, 3, 1, 2).to(self.device, non_blocking=True)
        self.model.eval()
        with torch.no_grad():
            y = self.model(x)
        self.model.train(training)
        return y.permute(0, 2, 3, 1)

    def calc_loss(self, y_hat, y_onehot, y_int):
        """
        Compute total loss including sigma weighting and optional regularization.

        y_hat:    (N,C,H,W) logits
        y_onehot: (N,C,H,W)
        y_int:    (N,H,W)
        """
        y_hat = y_hat.to(self.device)
        y_onehot = y_onehot.to(self.device)
        y_int = y_int.to(self.device)

        losses = self.loss_fn(y_hat, y_onehot, y_int)
        total_loss = torch.zeros(1, device=self.device)
        sigma_mult = torch.ones(1, device=self.device)

        for _loss, sig in zip(losses, self.sigma_list):
            weighted_loss = _loss / (len(self.sigma_list) * sig**2)
            total_loss += weighted_loss
            sigma_mult *= sig

        total_loss += torch.abs(torch.log(sigma_mult))

        if self.reg_opts:
            reg_map = {"l1_reg": l1_reg, "l2_reg": l2_reg}
            for reg_type, coeff in self.reg_opts.items():
                if reg_type not in reg_map:
                    raise ValueError(f"Unknown regularization type: {reg_type}")
                total_loss += reg_map[reg_type](self.model.parameters(), coeff, self.device)

        return total_loss

    def get_loss_alpha(self):
        return [sigma.item() for sigma in self.sigma_list]

    # ================================================================
    # METRICS HELPERS (for training-time patch metrics)
    # ================================================================
    def metrics(self, y_hat, y, mask, threshold):
        """
        Original metric computation (patch-level):

         - y_hat: after activation (softmax), NHWC
         - y:     one-hot GT, NHWC
         - mask:  boolean ignore mask (H,W) per batch element
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

        tp_t = torch.zeros(n_classes)
        fp_t = torch.zeros(n_classes)
        fn_t = torch.zeros(n_classes)
        for i in range(0, n_classes):
            _y_hat = (y_hat == i + 1).astype(np.uint8)
            _y = (y == i + 1).astype(np.uint8)
            _tp, _fp, _fn = tp_fp_fn(_y_hat, _y)
            tp_t[i] = _tp
            fp_t[i] = _fp
            fn_t[i] = _fn

        return tp_t, fp_t, fn_t

    def segment(self, y_hat):
        if self.multi_class:
            y_hat = torch.argmax(y_hat, axis=3)
            y_hat = torch.nn.functional.one_hot(
                y_hat, num_classes=self.num_classes
            )
        else:
            y_hat = torch.sigmoid(y_hat)
        return y_hat

    def act(self, logits):
        if self.multi_class:
            return torch.nn.Softmax(3)(logits)
        return torch.sigmoid(logits)

    # ================================================================
    # OTHER UTILS
    # ================================================================
    def log_metrics(self, writer, metrics, epoch, stage):
        for metric_name, values in metrics.items():
            for class_name, value in zip(self.mask_names, values):
                writer.add_scalar(f"{stage}/{metric_name}_{class_name}", float(value), epoch)

    def freeze_layers(self, layers=None):
        for i, layer in enumerate(self.model.parameters()):
            if layers is None:
                layer.requires_grad = False
            elif i < layers:
                layer.requires_grad = False

    def find_lr(self, train_loader, init_value, final_value):
        """
        LR finder (unchanged logic).
        """
        number_in_epoch = len(train_loader) - 1
        update_step = (final_value / init_value) ** (1 / number_in_epoch)
        lr = init_value
        self.optimizer.param_groups[0]["lr"] = lr
        best_loss = 0.0
        batch_num = 0
        losses = []
        log_lrs = []
        iterator = tqdm(
            train_loader,
            desc="Current lr=XX.XX Steps=XX Loss=XX.XX Best lr=XX.XX ",
        )
        for i, data in enumerate(iterator):
            batch_num += 1
            inputs, labels = data
            self.optimizer.zero_grad()
            inputs = inputs.permute(0, 3, 1, 2).to(self.device)
            labels = labels.permute(0, 3, 1, 2).to(self.device)
            outputs = self.model(inputs)
            # Note: this call assumes binary (old API); kept only for legacy debugging
            loss = self.calc_loss(outputs, labels, torch.zeros(labels.shape[0:3], dtype=torch.long, device=self.device))
            if batch_num > 1 and loss > 4 * best_loss:
                return log_lrs[10:-5], losses[10:-5]
            if loss < best_loss or batch_num == 1:
                best_loss = loss
                best_lr = lr
            loss.backward()
            self.optimizer.step()
            iterator.set_description(
                "Current lr=%5.9f Steps=%d Loss=%5.3f Best lr=%5.9f "
                % (lr, i, loss, best_lr)
            )
            losses.append(loss.detach())
            log_lrs.append(math.log10(lr))
            lr = lr * update_step
            self.optimizer.param_groups[0]["lr"] = lr
        return log_lrs[10:-5], losses[10:-5]

    def lr_finder(self, train_loader, init_value=1e-9, final_value=1.0):
        """
        Thin wrapper around Framework.find_lr().
        Produces an LR curve plot (lr_curve.png).
        """
        import matplotlib.pyplot as plt

        logs, losses = self.find_lr(train_loader, init_value, final_value)

        plt.figure(figsize=(8, 5))
        plt.plot(logs, losses)
        plt.xlabel("Learning Rate (log10)")
        plt.ylabel("Loss")
        plt.title("LR Finder")
        plt.grid(True)
        plt.savefig("lr_curve.png")
        print("LR curve saved to lr_curve.png")

        return logs, losses

    def compute_dataset_stats(self, name, loader):
        """
        Compute per-class pixel frequency statistics.
        Returns a summary dict.
        """
        from collections import defaultdict

        total_counts = defaultdict(int)
        total_pixels = 0
        num_images = 0

        for _, _, y_int in loader:
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

    def print_stats_table(self, results):
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

    def log_stats_tensorboard(self, writer, results):
        """
        Write dataset statistics to TensorBoard.
        """
        for res in results:
            prefix = f"dataset_stats/{res['dataset']}"
            for cls, info in res["stats"].items():
                cname = cls.replace(" ", "_").replace("(", "").replace(")", "")
                writer.add_scalar(f"{prefix}/{cname}_percent", info["percent"], 0)
                writer.add_scalar(f"{prefix}/{cname}_count", info["count"], 0)

    def print_epoch_summary(self, epoch, train_metric, val_metric, test_metric=None):
        """
        Print end-of-epoch table showing precision/recall/IoU per class.
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
            # ("Test",  test_metric if test_metric else val_metric),
        ]:
            for i, cname in enumerate(self.mask_names):
                print("{:<8} {:<12} {:<10} {:<10} {:<10}".format(
                    split,
                    cname,
                    fmt(metrics["precision"][i]),
                    fmt(metrics["recall"][i]),
                    fmt(metrics["IoU"][i]),
                ))
            print("-" * 25)
        print("")

    def log_images(self, writer, batch, epoch, stage, normalize):
        """
        Logs the SAME 8-panel composite used in predictor:
        TIFF | GT | PRED | CONF | TP | FP | FN | ENTROPY

        Includes a metrics header (precision/recall/IoU).
        """

        # ------------------------------------
        # 1. Fetch batch sample
        # ------------------------------------
        x, y_onehot, y_int = next(iter(batch))
        x = x.to(self.device)
        y_onehot = y_onehot.to(self.device)

        # Forward pass
        with torch.no_grad():
            y_hat = self.act(self.infer(x))

        # Convert to NumPy
        x_np = x.cpu().numpy()[0]
        yhat_np = y_hat.cpu().numpy()[0]
        y_gt = torch.argmax(y_onehot, dim=-1).cpu().numpy()[0]
        y_pred = torch.argmax(y_hat.cpu(), dim=-1).numpy()[0]

        # Ignore mask
        ignore_mask = (y_int.cpu().numpy()[0] == 255).squeeze()
        y_gt[ignore_mask] = 255
        y_pred[ignore_mask] = 255

        # ------------------------------------
        # 2. Categorical styling
        # ------------------------------------
        num_classes = self.num_classes
        is_binary = self.is_binary
        classname = self.mask_names[-1] if is_binary else None


        # cmap = build_cmap(num_classes, is_binary, classname)
        cmap = build_cmap_from_mask_names(self.mask_names)

        x_rgb = make_rgb_preview(x_np)
        gt_rgb = label_to_color(y_gt, cmap)
        pr_rgb = label_to_color(y_pred, cmap)

        # ------------------------------------
        # 3. Confidence & entropy maps
        # ------------------------------------
        if is_binary:
            conf = yhat_np[..., 1]
        else:
            conf = np.max(yhat_np, axis=-1)

        conf_rgb = make_confidence_map(conf, invalid_mask=ignore_mask)
        entropy_rgb = make_entropy_map(yhat_np, invalid_mask=ignore_mask)

        # ------------------------------------
        # 4. TP / FP / FN
        # ------------------------------------
        tp_mask = (y_pred == y_gt) & (~ignore_mask) & (y_gt != 0)
        fp_mask = (y_pred != y_gt) & (~ignore_mask) & (y_pred != 0)
        fn_mask = (y_pred != y_gt) & (~ignore_mask) & (y_gt != 0)

        tp_rgb, fp_rgb, fn_rgb = make_tp_fp_fn_masks(tp_mask, fp_mask, fn_mask)

        # ------------------------------------
        # 5. Metrics header
        # ------------------------------------
        from glacier_mapping.model import metrics as model_metrics

        metric_string_parts = []

        for ci, cname in enumerate(self.mask_names):
            pred_c = (y_pred == ci).astype(np.uint8)
            true_c = (y_gt == ci).astype(np.uint8)

            tp, fp, fn = model_metrics.tp_fp_fn(
                torch.from_numpy(pred_c),
                torch.from_numpy(true_c),
            )
            P = model_metrics.precision(tp, fp, fn)
            R = model_metrics.recall(tp, fp, fn)
            I = model_metrics.IoU(tp, fp, fn)

            metric_string_parts.append(f"{cname}: P={P:.3f} R={R:.3f} IoU={I:.3f}")

        metrics_text = " | ".join(metric_string_parts)

        # ------------------------------------
        # 6. Assemble panel
        # ------------------------------------
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

        # ------------------------------------
        # 7. Log to TensorBoard
        # ------------------------------------
        img_tensor = torch.tensor(composite).permute(2, 0, 1).float() / 255.0
        writer.add_image(f"{stage}/visualization", img_tensor, epoch)


    def get_model_device(self):
        return self.model, self.device

    # ================================================================
    # PREDICTION HELPERS
    # ================================================================
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
        whole_arr = whole_arr[:, :, self.use_channels]
        whole_arr = self.normalize(whole_arr)
        mask = np.sum(whole_arr, axis=2) == 0

        y_pred = np.zeros((whole_arr.shape[0], whole_arr.shape[1]), dtype=np.uint8)
        for row in range(0, whole_arr.shape[0], window_size[0]):
            for column in range(0, whole_arr.shape[1], window_size[1]):
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
                    temp[: current_slice.shape[0], : current_slice.shape[1], :] = (
                        current_slice
                    )
                    current_slice = temp

                pred = self.predict_slice(
                    current_slice, threshold, preprocess=False, use_mask=False
                )

                endrow_dest = row + window_size[0]
                endrow_source = window_size[0]
                endcolumn_dest = column + window_size[1]
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
        _mask = np.sum(slice_arr, axis=2) == 0

        if preprocess:
            slice_arr = slice_arr[:, :, self.use_channels]
            slice_arr = self.normalize(slice_arr)

        _x = torch.from_numpy(np.expand_dims(slice_arr, axis=0)).float().to(self.device)
        _y = self.infer(_x).to(self.device)  # NHWC logits

        if self.multi_class:
            # Softmax along channel axis (last dim)
            threshold = None
            _y = torch.nn.functional.softmax(_y, dim=3)
            _y = torch.squeeze(_y, dim=0)   # H,W,C
            _y = _y.cpu().numpy()
            cls = np.argmax(_y, axis=2)     # H,W
            y_pred = cls.astype(np.uint8) + 1  # 1..C
        else:
            if threshold is None:
                threshold = [0.5]
            elif isinstance(threshold, (int, float)):
                threshold = [threshold]
            assert isinstance(threshold, list) and len(threshold) == 1

            _y = torch.sigmoid(_y)
            _y = np.squeeze(_y.cpu())
            y_pred = np.zeros(_y.shape, dtype=np.uint8)
            y_pred[_y >= threshold[0]] = 1

        if use_mask:
            y_pred[_mask] = 0
            return y_pred, _mask
        return y_pred

    def get_y_true(self, label_mask: np.ndarray, mask=None):
        y_true = np.zeros((label_mask.shape[0], label_mask.shape[1]), dtype=np.uint8)
        if self.is_binary:
            assert self.binary_class_idx != 0, (
                "You are trying to predict BG instead of CI or DCG"
            )
            y_true[label_mask[:, :, self.binary_class_idx - 1] != 1] = 0
            y_true[label_mask[:, :, self.binary_class_idx - 1] == 1] = 1
        else:
            for i in range(label_mask.shape[2]):
                y_true[label_mask[:, :, i] == 1] = i + 1

        if mask is not None:
            y_true[mask] = 0
        return y_true

    # ================================================================
    # FULL-TILE EVAL WITH 8-PANEL VIZ
    # ================================================================
    def evaluate_full_test_tiles(self, writer, epoch, output_dir, num_samples=4):
        """
        Full-tile evaluation with the unified 8-panel layout.
        Writes a CSV and logs optional TensorBoard scalars + images.
        """
        import cv2
        from glacier_mapping.model.visualize import (
            build_cmap_from_mask_names,
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data_dir = Path(self.loader_opts.processed_dir)
        test_tiles = sorted((data_dir / "test").glob("tiff*"))

        n_classes = self.num_classes
        threshold = self.metrics_opts.threshold
        cmap = build_cmap_from_mask_names(self.mask_names)

        rows = []
        tp_sum = np.zeros(n_classes)
        fp_sum = np.zeros(n_classes)
        fn_sum = np.zeros(n_classes)

        # ------------------- Loop over tiles ---------------------------
        for x_path in tqdm(test_tiles, desc="Full-tile eval"):
            x = np.load(x_path)
            y_pred, invalid_mask = self.predict_slice(x, threshold)

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

            # Per-tile metrics row
            row = [x_path.name]
            for ci in range(n_classes):
                label = ci + 1
                p = (y_pred_valid == label).astype(np.uint8)
                t = (y_true_valid == label).astype(np.uint8)

                tp_, fp_, fn_ = tp_fp_fn(
                    torch.from_numpy(p),
                    torch.from_numpy(t),
                )
                tp_sum[ci] += tp_
                fp_sum[ci] += fp_
                fn_sum[ci] += fn_

                row += [
                    precision(tp_, fp_, fn_),
                    recall(tp_, fp_, fn_),
                    IoU(tp_, fp_, fn_),
                ]
            rows.append(row)

        # ------------------- CSV ---------------------------
        cols = ["tile"]
        for cname in self.mask_names:
            cols += [f"{cname}_precision", f"{cname}_recall", f"{cname}_IoU"]

        pd.DataFrame(rows, columns=cols).to_csv(
            output_dir / f"full_eval_epoch{epoch}.csv", index=False
        )

        # ------------------- Summary metrics ----------------
        totals = []
        for ci in range(n_classes):
            tp_, fp_, fn_ = tp_sum[ci], fp_sum[ci], fn_sum[ci]
            totals.append((precision(tp_, fp_, fn_),
                           recall(tp_, fp_, fn_),
                           IoU(tp_, fp_, fn_)))

        if writer is not None:
            for (prec, rec, iou), cname in zip(totals, self.mask_names):
                writer.add_scalar(f"fulltest_precision/{cname}", prec, epoch)
                writer.add_scalar(f"fulltest_recall/{cname}", rec, epoch)
                writer.add_scalar(f"fulltest_iou/{cname}", iou, epoch)

        # ------------------- Sample visualizations ---------------------
        num_samples = min(num_samples, len(test_tiles))

        # Get normalization parameters
        norm_type = self.loader_opts.normalize

        for idx, x_path in enumerate(test_tiles[:num_samples]):
            x_full = np.load(x_path)

            y_pred, invalid_mask = self.predict_slice(x_full, threshold)
            y_true_raw = np.load(
                x_path.with_name(x_path.name.replace("tiff", "mask"))
            ).astype(np.uint8)

            ignore = (y_true_raw == 255)
            if invalid_mask is not None:
                ignore |= invalid_mask

            # GT/PRED for visualization
            y_gt_vis = y_true_raw.copy()
            y_gt_vis[ignore] = 255
            y_pred_vis = y_pred.copy()
            y_pred_vis[ignore] = 255

            # ----------- Correct RGB visualization -----------
            x_rgb = make_rgb_preview(x_full)

            # ----------- Apply same normalization as training -----------
            x_use = x_full[..., self.loader_opts.use_channels].astype(np.float32)

            if norm_type == "mean-std":
                mean = self.norm_arr[0].astype(np.float32)
                std  = self.norm_arr[1].astype(np.float32)
                x_norm = (x_use - mean) / (std + 1e-6)

            elif norm_type == "min-max":
                minv = self.norm_arr[2].astype(np.float32)
                maxv = self.norm_arr[3].astype(np.float32)
                x_norm = (x_use - minv) / (maxv - minv + 1e-6)

            else:
                x_norm = x_use

            # ----------- Infer probability cube (correct!) -----------
            t_in = torch.from_numpy(x_norm[None, ...]).float().to(self.device)
            yhat_full = self.act(self.infer(t_in)).cpu().numpy()[0]

            # Confidence / entropy
            if self.is_binary:
                conf = yhat_full[..., 1]
            else:
                conf = np.max(yhat_full, axis=-1)

            conf_rgb = make_confidence_map(conf, invalid_mask=ignore)
            entropy_rgb = make_entropy_map(yhat_full, invalid_mask=ignore)

            # TP / FP / FN masks
            tp_mask = (y_pred == y_true_raw) & (~ignore) & (y_true_raw != 0)
            fp_mask = (y_pred != y_true_raw) & (~ignore) & (y_pred != 0)
            fn_mask = (y_pred != y_true_raw) & (~ignore) & (y_true_raw != 0)

            tp_rgb, fp_rgb, fn_rgb = make_tp_fp_fn_masks(tp_mask, fp_mask, fn_mask)

            # Per-class metrics string
            metric_string_parts = []
            for ci, cname in enumerate(self.mask_names):
                pred_c = (y_pred == ci + 1).astype(np.uint8)
                true_c = (y_true_raw == ci + 1).astype(np.uint8)

                tp_, fp_, fn_ = tp_fp_fn(
                    torch.from_numpy(pred_c),
                    torch.from_numpy(true_c),
                )
                P = precision(tp_, fp_, fn_)
                R = recall(tp_, fp_, fn_)
                I = IoU(tp_, fp_, fn_)
                metric_string_parts.append(f"{cname}: P={P:.3f} R={R:.3f} IoU={I:.3f}")

            metrics_text = " | ".join(metric_string_parts)

            composite = make_eight_panel(
                x_rgb=x_rgb,
                gt_rgb=label_to_color(y_gt_vis, cmap),
                pr_rgb=label_to_color(y_pred_vis, cmap),
                conf_rgb=conf_rgb,
                tp_rgb=tp_rgb,
                fp_rgb=fp_rgb,
                fn_rgb=fn_rgb,
                entropy_rgb=entropy_rgb,
                metrics_text=metrics_text,
            )

            out_path = output_dir / f"fulltile_{idx}_epoch{epoch}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

            if writer is not None:
                writer.add_image(
                    f"fulltest/fulltile_{idx}",
                    torch.tensor(composite).permute(2, 0, 1).float() / 255.0,
                    epoch,
                )

