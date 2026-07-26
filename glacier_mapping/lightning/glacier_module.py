import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torchmetrics import JaccardIndex as TorchMetricsIoU
from torchmetrics import Precision, Recall

import pytorch_lightning as pl

from glacier_mapping.model.losses import customloss
from glacier_mapping.model.unet import Unet as CustomUnet


class GlacierSegmentationModule(pl.LightningModule):
    def __init__(
        self,
        model_opts: Dict[str, Any],
        loss_opts: Dict[str, Any],
        optim_opts: Dict[str, Any],
        scheduler_opts: Optional[Dict[str, Any]] = None,
        metrics_opts: Optional[Dict[str, Any]] = None,
        training_opts: Optional[Dict[str, Any]] = None,
        reg_opts: Optional[Dict[str, Any]] = None,
        loader_opts: Optional[Dict[str, Any]] = None,
        class_names: List[str] = ["BG", "CleanIce", "Debris"],
        output_classes: List[int] = [0, 1, 2],
        landsat_channels=True,
        dem_channels=True,
        spectral_indices_channels=True,
        hsv_channels=True,
        physics_channels=False,
        velocity_channels=True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_opts = model_opts
        self.loss_opts = loss_opts
        self.use_velocity_loss = self.loss_opts.get("use_velocity_loss", False)
        self.optim_opts = optim_opts
        self.scheduler_opts = scheduler_opts
        self.metrics_opts = metrics_opts or {
            "metrics": ["IoU", "precision", "recall"],
            "threshold": [0.5, 0.5],
        }
        self.training_opts = training_opts or {}
        self.diagnostic_log_on_step = bool(
            self.training_opts.get("diagnostic_log_on_step", False)
        )
        self.reg_opts = reg_opts
        self.class_names = class_names
        self.output_classes = (
            loader_opts.get("output_classes", output_classes)
            if loader_opts
            else output_classes
        )

        self.landsat_channels = landsat_channels
        self.dem_channels = dem_channels
        self.spectral_indices_channels = spectral_indices_channels
        self.hsv_channels = hsv_channels
        self.physics_channels = physics_channels
        self.velocity_channels = velocity_channels

        if loader_opts:
            self.processed_dir = loader_opts.get("processed_dir", "/tmp")
            self.normalization = loader_opts.get("normalize", "mean-std")
            self.robust_scaling = loader_opts.get("robust_scaling", True)
        else:
            self.processed_dir = "/tmp"
            self.normalization = "mean-std"
            self.robust_scaling = True

        from glacier_mapping.data.data import load_band_names

        self.band_names = load_band_names(self.processed_dir)
        self.use_channels = list(range(len(self.band_names)))

        self._load_normalization_params()

        model_args = model_opts.get("args", {})
        out_channels = 2 if len(output_classes) == 1 else len(output_classes)
        n_channels = len(self.band_names)

        framework = model_opts.get("framework", "custom")
        if framework == "smp":
            import segmentation_models_pytorch as smp

            arch_name = model_opts.get("name", "Unet")
            smp_class = getattr(smp, arch_name)
            self.model = smp_class(
                in_channels=n_channels, classes=out_channels, **model_args
            )
        else:
            self.model = CustomUnet(
                inchannels=n_channels, outchannels=out_channels, **model_args
            )

        channels_last = bool(model_opts.get("channels_last", False))
        if channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.channels_last = channels_last

        supported_loss_args = {
            "act",
            "smooth",
            "label_smoothing",
            "theta0",
            "theta",
            "class_weights",
            "behavior",
            "binary_non_target_policy",
            "velocity_high_speed_threshold",
            "velocity_loss_weight",
            "velocity_loss_warmup_epochs",
            "velocity_loss_ramp_epochs",
        }
        loss_args = {k: v for k, v in loss_opts.items() if k in supported_loss_args}

        loss_args["output_classes"] = self.output_classes
        self.loss_fn = customloss(**loss_args)
        self.loss_behavior = str(loss_opts.get("behavior", "modern"))
        self.aryal_2023_behavior = self.loss_behavior == "aryal_2023"

        sigma_init = float(model_opts.get("sigma_init", 0.5))
        sigma_floor = float(model_opts.get("sigma_floor", 0.0))
        default_min_log_var = -3.0
        if sigma_floor > 0:
            default_min_log_var = math.log(sigma_floor**2)
        self.min_log_var = float(model_opts.get("min_log_var", default_min_log_var))
        self.max_log_var = float(model_opts.get("max_log_var", 2.0))
        if self.min_log_var >= self.max_log_var:
            raise ValueError(
                f"min_log_var ({self.min_log_var}) must be < max_log_var "
                f"({self.max_log_var})"
            )

        sigma_dice_init = float(model_opts.get("sigma_dice_init", sigma_init))
        sigma_boundary_init = float(model_opts.get("sigma_boundary_init", sigma_init))

        self.learnable_sigma_dice = bool(model_opts.get("learnable_sigma_dice", True))
        self.learnable_sigma_boundary = bool(
            model_opts.get("learnable_sigma_boundary", True)
        )

        if self.aryal_2023_behavior:
            self.sigma_dice = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            self.sigma_boundary = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            self.raw_log_var_dice = self._make_log_var(
                "dice", sigma_dice_init, self.learnable_sigma_dice
            )
            self.raw_log_var_boundary = self._make_log_var(
                "boundary", sigma_boundary_init, self.learnable_sigma_boundary
            )

        self.velocity_idx = None
        self.velocity_mask_idx = None

        used_band_names = self.band_names
        if "velocity" in used_band_names:
            self.velocity_idx = np.where(used_band_names == "velocity")[0][0]

        if "velocity_mask" in used_band_names:
            self.velocity_mask_idx = np.where(used_band_names == "velocity_mask")[0][0]

        self._setup_velocity_denormalization()
        self._setup_metrics()
        self.automatic_optimization = True
        self.class_name_to_short = {"BG": "bg", "CleanIce": "ci", "Debris": "dci"}

    def _make_log_var(
        self, name: str, sigma_init: float, learnable: bool
    ) -> torch.Tensor:
        if sigma_init <= 0:
            raise ValueError(f"sigma_{name}_init must be > 0, got {sigma_init}")
        init_log_var = math.log(sigma_init**2)
        log_var_tensor = torch.tensor(init_log_var, dtype=torch.float32)
        if learnable:
            return nn.Parameter(log_var_tensor)
        buffer_name = f"raw_log_var_{name}_buffer"
        self.register_buffer(buffer_name, log_var_tensor)
        return getattr(self, buffer_name)

    def _setup_velocity_denormalization(self) -> None:
        self.velocity_denorm_mode = "none"
        offset = torch.tensor(0.0, dtype=torch.float32)
        scale = torch.tensor(1.0, dtype=torch.float32)

        if self.velocity_idx is not None:
            channel_idx = int(self.use_channels[int(self.velocity_idx)])
            if self.robust_scaling:
                max_val = max(float(self.norm_arr_full[3, channel_idx]), 1e-6)
                self.velocity_denorm_mode = "robust"
                scale = torch.tensor(math.log1p(max_val), dtype=torch.float32)
            elif self.normalization == "mean-std":
                self.velocity_denorm_mode = "affine"
                offset = torch.tensor(
                    float(self.norm_arr_full[0, channel_idx]), dtype=torch.float32
                )
                scale = torch.tensor(
                    float(self.norm_arr_full[1, channel_idx]), dtype=torch.float32
                )
            elif self.normalization == "min-max":
                self.velocity_denorm_mode = "affine"
                min_val = float(self.norm_arr_full[2, channel_idx])
                max_val = float(self.norm_arr_full[3, channel_idx])
                offset = torch.tensor(min_val, dtype=torch.float32)
                scale = torch.tensor(max_val - min_val, dtype=torch.float32)
            else:
                raise ValueError(f"Invalid normalization: {self.normalization}")

        self.register_buffer("velocity_denorm_offset", offset, persistent=False)
        self.register_buffer("velocity_denorm_scale", scale, persistent=False)

    def _clamp_log_var(self, raw_log_var: torch.Tensor) -> torch.Tensor:
        return torch.clamp(raw_log_var, min=self.min_log_var, max=self.max_log_var)

    def _sigma_from_log_var(self, raw_log_var: torch.Tensor) -> torch.Tensor:
        return torch.exp(0.5 * self._clamp_log_var(raw_log_var))

    def _uncertainty_weighted_loss(
        self, component_loss: torch.Tensor, raw_log_var: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_var = self._clamp_log_var(raw_log_var)
        weighted_loss = 0.5 * torch.exp(-log_var) * component_loss
        uncertainty_penalty = 0.5 * log_var
        return weighted_loss + uncertainty_penalty, weighted_loss, uncertainty_penalty

    def _setup_metrics(self):
        self.train_metrics = nn.ModuleDict()
        self.val_metrics = nn.ModuleDict()

        for i, class_idx in enumerate(self.output_classes):
            if class_idx == 0:
                continue

            class_name = self.class_names[class_idx]

            self.train_metrics[f"{class_name}_iou"] = TorchMetricsIoU(
                task="binary", average="macro"
            )
            self.val_metrics[f"{class_name}_iou"] = TorchMetricsIoU(
                task="binary", average="macro"
            )

            self.train_metrics[f"{class_name}_precision"] = Precision(
                task="binary", average="macro"
            )
            self.val_metrics[f"{class_name}_precision"] = Precision(
                task="binary", average="macro"
            )

            self.train_metrics[f"{class_name}_recall"] = Recall(
                task="binary", average="macro"
            )
            self.val_metrics[f"{class_name}_recall"] = Recall(
                task="binary", average="macro"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y_uint8 = batch
        if x.dtype != torch.float32:
            x = x.float()
        y_hat = self(x)

        velocity = None
        velocity_mask = None

        if (
            self.use_velocity_loss
            and self.velocity_idx is not None
            and self.velocity_mask_idx is not None
        ):
            vel_norm = x[:, self.velocity_idx : self.velocity_idx + 1, :, :]
            velocity = self._denormalize_velocity(vel_norm)
            velocity_mask = x[
                :, self.velocity_mask_idx : self.velocity_mask_idx + 1, :, :
            ]

        loss = self.compute_loss(
            y_hat,
            y_uint8,
            velocity=velocity,
            velocity_mask=velocity_mask,
            stage="train",
        )

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=False,
        )

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y_uint8 = batch
        if x.dtype != torch.float32:
            x = x.float()
        with torch.no_grad():
            y_hat = self(x)

            velocity = None
            velocity_mask = None

            if (
                self.use_velocity_loss
                and self.velocity_idx is not None
                and self.velocity_mask_idx is not None
            ):
                vel_norm = x[:, self.velocity_idx : self.velocity_idx + 1, :, :]
                velocity = self._denormalize_velocity(vel_norm)
                velocity_mask = x[
                    :, self.velocity_mask_idx : self.velocity_mask_idx + 1, :, :
                ]

            loss = self.compute_loss(
                y_hat,
                y_uint8,
                velocity=velocity,
                velocity_mask=velocity_mask,
                stage="val",
            )

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._update_metrics(y_hat, y_uint8, self.val_metrics, "val")

        return loss

    def _update_metrics(
        self,
        y_hat: torch.Tensor,
        y_int: torch.Tensor,
        metrics_dict: nn.ModuleDict,
        prefix: str,
    ):
        y_prob = torch.softmax(y_hat, dim=1)
        y_pred_argmax = torch.argmax(y_prob, dim=1)
        thresholds = self.metrics_opts.get(
            "threshold", [0.5 for _ in range(len(self.class_names))]
        )

        valid_mask = y_int != 255
        if valid_mask.sum() == 0:
            return

        for i, class_idx in enumerate(self.output_classes):
            if class_idx == 0:
                continue

            class_name = self.class_names[class_idx]
            target = self.class_name_to_short.get(class_name, class_name.lower())

            if len(self.output_classes) == 1:
                pos_prob = y_prob[:, 1]
                thr = thresholds[class_idx] if class_idx < len(thresholds) else 0.5
                y_pred_class = (pos_prob >= thr).float()
                y_true_class = (y_int == class_idx).float()
            else:
                y_pred_class = (y_pred_argmax == class_idx).float()
                y_true_class = (y_int == class_idx).float()

            y_pred_class = y_pred_class[valid_mask]
            y_true_class = y_true_class[valid_mask]

            if y_true_class.numel() == 0:
                continue

            if f"{class_name}_iou" in metrics_dict:
                iou_metric: Any = metrics_dict[f"{class_name}_iou"]
                iou_metric.update(y_pred_class, y_true_class.int())
                iou_value = iou_metric.compute()
                self.log(
                    f"slice_{prefix}_{target}_iou",
                    iou_value,
                    on_step=False,
                    on_epoch=True,
                )

            if f"{class_name}_precision" in metrics_dict:
                precision_metric: Any = metrics_dict[f"{class_name}_precision"]
                precision_metric.update(y_pred_class, y_true_class.int())
                precision_value = precision_metric.compute()
                self.log(
                    f"slice_{prefix}_{target}_precision",
                    precision_value,
                    on_step=False,
                    on_epoch=True,
                )

            if f"{class_name}_recall" in metrics_dict:
                recall_metric: Any = metrics_dict[f"{class_name}_recall"]
                recall_metric.update(y_pred_class, y_true_class.int())
                recall_value = recall_metric.compute()
                self.log(
                    f"slice_{prefix}_{target}_recall",
                    recall_value,
                    on_step=False,
                    on_epoch=True,
                )

    def compute_loss(
        self,
        y_hat: torch.Tensor,
        y_uint8: torch.Tensor,
        velocity: Optional[torch.Tensor] = None,
        velocity_mask: Optional[torch.Tensor] = None,
        stage: Optional[str] = None,
    ) -> torch.Tensor:
        log_prefix = f"{stage}_" if stage else ""
        log_on_step = stage == "train" and self.diagnostic_log_on_step
        log_diagnostics = self.diagnostic_log_on_step

        if (
            log_diagnostics
            and self.use_velocity_loss
            and velocity_mask is not None
            and y_uint8 is not None
            and velocity_mask.numel() > 0
        ):
            ignore_mask = (y_uint8 != 255).unsqueeze(1)
            masked_velocity = velocity_mask * ignore_mask
            valid_pixels = masked_velocity.sum()
            total_pixels = ignore_mask.sum()
            coverage = valid_pixels.float() / total_pixels.float().clamp_min(1.0)
            self.log(
                f"{log_prefix}velocity_valid_fraction",
                coverage,
                on_step=log_on_step,
                on_epoch=True,
            )

        losses = self.loss_fn(
            y_hat,
            y_uint8,
            velocity=velocity,
            velocity_mask=velocity_mask,
            current_epoch=self.current_epoch,
        )

        total_loss = y_hat.new_zeros(())
        if log_diagnostics:
            self.log(
                f"{log_prefix}dice_loss_raw",
                losses[0].detach(),
                on_step=log_on_step,
                on_epoch=True,
            )
            self.log(
                f"{log_prefix}boundary_loss_raw",
                losses[1].detach(),
                on_step=log_on_step,
                on_epoch=True,
            )

        if (
            log_diagnostics
            and self.use_velocity_loss
            and getattr(self.loss_fn, "last_velocity_valid", False)
            and len(losses) >= 3
        ):
            velocity_base = self.loss_fn.last_velocity_base
            velocity_loss = self.loss_fn.last_velocity_loss
            if velocity_base is None or velocity_loss is None:
                raise RuntimeError("Velocity loss marked valid without logged values.")
            self.log(
                f"{log_prefix}velocity_loss_raw",
                velocity_base,
                on_step=log_on_step,
                on_epoch=True,
            )
            self.log(
                f"{log_prefix}velocity_loss_weighted",
                velocity_loss,
                on_step=log_on_step,
                on_epoch=True,
            )
            self.log(
                f"{log_prefix}velocity_loss_weight",
                y_hat.new_tensor(getattr(self.loss_fn, "last_velocity_weight", 0.0)),
                on_step=log_on_step,
                on_epoch=True,
            )

        if log_diagnostics:
            self.log(
                f"{log_prefix}sigma_dice",
                self._sigma_from_log_var(self.raw_log_var_dice),
                on_step=log_on_step,
                on_epoch=True,
            )
            self.log(
                f"{log_prefix}sigma_boundary",
                self._sigma_from_log_var(self.raw_log_var_boundary),
                on_step=log_on_step,
                on_epoch=True,
            )
            self.log(
                f"{log_prefix}log_var_dice",
                self._clamp_log_var(self.raw_log_var_dice),
                on_step=log_on_step,
                on_epoch=True,
            )
            self.log(
                f"{log_prefix}log_var_boundary",
                self._clamp_log_var(self.raw_log_var_boundary),
                on_step=log_on_step,
                on_epoch=True,
            )

        if self.aryal_2023_behavior:
            total_loss = (
                losses[0] / (2 * torch.square(self.sigma_dice))
                + losses[1] / (2 * torch.square(self.sigma_boundary))
                + torch.abs(torch.log(self.sigma_dice * self.sigma_boundary))
            )
            return total_loss.abs()

        weighted_components = [
            ("dice", losses[0], self.raw_log_var_dice),
            ("boundary", losses[1], self.raw_log_var_boundary),
        ]

        for name, component_loss, raw_log_var in weighted_components:
            (
                component_total,
                weighted_loss,
                uncertainty_penalty,
            ) = self._uncertainty_weighted_loss(component_loss, raw_log_var)
            total_loss = total_loss + component_total
            if log_diagnostics:
                self.log(
                    f"{log_prefix}{name}_loss_uncertainty_weighted",
                    weighted_loss.detach(),
                    on_step=log_on_step,
                    on_epoch=True,
                )
                self.log(
                    f"{log_prefix}{name}_loss_uncertainty_penalty",
                    uncertainty_penalty.detach(),
                    on_step=log_on_step,
                    on_epoch=True,
                )

        if self.use_velocity_loss and len(losses) >= 3:
            total_loss = total_loss + losses[2]
            if log_diagnostics:
                self.log(
                    f"{log_prefix}velocity_loss_fixed_weighted",
                    losses[2].detach(),
                    on_step=log_on_step,
                    on_epoch=True,
                )

        return total_loss

    def configure_optimizers(self) -> Any:
        optimizer_name = self.optim_opts.get("name", "AdamW")
        optimizer_args = self.optim_opts.get("args") or {}

        if "weight_decay" in optimizer_args and isinstance(
            optimizer_args["weight_decay"], str
        ):
            try:
                optimizer_args["weight_decay"] = float(optimizer_args["weight_decay"])
            except ValueError:
                optimizer_args["weight_decay"] = float(
                    str(optimizer_args["weight_decay"]).replace("E", "e")
                )

        if "lr" in optimizer_args and isinstance(optimizer_args["lr"], str):
            try:
                optimizer_args["lr"] = float(optimizer_args["lr"])
            except ValueError:
                optimizer_args["lr"] = float(optimizer_args["lr"].replace("e", "e-"))

        fused = optimizer_args.pop("fused", None)
        if fused is True and not torch.cuda.is_available():
            raise RuntimeError(
                "optimizer fused=True requires CUDA, but no GPU is available"
            )

        param_groups = [
            {"params": self.model.parameters(), **optimizer_args},
        ]
        log_var_optimizer_args = {**optimizer_args, "weight_decay": 0.0}
        if self.aryal_2023_behavior:
            param_groups.extend(
                [
                    {"params": [self.sigma_dice], **log_var_optimizer_args},
                    {"params": [self.sigma_boundary], **log_var_optimizer_args},
                ]
            )
        else:
            if isinstance(self.raw_log_var_dice, nn.Parameter):
                param_groups.append(
                    {"params": [self.raw_log_var_dice], **log_var_optimizer_args}
                )
            if isinstance(self.raw_log_var_boundary, nn.Parameter):
                param_groups.append(
                    {"params": [self.raw_log_var_boundary], **log_var_optimizer_args}
                )

        kw = {**optimizer_args}
        if fused is True:
            kw["fused"] = True

        if optimizer_name == "AdamW":
            try:
                optimizer = torch.optim.AdamW(param_groups, **kw)
            except (TypeError, RuntimeError) as e:
                raise RuntimeError(
                    f"Failed to create AdamW with fused=True: {e}. "
                    "Remove fused from the config or install a PyTorch build that "
                    "supports fused AdamW."
                ) from e
        elif optimizer_name == "Adam":
            if fused:
                raise RuntimeError(
                    "fused=True is not supported for Adam optimizer. "
                    "Use AdamW with fused=True instead."
                )
            optimizer = torch.optim.Adam(param_groups, **kw)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        self._optimizer_name = optimizer_name
        self._fused_enabled = fused is True

        scheduler = None
        if self.scheduler_opts:
            scheduler_name = self.scheduler_opts.get("name", "OneCycleLR")
            scheduler_args = self.scheduler_opts.get("args", {})

            if scheduler_name == "OneCycleLR":
                estimated_steps = int(self.trainer.estimated_stepping_batches)
                if estimated_steps <= 0:
                    total_steps = 1
                else:
                    total_steps = estimated_steps
                scheduler = OneCycleLR(
                    optimizer, total_steps=total_steps, **scheduler_args
                )
            elif scheduler_name == "ReduceLROnPlateau":
                scheduler = ReduceLROnPlateau(optimizer, **scheduler_args)

        if scheduler:
            if isinstance(scheduler, OneCycleLR):
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",
                    },
                }
            else:
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss",
                        "frequency": 1,
                    },
                }
        else:
            return optimizer

    def on_train_epoch_end(self):
        for metric_name, metric_obj in self.train_metrics.items():
            metric_obj: Any = metric_obj
            metric_obj.reset()

    def on_validation_model_eval(self) -> None:
        if self.aryal_2023_behavior:
            # Aryal Framework never switches the model to eval mode. Validation
            # therefore uses batch statistics and active spatial dropout.
            self.model.train()
        else:
            super().on_validation_model_eval()

    def _aryal_second_validation_forward_pass(self) -> None:
        """Preserve upstream's accidental second pass over val as "test"."""
        if not self.aryal_2023_behavior or self.trainer.sanity_checking:
            return
        loader = self.trainer.val_dataloaders
        if isinstance(loader, (list, tuple)):
            loader = loader[0]
        self.model.train()
        with torch.no_grad():
            for x, _target in loader:
                if x.dtype != torch.float32:
                    x = x.float()
                self(x.to(self.device))

    def on_validation_epoch_end(self):
        self._aryal_second_validation_forward_pass()
        for metric_name, metric_obj in self.val_metrics.items():
            metric_obj: Any = metric_obj
            metric_obj.reset()

    def _load_normalization_params(self):
        from glacier_mapping.data.data import (
            load_band_names,
            get_no_normalize_channel_names,
        )

        norm_path = Path(self.processed_dir) / "normalize_train.npy"
        if norm_path.exists():
            self.norm_arr_full = np.load(norm_path)
            self.norm_arr = self.norm_arr_full[:2, self.use_channels]
        else:
            num_channels = len(self.use_channels)
            self.norm_arr_full = np.array(
                [
                    [0] * num_channels,
                    [1] * num_channels,
                    [0] * num_channels,
                    [1] * num_channels,
                ]
            )
            self.norm_arr = self.norm_arr_full[:2, self.use_channels]

        band_names = load_band_names(self.processed_dir)
        no_norm_names = get_no_normalize_channel_names()
        used_band_names = band_names[self.use_channels].tolist()

        self.no_normalize_mask = np.array(
            [band_names[ch] in no_norm_names for ch in self.use_channels]
        )
        self.velocity_normalize_mask_idx = (
            used_band_names.index("velocity_mask")
            if "velocity_mask" in used_band_names
            else None
        )
        self.velocity_normalize_value_indices = [
            idx
            for idx, name in enumerate(used_band_names)
            if name in {"velocity", "velocity_x", "velocity_y"}
        ]

    def _denormalize_velocity(self, vel_norm):
        if self.velocity_idx is None:
            raise ValueError("velocity_idx is None, cannot denormalize velocity.")

        if self.velocity_denorm_mode == "robust":
            return torch.expm1(vel_norm * self.velocity_denorm_scale)

        if self.velocity_denorm_mode == "affine":
            return vel_norm * self.velocity_denorm_scale + self.velocity_denorm_offset

        raise ValueError("Velocity denormalization is not configured.")

    def normalize(self, x):
        x_no_norm = None
        if hasattr(self, "no_normalize_mask") and np.any(self.no_normalize_mask):
            x_no_norm = x[:, :, self.no_normalize_mask].copy()

        if self.normalization == "mean-std":
            _mean, _std = self.norm_arr[0], self.norm_arr[1]
            x_normalized = (x - _mean) / _std
        elif self.normalization == "min-max":
            _min = self.norm_arr_full[2, self.use_channels]
            _max = self.norm_arr_full[3, self.use_channels]
            x_normalized = (np.clip(x, _min, _max) - _min) / (_max - _min)
        else:
            raise Exception("Invalid normalization")

        if x_no_norm is not None:
            x_normalized[:, :, self.no_normalize_mask] = x_no_norm

        if (
            self.velocity_normalize_mask_idx is not None
            and self.velocity_normalize_value_indices
        ):
            missing_velocity = (
                x_normalized[:, :, self.velocity_normalize_mask_idx] <= 0.5
            )
            for channel_idx in self.velocity_normalize_value_indices:
                channel = x_normalized[:, :, channel_idx]
                channel[missing_velocity] = 0.0
                x_normalized[:, :, channel_idx] = channel

        return x_normalized

    def predict_slice(
        self, slice_arr, threshold=None, preprocess=True, use_mask=True, fill_holes=True
    ):
        from glacier_mapping.model.evaluation import predict_slice as eval_predict_slice

        y_pred, mask = eval_predict_slice(
            self, slice_arr, threshold, fill_holes=fill_holes
        )
        if not use_mask:
            return y_pred
        return y_pred, mask

    def freeze_layers(self, layers=None):
        for i, param in enumerate(self.model.parameters()):
            if layers is None:
                param.requires_grad = False
            elif i < layers:
                param.requires_grad = False
