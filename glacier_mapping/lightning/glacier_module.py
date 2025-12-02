"""Lightning module for glacier segmentation."""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torchmetrics import JaccardIndex as TorchMetricsIoU
from torchmetrics import Precision, Recall

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from glacier_mapping.model.losses import customloss
from glacier_mapping.model.metrics import IoU, precision, recall, tp_fp_fn
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


class GlacierSegmentationModule(pl.LightningModule):
    """Lightning module for glacier segmentation with U-Net."""

    def __init__(
        self,
        model_opts: Dict[str, Any],
        loss_opts: Dict[str, Any],
        optim_opts: Dict[str, Any],
        scheduler_opts: Optional[Dict[str, Any]] = None,
        metrics_opts: Optional[Dict[str, Any]] = None,
        training_opts: Optional[Dict[str, Any]] = None,
        reg_opts: Optional[Dict[str, Any]] = None,
        class_names: List[str] = ["BG", "CleanIce", "Debris"],
        output_classes: List[int] = [0, 1, 2],
        use_channels: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        **kwargs
    ):
        """
        Initialize Glacier segmentation module.

        Args:
            model_opts: Model configuration options
            loss_opts: Loss function configuration options
            optim_opts: Optimizer configuration options
            scheduler_opts: Learning rate scheduler options
            metrics_opts: Metrics configuration options
            training_opts: Training configuration options
            reg_opts: Regularization options
            class_names: Names for each class
            output_classes: Output class indices (0=BG, 1=CleanIce, 2=Debris)
            use_channels: Input channel indices to use
        """
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Model configuration
        self.model_opts = model_opts
        self.loss_opts = loss_opts
        self.optim_opts = optim_opts
        self.scheduler_opts = scheduler_opts
        self.metrics_opts = metrics_opts or {"metrics": ["IoU", "precision", "recall"], "threshold": [0.5, 0.5]}
        self.training_opts = training_opts or {}
        self.reg_opts = reg_opts
        self.class_names = class_names
        self.output_classes = output_classes
        self.use_channels = use_channels
        
        # Initialize model with proper parameters
        model_args = model_opts.get('args', {})
        # For binary classification, output_classes has 1 element but we need 2 output channels (BG, class)
        out_channels = 2 if len(output_classes) == 1 else len(output_classes)
        self.model = Unet(
            inchannels=len(use_channels),
            outchannels=out_channels,
            **model_args
        )
        
        # Initialize loss function - filter only supported args
        supported_loss_args = {'act', 'smooth', 'label_smoothing', 'masked', 'theta0', 'theta', 'foreground_classes', 'alpha'}
        loss_args = {k: v for k, v in loss_opts.items() if k in supported_loss_args}
        self.loss_fn = customloss(**loss_args)
        
        # Initialize learnable sigma parameters for loss weighting
        self.sigma_list = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(2)
        ])
        
        # Initialize metrics
        self._setup_metrics()
        
        # For mixed precision training - use automatic optimization
        self.automatic_optimization = True
        
        # Track best validation loss
        self.best_val_loss = float('inf')
        
    def _setup_metrics(self):
        """Setup torchmetrics for tracking."""
        self.train_metrics = nn.ModuleDict()
        self.val_metrics = nn.ModuleDict()
        
        # Setup metrics for each class
        for i, class_idx in enumerate(self.output_classes):
            if class_idx == 0:  # Skip background
                continue
                
            class_name = self.class_names[class_idx]
            
            # IoU (JaccardIndex)
            self.train_metrics[f'{class_name}_iou'] = TorchMetricsIoU(
                task="binary", average='macro'
            )
            self.val_metrics[f'{class_name}_iou'] = TorchMetricsIoU(
                task="binary", average='macro'
            )
            
            # Precision and Recall
            self.train_metrics[f'{class_name}_precision'] = Precision(
                task="binary", average='macro'
            )
            self.val_metrics[f'{class_name}_precision'] = Precision(
                task="binary", average='macro'
            )
            
            self.train_metrics[f'{class_name}_recall'] = Recall(
                task="binary", average='macro'
            )
            self.val_metrics[f'{class_name}_recall'] = Recall(
                task="binary", average='macro'
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step with automatic optimization."""
        x, y_onehot, y_int = batch
        
        # Convert from NHWC to NCHW format expected by Conv2D
        x = x.permute(0, 3, 1, 2)
        y_onehot = y_onehot.permute(0, 3, 1, 2)
        y_int = y_int.squeeze(-1)  # Remove last dimension
        
        # Forward pass
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y_onehot, y_int)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Update and log additional metrics
        self._update_metrics(y_hat, y_int, self.train_metrics, 'train')
        
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        x, y_onehot, y_int = batch
        
        # Convert from NHWC to NCHW format expected by Conv2D
        x = x.permute(0, 3, 1, 2)
        y_onehot = y_onehot.permute(0, 3, 1, 2)
        y_int = y_int.squeeze(-1)  # Remove last dimension
        
        with torch.no_grad():
            y_hat = self(x)
            loss = self.compute_loss(y_hat, y_onehot, y_int)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Update and log additional metrics
        self._update_metrics(y_hat, y_int, self.val_metrics, 'val')
        
        # Track best validation loss
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            self.log('best_val_loss', self.best_val_loss)
        
        return loss

    def _update_metrics(self, y_hat: torch.Tensor, y_int: torch.Tensor, 
                       metrics_dict: nn.ModuleDict, prefix: str):
        """Update metrics with predictions and targets."""
        # Convert predictions to probabilities
        y_prob = torch.softmax(y_hat, dim=1)
        
        # Update metrics for each class
        for i, class_idx in enumerate(self.output_classes):
            if class_idx == 0:  # Skip background
                continue
                
            class_name = self.class_names[class_idx]
            
            # Get binary predictions for this class
            if len(self.output_classes) == 2:  # Binary case
                y_pred_class = (y_prob[:, 1] > 0.5).float()
                y_true_class = (y_int == class_idx).float()
            else:  # Multi-class case
                y_pred_class = (y_prob[:, class_idx] > 0.5).float()
                y_true_class = (y_int == class_idx).float()
            
            # Update metrics
            if f'{class_name}_iou' in metrics_dict:
                metrics_dict[f'{class_name}_iou'].update(y_pred_class, y_true_class.int())
                iou_value = metrics_dict[f'{class_name}_iou'].compute()
                self.log(f'{prefix}_{class_name}_iou', iou_value, 
                        on_step=False, on_epoch=True)
            
            if f'{class_name}_precision' in metrics_dict:
                metrics_dict[f'{class_name}_precision'].update(y_pred_class, y_true_class.int())
                precision_value = metrics_dict[f'{class_name}_precision'].compute()
                self.log(f'{prefix}_{class_name}_precision', precision_value, 
                        on_step=False, on_epoch=True)
            
            if f'{class_name}_recall' in metrics_dict:
                metrics_dict[f'{class_name}_recall'].update(y_pred_class, y_true_class.int())
                recall_value = metrics_dict[f'{class_name}_recall'].compute()
                self.log(f'{prefix}_{class_name}_recall', recall_value, 
                        on_step=False, on_epoch=True)

    def compute_loss(self, y_hat: torch.Tensor, y_onehot: torch.Tensor, 
                   y_int: torch.Tensor) -> torch.Tensor:
        """Compute custom loss with sigma weighting."""
        # Compute custom loss (returns list of losses)
        losses = self.loss_fn(y_hat, y_onehot, y_int)
        
        # Apply sigma weighting like in original Framework
        total_loss = torch.zeros(1, device=y_hat.device)
        sigma_mult = torch.ones(1, device=y_hat.device)
        
        for _loss, sig in zip(losses, self.sigma_list):
            weighted_loss = _loss / (len(self.sigma_list) * sig**2)
            total_loss += weighted_loss
            sigma_mult *= sig
        
        return total_loss[0]

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Setup optimizer
        optimizer_name = self.optim_opts.get('name', 'AdamW')
        optimizer_args = self.optim_opts.get('args', {})
        
        if optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), **optimizer_args)
        elif optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), **optimizer_args)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Setup scheduler if specified
        scheduler = None
        if self.scheduler_opts:
            scheduler_name = self.scheduler_opts.get('name', 'OneCycleLR')
            scheduler_args = self.scheduler_opts.get('args', {})
            
            if scheduler_name == 'OneCycleLR':
                # Need total_steps for OneCycleLR
                total_steps = int(self.trainer.estimated_stepping_batches)
                scheduler = OneCycleLR(optimizer, total_steps=total_steps, **scheduler_args)
            elif scheduler_name == 'ReduceLROnPlateau':
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
        """Reset training metrics at end of epoch."""
        for metric in self.train_metrics.values():
            metric.reset()

    def on_validation_epoch_end(self):
        """Reset validation metrics at end of epoch."""
        for metric in self.val_metrics.values():
            metric.reset()