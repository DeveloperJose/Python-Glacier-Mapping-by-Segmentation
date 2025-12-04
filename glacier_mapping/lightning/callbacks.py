"""Custom callbacks for glacier mapping training."""

from pathlib import Path
from typing import Any, Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import MLFlowLogger

from glacier_mapping.model.visualize import make_eight_panel
import glacier_mapping.utils.logging as log

# Check if MLflow is available
try:
    from pytorch_lightning.loggers import MLFlowLogger

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    MLFlowLogger = None  # type: ignore


class GlacierVisualizationCallback(Callback):
    """Callback for generating visualization samples during training."""

    def __init__(
        self,
        num_samples: int = 4,
        log_every_n_epochs: int = 10,
        save_dir: Optional[str] = None,
    ):
        """
        Initialize visualization callback.

        Args:
            num_samples: Number of samples to visualize
            log_every_n_epochs: How often to generate visualizations
            save_dir: Directory to save visualizations
        """
        super().__init__()
        self.num_samples = num_samples
        self.log_every_n_epochs = log_every_n_epochs
        self.save_dir = Path(save_dir) if save_dir else None

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """Generate visualizations at end of validation epoch."""
        if (trainer.current_epoch + 1) % self.log_every_n_epochs == 0:
            self._generate_visualizations(trainer, pl_module)

    def _generate_visualizations(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """Generate and save visualization samples."""
        # Get a batch from validation dataloader
        # Handle both single DataLoader and list of DataLoaders (PL 2.x compatibility)
        val_dataloaders = trainer.val_dataloaders

        if val_dataloaders is None:
            log.warning("No validation dataloader available for visualization")
            return

        # Extract first dataloader (handle both single and multiple cases)
        if isinstance(val_dataloaders, list):
            if len(val_dataloaders) == 0:
                log.warning("No validation dataloader available for visualization")
                return
            val_dataloader = val_dataloaders[0]
        else:
            val_dataloader = val_dataloaders

        batch = next(iter(val_dataloader))

        x, y_onehot, y_int = batch

        # Convert from NHWC to NCHW format (matching training_step/validation_step)
        x = x.permute(0, 3, 1, 2)
        y_onehot = y_onehot.permute(0, 3, 1, 2)
        y_int = y_int.squeeze(-1)  # Remove last dimension: (N, H, W, 1) -> (N, H, W)

        # Move to correct device
        device = pl_module.device
        x = x.to(device)
        y_onehot = y_onehot.to(device)
        y_int = y_int.to(device)

        # Get model predictions
        pl_module.eval()
        with torch.no_grad():
            y_hat = pl_module(x)

        # Track saved paths for MLflow logging
        saved_paths = []

        # Generate visualizations for first N samples
        for i in range(min(self.num_samples, x.shape[0])):
            # Extract single sample and convert to numpy (H, W, C) format
            # x: (N, C, H, W) -> (H, W, C)
            sample_x_np = x[i].permute(1, 2, 0).cpu().numpy()

            # y_int: (N, H, W) -> (H, W)
            sample_y_np = y_int[i].cpu().numpy()

            # y_hat: (N, C, H, W) -> (H, W) prediction
            sample_pred_np = torch.argmax(y_hat[i], dim=0).cpu().numpy()

            # Create and save 8-panel visualization
            if self.save_dir:
                # Structure: slice_XXXX/epochYYYY.png
                slice_dir = self.save_dir / f"slice_{i:04d}"
                slice_dir.mkdir(parents=True, exist_ok=True)
                save_path = slice_dir / f"epoch{trainer.current_epoch + 1:04d}.png"

                from matplotlib import pyplot as plt
                from glacier_mapping.model.visualize import (
                    make_rgb_preview,
                    label_to_color,
                    build_cmap,
                )

                # Convert input to RGB preview
                x_rgb = make_rgb_preview(sample_x_np)

                # Build colormap for ground truth and prediction
                cmap = build_cmap(
                    num_classes=len(pl_module.output_classes),
                    is_binary=(len(pl_module.output_classes) == 1),
                    classname=pl_module.class_names[pl_module.output_classes[0]]
                    if len(pl_module.output_classes) == 1
                    else None,
                )

                # Convert labels to RGB
                gt_rgb = label_to_color(sample_y_np, cmap)
                pr_rgb = label_to_color(sample_pred_np, cmap)

                # Create placeholder images for None values (black images with same size as x_rgb)
                import numpy as np

                placeholder = np.zeros_like(x_rgb, dtype=np.uint8)

                viz = make_eight_panel(
                    x_rgb=x_rgb,
                    gt_rgb=gt_rgb,
                    pr_rgb=pr_rgb,
                    conf_rgb=placeholder,  # Placeholder for confidence map
                    tp_rgb=placeholder,  # Placeholder for TP
                    fp_rgb=placeholder,  # Placeholder for FP
                    fn_rgb=placeholder,  # Placeholder for FN
                    entropy_rgb=placeholder,  # Placeholder for entropy
                    metrics_text=None,
                )

                plt.imshow(viz)
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()

                saved_paths.append((slice_dir, save_path))

        # Log to MLflow
        self._log_to_mlflow(trainer, saved_paths)

    def _log_to_mlflow(self, trainer: pl.Trainer, saved_paths: list):
        """Log saved visualizations to MLflow."""
        for logger in trainer.loggers:
            if isinstance(logger, MLFlowLogger):
                try:
                    for slice_dir, save_path in saved_paths:
                        # Log individual PNG file
                        logger.experiment.log_artifact(
                            logger.run_id,
                            str(save_path),
                            artifact_path=f"slice_visualizations/{slice_dir.name}",
                        )
                except Exception as e:
                    log.warning(f"Failed to log slice visualization to MLflow: {e}")


class GlacierModelCheckpoint(ModelCheckpoint):
    """Enhanced model checkpointing for glacier mapping."""

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 3,
        save_last: bool = True,
        every_n_epochs: int = 1,
        filename: str = "epoch_{epoch:03d}_val_{val_loss:.4f}",
        **kwargs,
    ):
        """
        Initialize glacier model checkpoint.

        Args:
            monitor: Metric to monitor for checkpointing
            mode: 'min' or 'max' for monitored metric
            save_top_k: Number of best checkpoints to keep
            save_last: Whether to save last epoch checkpoint
            every_n_epochs: Checkpoint frequency
            filename: Checkpoint filename format
        """
        super().__init__(
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=save_last,
            every_n_epochs=every_n_epochs,
            filename=filename,
            **kwargs,
        )


class GlacierTrainingMonitor(Callback):
    """Monitor training progress and log additional metrics."""

    def __init__(self, log_every_n_steps: int = 50):
        """
        Initialize training monitor.

        Args:
            log_every_n_steps: How often to log additional metrics
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        """Log additional training metrics."""
        if batch_idx % self.log_every_n_steps == 0:
            # Log GPU memory usage
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                pl_module.log("gpu_memory_gb", gpu_memory, on_step=True, on_epoch=False)

            # Log learning rate
            optimizer = trainer.optimizers[0]
            current_lr = optimizer.param_groups[0]["lr"]
            pl_module.log("learning_rate", current_lr, on_step=True, on_epoch=False)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """Log validation summary metrics."""
        # Log epoch duration
        if (
            hasattr(trainer, "callback_metrics")
            and "val_loss" in trainer.callback_metrics
        ):
            val_loss = trainer.callback_metrics["val_loss"]
            pl_module.log("epoch", trainer.current_epoch, on_step=False, on_epoch=True)

            # Log improvement over best
            if (
                hasattr(pl_module, "best_val_loss")
                and pl_module.best_val_loss is not None
            ):
                improvement = pl_module.best_val_loss - val_loss
                pl_module.log(
                    "val_loss_improvement", improvement, on_step=False, on_epoch=True
                )
