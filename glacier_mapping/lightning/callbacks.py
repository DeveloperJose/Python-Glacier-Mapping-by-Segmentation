"""Custom callbacks for glacier mapping training."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from glacier_mapping.utils.callback_utils import (
    load_dataset_metadata,
    generate_single_visualization,
    select_slices_by_iou_thirds,
    parse_slice_path,
)
from glacier_mapping.utils import cleanup_gpu_memory
import glacier_mapping.utils.logging as log


class ValidationVisualizationCallback(Callback):
    """Periodic validation visualizations using IoU-based selection.

    Uses n-based thirds system: n=4 → 12 visualizations (4 top + 4 middle + 4 bottom).
    Tracks same slices across epochs for consistent comparison.
    """

    def __init__(
        self,
        viz_n: int = 4,
        log_every_n_epochs: int = 10,
        selection: str = "iou",
        save_dir: Optional[str] = None,
        image_dir: Optional[str] = None,
        scale_factor: float = 0.5,
    ):
        """
        Initialize validation visualization callback.

        Args:
            viz_n: Number of samples per third (top, middle, bottom).
                   Total visualizations = 3 * viz_n.
                   Set to 0 to disable.
            log_every_n_epochs: How often to generate visualizations
            selection: Selection method ("iou" or "random")
            save_dir: Directory to save visualizations
            image_dir: Path to raw Landsat image TIFFs (from servers.yaml)
        """
        super().__init__()
        self.viz_n = viz_n
        self.log_every_n_epochs = log_every_n_epochs
        self.selection = selection
        self.save_dir = Path(save_dir) if save_dir else None
        self.image_dir = Path(image_dir) if image_dir else None
        self.scale_factor = scale_factor

        # Track selected slices across epochs for consistency
        self.selected_slice_paths: Optional[List[Path]] = None
        self.slice_metadata: Dict[
            Path, Tuple[int, int]
        ] = {}  # {path: (tiff_num, slice_num)}
        self.tile_rank_map: Dict[Path, int] = {}  # {path: absolute_rank}

        # Cache for metadata to avoid reloading
        self._metadata_cache = None

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
        if self.viz_n < 1:
            return

        # Get validation data directory
        processed_dir = getattr(pl_module, "processed_dir", "data/processed")
        val_dir = Path(processed_dir) / "val"  # type: ignore[arg-type]

        if not val_dir.exists():
            log.warning(f"Validation directory not found: {val_dir}")
            return

        val_slices_all = sorted(val_dir.glob("tiff*"))

        if not val_slices_all:
            log.warning("No validation slices found")
            return

        # First epoch: select slices using IoU-based thirds
        if self.selected_slice_paths is None:
            num_samples = 3 * self.viz_n
            if self.selection == "iou":
                self.selected_slice_paths = select_slices_by_iou_thirds(
                    val_slices_all, pl_module, num_samples
                )
                self.tile_rank_map = {}
            else:  # random
                self.selected_slice_paths = val_slices_all[:num_samples]
                self.tile_rank_map = {}

            # Extract metadata for each selected slice
            for path in self.selected_slice_paths:
                tiff_num, slice_num = parse_slice_path(path)
                self.slice_metadata[path] = (tiff_num, slice_num)

            log.info(
                f"Selected {len(self.selected_slice_paths)} validation slices for tracking"
            )

        # Load metadata once
        if self._metadata_cache is None:
            self._metadata_cache = load_dataset_metadata(
                pl_module, "val", self.image_dir
            )

        # Generate visualizations for tracked slices
        output_dir = self.save_dir / "val_visualizations" if self.save_dir else None

        for slice_path in self.selected_slice_paths:
            try:
                generate_single_visualization(
                    x_path=slice_path,
                    pl_module=pl_module,
                    output_dir=output_dir,
                    epoch=trainer.current_epoch + 1,
                    title_prefix="VAL",
                    metadata_cache=self._metadata_cache,
                    image_dir=self.image_dir,
                    scale_factor=self.scale_factor,
                    tile_rank_map=self.tile_rank_map,
                )
            except Exception as e:
                import traceback

                log.error(f"Error generating visualization for {slice_path}: {e}")
                log.error(f"Full traceback: {traceback.format_exc()}")

        # GPU cleanup after visualization generation
        cleanup_gpu_memory()

        # Log to both MLflow and TensorBoard (if available)
        if self.save_dir:
            from glacier_mapping.utils.callback_utils import (
                log_visualizations_to_all_loggers,
            )

            # Use the actual output directory where visualizations are saved
            val_output_dir = self.save_dir / "val_visualizations"
            log_visualizations_to_all_loggers(
                trainer, val_output_dir, trainer.current_epoch + 1, "val_visualizations"
            )


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
            import torch

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
                # Type narrowing for best_val_loss
                best_loss = pl_module.best_val_loss
                if isinstance(best_loss, (int, float)):
                    improvement = best_loss - float(val_loss)  # type: ignore[arg-type]
                    pl_module.log(
                        "val_loss_improvement",
                        improvement,
                        on_step=False,
                        on_epoch=True,
                    )
