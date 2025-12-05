"""Custom callbacks for glacier mapping training."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from tqdm import tqdm

from glacier_mapping.model.metrics import IoU, precision, recall, tp_fp_fn
from glacier_mapping.model.visualize import (
    make_rgb_preview,
    make_overlay,
    make_redesigned_panel,
    make_confidence_map,
    build_binary_cmap,
    build_cmap_from_mask_names,
    calculate_slice_position,
    load_full_tiff_rgb,
    make_context_image,
    make_error_overlay,
    make_tp_fp_fn_masks,
)

from glacier_mapping.utils import cleanup_gpu_memory
import glacier_mapping.utils.logging as log
from glacier_mapping.utils.prediction import (
    calculate_binary_metrics,
    get_probabilities,
    predict_from_probs,
)

# Check if MLflow is available
try:
    from pytorch_lightning.loggers import MLFlowLogger

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    MLFlowLogger = None  # type: ignore


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

        # Track selected slices across epochs for consistency
        self.selected_slice_paths: Optional[List[Path]] = None
        self.slice_metadata: Dict[
            Path, Tuple[int, int]
        ] = {}  # {path: (tiff_num, slice_num)}

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """Generate visualizations at end of validation epoch."""
        if (trainer.current_epoch + 1) % self.log_every_n_epochs == 0:
            self._generate_visualizations(trainer, pl_module)

    def _load_dataset_metadata(self, pl_module: pl.LightningModule):
        """Load dataset metadata for context image generation."""
        import json

        import pandas as pd

        # Get processed dataset directory from model
        processed_dir = Path(getattr(pl_module, "processed_dir", "data/processed"))

        if not processed_dir.exists():
            log.warning(f"Processed directory not found: {processed_dir}")
            self._image_index_to_filename = {}
            self._window_size = (512, 512)
            self._overlap = 64
            self._tiff_cache = {}
            self._dataset_metadata_loaded = True
            return

        # Load slice_meta.csv to map image index → filename
        slice_meta_path = processed_dir / "slice_meta.csv"
        if slice_meta_path.exists():
            try:
                slice_meta = pd.read_csv(slice_meta_path)
                # Filter to validation split to get correct image mappings
                val_meta = slice_meta[slice_meta["split"] == "val"]
                self._image_index_to_filename = {}
                for _, row in val_meta.iterrows():
                    img_idx = int(row["Image"])
                    filename = str(row["Landsat ID"])
                    if img_idx not in self._image_index_to_filename:
                        self._image_index_to_filename[img_idx] = filename
                log.info(
                    f"Loaded {len(self._image_index_to_filename)} Landsat image mappings for validation split"
                )
            except Exception as e:
                log.warning(f"Failed to load slice_meta.csv: {e}")
                self._image_index_to_filename = {}
        else:
            log.warning(f"slice_meta.csv not found: {slice_meta_path}")
            self._image_index_to_filename = {}

        # Load dataset_statistics.json for window_size/overlap
        stats_path = processed_dir / "dataset_statistics.json"
        if stats_path.exists():
            try:
                with open(stats_path) as f:
                    stats = json.load(f)
                    config = stats.get("config", {})
                    self._window_size = tuple(config.get("window_size", [512, 512]))
                    self._overlap = config.get("overlap", 64)
                log.info(
                    f"Loaded dataset config: window_size={self._window_size}, overlap={self._overlap}"
                )
            except Exception as e:
                log.warning(f"Failed to load dataset_statistics.json: {e}")
                self._window_size = (512, 512)
                self._overlap = 64
        else:
            self._window_size = (512, 512)
            self._overlap = 64

        # Initialize TIFF cache
        self._tiff_cache = {}
        self._dataset_metadata_loaded = True

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
                self.selected_slice_paths = self._select_by_iou_thirds(
                    val_slices_all, pl_module, num_samples
                )
            else:  # random
                self.selected_slice_paths = val_slices_all[:num_samples]

            # Extract metadata for each selected slice
            for path in self.selected_slice_paths:
                tiff_num, slice_num = self._parse_slice_path(path)
                self.slice_metadata[path] = (tiff_num, slice_num)

            log.info(
                f"Selected {len(self.selected_slice_paths)} validation slices for tracking"
            )

        # Generate visualizations for tracked slices
        metrics_opts = getattr(pl_module, "metrics_opts", {"threshold": [0.5, 0.5]})
        threshold = metrics_opts.get("threshold", [0.5, 0.5])
        class_names = getattr(pl_module, "class_names", ["background", "target"])
        output_classes = getattr(pl_module, "output_classes", [1])

        # Build colormap based on task type
        if len(output_classes) == 1:  # Binary
            cmap = build_binary_cmap(output_classes)
        else:  # Multi-class
            cmap = build_cmap_from_mask_names(class_names)

        for slice_path in self.selected_slice_paths:
            tiff_num, slice_num = self.slice_metadata[slice_path]

            # Load data
            x_full = np.load(slice_path)
            y_true_raw = np.load(
                slice_path.with_name(slice_path.name.replace("tiff", "mask"))
            ).astype(np.uint8)

            # Get predictions
            probs = get_probabilities(pl_module, x_full)
            y_pred = predict_from_probs(
                probs, pl_module, threshold[0] if threshold else None
            )

            # Calculate confidence map
            if len(output_classes) == 1:  # Binary
                conf = probs[:, :, 1]  # Foreground probability
            else:  # Multi-class
                conf = np.max(probs, axis=-1)

            # Prepare visualization masks
            ignore = y_true_raw == 255

            y_gt_vis = y_true_raw.copy()
            y_pred_vis = y_pred.copy()

            if len(output_classes) == 1:  # Binary
                class_idx = output_classes[0]
                y_gt_vis_binary = np.zeros_like(y_true_raw)
                y_gt_vis_binary[y_true_raw == class_idx] = 1
                y_gt_vis_binary[y_true_raw == 255] = 255
                y_gt_vis = y_gt_vis_binary

            y_pred_vis[ignore] = 255

            # Create RGB visualizations
            x_rgb = make_rgb_preview(x_full)

            # Generate context image (full Landsat TIFF with slice location box)
            context_rgb = None
            tiff_filename = "unknown"  # Initialize for error messages
            try:
                # Load dataset metadata on first use
                if not hasattr(self, "_dataset_metadata_loaded"):
                    self._load_dataset_metadata(pl_module)

                # Check if image_dir is available
                if self.image_dir is None:
                    raise ValueError("image_dir not provided to callback")

                tiff_num, slice_num = self.slice_metadata[slice_path]

                # Get Landsat TIFF filename from index mapping
                if tiff_num not in self._image_index_to_filename:
                    raise ValueError(f"Image index {tiff_num} not in slice metadata")

                tiff_filename = self._image_index_to_filename[tiff_num]
                tiff_path = self.image_dir / tiff_filename

                if not tiff_path.exists():
                    raise FileNotFoundError(f"Landsat TIFF not found: {tiff_path}")

                # Load and cache TIFF
                if tiff_num not in self._tiff_cache:
                    self._tiff_cache[tiff_num] = load_full_tiff_rgb(str(tiff_path))

                tiff_full_rgb = self._tiff_cache[tiff_num]

                # Calculate slice position
                tiff_shape = tiff_full_rgb.shape[:2]
                row_start, col_start, row_end, col_end = calculate_slice_position(
                    slice_num, tiff_shape, self._window_size, self._overlap
                )

                # Generate context with yellow box
                context_rgb = make_context_image(
                    tiff_full_rgb,
                    row_start,
                    col_start,
                    row_end,
                    col_end,
                    target_size=(x_rgb.shape[1], x_rgb.shape[0]),
                    box_color=(255, 255, 0),
                )

            except FileNotFoundError as e:
                log.warning(f"Landsat TIFF not found: {e}")
                context_rgb = make_error_overlay(
                    x_rgb.shape[:2], f"Landsat TIFF missing:\n{tiff_filename}"
                )
            except ValueError as e:
                log.warning(f"Context metadata error: {e}")
                context_rgb = make_error_overlay(
                    x_rgb.shape[:2], f"Metadata error:\n{str(e)[:40]}"
                )
            except Exception as e:
                log.warning(f"Context generation failed: {e}")
                context_rgb = make_error_overlay(
                    x_rgb.shape[:2], f"Error:\n{str(e)[:40]}"
                )

            # Fallback if context is still None
            if context_rgb is None:
                context_rgb = make_error_overlay(x_rgb.shape[:2], "Context unavailable")

            # Create overlay visualizations
            pr_overlay_rgb = make_overlay(x_rgb, y_pred_vis, cmap, alpha=0.5)

            # Create confidence map visualization
            conf_rgb = make_confidence_map(conf, invalid_mask=ignore)

            # TP/FP/FN masks
            tp_mask = (
                (y_pred_vis == y_gt_vis)
                & (~ignore)
                & (y_gt_vis != 0)
                & (y_gt_vis != 255)
            )
            fp_mask = (
                (y_pred_vis != y_gt_vis)
                & (~ignore)
                & (y_pred_vis != 0)
                & (y_pred_vis != 255)
            )
            fn_mask = (
                (y_pred_vis != y_gt_vis)
                & (~ignore)
                & (y_gt_vis != 0)
                & (y_gt_vis != 255)
            )

            tp_rgb, fp_rgb, fn_rgb = make_tp_fp_fn_masks(tp_mask, fp_mask, fn_mask)

            # Calculate metrics for title
            metric_parts = []
            if len(output_classes) == 1:  # Binary
                target_class = output_classes[0]
                P, R, iou, _, _, _ = calculate_binary_metrics(
                    y_pred, y_true_raw, target_class, mask=ignore
                )
                target_class_name = class_names[target_class]
                metric_parts.append(
                    f"{target_class_name}: P={P:.3f} R={R:.3f} IoU={iou:.3f}"
                )
            else:  # Multi-class
                for ci, cname in enumerate(class_names):
                    if ci == 0:  # Skip background
                        continue
                    pred_c = (y_pred_vis == ci).astype(np.uint8)
                    true_c = (y_gt_vis == ci).astype(np.uint8)
                    tp_, fp_, fn_ = tp_fp_fn(
                        torch.from_numpy(pred_c), torch.from_numpy(true_c)
                    )
                    P_val = precision(tp_, fp_, fn_)
                    R_val = recall(tp_, fp_, fn_)
                    I_val = IoU(tp_, fp_, fn_)
                    metric_parts.append(
                        f"{cname}: P={P_val:.3f} R={R_val:.3f} IoU={I_val:.3f}"
                    )

            title_text = f"VAL TIFF {tiff_num:04d}, Slice {slice_num:02d}"
            metrics_text = title_text + " | " + " | ".join(metric_parts)

            # Create composite visualization with new layout
            composite = make_redesigned_panel(
                context_rgb=context_rgb,
                x_rgb=x_rgb,
                gt_labels=y_gt_vis,          # integer labels, masked version
                pr_labels=y_pred_vis,        # integer labels, masked version
                cmap=cmap,
                class_names=class_names,
                metrics_text=metrics_text,
                conf_rgb=conf_rgb,
                mask=~ignore,                # boolean valid mask
            )

            # Save to: val_visualizations/tiff_XXXX/slice_YY_epochZZZZ.png
            if self.save_dir:
                tiff_dir = self.save_dir / f"tiff_{tiff_num:04d}"
                tiff_dir.mkdir(parents=True, exist_ok=True)
                out_path = (
                    tiff_dir
                    / f"slice_{slice_num:02d}_epoch{trainer.current_epoch + 1:04d}.png"
                )

                try:
                    cv2.imwrite(
                        str(out_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
                    )
                except Exception as e:
                    log.error(f"Error saving validation visualization: {e}")

        # GPU cleanup after visualization generation
        cleanup_gpu_memory()

        # Log to MLflow (if available)
        if self.save_dir:
            self._log_to_mlflow(trainer)

    def _select_by_iou_thirds(
        self, slice_paths: List[Path], pl_module, num_samples: int
    ) -> List[Path]:
        """Select slices using IoU-based thirds distribution."""
        if num_samples < 3:
            return slice_paths[:num_samples]

        # Calculate IoU for each slice
        slice_ious = []
        output_classes = getattr(pl_module, "output_classes", [1])
        metrics_opts = getattr(pl_module, "metrics_opts", {"threshold": [0.5, 0.5]})
        threshold = metrics_opts.get("threshold", [0.5, 0.5])  # type: ignore[arg-type]

        log.info(f"Computing IoU for {len(slice_paths)} validation slices...")

        for idx, x_path in enumerate(tqdm(slice_paths, desc="Val IoU computation")):
            x = np.load(x_path)
            y_pred, invalid_mask = pl_module.predict_slice(x, threshold)  # type: ignore[arg-type]

            y_true_raw = np.load(
                x_path.with_name(x_path.name.replace("tiff", "mask"))
            ).astype(np.uint8)

            ignore = y_true_raw == 255
            if invalid_mask is not None:
                ignore |= invalid_mask

            # Calculate IoU
            if len(output_classes) == 1:  # Binary
                target_class = output_classes[0]
                _, _, iou, _, _, _ = calculate_binary_metrics(
                    y_pred, y_true_raw, target_class, mask=ignore
                )
            else:  # Multi-class
                valid = ~ignore
                y_pred_valid = y_pred[valid]
                y_true_valid = y_true_raw[valid]

                ious = []
                for ci in range(len(output_classes)):
                    label = ci
                    p = (y_pred_valid == label).astype(np.uint8)
                    t = (y_true_valid == label).astype(np.uint8)
                    tp_, fp_, fn_ = tp_fp_fn(torch.from_numpy(p), torch.from_numpy(t))
                    ious.append(IoU(tp_, fp_, fn_))
                iou = np.mean(ious)

            slice_ious.append((x_path, float(iou)))

            if (idx + 1) % 20 == 0:
                cleanup_gpu_memory(synchronize=False)

        cleanup_gpu_memory()

        # Sort by IoU (descending)
        slice_ious.sort(key=lambda x: x[1], reverse=True)

        # Split into thirds
        k = num_samples // 3
        remainder = num_samples % 3

        top_k = k + (1 if remainder > 0 else 0)
        bottom_k = k + (1 if remainder > 1 else 0)
        middle_k = k

        # Select slices
        top_slices = [path for path, iou in slice_ious[:top_k]]
        bottom_slices = [path for path, iou in slice_ious[-bottom_k:]]

        middle_start = len(slice_ious) // 2 - middle_k // 2
        middle_end = middle_start + middle_k
        middle_slices = [path for path, iou in slice_ious[middle_start:middle_end]]

        selected = top_slices + middle_slices + bottom_slices

        log.info(f"Selected {len(selected)} validation slices by IoU:")
        log.info(
            f"  Top {top_k}:    {[f'{slice_ious[i][1]:.3f}' for i in range(min(top_k, len(slice_ious)))]}"
        )
        log.info(
            f"  Middle {middle_k}: {[f'{slice_ious[middle_start + i][1]:.3f}' for i in range(min(middle_k, len(slice_ious) - middle_start))]}"
        )
        log.info(
            f"  Bottom {bottom_k}: {[f'{slice_ious[len(slice_ious) - bottom_k + i][1]:.3f}' for i in range(min(bottom_k, len(slice_ious)))]}"
        )

        return selected

    def _parse_slice_path(self, filepath: Path) -> Tuple[int, int]:
        """Extract (tiff_num, slice_num) from path."""
        filename = filepath.name
        parts = filename.replace(".npy", "").split("_")
        tiff_num = int(parts[1])  # tiff_{NUM}_slice_{SLICE}
        slice_num = int(parts[3])
        return tiff_num, slice_num

    def _log_to_mlflow(self, trainer: pl.Trainer):
        """Log saved visualizations to MLflow."""
        if not MLFLOW_AVAILABLE:
            return

        for logger in trainer.loggers:
            if MLFLOW_AVAILABLE and isinstance(logger, MLFlowLogger):  # type: ignore[misc]
                try:
                    for tiff_dir in self.save_dir.glob("tiff_*"):  # type: ignore[union-attr]
                        if tiff_dir.is_dir():
                            for png_file in tiff_dir.glob("*.png"):
                                logger.experiment.log_artifact(  # type: ignore[attr-defined]
                                    logger.run_id,  # type: ignore[attr-defined]
                                    str(png_file),
                                    artifact_path=f"val_visualizations/{tiff_dir.name}",
                                )
                except Exception as e:
                    log.warning(
                        f"Failed to log validation visualization to MLflow: {e}"
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
