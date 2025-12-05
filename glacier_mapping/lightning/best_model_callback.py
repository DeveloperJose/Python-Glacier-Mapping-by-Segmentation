"""Test set evaluation callback for glacier mapping (triggered on best model)."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from glacier_mapping.utils.callback_utils import (
    load_dataset_metadata,
    generate_single_visualization,
    select_informative_test_tiles,
    log_visualizations_to_all_loggers,
)
from glacier_mapping.utils import cleanup_gpu_memory
import glacier_mapping.utils.logging as log


class TestEvaluationCallback(Callback):
    """Test set evaluation triggered on best model improvement.

    Uses n-based thirds system: n=4 → 12 visualizations (4 top + 4 middle + 4 bottom).
    """

    def __init__(
        self, viz_n: int = 4, image_dir: str | None = None, scale_factor: float = 1
    ):
        """
        Initialize test evaluation callback.

        Args:
            viz_n: Number of samples per third (top, middle, bottom).
                   Total visualizations = 3 * viz_n.
                   Set to 0 to compute metrics only (no visualizations).
            image_dir: Path to raw Landsat image TIFFs (from servers.yaml)
        """
        super().__init__()
        self.viz_n = viz_n
        self.image_dir = Path(image_dir) if image_dir else None
        self.scale_factor = scale_factor
        self.best_val_loss = float("inf")
        self.best_test_metrics = {}  # Track best test metrics

        # Cache for metadata to avoid reloading
        self._metadata_cache = None

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """Trigger full-tile evaluation only on new best model."""
        # Skip evaluation during sanity check to prevent OOM
        if trainer.sanity_checking:
            log.info("Skipping full-tile evaluation during sanity check")
            return

        current_val_loss = trainer.callback_metrics.get("val_loss", float("inf"))

        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            log.info(f"🎯 New best model detected (val_loss: {current_val_loss:.4f})")
            log.info("Running test set evaluation...")
            self._run_full_evaluation(trainer, pl_module)

    def _run_full_evaluation(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Test set evaluation using Lightning module directly."""
        # Create output directory - use checkpoint directory as base
        checkpoint_callback = trainer.checkpoint_callback
        if checkpoint_callback is not None and hasattr(checkpoint_callback, "dirpath"):
            base_dir = Path(getattr(checkpoint_callback, "dirpath", ".")).parent
        else:
            # Fallback to default_root_dir or current directory
            base_dir = (
                Path(trainer.default_root_dir)
                if trainer.default_root_dir
                else Path(".")
            )

        output_dir = base_dir / "test_evaluations"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get test tiles
        processed_dir = getattr(pl_module, "processed_dir", "data/processed")
        data_dir = Path(processed_dir) / "test"  # type: ignore[arg-type]
        test_tiles_all = sorted(data_dir.glob("tiff*"))

        if not test_tiles_all:
            log.warning("No test tiles found for test evaluation")
            return

        # Select informative tiles for visualization and cache predictions
        # Calculate total samples: 3 * viz_n (top + middle + bottom)
        num_samples = 3 * self.viz_n if self.viz_n > 0 else 0
        test_tiles, tile_rank_map, prediction_cache = select_informative_test_tiles(
            test_tiles_all, pl_module, num_samples
        )

        # Setup metrics with proper type handling
        class_names = getattr(
            pl_module,
            "class_names",
            getattr(pl_module.hparams, "class_names", ["background", "target"]),
        )
        n_classes = len(class_names)
        metrics_opts = getattr(pl_module, "metrics_opts", {"threshold": [0.5, 0.5]})
        threshold = metrics_opts.get("threshold", [0.5, 0.5])
        output_classes = getattr(pl_module, "output_classes", [1])

        rows = []
        tp_sum = [0] * n_classes
        fp_sum = [0] * n_classes
        fn_sum = [0] * n_classes

        # Evaluate all test tiles (reuse cached predictions when available)
        from tqdm import tqdm
        import numpy as np
        import torch
        from glacier_mapping.model.metrics import IoU, precision, recall, tp_fp_fn
        from glacier_mapping.utils.prediction import calculate_binary_metrics

        for idx, x_path in enumerate(tqdm(test_tiles_all, desc="Test evaluation")):
            # Use cached prediction if available, otherwise compute
            if x_path not in prediction_cache:
                x = np.load(x_path)
                y_pred, invalid_mask = pl_module.predict_slice(x, threshold)  # type: ignore[call-arg]
            else:
                y_pred, invalid_mask = prediction_cache[x_path]

            y_true_raw = np.load(
                x_path.with_name(x_path.name.replace("tiff", "mask"))
            ).astype(np.uint8)

            ignore = y_true_raw == 255
            if invalid_mask is not None:
                ignore |= invalid_mask

            # Per-tile metrics
            row = [x_path.name]
            if len(output_classes) == 1:  # Binary
                # Use unified binary metrics calculation
                target_class = output_classes[0]  # 1 for CleanIce, 2 for Debris

                # Calculate target class metrics
                P_target, R_target, iou_target, tp_target, fp_target, fn_target = (
                    calculate_binary_metrics(
                        y_pred, y_true_raw, target_class, mask=ignore
                    )
                )

                # Calculate background metrics (everything NOT target class)
                # For binary: BG = all pixels that are neither target_class nor ignore
                y_pred_bg = (y_pred == 0).astype(np.uint8)
                y_true_bg = ((y_true_raw != target_class) & (y_true_raw != 255)).astype(
                    np.uint8
                )
                P_bg, R_bg, iou_bg, tp_bg, fp_bg, fn_bg = calculate_binary_metrics(
                    y_pred_bg, y_true_bg.astype(np.uint8), target_class=1, mask=ignore
                )

                # Store metrics in correct positions: [BG, CleanIce, Debris]
                if target_class == 1:  # CleanIce task
                    tp_sum[0] += tp_bg  # BG
                    fp_sum[0] += fp_bg
                    fn_sum[0] += fn_bg
                    tp_sum[1] += tp_target  # CleanIce
                    fp_sum[1] += fp_target
                    fn_sum[1] += fn_target
                    # tp_sum[2] stays 0 (Debris not evaluated)

                    row += [P_bg, R_bg, iou_bg]  # BG_precision, BG_recall, BG_IoU
                    row += [
                        P_target,
                        R_target,
                        iou_target,
                    ]  # CleanIce_precision, CleanIce_recall, CleanIce_IoU
                    row += [
                        0.0,
                        0.0,
                        0.0,
                    ]  # Debris_precision, Debris_recall, Debris_IoU (not evaluated)
                else:  # Debris task (target_class == 2)
                    tp_sum[0] += tp_bg  # BG
                    fp_sum[0] += fp_bg
                    fn_sum[0] += fn_bg
                    # tp_sum[1] stays 0 (CleanIce not evaluated)
                    tp_sum[2] += tp_target  # Debris
                    fp_sum[2] += fp_target
                    fn_sum[2] += fn_target

                    row += [P_bg, R_bg, iou_bg]  # BG_precision, BG_recall, BG_IoU
                    row += [
                        0.0,
                        0.0,
                        0.0,
                    ]  # CleanIce_precision, CleanIce_recall, CleanIce_IoU (not evaluated)
                    row += [
                        P_target,
                        R_target,
                        iou_target,
                    ]  # Debris_precision, Debris_recall, Debris_IoU
            else:  # Multi-class
                valid = ~ignore
                y_pred_valid = y_pred[valid]
                y_true_valid_raw = y_true_raw[valid]

                for ci in range(n_classes):
                    # Predictions are in 0,1,2 range (0=BG, 1=CI, 2=Debris)
                    # Ground truth is also in 0,1,2 range (after preprocessing)
                    label = ci  # Compare directly: 0=BG, 1=CI, 2=Debris
                    p = (y_pred_valid == label).astype(np.uint8)
                    t = (y_true_valid_raw == label).astype(np.uint8)

                    tp_, fp_, fn_ = tp_fp_fn(torch.from_numpy(p), torch.from_numpy(t))
                    tp_sum[ci] += tp_
                    fp_sum[ci] += fp_
                    fn_sum[ci] += fn_

                    row += [
                        precision(tp_, fp_, fn_),
                        recall(tp_, fp_, fn_),
                        IoU(tp_, fp_, fn_),
                    ]

            rows.append(row)

            # Periodic GPU cleanup every 20 tiles to prevent accumulation
            if (idx + 1) % 20 == 0:
                cleanup_gpu_memory(synchronize=False)

        # GPU cleanup after test evaluation to prevent OOM
        cleanup_gpu_memory()

        # Create CSV metrics subdirectory and save with new naming
        csv_dir = output_dir / "csv_metrics"
        csv_dir.mkdir(parents=True, exist_ok=True)

        import pandas as pd

        cols = ["tile"]
        for cname in class_names:
            cols += [f"{cname}_precision", f"{cname}_recall", f"{cname}_IoU"]

        df = pd.DataFrame(rows, columns=cols)  # type: ignore[arg-type]
        df.to_csv(csv_dir / f"epoch{trainer.current_epoch + 1:04d}.csv", index=False)

        # Log metrics to both loggers
        for ci, cname in enumerate(class_names):
            tp_, fp_, fn_ = tp_sum[ci], fp_sum[ci], fn_sum[ci]
            prec = precision(tp_, fp_, fn_)
            rec = recall(tp_, fp_, fn_)
            iou = IoU(tp_, fp_, fn_)

            pl_module.log(f"test_{cname}_precision", prec, on_step=False, on_epoch=True)
            pl_module.log(f"test_{cname}_recall", rec, on_step=False, on_epoch=True)
            pl_module.log(f"test_{cname}_iou", iou, on_step=False, on_epoch=True)

            # Track best test metrics (IoU as main comparison metric)
            if cname in self.best_test_metrics:
                best_iou = self.best_test_metrics[cname].get("iou", float("-inf"))
                if iou > best_iou:
                    self.best_test_metrics[cname] = {
                        "iou": iou,
                        "precision": prec,
                        "recall": rec,
                    }
                    # Log new best metrics
                    pl_module.log(
                        f"best_test_{cname}_iou", iou, on_step=False, on_epoch=True
                    )
                    pl_module.log(
                        f"best_test_{cname}_precision",
                        prec,
                        on_step=False,
                        on_epoch=True,
                    )
                    pl_module.log(
                        f"best_test_{cname}_recall",
                        rec,
                        on_step=False,
                        on_epoch=True,
                    )
            else:
                # Initialize tracking for this class
                self.best_test_metrics[cname] = {
                    "iou": iou,
                    "precision": prec,
                    "recall": rec,
                }
                # Log initial best metrics
                pl_module.log(
                    f"best_test_{cname}_iou", iou, on_step=False, on_epoch=True
                )
                pl_module.log(
                    f"best_test_{cname}_precision",
                    prec,
                    on_step=False,
                    on_epoch=True,
                )
                pl_module.log(
                    f"best_test_{cname}_recall", rec, on_step=False, on_epoch=True
                )

        # Generate visualizations for selected tiles (only if viz_n >= 1)
        if self.viz_n >= 1 and num_samples >= 1:
            log.info(
                f"Generating visualizations for {min(num_samples, len(test_tiles))} tiles (n={self.viz_n})..."
            )
            self._generate_visualizations(
                pl_module,
                test_tiles[:num_samples],
                output_dir,
                trainer.current_epoch + 1,
                tile_rank_map,
                len(test_tiles_all),
                prediction_cache,
            )
            log.info("Visualizations completed.")

            # GPU cleanup after visualization generation to prevent OOM
            cleanup_gpu_memory()

            # Log PNG files to both TensorBoard and MLflow
            log_visualizations_to_all_loggers(
                trainer, output_dir, trainer.current_epoch + 1, "test_evaluations"
            )

    def _generate_visualizations(
        self,
        pl_module: pl.LightningModule,
        test_tiles: List[Path],
        output_dir: Path,
        epoch: int,
        tile_rank_map: Dict[Path, int],
        total_tiles: int,
        prediction_cache: Dict[Path, Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Generate 8-panel visualizations using consolidated utilities."""
        # Load metadata once
        if self._metadata_cache is None:
            self._metadata_cache = load_dataset_metadata(
                pl_module, "test", self.image_dir
            )

        for idx, x_path in enumerate(test_tiles):
            try:
                generate_single_visualization(
                    x_path=x_path,
                    pl_module=pl_module,
                    output_dir=output_dir,
                    epoch=epoch,
                    title_prefix="TEST",
                    metadata_cache=self._metadata_cache,
                    image_dir=self.image_dir,
                    scale_factor=self.scale_factor,
                    tile_rank_map=tile_rank_map,
                )
            except Exception as e:
                log.error(f"Error generating visualization for {x_path}: {e}")
