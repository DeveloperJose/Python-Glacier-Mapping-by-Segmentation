"""Best-model full evaluation callback for glacier mapping."""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from tqdm import tqdm

from glacier_mapping.model.metrics import tp_fp_fn, precision, recall, IoU
from glacier_mapping.model.visualize import (
    build_cmap_from_mask_names,
    make_rgb_preview,
    label_to_color,
    make_confidence_map,
    make_entropy_map,
    make_tp_fp_fn_masks,
    make_eight_panel,
)
from glacier_mapping.utils.prediction import (
    get_probabilities,
    predict_from_probs,
    calculate_binary_metrics,
)

# Import MLflow utilities with error handling
try:
    import importlib.util

    MLFLOW_AVAILABLE = importlib.util.find_spec("mlflow") is not None
except ImportError:
    MLFLOW_AVAILABLE = False


class BestModelFullEvaluationCallback(Callback):
    """Full-tile evaluation triggered only on best model improvement."""

    def __init__(self, num_samples: int = 4):
        super().__init__()
        self.num_samples = num_samples
        self.best_val_loss = float("inf")
        self.best_full_metrics = {}  # Track best full-tile metrics

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """Trigger full-tile evaluation only on new best model."""
        current_val_loss = trainer.callback_metrics.get("val_loss", float("inf"))

        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            print(f"\n🎯 New best model detected (val_loss: {current_val_loss:.4f})")
            print("Running full-tile evaluation...")
            self._run_full_evaluation(trainer, pl_module)

    def _run_full_evaluation(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Full-tile evaluation using Lightning module directly."""
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

        output_dir = base_dir / "full_evaluations"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get test tiles
        processed_dir = getattr(pl_module, "processed_dir", "data/processed")
        data_dir = Path(processed_dir) / "test"  # type: ignore[arg-type]
        test_tiles_all = sorted(data_dir.glob("tiff*"))

        if not test_tiles_all:
            print("Warning: No test tiles found for full-tile evaluation")
            return

        # Select informative tiles for visualization
        test_tiles = self._select_informative_tiles(
            test_tiles_all, pl_module, self.num_samples
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
        tp_sum = np.zeros(n_classes)
        fp_sum = np.zeros(n_classes)
        fn_sum = np.zeros(n_classes)

        # Evaluate all test tiles
        for x_path in tqdm(test_tiles_all, desc="Full-tile evaluation"):
            x = np.load(x_path)
            y_pred, invalid_mask = pl_module.predict_slice(x, threshold)  # type: ignore[call-arg]

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
                P, R, iou, tp_, fp_, fn_ = calculate_binary_metrics(
                    y_pred, y_true_raw, target_class, ignore
                )

                # Store only target class metrics (skip background)
                tp_sum[0] += tp_
                fp_sum[0] += fp_
                fn_sum[0] += fn_

                # For binary, report all 3 classes but only target class has real metrics
                # Background metrics (zeros for binary classification)
                row += [0.0, 0.0, 0.0]  # BG_precision, BG_recall, BG_IoU

                # Target class metrics (place in correct position based on target_class)
                if target_class == 1:  # CleanIce
                    row += [
                        P,
                        R,
                        iou,
                    ]  # CleanIce_precision, CleanIce_recall, CleanIce_IoU
                    row += [
                        0.0,
                        0.0,
                        0.0,
                    ]  # Debris_precision, Debris_recall, Debris_IoU
                else:  # Debris
                    row += [
                        0.0,
                        0.0,
                        0.0,
                    ]  # CleanIce_precision, CleanIce_recall, CleanIce_IoU
                    row += [P, R, iou]  # Debris_precision, Debris_recall, Debris_IoU
            else:  # Multi-class
                valid = ~ignore
                y_pred_valid = y_pred[valid]
                y_true_valid_raw = y_true_raw[valid]

                for ci in range(n_classes):
                    label = ci + 1
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

        # Create CSV metrics subdirectory and save with new naming
        csv_dir = output_dir / "csv_metrics"
        csv_dir.mkdir(parents=True, exist_ok=True)

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

            pl_module.log(
                f"full_test_{cname}_precision", prec, on_step=False, on_epoch=True
            )
            pl_module.log(
                f"full_test_{cname}_recall", rec, on_step=False, on_epoch=True
            )
            pl_module.log(f"full_test_{cname}_iou", iou, on_step=False, on_epoch=True)

            # Track best full-tile metrics (IoU as main comparison metric)
            if cname in self.best_full_metrics:
                best_iou = self.best_full_metrics[cname].get("iou", float("-inf"))
                if iou > best_iou:
                    self.best_full_metrics[cname] = {
                        "iou": iou,
                        "precision": prec,
                        "recall": rec,
                    }
                    # Log new best metrics
                    pl_module.log(
                        f"best_full_test_{cname}_iou", iou, on_step=False, on_epoch=True
                    )
                    pl_module.log(
                        f"best_full_test_{cname}_precision",
                        prec,
                        on_step=False,
                        on_epoch=True,
                    )
                    pl_module.log(
                        f"best_full_test_{cname}_recall",
                        rec,
                        on_step=False,
                        on_epoch=True,
                    )
            else:
                # Initialize tracking for this class
                self.best_full_metrics[cname] = {
                    "iou": iou,
                    "precision": prec,
                    "recall": rec,
                }
                # Log initial best metrics
                pl_module.log(
                    f"best_full_test_{cname}_iou", iou, on_step=False, on_epoch=True
                )
                pl_module.log(
                    f"best_full_test_{cname}_precision",
                    prec,
                    on_step=False,
                    on_epoch=True,
                )
                pl_module.log(
                    f"best_full_test_{cname}_recall", rec, on_step=False, on_epoch=True
                )

        # Generate visualizations for selected tiles
        print(
            f"Generating visualizations for {min(self.num_samples, len(test_tiles))} tiles..."
        )
        self._generate_visualizations(
            pl_module,
            test_tiles[: self.num_samples],
            output_dir,
            trainer.current_epoch + 1,
        )
        print("Visualizations completed.")

        # Log PNG files to both TensorBoard and MLflow
        self._log_visualizations_to_all_loggers(
            trainer, output_dir, trainer.current_epoch + 1
        )

    def _log_visualizations_to_all_loggers(
        self, trainer: pl.Trainer, output_dir: Path, epoch: int
    ):
        """Log PNGs to both TensorBoard and MLflow."""
        import matplotlib.pyplot as plt

        for logger in trainer.loggers:
            try:
                # MLflow logging - Fixed structure: log each tile directory without double nesting
                if isinstance(logger, MLFlowLogger):
                    # Log CSV metrics directory
                    csv_dir = output_dir / "csv_metrics"
                    if csv_dir.exists():
                        logger.experiment.log_artifacts(
                            logger.run_id,
                            str(csv_dir),
                            artifact_path="csv_metrics",
                        )

                    # Log tile directories without double nesting
                    for tile_dir in output_dir.glob("fulltile_*"):
                        if tile_dir.is_dir():
                            logger.experiment.log_artifact(
                                logger.run_id,
                                str(tile_dir),
                                artifact_path=None,  # Fix: remove double nesting
                            )

                # TensorBoard logging
                elif isinstance(logger, TensorBoardLogger):
                    for tile_dir in output_dir.glob("fulltile_*"):
                        if tile_dir.is_dir():
                            for png_file in tile_dir.glob("*.png"):
                                # Load PNG and convert to tensor for TensorBoard
                                img = plt.imread(png_file)  # HWC format
                                img_tensor = (
                                    torch.from_numpy(img).permute(2, 0, 1).float()
                                    / 255.0
                                )
                                logger.experiment.add_image(
                                    f"evaluation/{tile_dir.name}/{png_file.stem}",
                                    img_tensor,
                                    global_step=epoch,
                                    dataformats="CHW",
                                )

            except Exception as e:
                print(f"Warning: Failed to log to {type(logger).__name__}: {e}")

    def _select_informative_tiles(
        self, tile_paths: List[Path], pl_module: pl.LightningModule, num_samples: int
    ) -> List[Path]:
        """Select tiles with most target class pixels."""
        tile_class_counts = []
        output_classes = getattr(pl_module, "output_classes", [1])

        for x_path in tile_paths:
            mask_path = x_path.with_name(x_path.name.replace("tiff", "mask"))
            mask = np.load(mask_path)

            if len(output_classes) == 1:  # Binary
                class_pixels = (mask == output_classes[0]).sum()
            else:  # Multi-class
                class_pixels = ((mask > 0) & (mask != 255)).sum()

            tile_class_counts.append((x_path, int(class_pixels)))

        tile_class_counts.sort(key=lambda x: x[1], reverse=True)
        selected = [path for path, count in tile_class_counts if count > 0]

        if len(selected) < num_samples:
            selected = [path for path, count in tile_class_counts]

        return selected[:num_samples]

    def _generate_visualizations(
        self,
        pl_module: pl.LightningModule,
        test_tiles: List[Path],
        output_dir: Path,
        epoch: int,
    ) -> None:
        """Generate 8-panel visualizations."""
        import cv2

        # Get module attributes with fallbacks
        metrics_opts = getattr(pl_module, "metrics_opts", {"threshold": [0.5, 0.5]})
        threshold = metrics_opts.get("threshold", [0.5, 0.5])
        class_names = getattr(pl_module, "class_names", ["background", "target"])
        output_classes = getattr(pl_module, "output_classes", [1])

        cmap = build_cmap_from_mask_names(class_names)

        for idx, x_path in enumerate(test_tiles):
            x_full = np.load(x_path)
            y_true_raw = np.load(
                x_path.with_name(x_path.name.replace("tiff", "mask"))
            ).astype(np.uint8)

            # Predict using shared utilities
            probs = get_probabilities(pl_module, x_full)
            y_pred_viz = predict_from_probs(
                probs, pl_module, threshold[0] if threshold else None
            )

            # Get confidence for visualization
            if len(output_classes) == 1:  # Binary
                conf = probs[:, :, 1]  # Foreground probability
            else:  # Multi-class
                conf = np.max(probs, axis=-1)

            # Generate 8-panel visualization
            ignore = y_true_raw == 255

            # GT/PRED for visualization
            y_gt_vis = y_true_raw.copy()
            y_pred_vis = y_pred_viz.copy()

            if len(output_classes) == 1:  # Binary
                class_idx = output_classes[0]
                y_gt_vis_binary = np.zeros_like(y_true_raw)
                y_gt_vis_binary[y_true_raw == class_idx] = 1
                y_gt_vis_binary[y_true_raw == 255] = 255
                y_gt_vis = y_gt_vis_binary
            else:  # Multi-class
                # Keep original 0,1,2 range - don't map to 1,2,3
                y_gt_vis = y_true_raw.copy()
                y_gt_vis[ignore] = 255

            y_pred_vis[ignore] = 255

            # Convert predictions from 1,2,3 back to 0,1,2 for consistent visualization (multi-class only)
            if len(output_classes) > 1:  # Multi-class only
                valid_pred = (y_pred_vis != 255) & (y_pred_vis > 0)
                y_pred_vis[valid_pred] = y_pred_vis[valid_pred] - 1  # 1→0, 2→1, 3→2

            x_rgb = make_rgb_preview(x_full)

            # Confidence / entropy
            conf_rgb = make_confidence_map(conf, invalid_mask=ignore)
            if len(output_classes) == 1:  # Binary
                # For binary, reshape probs to (H, W, 2) for entropy calculation
                if probs.shape[-1] == 2:
                    entropy_rgb = make_entropy_map(probs, invalid_mask=ignore)
                else:
                    # Handle case where probs might be in different format
                    entropy_rgb = make_confidence_map(
                        conf, invalid_mask=ignore
                    )  # Fallback
            else:  # Multi-class
                entropy_rgb = make_entropy_map(probs, invalid_mask=ignore)

            # TP / FP / FN masks
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

            # Per-class metrics string
            metric_string_parts = []
            if len(output_classes) == 1:  # Binary
                # For binary, only show target class metrics with proper class name
                target_class = output_classes[0]  # 1 for CleanIce, 2 for Debris

                # Calculate metrics for target class only
                P, R, iou, tp_, fp_, fn_ = calculate_binary_metrics(
                    y_pred_viz, y_true_raw, target_class, ignore
                )

                # Build metrics string with all class names but only target class has real values
                for ci, cname in enumerate(class_names):
                    if ci == 0:  # Background
                        metric_string_parts.append(
                            f"{cname}: P=0.000 R=0.000 IoU=0.000"
                        )
                    elif (
                        ci == target_class
                    ):  # Target class (ci=1 for class 1, ci=2 for class 2)
                        metric_string_parts.append(
                            f"{cname}: P={P:.3f} R={R:.3f} IoU={iou:.3f}"
                        )
                    else:  # Other class (not target)
                        metric_string_parts.append(
                            f"{cname}: P=0.000 R=0.000 IoU=0.000"
                        )
            else:  # Multi-class
                for ci, cname in enumerate(class_names):
                    pred_c = (y_pred_vis == ci).astype(np.uint8)
                    true_c = (y_gt_vis == ci).astype(np.uint8)

                    tp_, fp_, fn_ = tp_fp_fn(
                        torch.from_numpy(pred_c), torch.from_numpy(true_c)
                    )
                    P_val = precision(tp_, fp_, fn_)
                    R_val = recall(tp_, fp_, fn_)
                    I_val = IoU(tp_, fp_, fn_)
                    metric_string_parts.append(
                        f"{cname}: P={P_val:.3f} R={R_val:.3f} IoU={I_val:.3f}"
                    )

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

            # New structure: fulltile_XXXX/epochYYYY.png
            tile_dir = output_dir / f"fulltile_{idx:04d}"
            tile_dir.mkdir(parents=True, exist_ok=True)
            out_path = tile_dir / f"epoch{epoch:04d}.png"
            print(f"Saving visualization to: {out_path}")
            try:
                success = cv2.imwrite(
                    str(out_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
                )
                print(f"Save successful: {success}")
            except Exception as e:
                print(f"Error saving visualization: {e}")
