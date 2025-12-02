"""Best-model full evaluation callback for glacier mapping."""

from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from tqdm import tqdm

from glacier_mapping.model.metrics import tp_fp_fn, precision, recall, IoU
from glacier_mapping.model.visualize import (
    build_cmap_from_mask_names, make_rgb_preview, label_to_color, 
    make_confidence_map, make_entropy_map, make_tp_fp_fn_masks, make_eight_panel
)
from glacier_mapping.utils.prediction import get_probabilities, predict_from_probs, create_invalid_mask

# Import MLflow utilities with error handling
try:
    from glacier_mapping.utils.mlflow_utils import MLflowManager
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class BestModelFullEvaluationCallback(Callback):
    """Full-tile evaluation triggered only on best model improvement."""
    
    def __init__(self, num_samples: int = 4):
        super().__init__()
        self.num_samples = num_samples
        self.best_val_loss = float('inf')
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Trigger full-tile evaluation only on new best model."""
        current_val_loss = trainer.callback_metrics.get('val_loss', float('inf'))
        
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            print(f"\n🎯 New best model detected (val_loss: {current_val_loss:.4f})")
            print("Running full-tile evaluation...")
            self._run_full_evaluation(trainer, pl_module)
    
    def _run_full_evaluation(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Full-tile evaluation using Lightning module directly."""
        # Create output directory - use checkpoint directory as base
        checkpoint_callback = trainer.checkpoint_callback
        if checkpoint_callback is not None and hasattr(checkpoint_callback, 'dirpath'):
            base_dir = Path(getattr(checkpoint_callback, 'dirpath', '.')).parent
        else:
            # Fallback to default_root_dir or current directory
            base_dir = Path(trainer.default_root_dir) if trainer.default_root_dir else Path(".")
        
        output_dir = base_dir / "full_evaluations" / f"epoch_{trainer.current_epoch + 1}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get test tiles
        processed_dir = getattr(pl_module, 'processed_dir', 'data/processed')
        data_dir = Path(processed_dir) / "test"  # type: ignore[arg-type]
        test_tiles_all = sorted(data_dir.glob("tiff*"))
        
        if not test_tiles_all:
            print("Warning: No test tiles found for full-tile evaluation")
            return
        
        # Select informative tiles for visualization
        test_tiles = self._select_informative_tiles(test_tiles_all, pl_module, self.num_samples)
        
        # Setup metrics with proper type handling
        class_names = getattr(pl_module, 'class_names', getattr(pl_module.hparams, 'class_names', ['background', 'target']))
        n_classes = len(class_names)
        metrics_opts = getattr(pl_module, 'metrics_opts', {'threshold': [0.5, 0.5]})
        threshold = metrics_opts.get('threshold', [0.5, 0.5])
        output_classes = getattr(pl_module, 'output_classes', [1])
        
        rows = []
        tp_sum = np.zeros(n_classes)
        fp_sum = np.zeros(n_classes)
        fn_sum = np.zeros(n_classes)
        
        # Evaluate all test tiles
        for x_path in tqdm(test_tiles_all, desc="Full-tile evaluation"):
            x = np.load(x_path)
            y_pred, invalid_mask = pl_module.predict_slice(x, threshold)  # type: ignore[call-arg]
            
            y_true_raw = np.load(x_path.with_name(x_path.name.replace("tiff", "mask"))).astype(np.uint8)
            
            ignore = y_true_raw == 255
            if invalid_mask is not None:
                ignore |= invalid_mask
            
            valid = ~ignore
            y_pred_valid = y_pred[valid]
            y_true_valid_raw = y_true_raw[valid]
            
            # Per-tile metrics
            row = [x_path.name]
            for ci in range(n_classes):
                if len(output_classes) == 1:  # Binary
                    pred_label = ci + 1
                    p = (y_pred_valid == pred_label).astype(np.uint8)
                    if ci == 0:
                        t = (y_true_valid_raw != output_classes[0]).astype(np.uint8)
                    else:
                        t = (y_true_valid_raw == output_classes[0]).astype(np.uint8)
                else:  # Multi-class
                    label = ci + 1
                    p = (y_pred_valid == label).astype(np.uint8)
                    t = (y_true_valid_raw + 1 == label).astype(np.uint8)
                
                tp_, fp_, fn_ = tp_fp_fn(torch.from_numpy(p), torch.from_numpy(t))
                tp_sum[ci] += tp_
                fp_sum[ci] += fp_
                fn_sum[ci] += fn_
                
                row += [precision(tp_, fp_, fn_), recall(tp_, fp_, fn_), IoU(tp_, fp_, fn_)]
            
            rows.append(row)
        
        # Save CSV with proper column handling
        cols = ["tile"]
        for cname in class_names:
            cols += [f"{cname}_precision", f"{cname}_recall", f"{cname}_IoU"]
        
        df = pd.DataFrame(rows, columns=cols)  # type: ignore[arg-type]
        df.to_csv(output_dir / f"full_eval_epoch{trainer.current_epoch + 1}.csv", index=False)
        
        # Log metrics to both loggers
        for ci, cname in enumerate(class_names):
            tp_, fp_, fn_ = tp_sum[ci], fp_sum[ci], fn_sum[ci]
            prec = precision(tp_, fp_, fn_)
            rec = recall(tp_, fp_, fn_)
            iou = IoU(tp_, fp_, fn_)
            
            pl_module.log(f'full_test_{cname}_precision', prec, on_step=False, on_epoch=True)
            pl_module.log(f'full_test_{cname}_recall', rec, on_step=False, on_epoch=True)
            pl_module.log(f'full_test_{cname}_iou', iou, on_step=False, on_epoch=True)
        
        # Generate visualizations for selected tiles
        print(f"Generating visualizations for {min(self.num_samples, len(test_tiles))} tiles...")
        self._generate_visualizations(pl_module, test_tiles[:self.num_samples], output_dir, trainer.current_epoch + 1)
        print("Visualizations completed.")
        
        # Log PNG files to both TensorBoard and MLflow
        self._log_visualizations_to_all_loggers(trainer, output_dir, trainer.current_epoch + 1)
    
    def _log_visualizations_to_all_loggers(self, trainer: pl.Trainer, output_dir: Path, epoch: int):
        """Log PNGs to both TensorBoard and MLflow."""
        import matplotlib.pyplot as plt
        
        for logger in trainer.loggers:
            try:
                # MLflow logging
                if isinstance(logger, MLFlowLogger):
                    for png_file in output_dir.glob("*.png"):
                        logger.experiment.log_artifact(
                            logger.run_id,
                            str(png_file),
                            artifact_path=f"eval_epoch_{epoch}"
                        )
                
                # TensorBoard logging
                elif isinstance(logger, TensorBoardLogger):
                    for png_file in output_dir.glob("*.png"):
                        # Load PNG and convert to tensor for TensorBoard
                        img = plt.imread(png_file)  # HWC format
                        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                        logger.experiment.add_image(
                            f"evaluation/{png_file.stem}", 
                            img_tensor, 
                            global_step=epoch,
                            dataformats='CHW'
                        )
                        
            except Exception as e:
                print(f"Warning: Failed to log to {type(logger).__name__}: {e}")
    
    def _select_informative_tiles(self, tile_paths: List[Path], pl_module: pl.LightningModule, num_samples: int) -> List[Path]:
        """Select tiles with most target class pixels."""
        tile_class_counts = []
        output_classes = getattr(pl_module, 'output_classes', [1])
        
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
    
    def _generate_visualizations(self, pl_module: pl.LightningModule, test_tiles: List[Path], output_dir: Path, epoch: int) -> None:
        """Generate 8-panel visualizations."""
        import cv2
        
        # Get module attributes with fallbacks
        metrics_opts = getattr(pl_module, 'metrics_opts', {'threshold': [0.5, 0.5]})
        threshold = metrics_opts.get('threshold', [0.5, 0.5])
        class_names = getattr(pl_module, 'class_names', ['background', 'target'])
        output_classes = getattr(pl_module, 'output_classes', [1])
        
        cmap = build_cmap_from_mask_names(class_names)
        
        for idx, x_path in enumerate(test_tiles):
            x_full = np.load(x_path)
            y_true_raw = np.load(x_path.with_name(x_path.name.replace("tiff", "mask"))).astype(np.uint8)
            
            # Predict using shared utilities
            probs = get_probabilities(pl_module, x_full)
            y_pred_viz = predict_from_probs(probs, pl_module, threshold[0] if threshold else None)
            
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
                y_gt_vis[ignore] = 255
            
            y_pred_vis[ignore] = 255
            
            x_rgb = make_rgb_preview(x_full)
            
            # Confidence / entropy
            conf_rgb = make_confidence_map(conf, invalid_mask=ignore)
            if len(output_classes) == 1:  # Binary
                # For binary, reshape probs to (H, W, 2) for entropy calculation
                if probs.shape[-1] == 2:
                    entropy_rgb = make_entropy_map(probs, invalid_mask=ignore)
                else:
                    # Handle case where probs might be in different format
                    entropy_rgb = make_confidence_map(conf, invalid_mask=ignore)  # Fallback
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
            for ci, cname in enumerate(class_names):
                pred_c = (y_pred_vis == ci).astype(np.uint8)
                true_c = (y_gt_vis == ci).astype(np.uint8)
                
                tp_, fp_, fn_ = tp_fp_fn(torch.from_numpy(pred_c), torch.from_numpy(true_c))
                P_val = precision(tp_, fp_, fn_)
                R_val = recall(tp_, fp_, fn_)
                I_val = IoU(tp_, fp_, fn_)
                metric_string_parts.append(f"{cname}: P={P_val:.3f} R={R_val:.3f} IoU={I_val:.3f}")
            
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
            print(f"Saving visualization to: {out_path}")
            try:
                success = cv2.imwrite(str(out_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
                print(f"Save successful: {success}")
            except Exception as e:
                print(f"Error saving visualization: {e}")