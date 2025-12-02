"""Best-model full evaluation callback for glacier mapping."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import MLFlowLogger
from tqdm import tqdm

from glacier_mapping.model.metrics import tp_fp_fn, precision, recall, IoU
from glacier_mapping.model.visualize import (
    build_cmap_from_mask_names, make_rgb_preview, label_to_color, 
    make_confidence_map, make_entropy_map, make_tp_fp_fn_masks, make_eight_panel
)

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
        if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback:
            base_dir = Path(trainer.checkpoint_callback.dirpath).parent
        else:
            # Fallback to default_root_dir or current directory
            base_dir = Path(trainer.default_root_dir) if trainer.default_root_dir else Path(".")
        
        output_dir = base_dir / "full_evaluations" / f"epoch_{trainer.current_epoch + 1}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get test tiles
        data_dir = Path(pl_module.processed_dir) / "test"
        test_tiles_all = sorted(data_dir.glob("tiff*"))
        
        if not test_tiles_all:
            print("Warning: No test tiles found for full-tile evaluation")
            return
        
        # Select informative tiles for visualization
        test_tiles = self._select_informative_tiles(test_tiles_all, pl_module, self.num_samples)
        
        # Setup metrics
        n_classes = len(pl_module.class_names)
        threshold = pl_module.metrics_opts.get('threshold', [0.5, 0.5])
        cmap = build_cmap_from_mask_names(pl_module.class_names)
        
        rows = []
        tp_sum = np.zeros(n_classes)
        fp_sum = np.zeros(n_classes)
        fn_sum = np.zeros(n_classes)
        
        # Evaluate all test tiles
        for x_path in tqdm(test_tiles_all, desc="Full-tile evaluation"):
            x = np.load(x_path)
            y_pred, invalid_mask = pl_module.predict_slice(x, threshold)
            
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
                if len(pl_module.output_classes) == 1:  # Binary
                    pred_label = ci + 1
                    p = (y_pred_valid == pred_label).astype(np.uint8)
                    if ci == 0:
                        t = (y_true_valid_raw != pl_module.output_classes[0]).astype(np.uint8)
                    else:
                        t = (y_true_valid_raw == pl_module.output_classes[0]).astype(np.uint8)
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
        
        # Save CSV
        cols = ["tile"]
        for cname in pl_module.class_names:
            cols += [f"{cname}_precision", f"{cname}_recall", f"{cname}_IoU"]
        
        pd.DataFrame(rows, columns=cols).to_csv(output_dir / f"full_eval_epoch{trainer.current_epoch + 1}.csv", index=False)
        
        # Log metrics to both loggers
        for ci, cname in enumerate(pl_module.class_names):
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
        print(f"Visualizations completed.")
        
        # Log to MLflow if available
        for logger in trainer.loggers:
            if isinstance(logger, MLFlowLogger):
                try:
                    logger.experiment.log_artifact(
                        logger.run_id,
                        str(output_dir), 
                        artifact_path=f"full_evaluations/epoch_{trainer.current_epoch + 1}"
                    )
                except Exception as e:
                    print(f"Warning: Failed to log full evaluation to MLflow: {e}")
    
    def _select_informative_tiles(self, tile_paths, pl_module, num_samples):
        """Select tiles with most target class pixels."""
        tile_class_counts = []
        
        for x_path in tile_paths:
            mask_path = x_path.with_name(x_path.name.replace("tiff", "mask"))
            mask = np.load(mask_path)
            
            if len(pl_module.output_classes) == 1:  # Binary
                class_pixels = (mask == pl_module.output_classes[0]).sum()
            else:  # Multi-class
                class_pixels = ((mask > 0) & (mask != 255)).sum()
            
            tile_class_counts.append((x_path, int(class_pixels)))
        
        tile_class_counts.sort(key=lambda x: x[1], reverse=True)
        selected = [path for path, count in tile_class_counts if count > 0]
        
        if len(selected) < num_samples:
            selected = [path for path, count in tile_class_counts]
        
        return selected[:num_samples]
    
    def _generate_visualizations(self, pl_module, test_tiles, output_dir, epoch):
        """Generate 8-panel visualizations."""
        import cv2
        
        threshold = pl_module.metrics_opts.get('threshold', [0.5, 0.5])
        cmap = build_cmap_from_mask_names(pl_module.class_names)
        
        for idx, x_path in enumerate(test_tiles):
            x_full = np.load(x_path)
            y_true_raw = np.load(x_path.with_name(x_path.name.replace("tiff", "mask"))).astype(np.uint8)
            
            # Predict using Lightning module
            use_ch = pl_module.use_channels
            x = x_full[:, :, use_ch]
            x_norm = pl_module.normalize(x)
            
            inp = torch.from_numpy(np.expand_dims(x_norm, 0)).float().to(pl_module.device)
            logits = pl_module.forward(inp.permute(0, 3, 1, 2))
            
            if len(pl_module.output_classes) == 1:  # Binary
                probs = torch.sigmoid(logits)[0].cpu().numpy()  # (2, H, W)
                y_pred_viz = (probs[1] >= threshold[0]).astype(np.uint8)  # Use foreground channel
                conf = probs[1]
            else:  # Multi-class
                probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
                y_pred_viz = np.argmax(probs, axis=2).astype(np.uint8) + 1
                conf = np.max(probs, axis=-1)
            
            # Generate 8-panel visualization
            ignore = y_true_raw == 255
            invalid_mask = np.sum(x, axis=2) == 0
            
            # GT/PRED for visualization
            y_gt_vis = y_true_raw.copy()
            y_pred_vis = y_pred_viz.copy()
            
            if len(pl_module.output_classes) == 1:  # Binary
                class_idx = pl_module.output_classes[0]
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
            if len(pl_module.output_classes) == 1:  # Binary
                # For binary, transpose probs to (H, W, 2) for entropy calculation
                entropy_rgb = make_entropy_map(probs.transpose(1, 2, 0), invalid_mask=ignore)
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
            for ci, cname in enumerate(pl_module.class_names):
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