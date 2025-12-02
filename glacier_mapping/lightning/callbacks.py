"""Custom callbacks for glacier mapping training."""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import MLFlowLogger

from glacier_mapping.model.visualize import make_eight_panel

# Import MLflow utilities with error handling
try:
    from glacier_mapping.utils.mlflow_utils import MLflowManager
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


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
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Generate visualizations at end of validation epoch."""
        if (trainer.current_epoch + 1) % self.log_every_n_epochs == 0:
            self._generate_visualizations(trainer, pl_module)
    
    def _generate_visualizations(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Generate and save visualization samples."""
        # Get a batch from validation dataloader
        if trainer.val_dataloaders and len(trainer.val_dataloaders) > 0:
            val_dataloader = trainer.val_dataloaders[0]
            batch = next(iter(val_dataloader))
        else:
            print("Warning: No validation dataloader available for visualization")
            return
        
        x, y_onehot, y_int = batch
        
        # Get model predictions
        pl_module.eval()
        with torch.no_grad():
            y_hat = pl_module(x)
        
        # Generate visualizations for first N samples
        for i in range(min(self.num_samples, x.shape[0])):
            sample_x = x[i:i+1]
            sample_y = y_int[i:i+1]
            sample_pred = torch.argmax(y_hat[i:i+1], dim=1, keepdim=True)
            
            # Create and save 8-panel visualization
            if self.save_dir:
                self.save_dir.mkdir(parents=True, exist_ok=True)
                save_path = self.save_dir / f"epoch_{trainer.current_epoch}_sample_{i}.png"
                
                from matplotlib import pyplot as plt
                
                viz = make_eight_panel(
                    x_rgb=sample_x, gt_rgb=sample_y, pr_rgb=sample_pred,
                    conf_rgb=None, tp_rgb=None, fp_rgb=None, fn_rgb=None, entropy_rgb=None,
                    metrics_text=None
                )
                
                plt.imshow(viz)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Local save only - no MLflow logging to save storage


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
        **kwargs
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
            **kwargs
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
        batch_idx: int
    ):
        """Log additional training metrics."""
        if batch_idx % self.log_every_n_steps == 0:
            # Log GPU memory usage
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                pl_module.log('gpu_memory_gb', gpu_memory, on_step=True, on_epoch=False)
            
            # Log learning rate
            optimizer = trainer.optimizers[0]
            current_lr = optimizer.param_groups[0]['lr']
            pl_module.log('learning_rate', current_lr, on_step=True, on_epoch=False)
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Log validation summary metrics."""
        # Log epoch duration
        if hasattr(trainer, 'callback_metrics') and 'val_loss' in trainer.callback_metrics:
            val_loss = trainer.callback_metrics['val_loss']
            pl_module.log('epoch', trainer.current_epoch, on_step=False, on_epoch=True)
            
            # Log improvement over best
            if hasattr(pl_module, 'best_val_loss') and pl_module.best_val_loss is not None:
                improvement = pl_module.best_val_loss - val_loss
                pl_module.log('val_loss_improvement', improvement, on_step=False, on_epoch=True)



    
    def _reconstruct_config_from_module(self, pl_module: pl.LightningModule) -> Dict[str, Any]:
        """Reconstruct configuration dict from Lightning module hyperparameters."""
        config = {}
        
        # Training options
        if hasattr(pl_module, 'training_opts'):
            config['training_opts'] = pl_module.training_opts
        else:
            config['training_opts'] = {
                'full_eval_every': self.full_eval_every,
                'num_viz_samples': self.num_samples
            }
        
        # Model options
        config['model_opts'] = pl_module.model_opts
        
        # Loss options
        config['loss_opts'] = pl_module.loss_opts
        
        # Metrics options
        config['metrics_opts'] = pl_module.metrics_opts
        
        # Loader options
        if hasattr(pl_module, 'use_channels') and hasattr(pl_module, 'output_classes'):
            config['loader_opts'] = {
                'use_channels': pl_module.use_channels,
                'output_classes': pl_module.output_classes,
                'class_names': pl_module.class_names,
                'processed_dir': getattr(pl_module, 'processed_dir', '/tmp')
            }
        
        return config
    
    def _log_evaluation_results(self, trainer: pl.Trainer, pl_module: pl.LightningModule, output_dir: Path):
        """Log evaluation results to MLflow and TensorBoard."""
        epoch = trainer.current_epoch + 1
        
        # Find and log CSV results
        csv_files = list(output_dir.glob("*.csv"))
        for csv_file in csv_files:
            # Log to MLflow as artifact
            for logger in trainer.loggers:
                if isinstance(logger, MLFlowLogger):
                    try:
                        logger.experiment.log_artifact(
                            logger.run_id,
                            str(csv_file), 
                            artifact_path=f"test_evaluations/epoch_{epoch}"
                        )
                    except Exception as e:
                        print(f"Warning: Failed to log CSV to MLflow: {e}")
            
            # Parse CSV and log metrics
            try:
                import pandas as pd
                df = pd.read_csv(csv_file)
                
                # Log summary metrics to both loggers
                for class_name in ['CleanIce', 'Debris']:
                    if f'{class_name}_iou' in df.columns:
                        iou_mean = df[f'{class_name}_iou'].mean()
                        pl_module.log(f'test_{class_name}_iou', iou_mean, on_step=False, on_epoch=True)
                    
                    if f'{class_name}_precision' in df.columns:
                        precision_mean = df[f'{class_name}_precision'].mean()
                        pl_module.log(f'test_{class_name}_precision', precision_mean, on_step=False, on_epoch=True)
                    
                    if f'{class_name}_recall' in df.columns:
                        recall_mean = df[f'{class_name}_recall'].mean()
                        pl_module.log(f'test_{class_name}_recall', recall_mean, on_step=False, on_epoch=True)
                        
            except Exception as e:
                print(f"Warning: Failed to parse CSV metrics: {e}")
        
        # Log visualization images
        viz_files = list(output_dir.glob("*.png"))
        for viz_file in viz_files:
            # Log to MLflow as artifact
            for logger in trainer.loggers:
                if isinstance(logger, MLFlowLogger):
                    try:
                        logger.experiment.log_artifact(
                            logger.run_id,
                            str(viz_file), 
                            artifact_path=f"test_evaluations/epoch_{epoch}/visualizations"
                        )
                    except Exception as e:
                        print(f"Warning: Failed to log visualization to MLflow: {e}")
            
            # Log to TensorBoard
            if hasattr(trainer, 'logger') and trainer.logger:
                try:
                    from matplotlib import pyplot as plt
                    img = plt.imread(viz_file)
                    trainer.logger.experiment.add_image(
                        f"test_evaluation/{viz_file.stem}", 
                        img, 
                        dataformats='HWC',
                        global_step=epoch
                    )
                    plt.close()
                except Exception as e:
                    print(f"Warning: Failed to log visualization to TensorBoard: {e}")