#!/usr/bin/env python3
"""Simple working Lightning training script."""

import os
import sys
from pathlib import Path
from typing import Dict, Any

import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from glacier_mapping.core.frame import Framework


class SimpleGlacierModule(pl.LightningModule):
    """Simplified Lightning module for glacier segmentation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        
        # Load the original framework to get model and loss
        self.framework = Framework.from_dict(config)
        self.model = self.framework.model
        self.loss_fn = self.framework.loss_fn
        
        # Training config
        self.lr = config.get('optim_opts', {}).get('args', {}).get('lr', 0.0003)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y_onehot, y_int = batch
        y_hat = self(x)
        
        # Use framework's loss computation
        sigma_values = [sigma.item() for sigma in self.framework.sigma_list]
        loss = self.loss_fn(y_hat, y_onehot, y_int, sigma_values)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y_onehot, y_int = batch
        y_hat = self(x)
        
        sigma_values = [sigma.item() for sigma in self.framework.sigma_list]
        loss = self.loss_fn(y_hat, y_onehot, y_int, sigma_values)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        
        # Add scheduler if specified
        scheduler_opts = self.hparams.get('scheduler_opts')
        if scheduler_opts and scheduler_opts.get('name') == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=scheduler_opts.get('args', {}).get('max_lr', 0.001),
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=scheduler_opts.get('args', {}).get('pct_start', 0.3),
                anneal_strategy=scheduler_opts.get('args', {}).get('anneal_strategy', 'cos')
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        
        return optimizer


class SimpleDataModule(pl.LightningDataModule):
    """Simplified data module using existing data loaders."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.loader_opts = config.get('loader_opts', {})
        
        # Get processed data path
        self.processed_dir = os.environ.get(
            'GLACIER_PROCESSED_DATA', 
            "/home/devj/local-debian/processed_data/bibek_w512_o64_f1_v2"
        )
    
    def setup(self, stage=None):
        from glacier_mapping.data.data import fetch_loaders
        
        self.train_loader, self.val_loader, self.test_loader = fetch_loaders(
            processed_dir=self.processed_dir,
            **self.loader_opts
        )
    
    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return self.test_loader


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train glacier mapping with Lightning")
    parser.add_argument("--config", type=str, default="conf/unet_train.yaml")
    parser.add_argument("--experiment-name", type=str, default="lightning_test")
    parser.add_argument("--max-epochs", type=int, default=5)  # Short test
    parser.add_argument("--resume", type=str, default=None)
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
        
    config = load_config(str(config_path))
    print(f"Loaded config from: {config_path}")
    print(f"Experiment: {args.experiment_name}")
    
    # Create components
    print("Creating model...")
    model = SimpleGlacierModule(config)
    
    print("Creating data module...")
    datamodule = SimpleDataModule(config)
    
    # Setup logging
    print("Setting up logging...")
    mlflow_logger = MLFlowLogger(
        experiment_name="glacier_mapping",
        run_name=args.experiment_name,
        tracking_uri="http://localhost:5000",
        tags={"framework": "pytorch_lightning", "test": "simple"}
    )
    
    tb_logger = TensorBoardLogger(
        save_dir="./lightning_outputs/tensorboard",
        name=args.experiment_name
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            filename=f"{args.experiment_name}_{{epoch:03d}}_{{val_loss:.4f}}"
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Create trainer
    print("Creating trainer...")
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.max_epochs,
        logger=[mlflow_logger, tb_logger],
        callbacks=callbacks,
        precision="16-mixed",
        log_every_n_steps=10,
        val_check_interval=1.0,
        enable_progress_bar=True,
    )
    
    print(f"Starting training for {args.max_epochs} epochs...")
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"MLflow UI: http://localhost:5000")
    
    try:
        trainer.fit(
            model,
            datamodule=datamodule,
            ckpt_path=args.resume if args.resume else None
        )
        
        print("Training completed successfully!")
        
        # Test best model
        print("Running test evaluation...")
        trainer.test(model, datamodule=datamodule, ckpt_path="best")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()