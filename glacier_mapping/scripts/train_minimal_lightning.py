#!/usr/bin/env python3
"""Minimal working Lightning training script."""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

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

from glacier_mapping.model.unet import Unet
from glacier_mapping.model.losses import customloss
from glacier_mapping.data.data import fetch_loaders


class MinimalGlacierModule(pl.LightningModule):
    """Minimal Lightning module for glacier segmentation."""
    
    def __init__(self, model_config: Dict, loss_config: Dict, 
                 optim_config: Dict, scheduler_config: Optional[Dict] = None):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        model_args = model_config.get('args', {})
        self.model = Unet(
            inchannels=len(model_args.get('use_channels', [0, 1, 2])),
            outchannels=len(model_args.get('output_classes', [1, 2])),
            net_depth=model_args.get('net_depth', 4),
            dropout=model_args.get('dropout', 0.1),
            spatial=model_args.get('spatial', True),
            first_channel_output=model_args.get('first_channel_output', 32)
        )
        
        # Loss - filter out unsupported args
        supported_args = {'act', 'smooth', 'label_smoothing', 'masked', 'theta0', 'theta', 'foreground_classes', 'alpha'}
        loss_args = {k: v for k, v in loss_config.items() if k in supported_args}
        self.loss_fn = customloss(**loss_args)
        
        # Learnable sigma parameters
        self.sigma_list = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(2)
        ])
        
        # Optimizer config
        self.lr = optim_config.get('args', {}).get('lr', 0.0003)
        self.weight_decay = optim_config.get('args', {}).get('weight_decay', 5e-5)
        
        # Scheduler config
        self.scheduler_config = scheduler_config
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y_onehot, y_int = batch
        y_hat = self(x)
        
        # Compute loss with sigma
        sigma_values = [sigma.item() for sigma in self.sigma_list]
        loss = self.loss_fn(y_hat, y_onehot, y_int, sigma_values)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y_onehot, y_int = batch
        y_hat = self(x)
        
        sigma_values = [sigma.item() for sigma in self.sigma_list]
        loss = self.loss_fn(y_hat, y_onehot, y_int, sigma_values)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # Add scheduler if specified
        if self.scheduler_config and self.scheduler_config.get('name') == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.scheduler_config.get('args', {}).get('max_lr', 0.001),
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=self.scheduler_config.get('args', {}).get('pct_start', 0.3),
                anneal_strategy=self.scheduler_config.get('args', {}).get('anneal_strategy', 'cos')
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        
        return optimizer


class MinimalDataModule(pl.LightningDataModule):
    """Minimal data module."""
    
    def __init__(self, loader_config: Dict):
        super().__init__()
        self.loader_config = loader_config
        self.processed_dir = os.environ.get(
            'GLACIER_PROCESSED_DATA', 
            "/home/devj/local-debian/processed_data/bibek_w512_o64_f1_v2"
        )
    
    def setup(self, stage=None):
        self.train_loader, self.val_loader, self.test_loader = fetch_loaders(
            processed_dir=self.processed_dir,
            **self.loader_config
        )
    
    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return self.test_loader


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Minimal Lightning training")
    parser.add_argument("--config", type=str, default="conf/unet_train.yaml")
    parser.add_argument("--experiment-name", type=str, default="minimal_lightning")
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config from: {config_path}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Max epochs: {args.max_epochs}")
    
    # Create components
    print("Creating model...")
    model = MinimalGlacierModule(
        model_config=config.get('model_opts', {}),
        loss_config=config.get('loss_opts', {}),
        optim_config=config.get('optim_opts', {}),
        scheduler_config=config.get('scheduler_opts')
    )
    
    print("Creating data module...")
    datamodule = MinimalDataModule(config.get('loader_opts', {}))
    
    # Setup logging
    print("Setting up logging...")
    mlflow_logger = MLFlowLogger(
        experiment_name="glacier_mapping",
        run_name=args.experiment_name,
        tracking_uri="http://localhost:5000",
        tags={"framework": "pytorch_lightning", "test": "minimal"}
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
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()