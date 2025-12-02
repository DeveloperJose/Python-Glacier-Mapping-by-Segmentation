#!/usr/bin/env python3
"""Simple Lightning training for glacier mapping."""

import argparse
import pathlib
import sys
from typing import Dict, Any

import torch
import torch.nn as nn
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Add project root to path for imports
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from glacier_mapping.model.unet import Unet
from glacier_mapping.model.losses import customloss
from glacier_mapping.data.data import fetch_loaders


class SimpleGlacierModule(pl.LightningModule):
    """Simple Lightning module for glacier segmentation."""
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        loss_config: Dict[str, Any],
        optim_config: Dict[str, Any],
        scheduler_config: Dict[str, Any],
        use_channels: list,
        output_classes: list,
        class_names: list,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model setup
        model_args = model_config.get('args', {})
        # For binary classification, output_classes has 1 element but we need 2 output channels (BG, class)
        out_channels = 2 if len(output_classes) == 1 else len(output_classes)
        self.model = Unet(
            inchannels=len(use_channels),
            outchannels=out_channels,
            **model_args
        )
        
        # Loss setup - filter only supported args
        supported_loss_args = {'act', 'smooth', 'label_smoothing', 'masked', 'theta0', 'theta', 'foreground_classes', 'alpha'}
        loss_args = {k: v for k, v in loss_config.items() if k in supported_loss_args}
        self.loss_fn = customloss(**loss_args)
        
        # Learnable sigma parameters for loss weighting
        self.sigma_list = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(2)
        ])
        
        # Optimizer config
        self.lr = float(optim_config.get('args', {}).get('lr', 0.0003))
        self.weight_decay = float(optim_config.get('args', {}).get('weight_decay', 5e-5))
        
        # Scheduler config
        self.scheduler_config = scheduler_config
        
        # Class info
        self.class_names = class_names
        self.output_classes = output_classes
        self.use_channels = use_channels
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y_onehot, y_int = batch
        
        # Convert from NHWC to NCHW format expected by Conv2D
        x = x.permute(0, 3, 1, 2).contiguous()
        y_onehot = y_onehot.permute(0, 3, 1, 2).contiguous()
        y_int = y_int.squeeze(-1)  # Remove last dimension
        
        # Forward pass
        y_hat = self(x)
        
        # Compute loss (customloss returns [dice_loss, boundary_loss])
        loss_list = self.loss_fn(y_hat, y_onehot, y_int)
        loss = loss_list[0] + loss_list[1]  # Combine both losses
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y_onehot, y_int = batch
        
        # Convert from NHWC to NCHW format expected by Conv2D
        x = x.permute(0, 3, 1, 2).contiguous()
        y_onehot = y_onehot.permute(0, 3, 1, 2).contiguous()
        y_int = y_int.squeeze(-1)  # Remove last dimension
        
        # Forward pass
        with torch.no_grad():
            y_hat = self(x)
        
        # Compute loss
        loss_list = self.loss_fn(y_hat, y_onehot, y_int)
        loss = loss_list[0] + loss_list[1]  # Combine both losses
        
        # Log metrics
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
            total_steps = int(self.trainer.estimated_stepping_batches)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.scheduler_config.get('args', {}).get('max_lr', 0.001),
                total_steps=total_steps,
                pct_start=self.scheduler_config.get('args', {}).get('pct_start', 0.3),
                anneal_strategy=self.scheduler_config.get('args', {}).get('anneal_strategy', 'cos')
            )
            return [optimizer], [scheduler]
        
        return optimizer


class SimpleDataModule(pl.LightningDataModule):
    """Simple data module using existing fetch_loaders function."""
    
    def __init__(self, loader_config: Dict[str, Any]):
        super().__init__()
        self.loader_config = loader_config
        
    def setup(self, stage=None):
        # Use existing fetch_loaders function
        self.train_loader, self.val_loader, self.test_loader = fetch_loaders(
            **self.loader_config
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
    parser = argparse.ArgumentParser(description="Train glacier mapping with Lightning")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--max-epochs", type=int, default=3, help="Maximum epochs to train")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device to use")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = pathlib.Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    
    config = load_config(str(config_path))
    
    # Extract configuration sections
    training_opts = config.get('training_opts', {})
    loader_opts = config.get('loader_opts', {})
    model_opts = config.get('model_opts', {})
    loss_opts = config.get('loss_opts', {})
    optim_opts = config.get('optim_opts', {})
    scheduler_opts = config.get('scheduler_opts', {})
    
    # Get run name and output directory
    run_name = training_opts.get('run_name', 'experiment')
    output_dir = training_opts.get('output_dir', 'output/')
    
    print(f"Loaded config from: {config_path}")
    print(f"Run name: {run_name}")
    print(f"Output directory: {output_dir}")
    print(f"Data path: {loader_opts.get('processed_dir', 'NOT_SET')}")
    print(f"Using channels: {loader_opts.get('use_channels', 'NOT_SET')}")
    print(f"Output classes: {loader_opts.get('output_classes', 'NOT_SET')}")
    
    # Create data module
    print("Creating data module...")
    datamodule = SimpleDataModule(loader_opts)
    
    # Create model
    print("Creating model...")
    model = SimpleGlacierModule(
        model_config=model_opts,
        loss_config=loss_opts,
        optim_config=optim_opts,
        scheduler_config=scheduler_opts,
        use_channels=loader_opts.get('use_channels', [0, 1, 2]),
        output_classes=loader_opts.get('output_classes', [1]),
        class_names=loader_opts.get('class_names', ["BG", "CleanIce", "Debris"]),
    )
    
    # Setup logging
    print("Setting up logging...")
    logger = TensorBoardLogger(
        save_dir=f"{output_dir}/logs",
        name=run_name
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=f"{output_dir}/{run_name}/checkpoints",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            filename=f"{run_name}_{{epoch:03d}}_{{val_loss:.4f}}"
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Create trainer
    print("Creating trainer...")
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[args.gpu],
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=callbacks,
        precision="16-mixed",
        log_every_n_steps=10,
        val_check_interval=1.0,
        enable_progress_bar=True,
        num_sanity_val_steps=2,  # Quick sanity check
    )
    
    print(f"Starting training for {args.max_epochs} epochs...")
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"TensorBoard logs: {output_dir}/logs/{run_name}")
    print(f"Checkpoints: {output_dir}/{run_name}/checkpoints")
    
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