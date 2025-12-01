#!/usr/bin/env python3
"""Lightning training script for glacier mapping."""

import os
import sys
from pathlib import Path
from typing import Dict, Any

import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from glacier_mapping.lightning import (
    GlacierSegmentationModule,
    GlacierDataModule,
    GlacierVisualizationCallback,
    GlacierModelCheckpoint,
    GlacierTrainingMonitor,
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_trainer(
    config: Dict[str, Any],
    experiment_name: str,
    output_dir: str = "./lightning_outputs"
) -> pl.Trainer:
    """Create PyTorch Lightning trainer with MLflow logging."""
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup MLflow logger
    mlflow_logger = MLFlowLogger(
        experiment_name="glacier_mapping",
        run_name=experiment_name,
        tracking_uri="http://localhost:5000",  # MLflow server
        artifact_location=str(output_dir / "mlflow_artifacts"),
        tags={"framework": "pytorch_lightning", "config": experiment_name}
    )
    
    # Also setup TensorBoard logger for compatibility
    tb_logger = TensorBoardLogger(
        save_dir=str(output_dir / "tensorboard"),
        name=experiment_name
    )
    
    # Setup callbacks
    callbacks = [
        # Model checkpointing
        GlacierModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            dirpath=str(output_dir / experiment_name / "checkpoints"),
            filename=f"{experiment_name}_{{epoch:03d}}_{{val_loss:.4f}}"
        ),
        
        # Learning rate monitoring
        LearningRateMonitor(logging_interval='step'),
        
        # Training monitoring
        GlacierTrainingMonitor(log_every_n_steps=50),
        
        # Visualization (every 10 epochs)
        GlacierVisualizationCallback(
            num_samples=4,
            log_every_n_epochs=10,
            save_dir=str(output_dir / experiment_name / "visualizations")
        )
    ]
    
    # Determine accelerator and devices
    accelerator = "gpu" if pl.utilities.accelerator.is_accelerator_available("gpu") else "cpu"
    devices = 1 if accelerator == "gpu" else None  # Single GPU for now
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=config.get('training_opts', {}).get('epochs', 500),
        logger=[mlflow_logger, tb_logger],
        callbacks=callbacks,
        gradient_clip_val=0.5,  # Prevent gradient explosion
        accumulate_grad_batches=1,
        precision="16-mixed" if accelerator == "gpu" else 32,  # Mixed precision for GPU
        deterministic=True,
        benchmark=True,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,
        val_check_interval=1.0,  # Validate every epoch
        num_sanity_val_steps=2,  # Quick sanity check
    )
    
    return trainer


def create_datamodule(config: Dict[str, Any]) -> GlacierDataModule:
    """Create Glacier data module from config."""
    
    loader_opts = config.get('loader_opts', {})
    
    # Get processed data path - this needs to be configured
    # For now, use a default that matches your setup
    processed_dir = os.environ.get(
        'GLACIER_PROCESSED_DATA', 
        "/home/devj/local-debian/processed_data/bibek_w512_o64_f1_v2"
    )
    
    datamodule = GlacierDataModule(
        processed_dir=processed_dir,
        batch_size=loader_opts.get('batch_size', 8),
        use_channels=loader_opts.get('use_channels', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        output_classes=loader_opts.get('output_classes', [0, 1, 2]),
        class_names=loader_opts.get('class_names', ["BG", "CleanIce", "Debris"]),
        normalize=loader_opts.get('normalize', "mean-std"),
        num_workers=4,
        pin_memory=True,
    )
    
    return datamodule


def create_model(config: Dict[str, Any]) -> GlacierSegmentationModule:
    """Create Glacier segmentation module from config."""
    
    return GlacierSegmentationModule(
        model_opts=config.get('model_opts', {}),
        loss_opts=config.get('loss_opts', {}),
        optim_opts=config.get('optim_opts', {}),
        scheduler_opts=config.get('scheduler_opts', {}),
        metrics_opts=config.get('metrics_opts', {}),
        training_opts=config.get('training_opts', {}),
        reg_opts=config.get('reg_opts', None),
        class_names=config.get('loader_opts', {}).get('class_names', ["BG", "CleanIce", "Debris"]),
        output_classes=config.get('loader_opts', {}).get('output_classes', [0, 1, 2]),
        use_channels=config.get('loader_opts', {}).get('use_channels', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    )


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train glacier mapping with PyTorch Lightning")
    parser.add_argument(
        "--config", 
        type=str, 
        default="conf/unet_train.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="lightning_experiment",
        help="Experiment name for logging"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./lightning_outputs",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")
    print(f"Experiment name: {args.experiment_name}")
    
    # Create components
    print("Creating data module...")
    datamodule = create_datamodule(config)
    
    print("Creating model...")
    model = create_model(config)
    
    print("Creating trainer...")
    trainer = create_trainer(config, args.experiment_name, args.output_dir)
    
    # Start training
    print(f"Starting training for {config.get('training_opts', {}).get('epochs', 500)} epochs...")
    print(f"Training with accelerator: {trainer.accelerator}, devices: {trainer.devices}")
    
    try:
        trainer.fit(
            model,
            datamodule=datamodule,
            ckpt_path=args.resume if args.resume else None
        )
        
        print("Training completed successfully!")
        
        # Test the best model
        print("Running test evaluation...")
        trainer.test(model, datamodule=datamodule, ckpt_path="best")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()