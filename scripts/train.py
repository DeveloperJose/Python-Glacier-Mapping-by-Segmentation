#!/usr/bin/env python3
"""Simple Lightning training for glacier mapping."""

import argparse
import pathlib
from typing import Dict, Any

import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from glacier_mapping.lightning.glacier_module import GlacierSegmentationModule
from glacier_mapping.lightning.glacier_datamodule import GlacierDataModule
from glacier_mapping.lightning.best_model_callback import (
    BestModelFullEvaluationCallback,
)

# Import MLflow utilities
try:
    from glacier_mapping.utils.mlflow_utils import MLflowManager

    MLFLOW_AVAILABLE = True
except ImportError as e:
    print(f"Warning: MLflow utilities not available: {e}")
    MLFLOW_AVAILABLE = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train glacier mapping with Lightning")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--max-epochs", type=int, default=3, help="Maximum epochs to train"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device to use")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    # MLflow arguments
    parser.add_argument(
        "--server",
        type=str,
        required=True,
        help="Server name (must exist in servers.yaml)",
    )
    parser.add_argument(
        "--mlflow-enabled",
        type=str,
        default="true",
        help="Enable MLflow logging (true/false)",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="https://mlflow.developerjose.duckdns.org/",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Override MLflow experiment name (bypasses automatic categorization)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from server config",
    )

    args = parser.parse_args()

    # Load configuration
    config_path = pathlib.Path(args.config)

    config = load_config(str(config_path))

    # Parse MLflow arguments
    mlflow_enabled = args.mlflow_enabled.lower() == "true"

    # Load server configuration (explicit, no defaults)
    servers_yaml_path = pathlib.Path("configs") / "servers.yaml"
    if MLFLOW_AVAILABLE:
        server_config = MLflowManager.load_server_config(
            str(servers_yaml_path), args.server
        )  # type: ignore[arg-type]
    else:
        # Fallback server config for when MLflow unavailable
        with open(servers_yaml_path, "r") as f:
            servers = yaml.safe_load(f)
        if args.server not in servers:
            raise ValueError(f"Server '{args.server}' not found in {servers_yaml_path}")
        server_config = servers[args.server]

    # Extract configuration sections
    training_opts = config.get("training_opts", {})
    loader_opts = config.get("loader_opts", {})
    model_opts = config.get("model_opts", {})
    loss_opts = config.get("loss_opts", {})
    optim_opts = config.get("optim_opts", {})
    scheduler_opts = config.get("scheduler_opts", {})
    metrics_opts = config.get("metrics_opts", {})

    # Get run name and output directory
    base_run_name = training_opts.get("run_name", "experiment")
    output_dir = args.output_dir or training_opts.get("output_dir", "output/")

    # Generate MLflow experiment name and run name
    if mlflow_enabled and MLFLOW_AVAILABLE:
        experiment_name = args.experiment_name or MLflowManager.categorize_experiment(
            config
        )  # type: ignore[assignment]
        run_name = MLflowManager.generate_run_name(base_run_name, args.server)  # type: ignore[assignment]
        mlflow_params = MLflowManager.extract_mlflow_params(config, server_config)  # type: ignore[assignment]
        mlflow_tags = MLflowManager.generate_run_tags(
            config, server_config, str(config_path)
        )  # type: ignore[assignment]
    else:
        experiment_name = None
        run_name = base_run_name
        mlflow_params = {}
        mlflow_tags = {}

    print(f"Loaded config from: {config_path}")
    print(f"Server: {args.server}")
    print(f"MLflow enabled: {mlflow_enabled}")
    if mlflow_enabled and MLFLOW_AVAILABLE:
        print(f"MLflow experiment: {experiment_name}")
        print(f"MLflow run name: {run_name}")
    print(f"Base run name: {base_run_name}")
    print(f"Output directory: {output_dir}")
    print(f"Data path: {loader_opts.get('processed_dir', 'NOT_SET')}")
    print(f"Using channels: {loader_opts.get('use_channels', 'NOT_SET')}")
    print(f"Output classes: {loader_opts.get('output_classes', 'NOT_SET')}")

    # Save config as JSON for upload script
    import json

    config_output_dir = pathlib.Path(output_dir) / run_name
    config_output_dir.mkdir(parents=True, exist_ok=True)
    config_json_path = config_output_dir / "conf.json"
    with open(config_json_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_json_path}")

    # Create data module
    print("Creating data module...")
    datamodule = GlacierDataModule(
        processed_dir=loader_opts.get("processed_dir", "/tmp"),
        batch_size=loader_opts.get("batch_size", 8),
        use_channels=loader_opts.get("use_channels", [0, 1, 2]),
        output_classes=loader_opts.get("output_classes", [1]),
        class_names=loader_opts.get("class_names", ["BG", "CleanIce", "Debris"]),
        normalize=loader_opts.get("normalize", "mean-std"),
    )

    # Create model
    print("Creating model...")
    model = GlacierSegmentationModule(
        model_opts=model_opts,
        loss_opts=loss_opts,
        optim_opts=optim_opts,
        scheduler_opts=scheduler_opts,
        metrics_opts=metrics_opts,
        training_opts=training_opts,
        loader_opts=loader_opts,
        use_channels=loader_opts.get("use_channels", [0, 1, 2]),
        output_classes=loader_opts.get("output_classes", [1]),
        class_names=loader_opts.get("class_names", ["BG", "CleanIce", "Debris"]),
    )

    # Setup logging
    print("Setting up logging...")
    from pytorch_lightning.loggers import Logger

    loggers: list[Logger] = [
        TensorBoardLogger(save_dir=f"{output_dir}/{run_name}/logs", name="")
    ]

    # Add MLflow logger if enabled and available
    mlflow_logger = None
    if mlflow_enabled and MLFLOW_AVAILABLE and experiment_name:
        try:
            from pytorch_lightning.loggers import MLFlowLogger

            mlflow_logger = MLFlowLogger(
                experiment_name=experiment_name,
                run_name=run_name,
                tracking_uri=args.tracking_uri,
                tags=mlflow_tags,
                log_model=False,  # Disable automatic model logging to save MLflow storage
            )
            loggers.append(mlflow_logger)
            print(f"MLflow logger setup complete for experiment: {experiment_name}")
        except Exception as e:
            print(f"Warning: Failed to setup MLflow logger: {e}")
            mlflow_logger = None

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=f"{output_dir}/{run_name}/checkpoints",
            monitor="val_loss",
            mode="min",
            save_top_k=-1,  # Keep all improving checkpoints
            save_last=True,
            filename=f"{run_name}_{{epoch:03d}}_{{val_loss:.4f}}",
        ),
        LearningRateMonitor(logging_interval="step"),
        # DeviceStatsMonitor(cpu_stats=True),  # Automatic system monitoring
        BestModelFullEvaluationCallback(
            num_samples=training_opts.get("num_viz_samples", 4)
        ),
    ]

    # Create trainer
    print("Creating trainer...")
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[args.gpu],
        max_epochs=args.max_epochs,
        logger=loggers,  # Support multiple loggers
        callbacks=callbacks,
        precision="16-mixed",
        log_every_n_steps=10,
        val_check_interval=1.0,
        enable_progress_bar=True,
        num_sanity_val_steps=2,  # Quick sanity check
    )

    print(f"Starting training for {args.max_epochs} epochs...")
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"TensorBoard logs: {output_dir}/{run_name}/logs")
    print(f"Checkpoints: {output_dir}/{run_name}/checkpoints")

    try:
        trainer.fit(
            model, datamodule=datamodule, ckpt_path=args.resume if args.resume else None
        )

        print("Training completed successfully!")

    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        # Lightning's DeviceStatsMonitor and dual loggers handle error logging automatically
        raise


if __name__ == "__main__":
    main()
