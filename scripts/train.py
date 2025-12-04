#!/usr/bin/env python3
"""Simple Lightning training for glacier mapping."""

import argparse
import pathlib
from typing import Dict, Any

import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import TensorBoardLogger

from glacier_mapping.lightning.glacier_module import GlacierSegmentationModule
from glacier_mapping.lightning.glacier_datamodule import GlacierDataModule
from glacier_mapping.lightning.callbacks import GlacierVisualizationCallback
from glacier_mapping.lightning.best_model_callback import (
    BestModelFullEvaluationCallback,
)

# Import MLflow utilities
try:
    import glacier_mapping.utils.mlflow_utils as mlflow_utils

    MLFLOW_AVAILABLE = True
except ImportError as e:
    print(f"Warning: MLflow utilities not available: {e}")
    MLFLOW_AVAILABLE = False

# Import error handling
try:
    from glacier_mapping.utils.error_handler import setup_error_handler

    ERROR_HANDLER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Error handler not available: {e}")
    ERROR_HANDLER_AVAILABLE = False


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
        "--max-epochs",
        type=int,
        default=None,
        help="Maximum epochs to train (overrides config value if specified)",
    )
    parser.add_argument(
        "--gpu", type=int, default=None, help="GPU device to use (default: auto-detect)"
    )
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
    parser.add_argument(
        "--no-output",
        action="store_true",
        default=False,
        help="Disable all output (checkpoints, logs, config). Used for Ray hyperparameter search.",
    )
    parser.add_argument(
        "--skip-full-eval",
        action="store_true",
        default=False,
        help="Skip full-tile test evaluation (overrides config)",
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
        server_config = mlflow_utils.load_server_config(
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

    # Determine max_epochs: CLI argument overrides config, otherwise use config value
    if args.max_epochs is not None:
        max_epochs = args.max_epochs
        print(f"Using max_epochs from CLI argument: {max_epochs}")
    else:
        max_epochs = training_opts.get("epochs", 100)  # Default to 100 if not in config
        print(f"Using max_epochs from config: {max_epochs}")

    # Auto-construct processed_dir from server config + dataset_name
    # This matches the pattern used in ray_train.py and preprocess.py
    if "processed_dir" not in loader_opts or not loader_opts["processed_dir"]:
        dataset_name = training_opts.get("dataset_name")
        if not dataset_name:
            raise ValueError(
                "Either 'training_opts.dataset_name' or 'loader_opts.processed_dir' "
                "must be specified in the config"
            )
        if "processed_data_path" not in server_config:
            raise ValueError(
                f"Server '{args.server}' config must include 'processed_data_path'"
            )
        loader_opts["processed_dir"] = (
            f"{server_config['processed_data_path']}/{dataset_name}/"
        )
        print(
            f"✓ Auto-constructed data path from server config: {loader_opts['processed_dir']}"
        )

    # Validate that the constructed/specified path exists
    data_path = pathlib.Path(loader_opts["processed_dir"])
    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed data directory does not exist: {data_path}\n"
            f"Please run preprocessing first or check your server config."
        )

    # Check for required normalization file
    norm_file = data_path / "normalize_train.npy"
    if not norm_file.exists():
        raise FileNotFoundError(
            f"Normalization file not found: {norm_file}\n"
            f"Please run preprocessing to generate normalization statistics."
        )

    # Get run name and output directory
    base_run_name = training_opts.get("run_name", "experiment")
    # Priority: CLI arg > config file > server config > fallback
    output_dir_source: str
    if args.output_dir:
        output_dir: str = args.output_dir
        output_dir_source = "CLI argument"
    elif training_opts.get("output_dir"):
        output_dir = str(training_opts.get("output_dir"))
        output_dir_source = "config file"
    elif server_config.get("output_path"):
        output_dir = str(server_config.get("output_path"))
        output_dir_source = f"server config ({args.server})"
    else:
        output_dir = "output/"
        output_dir_source = "default fallback"

    # Generate MLflow experiment name and run name
    if mlflow_enabled and MLFLOW_AVAILABLE:
        experiment_name = args.experiment_name or mlflow_utils.categorize_experiment(
            config
        )  # type: ignore[assignment]
        run_name = mlflow_utils.generate_run_name(base_run_name, args.server)  # type: ignore[assignment]
        mlflow_params = mlflow_utils.extract_mlflow_params(config, server_config)  # type: ignore[assignment]
        mlflow_tags = mlflow_utils.generate_run_tags(
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
    print(f"Output directory: {output_dir} (source: {output_dir_source})")
    print(f"Data path: {loader_opts.get('processed_dir', 'NOT_SET')}")
    print(f"Using channels: {loader_opts.get('use_channels', 'NOT_SET')}")
    print(f"Output classes: {loader_opts.get('output_classes', 'NOT_SET')}")

    # Save config as JSON for upload script (skip if --no-output)
    if not args.no_output:
        import json

        config_output_dir = pathlib.Path(output_dir) / run_name
        config_output_dir.mkdir(parents=True, exist_ok=True)
        config_json_path = config_output_dir / "conf.json"
        with open(config_json_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to: {config_json_path}")
    else:
        print("Skipping config save (--no-output mode)")

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

    # Setup logging (skip if --no-output for Ray hyperparameter search)
    print("Setting up logging...")
    from pytorch_lightning.loggers import Logger

    mlflow_logger = None
    if args.no_output:
        # No output mode: no loggers, no callbacks
        loggers: list[Logger] = []
        print("Skipping loggers (--no-output mode)")
    else:
        loggers = [TensorBoardLogger(save_dir=f"{output_dir}/{run_name}/logs", name="")]

        # Add MLflow logger if enabled and available
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

    # Setup error handler (skip if --no-output)
    error_handler = None
    if not args.no_output and ERROR_HANDLER_AVAILABLE:
        error_handler = setup_error_handler(
            output_dir=output_dir,
            run_name=run_name,
            mlflow_logger=mlflow_logger,
        )

    # Setup callbacks
    callbacks = []

    if not args.no_output:
        callbacks.extend(
            [
                ModelCheckpoint(
                    dirpath=f"{output_dir}/{run_name}/checkpoints",
                    monitor="val_loss",
                    mode="min",
                    save_top_k=3,  # Keep only the best checkpoint
                    save_last=True,
                    filename=f"{run_name}_{{epoch:03d}}_{{val_loss:.4f}}",
                ),
                LearningRateMonitor(logging_interval="step"),
            ]
        )

        # Early stopping callback (Lightning only - no manual logic)
        early_stopping_patience = training_opts.get("early_stopping", None)
        if early_stopping_patience and early_stopping_patience > 0:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    mode="min",
                    verbose=True,
                )
            )
            print(
                f"✓ Early stopping enabled (patience={early_stopping_patience} epochs)"
            )

        # Slice visualization callback (only if output enabled and num >= 1)
        num_slice_viz = training_opts.get("num_slice_viz", 4)
        if num_slice_viz >= 1:
            callbacks.append(
                GlacierVisualizationCallback(
                    num_samples=num_slice_viz,
                    log_every_n_epochs=training_opts.get(
                        "slice_viz_every_n_epochs", 10
                    ),
                    save_dir=f"{output_dir}/{run_name}/slice_visualizations",
                )
            )

    # Full-tile evaluation (runs even with --no-output for metrics, but no PNGs)
    run_full_eval = training_opts.get("run_full_eval", True) and not args.skip_full_eval
    if run_full_eval:
        num_full_viz = training_opts.get("num_full_viz", 4) if not args.no_output else 0
        callbacks.append(BestModelFullEvaluationCallback(num_samples=num_full_viz))

    if not callbacks:
        print("No callbacks enabled")

    # Create trainer
    print("Creating trainer...")

    # Handle GPU device selection
    if args.gpu is not None:
        # Explicit GPU specified
        devices = [args.gpu]
        print(f"Using explicit GPU: {args.gpu}")
    else:
        # Auto-detect (for Ray or single-GPU systems)
        devices = 1  # Use 1 GPU (Ray sets CUDA_VISIBLE_DEVICES)
        print("Using auto-detected GPU (Ray controlled)")

    # Set default_root_dir to ensure all Lightning outputs go to output/
    default_root = f"{output_dir}/{run_name}" if not args.no_output else None
    if default_root:
        print(f"Lightning default_root_dir: {default_root}")

    trainer = pl.Trainer(
        default_root_dir=default_root,
        accelerator="gpu",
        devices=devices,
        max_epochs=max_epochs,
        logger=loggers,  # Support multiple loggers
        callbacks=callbacks,
        precision="16-mixed",
        log_every_n_steps=10,
        val_check_interval=1.0,
        enable_progress_bar=True,
        num_sanity_val_steps=2,  # Quick sanity check
    )

    print(f"Starting training for {max_epochs} epochs...")
    print(f"GPU available: {torch.cuda.is_available()}")
    if args.no_output:
        print("No-output mode: skipping all disk writes (Ray hyperparameter search)")
    else:
        print(f"TensorBoard logs: {output_dir}/{run_name}/logs")
        print(f"Checkpoints: {output_dir}/{run_name}/checkpoints")

    try:
        trainer.fit(
            model, datamodule=datamodule, ckpt_path=args.resume if args.resume else None
        )

        print("Training completed successfully!")

        # Extract final validation loss for Ray Tune integration
        final_val_loss = float(trainer.callback_metrics.get("val_loss", 999.0))
        print(f"Final validation loss: {final_val_loss:.4f}")
        return final_val_loss

    except KeyboardInterrupt as e:
        print("Training interrupted by user")
        if error_handler:
            error_handler.log_error(
                Exception(f"Training interrupted by user: {e}"),
                {"message": "Training interrupted by user"},
            )
    except Exception as e:
        print(f"Training failed with error: {e}")
        if error_handler:
            error_handler.log_error(
                e,
                {
                    "epoch": trainer.current_epoch
                    if hasattr(trainer, "current_epoch")
                    else "unknown",
                    "global_step": trainer.global_step
                    if hasattr(trainer, "global_step")
                    else "unknown",
                    "config_file": str(config_path),
                },
            )
        raise


if __name__ == "__main__":
    main()
