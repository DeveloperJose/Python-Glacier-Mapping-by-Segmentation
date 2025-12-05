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
from glacier_mapping.lightning.callbacks import ValidationVisualizationCallback
from glacier_mapping.lightning.best_model_callback import TestEvaluationCallback

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


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, override takes precedence.

    Args:
        base: Base dictionary
        override: Override dictionary (takes precedence)

    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str, server: str) -> Dict[str, Any]:
    """
    Load YAML configuration with 4-level hierarchy:
    1. train.yaml (global defaults)
    2. servers.yaml[server] -> loader_opts/training_opts
    3. tasks/{task}.yaml (inferred from path)
    4. {experiment_path} (specific experiment)

    Args:
        config_path: Path to experiment config (e.g., "configs/frodo/clean_ice/base.yaml")
        server: Server name (e.g., "frodo")

    Returns:
        Merged configuration dictionary
    """
    config_path_obj = pathlib.Path(config_path)

    # Parse task from path structure
    path_parts = config_path_obj.parts
    if len(path_parts) >= 3 and path_parts[0] == "configs":
        server_from_path = path_parts[1]  # e.g., "frodo"
        task = path_parts[2]  # e.g., "clean_ice"

        # Validate server consistency
        if server != server_from_path:
            raise ValueError(
                f"Server mismatch: CLI arg '{server}' != path '{server_from_path}'\n"
                f"Config path: {config_path}"
            )
    else:
        # Fallback for configs not following new structure
        print(f"Warning: Config path doesn't follow new structure: {config_path}")
        print("Loading as standalone config without hierarchy")
        with open(config_path) as f:
            return yaml.safe_load(f)

    # 1. Load train.yaml (global base config)
    base_config_path = pathlib.Path("configs/train.yaml")
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    with open(base_config_path) as f:
        merged = yaml.safe_load(f)

    # 2. Load server-specific training settings from servers.yaml
    servers_yaml_path = pathlib.Path("configs/servers.yaml")
    if servers_yaml_path.exists():
        with open(servers_yaml_path) as f:
            servers = yaml.safe_load(f)

        if server in servers:
            server_config = servers[server]
            # Extract training-relevant fields (batch_size, epochs, num_workers)
            if "batch_size" in server_config:
                if "loader_opts" not in merged:
                    merged["loader_opts"] = {}
                merged["loader_opts"]["batch_size"] = server_config["batch_size"]
            if "epochs" in server_config:
                if "training_opts" not in merged:
                    merged["training_opts"] = {}
                merged["training_opts"]["epochs"] = server_config["epochs"]
            if "num_workers" in server_config:
                if "loader_opts" not in merged:
                    merged["loader_opts"] = {}
                merged["loader_opts"]["num_workers"] = server_config["num_workers"]

    # 3. Load task config
    task_config_path = pathlib.Path(f"configs/tasks/{task}.yaml")
    if task_config_path.exists():
        with open(task_config_path) as f:
            task_config = yaml.safe_load(f)
        merged = deep_merge(merged, task_config)

    # 4. Load experiment-specific config
    with open(config_path) as f:
        experiment_config = yaml.safe_load(f)
    merged = deep_merge(merged, experiment_config)

    return merged


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
        "--skip-test-eval",
        action="store_true",
        default=False,
        help="Skip test set evaluation (overrides config)",
    )

    args = parser.parse_args()

    # Load configuration with hierarchical merging
    config_path = pathlib.Path(args.config)

    config = load_config(str(config_path), args.server)

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

    # Extract channel group selections from loader_opts
    landsat_channels = loader_opts.get("landsat_channels", True)
    dem_channels = loader_opts.get("dem_channels", True)
    spectral_indices_channels = loader_opts.get("spectral_indices_channels", True)
    hsv_channels = loader_opts.get("hsv_channels", True)
    physics_channels = loader_opts.get("physics_channels", False)
    velocity_channels = loader_opts.get("velocity_channels", True)

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
        mlflow_utils.extract_mlflow_params(config, server_config)  # type: ignore[assignment]
        mlflow_tags = mlflow_utils.generate_run_tags(
            config, server_config, str(config_path)
        )  # type: ignore[assignment]
    else:
        experiment_name = None
        run_name = base_run_name
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
    print("\nChannel Selection:")
    print(f"  Landsat: {landsat_channels}")
    print(f"  DEM: {dem_channels}")
    print(f"  Spectral Indices: {spectral_indices_channels}")
    print(f"  HSV: {hsv_channels}")
    print(f"  Physics: {physics_channels}")
    print(f"  Velocity: {velocity_channels}")
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
        landsat_channels=landsat_channels,
        dem_channels=dem_channels,
        spectral_indices_channels=spectral_indices_channels,
        hsv_channels=hsv_channels,
        physics_channels=physics_channels,
        velocity_channels=velocity_channels,
        output_classes=loader_opts.get("output_classes", [1]),
        class_names=loader_opts.get("class_names", ["BG", "CleanIce", "Debris"]),
        normalize=loader_opts.get("normalize", "mean-std"),
        num_workers=loader_opts.get("num_workers", 4),
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
        landsat_channels=landsat_channels,
        dem_channels=dem_channels,
        spectral_indices_channels=spectral_indices_channels,
        hsv_channels=hsv_channels,
        physics_channels=physics_channels,
        velocity_channels=velocity_channels,
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

        # Validation visualization callback (only if output enabled and viz_n >= 1)
        val_viz_n = training_opts.get("val_viz_n", 4)
        viz_scale_factor = training_opts.get("viz_scale_factor", 1)
        if val_viz_n >= 1:
            callbacks.append(
                ValidationVisualizationCallback(
                    viz_n=val_viz_n,
                    log_every_n_epochs=training_opts.get("val_viz_every_n_epochs", 10),
                    selection=training_opts.get("val_viz_selection", "iou"),
                    save_dir=f"{output_dir}/{run_name}",
                    image_dir=server_config.get("image_dir"),
                    scale_factor=viz_scale_factor,
                )
            )

    # Test evaluation (runs even with --no-output for metrics, but no PNGs)
    run_test_eval = training_opts.get("run_test_eval", True) and not args.skip_test_eval
    if run_test_eval:
        test_eval_n = training_opts.get("test_eval_n", 4) if not args.no_output else 0
        callbacks.append(
            TestEvaluationCallback(
                viz_n=test_eval_n,
                image_dir=server_config.get("image_dir"),
                scale_factor=viz_scale_factor,
            )
        )

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
