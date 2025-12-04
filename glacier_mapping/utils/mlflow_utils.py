"""MLflow utilities for glacier mapping experiment tracking.

This module provides functions for MLflow experiment tracking, including:
- Experiment categorization
- Parameter extraction and logging
- Tag generation
- Connection management with retry logic

All functions are module-level (converted from MLflowManager static methods).
"""

import warnings
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional

import mlflow


def load_server_config(servers_yaml_path: str, server_name: str) -> dict:
    """Load explicit server configuration - HARD ERROR if missing."""
    if not Path(servers_yaml_path).exists():
        raise FileNotFoundError(
            f"Server configuration file not found: {servers_yaml_path}"
        )
    with open(servers_yaml_path, "r") as f:
        servers = yaml.safe_load(f)

    if server_name not in servers:
        raise ValueError(
            f"Server '{server_name}' not found in {servers_yaml_path}. "
            f"Available servers: {list(servers.keys())}"
        )

    return servers[server_name]


def categorize_experiment(config: dict) -> str:
    """Categorize experiment based on config content, not run_name.

    Returns:
        str: One of - baseline_ci, baseline_debris, baseline_multiclass,
             physics_ci, physics_debris, architecture_study, dataset_study,
             glacier_mapping_general
    """
    # Extract key configuration sections
    loader_opts = config.get("loader_opts", {})
    output_classes = loader_opts.get("output_classes", [])

    # Physics experiments: use channels beyond index 15 (physics channels start at 16)
    # use_channels = loader_opts.get("use_channels", [])
    # if any(ch >= 16 for ch in use_channels):
    #     if output_classes == [1]:
    #         return "physics_ci"
    #     elif output_classes == [2]:
    #         return "physics_debris"

    # Baseline experiments
    if output_classes == [1]:
        return "clean_ice"
    elif output_classes == [2]:
        return "debris_ice"
    elif output_classes == [0, 1, 2]:
        return "multi_class"

    # Architecture variations (check model parameters)
    # model_opts = config.get("model_opts", {})
    # model_args = model_opts.get("args", {})
    # training_opts = config.get("training_opts", {})
    # dataset_name = training_opts.get("dataset_name", "")
    # loss_opts = config.get("loss_opts", {})
    # net_depth = model_args.get("net_depth", 4)
    # first_channel_output = model_args.get("first_channel_output", 32)
    # label_smoothing = loss_opts.get("label_smoothing", 0)
    # if net_depth != 4 or first_channel_output != 32 or label_smoothing != 0:
    #     return "architecture_study"
    #
    # # Data configuration variations (check dataset name patterns)
    # if any(
    #     pattern in dataset_name
    #     for pattern in ["w256", "w1024", "o32", "o128", "f20", "f15"]
    # ):
    #     return "dataset_study"

    return "glacier_mapping_general"  # fallback


def extract_mlflow_params(config: dict, server_config: dict) -> dict:
    """Extract all relevant parameters for MLflow logging."""
    params = {}

    # Basic training parameters
    training_opts = config.get("training_opts", {})
    params.update(
        {
            "dataset_name": training_opts.get("dataset_name"),
            "epochs": training_opts.get("epochs"),
            "early_stopping": training_opts.get("early_stopping"),
            "full_eval_every": training_opts.get("full_eval_every"),
            "num_viz_samples": training_opts.get("num_viz_samples"),
        }
    )

    # Model architecture parameters
    model_args = config.get("model_opts", {}).get("args", {})
    params.update(
        {
            "net_depth": model_args.get("net_depth"),
            "first_channel_output": model_args.get("first_channel_output"),
            "dropout": model_args.get("dropout"),
            "spatial": model_args.get("spatial"),
        }
    )

    # Data configuration parameters
    loader_opts = config.get("loader_opts", {})
    params.update(
        {
            "batch_size": loader_opts.get("batch_size"),
            "use_channels_count": len(loader_opts.get("use_channels", [])),
            "output_classes": str(loader_opts.get("output_classes")),
            "normalize": loader_opts.get("normalize"),
            "class_names": ",".join(loader_opts.get("class_names", [])),
        }
    )

    # Loss configuration
    loss_opts = config.get("loss_opts", {})
    params.update(
        {
            "loss_name": loss_opts.get("name"),
            "label_smoothing": loss_opts.get("label_smoothing"),
            "masked": loss_opts.get("masked"),
            "gaussian_blur_sigma": loss_opts.get("gaussian_blur_sigma"),
            "use_unified": loss_opts.get("use_unified"),
        }
    )

    # Optimizer configuration
    optim_opts = config.get("optim_opts", {})
    optim_args = optim_opts.get("args", {})
    params.update(
        {
            "optimizer_name": optim_opts.get("name"),
            "learning_rate": optim_args.get("lr"),
            "weight_decay": optim_args.get("weight_decay"),
        }
    )

    # Scheduler configuration
    scheduler_opts = config.get("scheduler_opts", {})
    scheduler_args = scheduler_opts.get("args", {})
    params.update(
        {
            "scheduler_name": scheduler_opts.get("name"),
            "max_lr": scheduler_args.get("max_lr"),
            "pct_start": scheduler_args.get("pct_start"),
            "anneal_strategy": scheduler_args.get("anneal_strategy"),
        }
    )

    # Server environment
    params.update(
        {
            "server": server_config.get("hostname"),
            "code_path": server_config.get("code_path"),
            "output_path": server_config.get("output_path"),
        }
    )

    # Filter out None values
    return {k: v for k, v in params.items() if v is not None}


def generate_run_tags(config: dict, server_config: dict, config_file: str) -> dict:
    """Generate comprehensive tags for MLflow runs."""
    training_opts = config.get("training_opts", {})
    loader_opts = config.get("loader_opts", {})
    model_opts = config.get("model_opts", {})
    loss_opts = config.get("loss_opts", {})

    # Extract key characteristics for tagging
    dataset_name = training_opts.get("dataset_name", "")
    output_classes = loader_opts.get("output_classes", [])
    model_args = model_opts.get("args", {})

    tags = {
        "experiment_type": "physics"
        if any(ch >= 16 for ch in loader_opts.get("use_channels", []))
        else "baseline",
        "target_class": _get_target_class_name(output_classes),
        "dataset_name": dataset_name,
        "server": server_config.get("hostname"),
        "config_file": Path(config_file).name,
        "model_depth": str(model_args.get("net_depth", 4)),
        "model_width": str(model_args.get("first_channel_output", 32)),
        "label_smoothing": str(loss_opts.get("label_smoothing", 0)),
        "batch_size": str(loader_opts.get("batch_size", 8)),
    }

    # Add dataset-specific tags
    if "w256" in dataset_name:
        tags["window_size"] = "256"
    elif "w512" in dataset_name:
        tags["window_size"] = "512"
    elif "w1024" in dataset_name:
        tags["window_size"] = "1024"

    if "o32" in dataset_name:
        tags["overlap"] = "32"
    elif "o64" in dataset_name:
        tags["overlap"] = "64"
    elif "o128" in dataset_name:
        tags["overlap"] = "128"

    if "f1" in dataset_name:
        tags["frequency"] = "1"
    elif "f2" in dataset_name:
        tags["frequency"] = "2"
    elif "f15" in dataset_name:
        tags["frequency"] = "15"
    elif "f20" in dataset_name:
        tags["frequency"] = "20"

    # Add physics channel info
    use_channels = loader_opts.get("use_channels", [])
    if any(ch >= 16 for ch in use_channels):
        tags["physics_channels"] = "enhanced"
    else:
        tags["physics_channels"] = "standard"

    return tags


def _get_target_class_name(output_classes: list) -> str:
    """Convert output classes to readable name."""
    if output_classes == [1]:
        return "ci"
    elif output_classes == [2]:
        return "debris"
    elif output_classes == [0, 1, 2]:
        return "multiclass"
    else:
        return "unknown"


def generate_run_name(base_run_name: str, server_name: str) -> str:
    """Generate run name with server and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_run_name}_{server_name}_{timestamp}"


def setup_mlflow_logger(
    tracking_uri: str, experiment_name: str, run_name: str, tags: dict
):
    """Setup MLflow logger with connection testing."""
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        # Test connection
        mlflow.get_experiment_by_name(experiment_name)

        # Start run
        mlflow.start_run(run_name=run_name, tags=tags)

        return True

    except Exception as e:
        warnings.warn(f"Failed to setup MLflow logger: {e}")
        return False


def attempt_connection_with_retry(
    tracking_uri: str,
    experiment_name: str,
    current_epoch: int,
    last_attempt_epoch: int = -1,
    failed_attempts: int = 0,
    max_retries: int = 10,
) -> tuple[bool, int, int]:
    """Handle connection logic with retry limit.

    Returns:
        tuple: (success, last_attempt_epoch, failed_attempts)
    """
    if failed_attempts >= max_retries:
        return False, last_attempt_epoch, failed_attempts

    # Only retry if we haven't tried this epoch
    if current_epoch <= last_attempt_epoch:
        return False, last_attempt_epoch, failed_attempts

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        # Test connection
        mlflow.get_experiment_by_name(experiment_name)

        return True, current_epoch, 0  # Reset failed_attempts on success

    except Exception as e:
        warnings.warn(
            f"MLflow connection failed (attempt {failed_attempts + 1}/{max_retries}): {e}"
        )
        return False, current_epoch, failed_attempts + 1


def log_params_safe(params: dict, logger_name: str = "MLflow"):
    """Log parameters with error handling."""
    try:
        mlflow.log_params(params)
    except Exception as e:
        warnings.warn(f"Failed to log parameters to {logger_name}: {e}")


def log_metrics_safe(
    metrics: dict, step: Optional[int] = None, logger_name: str = "MLflow"
):
    """Log metrics with error handling."""
    try:
        mlflow.log_metrics(metrics, step=step)
    except Exception as e:
        warnings.warn(f"Failed to log metrics to {logger_name}: {e}")


def log_artifact_safe(
    local_path: str,
    artifact_path: Optional[str] = None,
    logger_name: str = "MLflow",
):
    """Log artifact with error handling."""
    try:
        mlflow.log_artifact(local_path, artifact_path=artifact_path)
    except Exception as e:
        warnings.warn(f"Failed to log artifact to {logger_name}: {e}")
