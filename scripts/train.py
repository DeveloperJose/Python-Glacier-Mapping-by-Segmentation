#!/usr/bin/env python3
"""Simple Lightning training for glacier mapping."""

import argparse
import json
import os
import pathlib
import random
import socket
import warnings
from typing import Dict, Any
from types import SimpleNamespace
from urllib.parse import urlparse

import numpy as np
import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    BatchSizeFinder,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import (
    AdvancedProfiler,
    PyTorchProfiler,
    SimpleProfiler,
)
from glacier_mapping.lightning.glacier_module import GlacierSegmentationModule
from glacier_mapping.lightning.glacier_datamodule import GlacierDataModule
from glacier_mapping.lightning.callbacks import (
    GlacierModelCheckpoint,
    ValidationVisualizationCallback,
    TestEvaluationCallback,
)
import glacier_mapping.utils.mlflow_utils as mlflow_utils
from glacier_mapping.utils.error_handler import setup_error_handler
from glacier_mapping.utils.config import load_server_config, load_servers_config
import glacier_mapping.utils.logging as log

MLFLOW_AVAILABLE = True
ERROR_HANDLER_AVAILABLE = True

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*Detected call of `lr_scheduler.step()` before `optimizer.step()`.*",
)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    if override.get("_replace_", False):
        return {key: value for key, value in override.items() if key != "_replace_"}

    result = base.copy()
    for key, value in override.items():
        if key == "_replace_":
            continue
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str, server: str) -> Dict[str, Any]:
    config_path_obj = pathlib.Path(config_path)
    path_parts = config_path_obj.parts
    if len(path_parts) >= 3 and path_parts[0] == "configs":
        server_from_path = path_parts[1]
        task = path_parts[2]

        if server != server_from_path:
            raise ValueError(
                f"Server mismatch: CLI arg '{server}' != path '{server_from_path}'\n"
                f"Config path: {config_path}"
            )
    else:
        log.warning(f"Config path doesn't follow new structure: {config_path}")
        log.info("Loading as standalone config without hierarchy")
        with open(config_path) as f:
            return yaml.safe_load(f)

    base_config_path = pathlib.Path("configs/train.yaml")
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    with open(base_config_path) as f:
        merged = yaml.safe_load(f)

    servers_yaml_path = pathlib.Path("configs/servers.yaml")
    if servers_yaml_path.exists():
        servers = load_servers_config(str(servers_yaml_path))

        if server in servers:
            server_config = servers[server]
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

    task_config_path = pathlib.Path(f"configs/tasks/{task}.yaml")
    if task_config_path.exists():
        with open(task_config_path) as f:
            task_config = yaml.safe_load(f)
        merged = deep_merge(merged, task_config)

    with open(config_path) as f:
        experiment_config = yaml.safe_load(f)
    merged = deep_merge(merged, experiment_config)

    return merged


def seed_reproducibility(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=not deterministic)
    except TypeError:
        torch.use_deterministic_algorithms(deterministic)


def configure_torch_performance() -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


class TrainingLogUploadCallback(pl.Callback):
    def __init__(self, log_file_path: pathlib.Path, upload_best_model: bool = True):
        super().__init__()
        self.log_file_path = pathlib.Path(log_file_path)
        self.upload_best_model = upload_best_model

    def _get_mlflow_logger(self, trainer: pl.Trainer):
        loggers = []
        if hasattr(trainer, "loggers") and trainer.loggers:
            loggers.extend(trainer.loggers)
        elif getattr(trainer, "logger", None):
            loggers.append(trainer.logger)

        for logger in loggers:
            if hasattr(logger, "experiment") and hasattr(logger, "run_id"):
                return logger
        return None

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        for handler in log.LOGGER.handlers:
            if hasattr(handler, "flush"):
                handler.flush()

        if not mlflow_utils.MLFLOW_ARTIFACT_UPLOAD_ENABLED:
            log.info("MLflow artifact upload disabled; scalar metrics remain enabled")
            return

        if not self.log_file_path.exists():
            log.warning(
                f"Training log not found for MLflow upload: {self.log_file_path}"
            )
            return

        mlflow_logger = self._get_mlflow_logger(trainer)
        if mlflow_logger is None:
            log.warning("MLflow logger missing; skipping training log upload")
            return

        # Check artifact URI before uploading. The MLflow server may return a
        # local path that the client cannot write to (e.g. /mlflow).
        artifact_uri = mlflow_logger.experiment.get_run(
            mlflow_logger.run_id
        ).info.artifact_uri
        if artifact_uri.startswith("file://") or artifact_uri.startswith("/"):
            local_path = artifact_uri.replace("file://", "")
            if not os.access(local_path, os.W_OK):
                log.warning(
                    f"MLflow artifact URI ({artifact_uri}) not writable; "
                    "skipping training log upload"
                )
                return

        try:
            mlflow_logger.experiment.log_artifact(
                mlflow_logger.run_id,
                str(self.log_file_path),
                artifact_path="logs",
            )
            log.info(f"Uploaded training log to MLflow: {self.log_file_path}")
        except Exception as e:
            log.warning(f"Failed to upload training log to MLflow: {e}")

        if not self.upload_best_model:
            return

        best_checkpoint = self._get_best_checkpoint(trainer)
        if not best_checkpoint:
            log.warning("Best checkpoint path not found; skipping MLflow upload")
            return

    def _get_best_checkpoint(self, trainer: pl.Trainer) -> pathlib.Path | None:
        from pytorch_lightning.callbacks import ModelCheckpoint

        best_path: pathlib.Path | None = None
        for cb in getattr(trainer, "callbacks", []):
            if isinstance(cb, ModelCheckpoint):
                candidate = getattr(cb, "best_model_path", "")
                if candidate:
                    best_path = pathlib.Path(candidate)
                    break

        if best_path is None:
            checkpoint_cb = getattr(trainer, "checkpoint_callback", None)
            if (
                checkpoint_cb
                and isinstance(checkpoint_cb, ModelCheckpoint)
                and getattr(checkpoint_cb, "best_model_path", "")
            ):
                best_path = pathlib.Path(checkpoint_cb.best_model_path)

        if best_path and best_path.exists():
            return best_path
        return None


class _NoopMLflowClient:
    """Minimal MLflow client stand-in used when remote MLflow is unavailable."""

    def __init__(self, run_id: str):
        self._run_id = run_id

    def get_run(self, run_id: str):
        return SimpleNamespace(
            info=SimpleNamespace(
                run_id=run_id,
                experiment_id="mlflow_unavailable",
                artifact_uri="",
            )
        )

    def log_batch(self, *args, **kwargs) -> None:
        return None

    def log_artifact(self, *args, **kwargs) -> None:
        return None

    def log_artifacts(self, *args, **kwargs) -> None:
        return None

    def set_terminated(self, *args, **kwargs) -> None:
        return None


class ResilientMLFlowLogger:
    """MLflow logger wrapper that never lets tracking outages stop training."""

    def __init__(self, *args, **kwargs):
        from pytorch_lightning.loggers import MLFlowLogger

        self._logger = MLFlowLogger(*args, **kwargs)
        self._disabled = False
        self._disabled_reason: str | None = None
        self._noop_run_id = "mlflow_unavailable"
        self._noop_client = _NoopMLflowClient(self._noop_run_id)

    def _disable(self, error: Exception) -> None:
        if not self._disabled:
            self._disabled_reason = str(error)
            log.warning(
                f"MLflow logging disabled for this run after tracking failure: {error}"
            )
        self._disabled = True

    @property
    def experiment(self):
        if self._disabled:
            return self._noop_client
        try:
            return self._logger.experiment
        except Exception as e:
            self._disable(e)
            return self._noop_client

    @property
    def run_id(self) -> str:
        if self._disabled:
            return self._noop_run_id
        try:
            run_id = self._logger.run_id
            return run_id or self._noop_run_id
        except Exception as e:
            self._disable(e)
            return self._noop_run_id

    @property
    def name(self) -> str:
        return getattr(self._logger, "name", "mlflow")

    @property
    def version(self) -> str:
        return str(self.run_id)

    def __getattr__(self, name: str):
        try:
            target = getattr(self._logger, name)
        except Exception as e:
            self._disable(e)

            def noop(*args, **kwargs):
                return None

            return noop

        if not callable(target):
            return target

        def guarded(*args, **kwargs):
            if self._disabled:
                return None
            try:
                return target(*args, **kwargs)
            except Exception as e:
                self._disable(e)
                return None

        return guarded


def mlflow_tracking_uri_available(
    tracking_uri: str, timeout_seconds: float = 2.0
) -> bool:
    parsed = urlparse(tracking_uri)
    if parsed.scheme in {"", "file"}:
        return True
    if not parsed.hostname:
        log.warning(f"MLflow tracking URI has no hostname: {tracking_uri}")
        return False

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout_seconds)
    try:
        socket.getaddrinfo(parsed.hostname, port)
        return True
    except OSError as e:
        log.warning(
            "MLflow tracking URI unavailable; continuing with local logging only: "
            f"{tracking_uri} ({e})"
        )
        return False
    finally:
        socket.setdefaulttimeout(previous_timeout)


def build_lightning_profiler(profiling_opts: dict, profile_dir: pathlib.Path):
    if not profiling_opts.get("enabled", False):
        return None

    mode = str(profiling_opts.get("mode", "simple")).lower()
    filename = f"{mode}_profiler"
    profile_dir.mkdir(parents=True, exist_ok=True)

    if mode == "simple":
        return SimpleProfiler(
            dirpath=profile_dir,
            filename=filename,
            extended=bool(profiling_opts.get("extended", True)),
        )
    if mode == "advanced":
        return AdvancedProfiler(
            dirpath=profile_dir,
            filename=filename,
            line_count_restriction=float(
                profiling_opts.get("line_count_restriction", 1.0)
            ),
            dump_stats=bool(profiling_opts.get("dump_stats", False)),
        )
    if mode in {"pytorch", "torch"}:
        profiler_kwargs = {
            "profile_memory": bool(profiling_opts.get("profile_memory", False)),
            "record_shapes": bool(profiling_opts.get("record_shapes", False)),
            "with_stack": bool(profiling_opts.get("with_stack", False)),
            "with_flops": bool(profiling_opts.get("with_flops", False)),
        }
        if "schedule" in profiling_opts:
            schedule_opts = profiling_opts["schedule"]
            profiler_kwargs["schedule"] = torch.profiler.schedule(
                wait=int(schedule_opts.get("wait", 1)),
                warmup=int(schedule_opts.get("warmup", 1)),
                active=int(schedule_opts.get("active", 3)),
                repeat=int(schedule_opts.get("repeat", 1)),
                skip_first=int(schedule_opts.get("skip_first", 0)),
            )
        return PyTorchProfiler(
            dirpath=profile_dir,
            filename=filename,
            export_to_chrome=bool(profiling_opts.get("export_to_chrome", True)),
            row_limit=int(profiling_opts.get("row_limit", 20)),
            **profiler_kwargs,
        )

    raise ValueError(
        f"Unsupported profiling mode '{mode}'. Use simple, advanced, or pytorch."
    )


def main():
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

    parser.add_argument(
        "--server",
        type=str,
        required=True,
        help="Server name (must exist in servers.yaml)",
    )
    parser.add_argument(
        "--mlflow-enabled",
        type=str,
        choices=("true", "false"),
        default=None,
        help="Enable MLflow logging (default: training_opts.mlflow_enabled=false)",
    )
    parser.add_argument(
        "--mlflow-artifacts-enabled",
        type=str,
        default=None,
        help=(
            "Enable MLflow artifact uploads while keeping metrics enabled "
            "(true/false; defaults to training_opts.mlflow_artifacts_enabled)"
        ),
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=os.environ.get("MLFLOW_TRACKING_URI"),
        help="MLflow tracking URI (or set MLFLOW_TRACKING_URI)",
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

    config_path = pathlib.Path(args.config)

    config = load_config(str(config_path), args.server)

    servers_yaml_path = pathlib.Path("configs") / "servers.yaml"
    server_config = load_server_config(args.server, str(servers_yaml_path))

    training_opts = config.get("training_opts", {})
    loader_opts = config.get("loader_opts", {})
    model_opts = config.get("model_opts", {})
    loss_opts = config.get("loss_opts", {})
    optim_opts = config.get("optim_opts", {})
    scheduler_opts = config.get("scheduler_opts", {})
    metrics_opts = config.get("metrics_opts", {})
    profiling_opts = config.get("profiling_opts", training_opts.get("profiling", {}))
    batch_size_finder_opts = config.get(
        "batch_size_finder_opts", training_opts.get("batch_size_finder", {})
    )

    if args.mlflow_enabled is None:
        mlflow_enabled = bool(training_opts.get("mlflow_enabled", False))
    else:
        mlflow_enabled = args.mlflow_enabled == "true"
    if mlflow_enabled and not args.tracking_uri:
        raise ValueError(
            "MLflow was enabled but no tracking URI was provided. Use "
            "--tracking-uri or set MLFLOW_TRACKING_URI."
        )
    if args.mlflow_artifacts_enabled is not None:
        mlflow_artifacts_enabled = args.mlflow_artifacts_enabled.lower() == "true"
    else:
        mlflow_artifacts_enabled = bool(
            training_opts.get("mlflow_artifacts_enabled", False)
        )
    training_opts["mlflow_artifacts_enabled"] = mlflow_artifacts_enabled
    mlflow_utils.MLFLOW_ARTIFACT_UPLOAD_ENABLED = mlflow_artifacts_enabled

    seed = int(training_opts.get("seed", 42))
    augmentation_seed = training_opts.get("augmentation_seed", None)
    deterministic = bool(training_opts.get("deterministic", True))
    seed_reproducibility(seed, deterministic)
    configure_torch_performance()

    landsat_channels = loader_opts.get("landsat_channels", True)
    dem_channels = loader_opts.get("dem_channels", True)
    spectral_indices_channels = loader_opts.get("spectral_indices_channels", True)
    hsv_channels = loader_opts.get("hsv_channels", True)
    physics_channels = loader_opts.get("physics_channels", False)
    velocity_channels = loader_opts.get("velocity_channels", True)

    if args.max_epochs is not None:
        max_epochs = args.max_epochs
        log.info(f"Using max_epochs from CLI argument: {max_epochs}")
    else:
        max_epochs = training_opts.get("epochs", 100)
        log.info(f"Using max_epochs from config: {max_epochs}")

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
        log.info(
            f"Auto-constructed data path from server config: {loader_opts['processed_dir']}"
        )

    data_path = pathlib.Path(loader_opts["processed_dir"])
    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed data directory does not exist: {data_path}\n"
            f"Please run preprocessing first or check your server config."
        )

    norm_file = data_path / "normalize_train.npy"
    if not norm_file.exists():
        raise FileNotFoundError(
            f"Normalization file not found: {norm_file}\n"
            f"Please run preprocessing to generate normalization statistics."
        )

    base_run_name = training_opts.get("run_name", "experiment")
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

    if mlflow_enabled and MLFLOW_AVAILABLE:
        experiment_name = args.experiment_name or mlflow_utils.categorize_experiment(
            config
        )
        run_name = mlflow_utils.generate_run_name(base_run_name, args.server)
        mlflow_utils.extract_mlflow_params(config, server_config)
        mlflow_tags = mlflow_utils.generate_run_tags(
            config, server_config, str(config_path)
        )
    else:
        experiment_name = None
        run_name = base_run_name
        mlflow_tags = {}

    config_output_dir = pathlib.Path(output_dir) / run_name
    config_output_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = None
    if not args.no_output:
        log_file_path = config_output_dir / "training.log"
        log.configure_file_logging(str(log_file_path))

    log.info(f"Loaded config from: {config_path}")
    log.info(f"Server: {args.server}")
    log.info(f"Seed: {seed}")
    log.info(f"Deterministic mode: {deterministic}")
    log.info(f"MLflow enabled: {mlflow_enabled}")
    log.info(f"MLflow artifact upload enabled: {mlflow_artifacts_enabled}")
    if profiling_opts.get("enabled", False):
        log.info(f"Profiling enabled: {profiling_opts}")
    if mlflow_enabled and MLFLOW_AVAILABLE:
        log.info(f"MLflow experiment: {experiment_name}")
        log.info(f"MLflow run name: {run_name}")
    log.info(f"Base run name: {base_run_name}")
    log.info(f"Output directory: {output_dir} (source: {output_dir_source})")
    log.info(f"Data path: {loader_opts.get('processed_dir', 'NOT_SET')}")
    band_metadata_path = data_path / "band_metadata.json"
    if band_metadata_path.exists():
        with open(band_metadata_path, "r") as f:
            band_metadata = json.load(f)
        band_names = band_metadata.get("band_names", [])
        log.info(
            f"Packed dataset channels ({len(band_names)}): {band_names}. "
            "Training uses all stored channels."
        )
    log.info(f"Output classes: {loader_opts.get('output_classes', 'NOT_SET')}")

    if not args.no_output:
        config_json_path = config_output_dir / "conf.json"
        with open(config_json_path, "w") as f:
            json.dump(config, f, indent=2)
        log.info(f"Config saved to: {config_json_path}")
    else:
        log.info("Skipping config save (--no-output mode)")

    log.info("Creating data module...")
    datamodule = GlacierDataModule(
        processed_dir=loader_opts.get("processed_dir", "/tmp"),
        batch_size=loader_opts.get("batch_size", 8),
        landsat_channels=landsat_channels,
        dem_channels=dem_channels,
        spectral_indices_channels=spectral_indices_channels,
        hsv_channels=hsv_channels,
        physics_channels=physics_channels,
        velocity_channels=velocity_channels,
        augmentations=loader_opts.get("augmentations", None),
        output_classes=loader_opts.get("output_classes", [1]),
        class_names=loader_opts.get("class_names", ["BG", "CleanIce", "Debris"]),
        normalize=loader_opts.get("normalize", "mean-std"),
        robust_scaling=loader_opts.get("robust_scaling", True),
        num_workers=loader_opts.get("num_workers", 4),
        pin_memory=loader_opts.get("pin_memory", True),
        persistent_workers=loader_opts.get("persistent_workers", None),
        prefetch_factor=loader_opts.get("prefetch_factor", None),
        seed=int(loader_opts.get("seed", seed)),
        augmentation_seed=augmentation_seed,
        val_shuffle=bool(loader_opts.get("val_shuffle", False)),
        shared_loader_generator=bool(loader_opts.get("shared_loader_generator", False)),
    )

    log.info("Creating model...")
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

    log.info(f"optimizer = {optim_opts.get('name', 'AdamW')}")
    optim_args = optim_opts.get("args", {})
    log.info(
        f"optimizer args: lr={optim_args.get('lr')} "
        f"weight_decay={optim_args.get('weight_decay')} "
        f"fused={optim_args.get('fused', False)}"
    )

    log.info("Setting up logging...")
    from pytorch_lightning.loggers import Logger

    mlflow_logger = None
    if args.no_output:
        loggers: list[Logger] = []
        log.info("Skipping loggers (--no-output mode)")
    else:
        loggers = [TensorBoardLogger(save_dir=f"{output_dir}/{run_name}/logs", name="")]

        mlflow_ready = (
            mlflow_enabled
            and MLFLOW_AVAILABLE
            and experiment_name
            and mlflow_tracking_uri_available(args.tracking_uri)
        )
        if mlflow_ready:
            try:
                mlflow_logger = ResilientMLFlowLogger(
                    experiment_name=experiment_name,
                    run_name=run_name,
                    tracking_uri=args.tracking_uri,
                    tags=mlflow_tags,
                    log_model=False,
                )
                loggers.append(mlflow_logger)
                log.info(
                    f"MLflow logger setup complete for experiment: {experiment_name}"
                )
            except Exception as e:
                log.warning(f"Failed to setup MLflow logger: {e}")
                mlflow_logger = None
        elif mlflow_enabled and MLFLOW_AVAILABLE and experiment_name:
            log.warning(
                "MLflow logger not added; training will continue with TensorBoard "
                "and local checkpoints/test metrics"
            )

    error_handler = None
    if not args.no_output and ERROR_HANDLER_AVAILABLE:
        error_handler = setup_error_handler(
            output_dir=output_dir,
            run_name=run_name,
            mlflow_logger=mlflow_logger,
        )

    callbacks = []
    viz_scale_factor = 1

    if not args.no_output:
        callbacks.append(
            GlacierModelCheckpoint(
                dirpath=f"{output_dir}/{run_name}/checkpoints",
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                save_last=True,
                filename=f"{run_name}_{{epoch:03d}}_{{val_loss:.4f}}",
                start_epoch=training_opts.get("checkpoint_monitor_start_epoch", 0),
            )
        )

        if training_opts.get("lr_monitor", True):
            callbacks.append(
                LearningRateMonitor(
                    logging_interval=training_opts.get(
                        "lr_monitor_logging_interval", "step"
                    )
                )
            )
        else:
            log.info("LearningRateMonitor disabled by config")

        early_stopping_patience = training_opts.get("early_stopping", None)
        if early_stopping_patience and early_stopping_patience > 0:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    mode="min",
                    min_delta=training_opts.get("early_stopping_min_delta", 0.0),
                    verbose=True,
                )
            )
            log.info(
                f"Early stopping enabled (patience={early_stopping_patience} epochs)"
            )

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

        if mlflow_logger and log_file_path:
            callbacks.append(
                TrainingLogUploadCallback(log_file_path, upload_best_model=False)
            )

    run_test_eval = (
        training_opts.get("run_test_eval", True)
        and not args.skip_test_eval
        and not args.no_output
    )
    if run_test_eval:
        test_eval_n = training_opts.get("test_eval_n", 4) if not args.no_output else 0
        callbacks.append(
            TestEvaluationCallback(
                viz_n=test_eval_n,
                image_dir=server_config.get("image_dir"),
                scale_factor=viz_scale_factor,
            )
        )

    if batch_size_finder_opts.get("enabled", False):
        callbacks.append(
            BatchSizeFinder(
                mode=batch_size_finder_opts.get("mode", "binsearch"),
                steps_per_trial=batch_size_finder_opts.get("steps_per_trial", 3),
                init_val=batch_size_finder_opts.get(
                    "init_val", loader_opts.get("batch_size", 8)
                ),
                max_trials=batch_size_finder_opts.get("max_trials", 25),
                batch_arg_name=batch_size_finder_opts.get(
                    "batch_arg_name", "batch_size"
                ),
                margin=batch_size_finder_opts.get("margin", 0.05),
                max_val=batch_size_finder_opts.get("max_val", 8192),
            )
        )
        log.info(f"BatchSizeFinder enabled: {batch_size_finder_opts}")

    if not callbacks:
        log.info("No callbacks enabled")

    log.info("Creating trainer...")

    if args.gpu is not None:
        devices = [args.gpu]
        log.info(f"Using explicit GPU: {args.gpu}")
    else:
        devices = 1
        log.info("Using auto-detected GPU (Ray controlled)")

    default_root = f"{output_dir}/{run_name}" if not args.no_output else None
    if default_root:
        log.info(f"Lightning default_root_dir: {default_root}")

    profiler = None
    if profiling_opts.get("enabled", False) and not args.no_output:
        profile_dir = config_output_dir / "profiling"
        profiler = build_lightning_profiler(profiling_opts, profile_dir)
        log.info(f"Lightning profiler enabled: {type(profiler).__name__}")

    trainer = pl.Trainer(
        default_root_dir=default_root,
        accelerator="gpu",
        devices=devices,
        max_epochs=max_epochs,
        logger=loggers,
        callbacks=callbacks,
        precision=training_opts.get("precision", "16-mixed"),
        accumulate_grad_batches=int(training_opts.get("accumulate_grad_batches", 1)),
        log_every_n_steps=training_opts.get("log_every_n_steps", 10),
        val_check_interval=1.0,
        check_val_every_n_epoch=training_opts.get("check_val_every_n_epoch", 1),
        enable_progress_bar=training_opts.get("progress_bar", False),
        num_sanity_val_steps=training_opts.get("num_sanity_val_steps", 2),
        deterministic=deterministic,
        profiler=profiler,
    )

    log.info(f"Starting training for {max_epochs} epochs...")
    log.info(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if args.no_output:
        log.info("No-output mode: skipping all disk writes (Ray hyperparameter search)")
    else:
        log.info(f"TensorBoard logs: {output_dir}/{run_name}/logs")
        log.info(f"Checkpoints: {output_dir}/{run_name}/checkpoints")

    try:
        trainer.fit(
            model, datamodule=datamodule, ckpt_path=args.resume if args.resume else None
        )

        log.info("Training completed successfully!")
        if torch.cuda.is_available():
            max_allocated_gb = torch.cuda.max_memory_allocated() / 1024**3
            max_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
            current_allocated_gb = torch.cuda.memory_allocated() / 1024**3
            current_reserved_gb = torch.cuda.memory_reserved() / 1024**3
            log.info(
                "CUDA memory GB: "
                f"max_allocated={max_allocated_gb:.3f}, "
                f"max_reserved={max_reserved_gb:.3f}, "
                f"current_allocated={current_allocated_gb:.3f}, "
                f"current_reserved={current_reserved_gb:.3f}"
            )

        final_val_loss = float(trainer.callback_metrics.get("val_loss", 999.0))
        log.info(f"Final validation loss: {final_val_loss:.4f}")

        return final_val_loss

    except KeyboardInterrupt as e:
        log.info("Training interrupted by user")
        if error_handler:
            error_handler.log_error(
                Exception(f"Training interrupted by user: {e}"),
                {"message": "Training interrupted by user"},
            )
    except Exception as e:
        log.error(f"Training failed with error: {e}")
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
