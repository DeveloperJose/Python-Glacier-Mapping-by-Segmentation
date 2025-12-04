#!/usr/bin/env python3
"""Upload completed training runs to MLflow."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

try:
    from glacier_mapping.utils.mlflow_utils import MLflowManager

    MLFLOW_AVAILABLE = True
except ImportError as e:
    print(f"Error: MLflow utilities not available: {e}")
    MLFLOW_AVAILABLE = False

try:
    from tensorboard.backend.event_processing import event_file_loader

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def parse_tensorboard_logs(log_dir: Path) -> Dict:
    """Parse TensorBoard logs to extract metrics."""
    if not TENSORBOARD_AVAILABLE:
        print("Warning: TensorBoard not available, cannot parse logs")
        return {}

    metrics = {}
    try:
        # Find all event files
        event_files = list(log_dir.rglob("events.out.tfevents.*"))
        if not event_files:
            print(f"Warning: No TensorBoard event files found in {log_dir}")
            return metrics

        # Load events from the most recent file
        event_file = max(event_files, key=os.path.getctime)

        for event in event_file_loader.EventFileLoader(event_file).Load():
            for value in event.step:
                if value.HasField("tag") and value.HasField("simple_value"):
                    tag = value.tag
                    simple_value = value.simple_value
                    if tag not in metrics or simple_value < metrics[tag]:
                        metrics[tag] = simple_value
                    elif tag in ["epoch", "val_loss", "train_loss_epoch"]:
                        # Keep the latest for these important metrics
                        metrics[tag] = simple_value

        return metrics

    except Exception as e:
        print(f"Warning: Failed to parse TensorBoard logs: {e}")
        return {}


def load_run_config(run_dir: Path) -> Optional[Dict]:
    """Load configuration from saved run."""
    config_path = run_dir / "conf.json"
    if not config_path.exists():
        print(f"Warning: No config found at {config_path}")
        return None

    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load config from {config_path}: {e}")
        return None


def detect_experiment_type(config: Dict) -> str:
    """Detect experiment type from config using MLflowManager."""
    if not MLFLOW_AVAILABLE:
        return "unknown"

    return MLflowManager.categorize_experiment(config)


def upload_single_run(run_dir: Path, tracking_uri: str, force: bool = False) -> bool:
    """Upload a single run to MLflow."""
    print(f"Processing run: {run_dir.name}")

    # Load configuration
    config = load_run_config(run_dir)
    if not config:
        print(f"Skipping {run_dir.name}: No configuration found")
        return False

    # Parse TensorBoard logs
    log_dir = run_dir / "logs"
    if not log_dir.exists():
        print(f"Skipping {run_dir.name}: No logs directory found")
        return False

    # Find the actual log subdirectory
    log_subdirs = [d for d in log_dir.iterdir() if d.is_dir()]
    if not log_subdirs:
        print(f"Skipping {run_dir.name}: No log subdirectories found")
        return False

    # Use the most recent log directory
    log_dir = max(log_subdirs, key=os.path.getctime)

    metrics = parse_tensorboard_logs(log_dir)
    if not metrics:
        print(f"Skipping {run_dir.name}: No metrics found")
        return False

    # Detect experiment type
    experiment_name = detect_experiment_type(config)
    run_name = run_dir.name

    try:
        # Setup MLflow
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        # Check if run already exists
        existing_runs = mlflow.search_runs(filter_string=f"run_name = '{run_name}'")
        if existing_runs and not force:
            print(
                f"Skipping {run_dir.name}: Run already exists in MLflow (use --force to override)"
            )
            return False

        # Start run
        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            params = (
                MLflowManager.extract_mlflow_params(config, {})
                if MLFLOW_AVAILABLE
                else {}
            )
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log checkpoints as artifacts
            checkpoint_dir = run_dir / "checkpoints"
            if checkpoint_dir.exists():
                mlflow.log_artifacts(str(checkpoint_dir), artifact_path="checkpoints")

            # Log config as artifact
            config_path = run_dir / "conf.json"
            if config_path.exists():
                mlflow.log_artifact(str(config_path), artifact_path="config")

        print(f"✅ Successfully uploaded {run_dir.name} to MLflow")
        print(f"   Experiment: {experiment_name}")
        print(f"   Metrics: {len(metrics)} logged")
        return True

    except Exception as e:
        print(f"❌ Failed to upload {run_dir.name}: {e}")
        return False


def upload_batch_runs(
    output_dir: Path,
    tracking_uri: str,
    experiment_type: Optional[str] = None,
    force: bool = False,
) -> int:
    """Upload multiple runs to MLflow."""
    print(f"Scanning for runs in: {output_dir}")

    # Find all run directories
    run_dirs = [
        d for d in output_dir.iterdir() if d.is_dir() and (d / "conf.json").exists()
    ]

    if not run_dirs:
        print("No runs found with configuration files")
        return 0

    # Filter by experiment type if specified
    if experiment_type:
        filtered_runs = []
        for run_dir in run_dirs:
            config = load_run_config(run_dir)
            if config and detect_experiment_type(config) == experiment_type:
                filtered_runs.append(run_dir)
        run_dirs = filtered_runs
        print(f"Found {len(run_dirs)} runs of type '{experiment_type}'")
    else:
        print(f"Found {len(run_dirs)} total runs")

    # Upload each run
    successful_uploads = 0
    for run_dir in sorted(run_dirs):
        if upload_single_run(run_dir, tracking_uri, force):
            successful_uploads += 1

    print(
        f"\n📊 Upload Summary: {successful_uploads}/{len(run_dirs)} runs uploaded successfully"
    )
    return successful_uploads


def main():
    """Main upload function."""
    parser = argparse.ArgumentParser(description="Upload completed runs to MLflow")
    parser.add_argument("run_dir", nargs="?", help="Single run directory to upload")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory containing runs",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="https://mlflow.developerjose.duckdns.org/",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--experiment-type",
        type=str,
        help="Filter by experiment type (e.g., baseline_ci, baseline_debris)",
    )
    parser.add_argument("--force", action="store_true", help="Override existing runs")
    parser.add_argument(
        "--batch", action="store_true", help="Upload all runs from output directory"
    )

    args = parser.parse_args()

    if not MLFLOW_AVAILABLE:
        print("Error: MLflow utilities not available. Cannot proceed.")
        return 1

    if args.batch:
        # Batch upload mode
        output_dir = Path(args.output_dir)
        if not output_dir.exists():
            print(f"Error: Output directory {output_dir} does not exist")
            return 1

        successful_uploads = upload_batch_runs(
            output_dir, args.tracking_uri, args.experiment_type, args.force
        )
        return 0 if successful_uploads > 0 else 1

    elif args.run_dir:
        # Single run upload mode
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"Error: Run directory {run_dir} does not exist")
            return 1

        success = upload_single_run(run_dir, args.tracking_uri, args.force)
        return 0 if success else 1

    else:
        print("Error: Must specify either a run directory or use --batch")
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
