#!/usr/bin/env python3
"""
Monitor experiment status across all servers.

Usage:
  uv run python -m glacier_mapping.distributed.monitor
  uv run python -m glacier_mapping.distributed.monitor --detailed
  uv run python -m glacier_mapping.distributed.monitor --server bilbo
"""

import argparse
from pathlib import Path

import yaml


def get_experiment_status(exp_id: str, server_name: str, gpu_rank: int) -> str:
    """
    Determine experiment status from filesystem.

    Returns: 'pending' | 'running' | 'completed' | 'failed'
    """
    results_dir = Path("output/runs") / f"{exp_id}_{server_name}_gpu{gpu_rank}"

    if not results_dir.exists():
        return "pending"

    if (results_dir / "FAILED").exists():
        return "failed"

    if (results_dir / "checkpoints_summary.json").exists():
        return "completed"

    return "running"


def monitor_experiments(
    detailed: bool = False, server_filter: str | None = None
) -> None:
    """Display experiment status."""
    experiments_dir = Path("experiments")
    conf_dir = experiments_dir / "conf"

    # Find all experiment YAML files
    exp_files = sorted(conf_dir.glob("exp_*.yaml"))

    if not exp_files:
        print("No experiments found in experiments/conf/")
        print("\nTo create an experiment:")
        print("  uv run python experiments/submit.py --server bilbo --gpu 0")
        return

    print("\n" + "=" * 80)
    print("EXPERIMENT STATUS")
    print("=" * 80 + "\n")

    # Group by status
    status_groups: dict[str, list[tuple[str, str, dict, str]]] = {
        "pending": [],
        "running": [],
        "completed": [],
        "failed": [],
    }

    for exp_file in exp_files:
        exp_id = exp_file.stem
        exp_config = yaml.safe_load(exp_file.read_text())

        server = exp_config.get("server", "unknown")
        gpu_rank = exp_config.get("gpu_rank", 0)

        # Apply server filter if specified
        if server_filter and server != server_filter:
            continue

        status = get_experiment_status(exp_id, server, gpu_rank)
        status_groups[status].append((exp_id, server, exp_config, exp_file.name))

    # Display each status group
    for status_name, exps in status_groups.items():
        if not exps:
            continue

        print(f"\n{status_name.upper()} ({len(exps)}):")
        print("-" * 80)

        for exp_id, server, exp_config, filename in exps:
            gpu = exp_config.get("gpu_rank", "?")
            run_name = exp_config.get("training_opts", {}).get("run_name", "?")

            line = f"  {exp_id:<12} {server:<10} GPU:{gpu}  {run_name}"

            if detailed:
                line += f"\n    File: experiments/conf/{filename}"
                processed_dir = exp_config.get("loader_opts", {}).get(
                    "processed_dir", "?"
                )
                line += f"\n    Dataset: {processed_dir}"
                epochs = exp_config.get("training_opts", {}).get("epochs", "?")
                line += f"\n    Epochs: {epochs}"

                # Show results path if exists
                gpu = exp_config.get("gpu_rank", 0)
                results_path = Path("output/runs") / f"{exp_id}_{server}_gpu{gpu}"
                if results_path.exists():
                    line += f"\n    Results: {results_path}"

            print(line)

    # Summary
    total = sum(len(exps) for exps in status_groups.values())
    print("\n" + "=" * 80)
    print(f"Total experiments: {total}")
    if server_filter:
        print(f"Filtered by server: {server_filter}")

    # Show status counts
    counts = {status: len(exps) for status, exps in status_groups.items() if exps}
    if counts:
        print(
            f"Status: {', '.join(f'{status}={count}' for status, count in counts.items())}"
        )

    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor distributed experiments")
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed information",
    )
    parser.add_argument(
        "--server",
        choices=["desktop", "bilbo"],
        help="Filter by server",
    )
    args = parser.parse_args()

    monitor_experiments(args.detailed, args.server)
