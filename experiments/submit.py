#!/usr/bin/env python3
"""
Create a new experiment config with server metadata.

Usage:
  uv run python experiments/submit.py --server bilbo --gpu 0 --conf my_experiment.yaml
  uv run python experiments/submit.py --server desktop --gpu 0
"""

import argparse
from pathlib import Path

import yaml


def submit_experiment(server: str, gpu_rank: int, conf_name: str | None = None) -> None:
    """Create an experiment config with server metadata."""
    repo_root = Path(__file__).parent.parent
    servers_path = repo_root / "conf" / "servers.yaml"
    experiments_dir = repo_root / "experiments"

    # Validate server exists
    servers = yaml.safe_load(servers_path.read_text())
    if server not in servers:
        raise ValueError(
            f"Server '{server}' not found in conf/servers.yaml. "
            f"Available: {list(servers.keys())}"
        )

    # Generate experiment ID
    conf_dir = experiments_dir / "conf"
    existing_exps = list(conf_dir.glob("exp_*.yaml"))
    if existing_exps:
        existing_nums = [
            int(f.stem.split("_")[1])
            for f in existing_exps
            if f.stem.startswith("exp_")
        ]
        next_num = max(existing_nums) + 1
    else:
        next_num = 1
    exp_id = f"exp_{next_num:03d}"

    # Default config name if not provided
    if conf_name is None:
        conf_name = f"{exp_id}.yaml"

    exp_conf_path = conf_dir / conf_name

    # Check if config already exists
    if exp_conf_path.exists():
        # If it exists, just add server metadata to it
        print(f"Config already exists: experiments/{conf_name}")
        print(f"Adding server metadata: {server}")
        exp_config = yaml.safe_load(exp_conf_path.read_text())
    else:
        # Create from template
        template_path = repo_root / "glacier_mapping" / "conf" / "unet_train.yaml"
        if not template_path.exists():
            raise FileNotFoundError(
                f"Template not found: {template_path}\n"
                f"Please ensure glacier_mapping/conf/unet_train.yaml exists"
            )
        exp_config = yaml.safe_load(template_path.read_text())
        print(f"Created new config: experiments/{conf_name}")

    # Add server metadata at top level
    exp_config["server"] = server
    exp_config["gpu_rank"] = gpu_rank

    # Update paths for this server
    servers_cfg = yaml.safe_load(servers_path.read_text())
    server_cfg = servers_cfg[server]

    # Update output_dir to use server's path
    if "training_opts" in exp_config and "output_dir" in exp_config["training_opts"]:
        desktop_output_dir = exp_config["training_opts"]["output_dir"]
        desktop_cfg = servers_cfg["desktop"]
        desktop_base = desktop_cfg["output_path"]
        server_base = server_cfg["output_path"]

        if desktop_output_dir.startswith(desktop_base):
            # Replace desktop base with server base
            relative_path = desktop_output_dir[len(desktop_base) :].lstrip("/")
            exp_config["training_opts"]["output_dir"] = f"{server_base}/{exp_id}"
        else:
            # Fallback: use server base + exp_id
            exp_config["training_opts"]["output_dir"] = f"{server_base}/{exp_id}"

    # Update processed_dir to use server's paths
    if "loader_opts" in exp_config and "processed_dir" in exp_config["loader_opts"]:
        desktop_processed_dir = exp_config["loader_opts"]["processed_dir"]
        desktop_cfg = servers_cfg["desktop"]
        desktop_base = desktop_cfg["processed_data_path"]
        server_base = server_cfg["processed_data_path"]

        if desktop_processed_dir.startswith(desktop_base):
            # Replace desktop base with server base
            relative_path = desktop_processed_dir[len(desktop_base) :].lstrip("/")
            exp_config["loader_opts"]["processed_dir"] = (
                f"{server_base}/{relative_path}"
            )
        # Note: If processed_dir doesn't start with desktop_base, leave as-is (user custom path)

    # Update run_name to match experiment ID
    if "training_opts" in exp_config:
        exp_config["training_opts"]["run_name"] = exp_id

    # Save config
    exp_conf_path.write_text(
        yaml.dump(exp_config, sort_keys=False, default_flow_style=False)
    )

    print(f"✓ Created {exp_id} for {server} GPU {gpu_rank}")
    print(f"  Config: experiments/conf/{conf_name}")
    print(f"\nNext steps:")
    print(f"  1. Edit config: vim experiments/conf/{conf_name}")
    print(f"  2. Commit: git add experiments/conf/{conf_name}")
    print(f"  3. Push: git commit -m 'Add {exp_id}' && git push")
    if server != "desktop":
        print(
            f"  4. On {server}: git pull && uv run python experiments/worker.py --server {server} --gpu {gpu_rank} --once"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create experiment config with server metadata"
    )
    parser.add_argument(
        "--server",
        required=True,
        choices=["desktop", "bilbo"],
        help="Target server for this experiment",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        required=True,
        help="GPU rank to use (0 or 1 for bilbo, 0 for desktop)",
    )
    parser.add_argument(
        "--conf",
        help="Experiment config filename (optional, defaults to exp_XXX.yaml)",
    )
    args = parser.parse_args()

    submit_experiment(args.server, args.gpu, args.conf)
