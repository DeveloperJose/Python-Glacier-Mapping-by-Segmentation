#!/usr/bin/env python3
"""Ray Tune training orchestration for glacier mapping experiments."""

import argparse
import glob
import os
import sys
from typing import Dict, List, Any

import yaml

try:
    import ray
    from ray import tune
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Warning: Ray not available. Install with: uv pip install ray[tune]")


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def load_server_config(servers_yaml_path: str, server_name: str) -> Dict[str, Any]:
    with open(servers_yaml_path, "r") as f:
        servers = yaml.safe_load(f)
    if server_name not in servers:
        raise ValueError(f"Server '{server_name}' not found in {servers_yaml_path}")
    return servers[server_name]


def get_configs_for_server(server_name: str, patterns: List[str] | None = None) -> List[str]:
    configs = []
    config_files = glob.glob("configs/experiments/*.yaml")

    for config_path in config_files:
        if any(x in config_path for x in ["debug_", "initial_validation"]) and not patterns:
            continue

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # server filter
        if cfg.get("server") != server_name:
            continue

        # optional pattern filters
        if patterns:
            base = os.path.basename(config_path)
            if not any(p.replace("*", "") in base for p in patterns):
                continue

        configs.append(config_path)

    return sorted(configs)


def get_gpu_resources(server_name: str, gpu_per_trial: float | None = None):
    if gpu_per_trial is not None:
        return {"gpu": gpu_per_trial}

    defaults = {
        "desktop": 0.5,
        "bilbo":   0.5,
        "frodo":   0.33,
    }
    return {"gpu": defaults[server_name]}


# -------------------------------------------------------------
# Trainable
# -------------------------------------------------------------
def ray_train(config: Dict[str, Any]):
    """Ray trainable. Config is FLAT — no nested dicts."""
    from ray import tune  # Import inside function to ensure it's available
    
    config_path   = config["config_path"]
    server        = config["server"]
    max_epochs    = config["max_epochs"]
    tracking_uri  = config["tracking_uri"]
    output_path   = config["output_path"]

    print(f"[RAY] Running: {config_path}")
    
    # Find the actual working_dir (where Ray extracted the packaged code)
    current_dir = os.getcwd()
    print(f"[RAY] Current working directory: {current_dir}")
    
    # Find the packaged code directory by looking for where scripts module is
    project_root = None
    for path_entry in sys.path:
        test_path = os.path.join(path_entry, "scripts", "train.py")
        if os.path.exists(test_path):
            project_root = path_entry
            break
    
    if not project_root:
        raise RuntimeError("Could not find project root in Ray worker")
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"[RAY] Added {project_root} to sys.path")
    else:
        print(f"[RAY] Found project root at: {project_root} (already in sys.path)")
    
    # Make config path absolute relative to project root
    absolute_config_path = os.path.join(project_root, config_path)
    if not os.path.exists(absolute_config_path):
        raise FileNotFoundError(f"Config not found: {absolute_config_path}")
    
    print(f"[RAY] Using config: {absolute_config_path}")

    original_argv = sys.argv.copy()
    original_cwd = os.getcwd()
    
    try:
        # Change to project root so relative paths in train.py work
        os.chdir(project_root)
        print(f"[RAY] Changed to project root: {project_root}")
        
        # Set up arguments as if we called train.py from command line
        sys.argv = [
            "train.py",
            "--config", absolute_config_path,
            "--server", server,
            "--max-epochs", str(max_epochs),
            "--mlflow-enabled", "true",
            "--tracking-uri", tracking_uri,
            "--output-dir", output_path,
            "--gpu", "0",  # Default to GPU 0 for Ray workers
        ]
        
        # Import and call the train script's main function
        from scripts.train import main as train_main
        
        train_main()
        
        print(f"[RAY] Success: {config_path}")
        tune.report({"result": 1})
        
    except Exception as e:
        print(f"[RAY] Failure: {config_path} → {str(e)}")
        import traceback
        traceback.print_exc()
        tune.report({"result": 0})
        
    finally:
        # Always restore original sys.argv and working directory
        sys.argv = original_argv
        os.chdir(original_cwd)


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    if not RAY_AVAILABLE:
        print("Ray not installed. Run: uv pip install ray[tune]")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, choices=["desktop", "bilbo", "frodo"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--gpu-per-trial", type=float)
    parser.add_argument("--patterns", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tracking-uri", default="https://mlflow.developerjose.duckdns.org/")
    parser.add_argument("--num-samples", type=int, default=1)
    args = parser.parse_args()

    # Load server and configs
    server_cfg = load_server_config("configs/servers.yaml", args.server)
    configs = get_configs_for_server(args.server, args.patterns)

    if not configs:
        print("No configs found for server.")
        sys.exit(1)

    print("\nFound experiment configs:")
    for c in configs:
        print("  -", c)

    if args.dry_run:
        print("\nDry run complete.")
        sys.exit(0)

    # Compute project root for Ray working_dir
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    print(f"[RAY] Project root: {project_root}")

    # Ray init (ray is guaranteed to be available due to RAY_AVAILABLE check above)
    if not ray.is_initialized():  # type: ignore[possibly-unbound]
        ray.init(  # type: ignore[possibly-unbound]
            runtime_env={
                # Pack the ENTIRE repo as working directory
                "working_dir": project_root,

                # Things to exclude to keep packaging fast
                # Note: Be careful with excludes - they match recursively!
                # We exclude large data directories but NOT glacier_mapping/data (the code module)
                "excludes": [
                    ".git",
                    ".venv",
                    "datasets/*",  # Exclude datasets directory contents
                    "output/*",    # Exclude output directory contents
                    "mlruns/*",    # Exclude mlruns directory contents
                    "__pycache__",
                    "*.pyc",
                ],
            }
        )

    # Build flat configs for Ray
    experiment_configs = []
    for c in configs:
        experiment_configs.append({
            "config_path": c,
            "server": args.server,
            "max_epochs": args.max_epochs,
            "tracking_uri": args.tracking_uri,
            "output_path": server_cfg["output_path"],
        })

    # Get GPU resource allocation
    gpu_resources = get_gpu_resources(args.server, args.gpu_per_trial)
    print(f"[RAY] GPU resources per trial: {gpu_resources}")
    
    tuner = tune.Tuner(  # type: ignore[possibly-unbound]
        tune.with_resources(  # type: ignore[possibly-unbound]
            ray_train,
            resources=gpu_resources
        ),
        param_space=tune.grid_search(experiment_configs),  # type: ignore[possibly-unbound]
        tune_config=tune.TuneConfig(  # type: ignore[possibly-unbound]
            metric="result",
            mode="max",
        ),
        run_config=tune.RunConfig(  # type: ignore[possibly-unbound]
            name=f"glacier_experiments_{args.server}",
            verbose=1,
        )
    )

    print("\nStarting Ray Tune...\n")
    results = tuner.fit()

    # Summaries
    completed = 0
    failed = 0

    for r in results:
        metrics = r.metrics or {}
        cfg      = r.config or {}
        cfg_path = cfg.get("config_path", "<unknown>")
        val      = metrics.get("result", 0)

        if val == 1:
            completed += 1
        else:
            failed += 1
            print("FAILED:", cfg_path)

    print(f"\nCompleted: {completed}, Failed: {failed}")


if __name__ == "__main__":
    main()

