#!/usr/bin/env python3
"""
Interactive worker script that runs on GPU servers.
Finds and runs pending experiments assigned to this server and GPU with full interactive control.

Usage on server:
  # Interactive mode (default)
  uv run python -m glacier_mapping.distributed.worker --server bilbo --gpu 0
"""

import argparse
import contextlib
import gc
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# ---------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------
OOM_RETRY_INTERVAL_MINUTES = 30   # Check OOM every 30 minutes
OOM_TIMEOUT_HOURS = 12            # Fail OOM after 12 hours
ERROR_RETRY_INTERVAL_MINUTES = 60 # Check all other errors every 1 hour
ERROR_TIMEOUT_HOURS = 12          # Fail all other errors after 12 hours
OTHER_MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 60
PROCESS_CHECK_INTERVAL = 5        # seconds (not currently used, but kept for future)
REFRESH_INTERVAL = 10             # seconds (reload experiments / UI)
CONFIRMATION_TIMEOUT = 30         # seconds
LINE_WIDTH = 80
EXPERIMENT_START_DELAY_SECONDS = 300  # 5 minutes between experiment starts (only if something is already running)

# Status values (string-based state machine)
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_OOM_WAITING = "oom_waiting"
STATUS_ERROR_WAITING = "error_waiting"
STATUS_STOPPED = "stopped"
STATUS_FAILED = "failed"
STATUS_COMPLETED = "completed"


# ---------------------------------------------------------------------
# Experiment state
# ---------------------------------------------------------------------
@dataclass
class ExperimentState:
    """Track the full state of a single experiment."""

    exp_id: str
    process: Optional[subprocess.Popen] = None
    status: str = STATUS_PENDING  # see constants above
    start_time: Optional[datetime] = None
    stop_time: Optional[datetime] = None
    retry_count: int = 0

    # OOM / error waiting
    oom_wait_start: Optional[datetime] = None
    error_wait_start: Optional[datetime] = None
    oom_last_retry_time: Optional[datetime] = None
    error_last_retry_time: Optional[datetime] = None

    # Error metadata
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    error_type: Optional[str] = None  # "oom", "cuda", "data", "import", "config", "unknown"
    is_recoverable: bool = False

    # Runtime metrics and bookkeeping
    pid: Optional[int] = None
    gpu_memory_gb: float = 0.0
    current_epoch: int = 0
    current_loss: float = 0.0
    log_file: Optional[Path] = None
    last_log_pos: int = 0
    user_stopped: bool = False    # track user-initiated stops
    total_epochs: Optional[int] = None  # <-- for ETA computation


# ---------------------------------------------------------------------
# Worker implementation
# ---------------------------------------------------------------------
class InteractiveWorker:
    """Interactive worker with concurrent experiment execution."""

    def __init__(self, server_name: str, gpu_rank: int):
        self.server_name = server_name
        self.gpu_rank = gpu_rank
        self.experiments: Dict[str, ExperimentState] = {}
        self.selected_exp_id: Optional[str] = None
        self.worker_paused = False
        self.running = True
        self.display_needs_refresh = True
        self.total_gpu_memory_gb = 0.0
        self.last_experiment_start_time: Optional[datetime] = None

        # VALIDATE: Must be run from glacier_mapping/ directory
        if not Path("conf/servers.yaml").exists():
            print("\n" + "=" * 80)
            print("ERROR: Worker must be run from glacier_mapping/ directory")
            print("=" * 80)
            print("\nExpected usage:")
            print("  cd glacier_mapping")
            print(
                "  uv run python -m glacier_mapping.distributed.worker --server <name> --gpu <N>"
            )
            print("\nCurrent directory:", Path.cwd())
            print("=" * 80 + "\n")
            sys.exit(1)

        # Load server configuration
        self.servers_cfg = yaml.safe_load(Path("conf/servers.yaml").read_text())
        try:
            self.server = self.servers_cfg[server_name]
        except KeyError:
            print(f"ERROR: Server '{server_name}' not found in conf/servers.yaml")
            sys.exit(1)

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Initialize total GPU memory for percentage calculations
        self._get_total_gpu_memory()

    # ------------------------------------------------------------------
    # System / process management
    # ------------------------------------------------------------------
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.running = False
        self._cleanup_all_processes()
        # termios cleanup happens in run_interactive finally
        sys.exit(0)

    def _cleanup_all_processes(self):
        """Clean up all running processes."""
        for exp_state in self.experiments.values():
            if exp_state.process and exp_state.process.poll() is None:
                try:
                    exp_state.process.terminate()
                    exp_state.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    exp_state.process.kill()

    # ------------------------------------------------------------------
    # Terminal helpers
    # ------------------------------------------------------------------
    def get_key(self) -> Optional[str]:
        """Get a single keypress if available (non-blocking)."""
        import select

        if sys.stdin in select.select([sys.stdin], [], [], 0.05)[0]:
            ch = sys.stdin.read(1)
            return ch
        return None

    def clear_screen(self):
        """Clear screen and move cursor to top-left."""
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

    def print_raw(self, text: str = ""):
        """Print in raw mode with proper line endings."""
        # Use CRLF so cursor returns to column 0 in raw mode
        sys.stdout.write(text + "\r\n")
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # Experiment discovery and status derivation
    # ------------------------------------------------------------------
    def load_experiments(self):
        """Load all experiments from config files and derive initial status."""
        conf_dir = Path("conf/experiments")
        exp_files = sorted(conf_dir.glob("exp_*.yaml"))

        for exp_file in exp_files:
            exp_id = exp_file.stem
            exp_config = yaml.safe_load(exp_file.read_text())

            # Check assignment
            server = exp_config.get("server")
            gpu_rank = exp_config.get("gpu_rank")

            if server is None or gpu_rank is None:
                # Misconfigured experiment; we just skip it here
                continue

            if server != self.server_name or gpu_rank != self.gpu_rank:
                continue

            # New experiment or existing one?
            if exp_id not in self.experiments:
                exp_state = ExperimentState(exp_id=exp_id)
                exp_state.status = self._get_experiment_status_from_fs(exp_id, exp_config)
                
                # Load total epochs for ETA computation with error handling
                epochs = exp_config["training_opts"].get("epochs")
                if epochs is not None:
                    try:
                        exp_state.total_epochs = int(epochs)
                        if exp_state.total_epochs <= 0:
                            exp_state.total_epochs = None
                    except (ValueError, TypeError):
                        exp_state.total_epochs = None
                
                self.experiments[exp_id] = exp_state
            else:
                # Existing: update status if experiment is not currently running
                exp_state = self.experiments[exp_id]
                if exp_state.status not in (STATUS_RUNNING, STATUS_PENDING, STATUS_OOM_WAITING, STATUS_ERROR_WAITING):
                    exp_state.status = self._get_experiment_status_from_fs(exp_id, exp_config)

                # If status is PENDING and we see stale STOPPED/RUNNING markers, we clean them
                if exp_state.status == STATUS_PENDING:
                    run_name = exp_config["training_opts"]["run_name"]
                    results_dir = (
                        Path(self.server["output_path"])
                        / "runs"
                        / f"{run_name}_{self.server_name}_gpu{self.gpu_rank}"
                    )
                    stopped_marker = results_dir / "STOPPED"
                    if stopped_marker.exists():
                        stopped_marker.unlink()
                    running_marker = results_dir / "RUNNING"
                    if running_marker.exists():
                        running_marker.unlink()

    def _get_experiment_status_from_fs(self, exp_id: str, exp_config: dict) -> str:
        """Determine experiment status from filesystem markers."""
        run_name = exp_config["training_opts"]["run_name"]
        results_dir = (
            Path(self.server["output_path"])
            / "runs"
            / f"{run_name}_{self.server_name}_gpu{self.gpu_rank}"
        )

        if not results_dir.exists():
            return STATUS_PENDING

        # Explicit markers
        if (results_dir / "STOPPED").exists():
            disable_resume = exp_config.get("training_opts", {}).get("disable_resume", False)
            if disable_resume:
                return STATUS_STOPPED
            else:
                # auto-resume previously stopped runs
                return STATUS_PENDING

        if (results_dir / "FAILED").exists():
            return STATUS_FAILED

        if (results_dir / "checkpoints_summary.json").exists():
            return STATUS_COMPLETED

        # RUNNING marker with PID validation
        running_marker = results_dir / "RUNNING"
        if running_marker.exists():
            try:
                running_content = running_marker.read_text().strip()
                pid = None
                for line in running_content.splitlines():
                    if line.strip().startswith("PID:"):
                        pid_str = line.split(":", 1)[1].strip()
                        pid = int(pid_str)
                        break

                if pid is not None and PSUTIL_AVAILABLE:
                    if psutil.pid_exists(pid):
                        return STATUS_RUNNING
                    # else stale marker
                # If no psutil, we err on the side of treating it as stale,
                # and will re-launch if resume is allowed.
            except Exception:
                # Any parsing failure -> treat marker as stale
                pass

        # Incomplete run with checkpoints but no final markers -> resume or stop
        models_dir = results_dir / "models"
        if models_dir.exists() and list(models_dir.glob("model_*.pt")):
            disable_resume = exp_config.get("training_opts", {}).get("disable_resume", False)
            if disable_resume:
                return STATUS_STOPPED
            else:
                return STATUS_PENDING

        # Default: pending
        return STATUS_PENDING

    def get_run_name(self, exp_id: str) -> str:
        """Get run name for experiment."""
        exp_file = Path(f"conf/experiments/{exp_id}.yaml")
        if exp_file.exists():
            exp_config = yaml.safe_load(exp_file.read_text())
            return exp_config["training_opts"]["run_name"]
        return "unknown"

    def get_output_path(self, exp_id: str) -> Path:
        """Get output path for experiment."""
        run_name = self.get_run_name(exp_id)
        return (
            Path(self.server["output_path"])
            / "runs"
            / f"{run_name}_{self.server_name}_gpu{self.gpu_rank}"
        )

    # ------------------------------------------------------------------
    # Memory test + launch decision
    # ------------------------------------------------------------------
    def test_experiment_memory(self, exp_config: dict) -> Tuple[bool, str]:
        """
        Test if an experiment can fit in GPU memory.

        Returns:
            (True, "success") on success,
            (False, "out_of_memory") on CUDA OOM,
            (False, "error: <message>") on other errors.
        """
        # Add glacier_mapping to path if needed
        code_path = Path.cwd()
        if str(code_path) not in sys.path:
            sys.path.insert(0, str(code_path))

        # Lazy imports so worker startup is cheap
        from glacier_mapping.data.data import fetch_loaders
        from glacier_mapping.core.frame import Framework  # per your refactor

        try:
            # Set GPU device mapping
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_rank)
            torch.cuda.set_device(0)  # logical device 0 after masking

            # Sanity checks on loader_opts
            loader_opts = exp_config.get("loader_opts", {})
            try:
                processed_dir = loader_opts["processed_dir"]
            except KeyError:
                raise KeyError("loader_opts.processed_dir is missing")

            processed_dir = str(processed_dir)
            batch_size = int(loader_opts["batch_size"])
            use_channels = list(loader_opts["use_channels"])
            output_classes = list(loader_opts["output_classes"])
            class_names = list(loader_opts["class_names"])
            normalize = loader_opts["normalize"]

            # Silence framework/log output during memory test
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                frame = Framework.from_dict(exp_config, device=0)

                train_loader, val_loader, test_loader = fetch_loaders(
                    processed_dir=processed_dir,
                    batch_size=batch_size,
                    use_channels=use_channels,
                    output_classes=output_classes,
                    class_names=class_names,
                    normalize=normalize,
                )

                # Test one training step
                for x, y_onehot, y_int in train_loader:
                    y_hat, loss = frame.optimize(x, y_onehot, y_int.squeeze(-1))
                    frame.step()
                    break

            # Clean up
            del frame
            del train_loader, val_loader, test_loader
            torch.cuda.empty_cache()
            gc.collect()

            return True, "success"

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            return False, "out_of_memory"

        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
            return False, f"error: {str(e)}"

    def _can_retry_error_waiting_experiment(self, exp_id: str) -> Tuple[bool, str]:
        """Direct memory test for error-waiting experiments (bypasses status checks)."""
        exp_file = Path(f"conf/experiments/{exp_id}.yaml")
        exp_config = yaml.safe_load(exp_file.read_text())
        can_run, message = self.test_experiment_memory(exp_config)

        if can_run:
            return True, "fits"
        elif message == "out_of_memory":
            return False, "oom"
        else:
            return False, message

    def _can_retry_oom_experiment(self, exp_id: str) -> Tuple[bool, str]:
        """Direct memory test for OOM-waiting experiments (bypasses status checks)."""
        exp_file = Path(f"conf/experiments/{exp_id}.yaml")
        exp_config = yaml.safe_load(exp_file.read_text())
        can_run, message = self.test_experiment_memory(exp_config)

        if can_run:
            return True, "fits"
        elif message == "out_of_memory":
            return False, "oom"
        else:
            return False, message

    def can_launch_experiment(self, exp_id: str) -> Tuple[bool, str]:
        """Check if experiment can be launched, based on its config and a memory probe."""
        exp_state = self.experiments[exp_id]

        if exp_state.status != STATUS_PENDING:
            return False, "not_pending"

        exp_file = Path(f"conf/experiments/{exp_id}.yaml")
        exp_config = yaml.safe_load(exp_file.read_text())

        can_run, message = self.test_experiment_memory(exp_config)

        if can_run:
            return True, "fits"
        elif message == "out_of_memory":
            return False, "oom"
        else:
            return False, message

    # ------------------------------------------------------------------
    # Launch and process tracking
    # ------------------------------------------------------------------
    def launch_experiment_async(self, exp_id: str) -> bool:
        """Launch experiment asynchronously (one at a time per GPU)."""
        exp_file = Path(f"conf/experiments/{exp_id}.yaml")
        exp_config = yaml.safe_load(exp_file.read_text())

        # Set up log file
        run_name = exp_config["training_opts"]["run_name"]
        base_output_path = Path(self.server["output_path"])
        log_dir = (
            base_output_path
            / "runs"
            / f"{run_name}_{self.server_name}_gpu{self.gpu_rank}"
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "worker_training.log"

        # Set up environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_rank)
        env["GLACIER_MAPPING_DEVICE_OVERRIDE"] = "0"

        # Inject explicit output_dir into config dict
        exp_config_with_output = json.loads(json.dumps(exp_config))  # deep-ish copy
        exp_config_with_output.setdefault("training_opts", {})["output_dir"] = str(log_dir)

        try:
            log_f = open(log_file, "w")

            process = subprocess.Popen(
                [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "glacier_mapping.scripts.train",
                    "--config-dict",
                    json.dumps(exp_config_with_output),
                ],
                cwd=self.server["code_path"],
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Close our handle; child keeps its inherited FD
            log_f.close()

            exp_state = self.experiments[exp_id]
            exp_state.process = process
            exp_state.pid = process.pid
            exp_state.status = STATUS_RUNNING
            exp_state.start_time = datetime.now()
            exp_state.log_file = log_file
            exp_state.last_log_pos = 0
            exp_state.user_stopped = False

            # Create RUNNING marker with PID
            (log_dir / "RUNNING").write_text(
                f"Started: {datetime.now().isoformat()}\nPID: {process.pid}\n"
            )

            self.display_needs_refresh = True
            return True

        except Exception as e:
            # Log the launch failure into a FAILED marker
            exp_state = self.experiments[exp_id]
            exp_state.status = STATUS_FAILED
            exp_state.last_error = f"Launch failed: {e}"
            exp_state.last_error_time = datetime.now()

            run_name = self.get_run_name(exp_id)
            results_dir = (
                Path(self.server["output_path"])
                / "runs"
                / f"{run_name}_{self.server_name}_gpu{self.gpu_rank}"
            )
            results_dir.mkdir(parents=True, exist_ok=True)
            (results_dir / "FAILED").write_text(
                f"Failed: {datetime.now().isoformat()}\nLaunch error: {e}\n"
            )

            self.display_needs_refresh = True
            return False

    # ------------------------------------------------------------------
    # Error detection from logs
    # ------------------------------------------------------------------
    def _detect_error_in_log(self, log_file: Path) -> Tuple[str, str, bool]:
        """
        Detect and categorize errors from a log file.

        Returns:
            error_type: "oom", "cuda", "data", "import", "config", "unknown"
            error_message: Short error details for display
            is_recoverable: Whether this error should be retried
        """
        if not log_file or not log_file.exists():
            return "unknown", "Log file not found", False

        error_patterns = [
            # OOM errors - recoverable
            (r"torch\.OutOfMemoryError", "oom", True, "Out of GPU memory"),
            (r"CUDA out of memory", "oom", True, "CUDA out of memory"),
            # CUDA/driver errors - recoverable
            (r"RuntimeError.*CUDA.*device-side assert", "cuda", True, "CUDA device assertion"),
            (r"RuntimeError.*CUDA", "cuda", True, "CUDA runtime error"),
            (r"CUDA error.*no kernel image", "cuda", True, "CUDA kernel error"),
            # Data loading errors - recoverable
            (r"FileNotFoundError", "data", True, "File not found"),
            (r"PermissionError", "data", True, "Permission denied"),
            (r"IsADirectoryError", "data", True, "Path is directory"),
            # Import/module errors - not recoverable
            (r"ModuleNotFoundError", "import", False, "Module not found"),
            (r"ImportError", "import", False, "Import error"),
            # Configuration errors - not recoverable
            (r"KeyError", "config", False, "Configuration key error"),
            (r"TypeError", "config", False, "Type error"),
            (r"ValueError", "config", False, "Value error"),
            # Network/IO errors - recoverable
            (r"ConnectionError", "data", True, "Connection error"),
            (r"TimeoutError", "data", True, "Timeout error"),
            (r"URLError", "data", True, "URL error"),
            # Memory allocation errors (non-OOM) - recoverable
            (r"RuntimeError.*memory", "cuda", True, "Memory allocation error"),
            (r"alloc.*failed", "cuda", True, "Memory allocation failed"),
        ]

        try:
            with open(log_file, "r") as f:
                log_content = f.read()

            lines = log_content.strip().splitlines()
            for i, line in enumerate(lines):
                for pattern, error_type, is_recoverable, _ in error_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        error_msg = line.strip()
                        # If next line has content, include it as context
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].rstrip()
                            if next_line:
                                error_msg += "\n" + next_line
                        return error_type, error_msg, is_recoverable

            # Fallback: last few lines
            if len(lines) > 5:
                context = "\n".join(lines[-5:])
                return "unknown", context, False

            return "unknown", log_content, False

        except Exception as e:
            return "unknown", f"Failed to read log: {e}", False

    # ------------------------------------------------------------------
    # Lifecycle: cleanup, waiting retries, OOM handling
    # ------------------------------------------------------------------
    def cleanup_finished_experiments(self):
        """Check and clean up finished processes, transition statuses."""
        for exp_id, exp_state in list(self.experiments.items()):
            if not exp_state.process:
                continue

            if exp_state.process.poll() is None:
                continue  # still running

            # Process finished
            return_code = exp_state.process.returncode
            exp_state.stop_time = datetime.now()
            exp_state.process = None
            exp_state.pid = None

            run_name = self.get_run_name(exp_id)
            results_dir = (
                Path(self.server["output_path"])
                / "runs"
                / f"{run_name}_{self.server_name}_gpu{self.gpu_rank}"
            )
            (results_dir / "RUNNING").unlink(missing_ok=True)

            if exp_state.user_stopped:
                exp_state.status = STATUS_STOPPED
                (results_dir / "STOPPED").write_text(
                    f"Stopped: {datetime.now().isoformat()}\nBy user request\n"
                )
                self.display_needs_refresh = True
                continue

            if return_code == 0:
                exp_state.status = STATUS_COMPLETED
                self.display_needs_refresh = True
                continue

            # Non-zero return code → categorize via log
            if exp_state.log_file:
                error_type, error_message, is_recoverable = self._detect_error_in_log(
                    exp_state.log_file
                )
            else:
                error_type, error_message, is_recoverable = (
                    "unknown",
                    "No log file available",
                    False,
                )

            exp_state.error_type = error_type
            exp_state.is_recoverable = is_recoverable
            exp_state.last_error = error_message
            exp_state.last_error_time = datetime.now()

            if is_recoverable:
                exp_state.status = STATUS_ERROR_WAITING
                exp_state.error_wait_start = datetime.now()
                exp_state.error_last_retry_time = None
            else:
                exp_state.status = STATUS_FAILED
                (results_dir / "FAILED").write_text(
                    f"Failed: {datetime.now().isoformat()}\n{error_message}\n"
                )

            self.display_needs_refresh = True

    def handle_error_waiting_experiments(self):
        """Handle recoverable errors - retry when conditions improve."""
        for exp_id, exp_state in self.experiments.items():
            if exp_state.status != STATUS_ERROR_WAITING:
                continue

            if exp_state.error_wait_start is None:
                exp_state.error_wait_start = datetime.now()

            wait_time = datetime.now() - exp_state.error_wait_start

            if not exp_state.is_recoverable:
                continue

            now = datetime.now()
            retry_interval_sec = ERROR_RETRY_INTERVAL_MINUTES * 60
            should_retry = (
                exp_state.error_last_retry_time is None
                or (now - exp_state.error_last_retry_time).total_seconds()
                >= retry_interval_sec
            )

            if should_retry:
                exp_state.error_last_retry_time = now
                can_launch, reason = self._can_retry_error_waiting_experiment(exp_id)

                if can_launch:
                    exp_state.status = STATUS_PENDING
                    exp_state.error_wait_start = None
                    exp_state.error_last_retry_time = None
                    exp_state.error_type = None
                    print(f"[{exp_id}] Issue resolved - ready to launch")
                    self.display_needs_refresh = True
                    continue

                # If it's now an OOM, move to OOM waiting
                if reason == "oom":
                    exp_state.status = STATUS_OOM_WAITING
                    exp_state.oom_wait_start = datetime.now()
                    exp_state.oom_last_retry_time = None
                    print(f"[{exp_id}] Now failing with OOM - switching to OOM wait")
                    self.display_needs_refresh = True
                    continue

            # Hard timeout: mark as failed if only one waiting + nothing running
            if wait_time.total_seconds() > ERROR_TIMEOUT_HOURS * 3600:
                waiting_count = len(
                    [
                        s
                        for s in self.experiments.values()
                        if s.status in [STATUS_ERROR_WAITING, STATUS_PENDING]
                    ]
                )
                running_count = len(
                    [s for s in self.experiments.values() if s.status == STATUS_RUNNING]
                )

                if waiting_count == 1 and running_count == 0:
                    exp_state.status = STATUS_FAILED
                    exp_state.last_error = (
                        f"Timeout after {ERROR_TIMEOUT_HOURS}h waiting for error resolution"
                    )
                    exp_state.last_error_time = datetime.now()
                    print(f"[{exp_id}] Failed: error wait timeout")
                    self.display_needs_refresh = True

    def handle_oom_experiments(self):
        """Handle OOM experiments - retry when memory frees."""
        for exp_id, exp_state in self.experiments.items():
            if exp_state.status != STATUS_OOM_WAITING:
                continue

            if exp_state.oom_wait_start is None:
                exp_state.oom_wait_start = datetime.now()

            wait_time = datetime.now() - exp_state.oom_wait_start

            now = datetime.now()
            retry_interval_sec = OOM_RETRY_INTERVAL_MINUTES * 60
            should_retry = (
                exp_state.oom_last_retry_time is None
                or (now - exp_state.oom_last_retry_time).total_seconds()
                >= retry_interval_sec
            )

            if should_retry:
                exp_state.oom_last_retry_time = now
                can_launch, reason = self._can_retry_oom_experiment(exp_id)

                if can_launch:
                    exp_state.status = STATUS_PENDING
                    exp_state.oom_wait_start = None
                    exp_state.oom_last_retry_time = None
                    print(f"[{exp_id}] Memory freed - ready to launch")
                    self.display_needs_refresh = True
                    continue

                # If it's now a different error instead of OOM, push to error_waiting
                if reason not in ("oom", "out_of_memory"):
                    exp_state.status = STATUS_ERROR_WAITING
                    exp_state.error_wait_start = datetime.now()
                    exp_state.error_last_retry_time = None
                    exp_state.error_type = "config"
                    exp_state.is_recoverable = True
                    exp_state.last_error = f"Memory test failed: {reason}"
                    exp_state.last_error_time = datetime.now()
                    print(f"[{exp_id}] OOM resolved but now failing: {reason}")
                    self.display_needs_refresh = True
                    continue

            # Hard timeout after OOM_TIMEOUT_HOURS
            if wait_time.total_seconds() > OOM_TIMEOUT_HOURS * 3600:
                waiting_count = len(
                    [
                        s
                        for s in self.experiments.values()
                        if s.status in [STATUS_OOM_WAITING, STATUS_PENDING]
                    ]
                )
                running_count = len(
                    [s for s in self.experiments.values() if s.status == STATUS_RUNNING]
                )

                if waiting_count == 1 and running_count == 0:
                    exp_state.status = STATUS_FAILED
                    exp_state.last_error = (
                        f"OOM after {OOM_TIMEOUT_HOURS}h with no memory freed"
                    )
                    exp_state.last_error_time = datetime.now()
                    print(f"[{exp_id}] Failed: OOM timeout")
                    self.display_needs_refresh = True

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------
    def get_experiments_by_status(self) -> Dict[str, List[ExperimentState]]:
        """Group experiments by status."""
        groups: Dict[str, List[ExperimentState]] = {
            STATUS_RUNNING: [],
            STATUS_PENDING: [],
            STATUS_OOM_WAITING: [],
            STATUS_ERROR_WAITING: [],
            STATUS_STOPPED: [],
            STATUS_FAILED: [],
            STATUS_COMPLETED: [],
        }

        for exp_state in self.experiments.values():
            if exp_state.status in groups:
                groups[exp_state.status].append(exp_state)

        return groups

    def get_navigation_list(self) -> List[ExperimentState]:
        """
        Return experiments in the same logical order as the UI:
        running → pending → oom_waiting → error_waiting → stopped → failed → completed.
        This ensures navigation matches what you see on screen.
        """
        groups = self.get_experiments_by_status()
        order = [
            STATUS_RUNNING,
            STATUS_PENDING,
            STATUS_OOM_WAITING,
            STATUS_ERROR_WAITING,
            STATUS_STOPPED,
            STATUS_FAILED,
            STATUS_COMPLETED,
        ]
        nav_list: List[ExperimentState] = []
        for status in order:
            for exp in groups[status]:
                nav_list.append(exp)
        return nav_list

    def try_launch_experiments(self):
        """Try to launch pending experiments that fit in memory."""
        if self.worker_paused:
            return

        groups = self.get_experiments_by_status()
        any_running = len(groups[STATUS_RUNNING]) > 0

        # Only enforce delay if at least one experiment is already running
        if any_running and self.last_experiment_start_time:
            time_since_last = datetime.now() - self.last_experiment_start_time
            if time_since_last.total_seconds() < EXPERIMENT_START_DELAY_SECONDS:
                return

        # Get list of pending experiments
        pending_experiments = [
            exp_id
            for exp_id, exp_state in self.experiments.items()
            if exp_state.status == STATUS_PENDING
        ]

        for exp_id in pending_experiments:
            can_launch, reason = self.can_launch_experiment(exp_id)

            if can_launch:
                if self.launch_experiment_async(exp_id):
                    self.last_experiment_start_time = datetime.now()
                    self.display_needs_refresh = True
                    break  # Only launch one experiment per cycle
            elif reason == "oom":
                # Mark as OOM waiting
                exp_state = self.experiments[exp_id]
                if exp_state.status != STATUS_OOM_WAITING:
                    exp_state.status = STATUS_OOM_WAITING
                    exp_state.oom_wait_start = datetime.now()
                    # Set last retry time to now so we don't immediately re-test
                    # in handle_oom_experiments on the very next loop.
                    exp_state.oom_last_retry_time = datetime.now()
                    print(f"[{exp_id}] Out of memory, waiting...")
                    self.display_needs_refresh = True
            else:
                # Other error - mark as error_waiting (will retry)
                exp_state = self.experiments[exp_id]
                exp_state.status = STATUS_ERROR_WAITING
                exp_state.error_wait_start = datetime.now()
                # Set last retry time to now so we don't immediately re-test
                # in handle_error_waiting_experiments on the very next loop.
                exp_state.error_last_retry_time = datetime.now()
                exp_state.error_type = "config"
                exp_state.is_recoverable = True
                exp_state.last_error = f"Memory test failed: {reason}"
                exp_state.last_error_time = datetime.now()
                print(
                    f"[{exp_id}] Memory test failed, will retry in {ERROR_RETRY_INTERVAL_MINUTES} min..."
                )
                self.display_needs_refresh = True

    # ------------------------------------------------------------------
    # GPU metrics & training log parsing
    # ------------------------------------------------------------------
    def _get_total_gpu_memory(self):
        """Get total GPU memory for percentage calculations."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                    f"--id={self.gpu_rank}",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.stdout.strip():
                total_mb = float(result.stdout.strip())
                self.total_gpu_memory_gb = total_mb / 1024.0
        except Exception:
            # Fallback: query all GPUs and pick index
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if result.stdout.strip():
                    lines = result.stdout.strip().splitlines()
                    if len(lines) > self.gpu_rank:
                        total_mb = float(lines[self.gpu_rank].strip())
                        self.total_gpu_memory_gb = total_mb / 1024.0
            except Exception:
                # Silently fail; we'll still display GB without %
                pass

    def _update_gpu_memory(self):
        """Query GPU memory for all running processes."""
        query = [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(query, capture_output=True, text=True, timeout=2)
            if not result.stdout:
                return

            for line in result.stdout.strip().splitlines():
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                try:
                    pid = int(parts[0])
                    mem_mb = float(parts[1])
                    mem_gb = mem_mb / 1024.0
                except ValueError:
                    continue

                for exp_state in self.experiments.values():
                    if exp_state.pid == pid:
                        exp_state.gpu_memory_gb = mem_gb
                        break
        except Exception:
            # GPU monitoring is non-critical
            pass

    def _parse_training_logs(self, exp_id: str):
        """Parse recent training output for metrics (epoch/loss)."""
        exp_state = self.experiments[exp_id]

        if not exp_state.log_file or not exp_state.log_file.exists():
            return

        try:
            with open(exp_state.log_file, "r") as f:
                f.seek(exp_state.last_log_pos)
                new_lines = f.readlines()
                exp_state.last_log_pos = f.tell()

            for line in new_lines:
                epoch_match = re.search(r"Epoch\s+(\d+)\s+Summary", line)
                if epoch_match:
                    exp_state.current_epoch = int(epoch_match.group(1))

                loss_match = re.search(r"Avg=(\d+\.\d+)", line)
                if loss_match:
                    exp_state.current_loss = float(loss_match.group(1))

        except Exception:
            # Not critical
            pass

    # ------------------------------------------------------------------
    # UI formatting
    # ------------------------------------------------------------------
    def format_experiment_line(self, exp_state: ExperimentState, is_selected: bool) -> str:
        """Format a single experiment line for display."""
        prefix = "> " if is_selected else "  "
        line = f"{prefix}{exp_state.exp_id:<30}"

        if exp_state.status == STATUS_RUNNING:
            line += f" [PID: {exp_state.pid or '???':<6}]"

            if exp_state.gpu_memory_gb > 0:
                if self.total_gpu_memory_gb > 0:
                    percentage = (exp_state.gpu_memory_gb / self.total_gpu_memory_gb) * 100
                    line += f" GPU: {exp_state.gpu_memory_gb:.1f}/{self.total_gpu_memory_gb:.1f}GB ({percentage:.0f}%)"
                else:
                    line += f" GPU: {exp_state.gpu_memory_gb:.1f}GB"

            if exp_state.start_time:
                elapsed = datetime.now() - exp_state.start_time
                hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                line += f" Time: {hours:02d}:{minutes:02d}:{seconds:02d}"

                # =============================
                #   EPOCH-BASED ETA (RIGHT EDGE)
                # =============================

                if (
                    exp_state.current_epoch > 0
                    and exp_state.total_epochs
                    and exp_state.total_epochs > exp_state.current_epoch
                ):
                    # Fraction completed
                    frac = exp_state.current_epoch / exp_state.total_epochs
                    total_elapsed = elapsed.total_seconds()

                    if frac > 0:
                        eta_seconds = total_elapsed * (1 - frac) / frac

                        eta_h = int(eta_seconds // 3600)
                        eta_m = int((eta_seconds % 3600) // 60)
                        eta_s = int(eta_seconds % 60)

                        eta_str = f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}"

                        # Right-align the ETA using a spacing block
                        # (keeps the UI stable regardless of line length)
                        padding = max(1, 80 - len(line) - len(" ETA: ") - len(eta_str))

                        line += " " * padding + f"ETA: {eta_str}"

                # END ETA

            if exp_state.current_loss > 0:
                line += f" Loss: {exp_state.current_loss:.3f}"

        elif exp_state.status == STATUS_PENDING:
            line += " Memory test: pending..."

        elif exp_state.status == STATUS_OOM_WAITING:
            if exp_state.oom_wait_start:
                wait_time = datetime.now() - exp_state.oom_wait_start
                hours, remainder = divmod(int(wait_time.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                line += f" Waiting: {hours:02d}:{minutes:02d}:{seconds:02d}"
            line += f" (max: {OOM_TIMEOUT_HOURS:02d}:00:00)"

        elif exp_state.status == STATUS_ERROR_WAITING:
            if exp_state.error_wait_start:
                wait_time = datetime.now() - exp_state.error_wait_start
                hours, remainder = divmod(int(wait_time.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                line += f" Waiting: {hours:02d}:{minutes:02d}:{seconds:02d}"
            line += f" (max: {ERROR_TIMEOUT_HOURS:02d}:00:00)"
            if exp_state.error_type:
                msg = exp_state.last_error or "Unknown error"
                msg = (msg[:80] + "...") if len(msg) > 80 else msg
                line += f" [{exp_state.error_type.upper()}: {msg}]"

        elif exp_state.status == STATUS_STOPPED:
            line += " Stopped by user"

        elif exp_state.status == STATUS_FAILED:
            if exp_state.last_error:
                msg = exp_state.last_error
                msg = (msg[:80] + "...") if len(msg) > 80 else msg
                if exp_state.error_type:
                    line += f" [{exp_state.error_type.upper()}: {msg}]"
                else:
                    line += f" [Error: {msg}]"
            if exp_state.retry_count > 0:
                line += f" Retries: {exp_state.retry_count}/{OTHER_MAX_RETRIES}"

        elif exp_state.status == STATUS_COMPLETED:
            line += " ✓ Completed"

        return line

    def display_status(self):
        """Display main status screen."""
        self.clear_screen()

        timestamp = datetime.now().strftime("%H:%M:%S")
        groups = self.get_experiments_by_status()

        status_line = f"Status: {'PAUSED' if self.worker_paused else 'RUNNING'}"
        status_line += f" ({len(groups[STATUS_RUNNING])} active, {len(groups[STATUS_PENDING])} pending"
        status_line += f", {len(groups[STATUS_OOM_WAITING])} OOM-waiting"
        status_line += f", {len(groups[STATUS_ERROR_WAITING])} error-waiting)"

        self.print_raw("=" * LINE_WIDTH)
        self.print_raw(
            f"INTERACTIVE WORKER MODE - {self.server_name} GPU {self.gpu_rank} - Last updated: {timestamp}"
        )
        self.print_raw(status_line)
        self.print_raw("=" * LINE_WIDTH)
        self.print_raw()

        status_order = [
            STATUS_RUNNING,
            STATUS_PENDING,
            STATUS_OOM_WAITING,
            STATUS_ERROR_WAITING,
            STATUS_STOPPED,
            STATUS_FAILED,
            STATUS_COMPLETED,
        ]
        status_labels = {
            STATUS_RUNNING: "RUNNING",
            STATUS_PENDING: "PENDING",
            STATUS_OOM_WAITING: "OOM-WAITING",
            STATUS_ERROR_WAITING: "ERROR-WAITING",
            STATUS_STOPPED: "STOPPED",
            STATUS_FAILED: "FAILED",
            STATUS_COMPLETED: "COMPLETED",
        }

        for status in status_order:
            exp_list = groups[status]
            if exp_list:
                self.print_raw(f"{status_labels[status]} ({len(exp_list)}):")
                for exp_state in exp_list:
                    is_selected = exp_state.exp_id == self.selected_exp_id
                    line = self.format_experiment_line(exp_state, is_selected)
                    self.print_raw(line)
                self.print_raw()

        self.print_raw("=" * LINE_WIDTH)
        self.print_raw("Experiment Controls: [s]top | [R]estart | [c]ontinue")
        self.print_raw("Worker Controls:    [p]ause | [r]efresh | [q]uit")
        self.print_raw("Navigation:         [ prev , next ]")
        self.print_raw("=" * LINE_WIDTH)

    # ------------------------------------------------------------------
    # UI actions (confirmation + user controls)
    # ------------------------------------------------------------------
    def confirm_action(
        self,
        action_description: str,
        experiment_info: Optional[dict] = None,
        warning: Optional[str] = None,
    ) -> bool:
        """Show confirmation dialog and return True/False."""
        self.clear_screen()
        self.print_raw("=" * LINE_WIDTH)
        self.print_raw("CONFIRMATION REQUIRED")
        self.print_raw("=" * LINE_WIDTH)
        self.print_raw(f"Action: {action_description}")

        if experiment_info:
            self.print_raw(
                f"Experiment: {experiment_info['exp_id']} ({experiment_info['run_name']})"
            )
            if "output_path" in experiment_info:
                self.print_raw(f"Output path: {experiment_info['output_path']}/")

        if warning:
            self.print_raw(f"Warning: {warning}")

        self.print_raw()
        self.print_raw("Are you sure? [y]es / [n]o (default: no)")
        self.print_raw("=" * LINE_WIDTH)

        start_time = time.time()
        while time.time() - start_time < CONFIRMATION_TIMEOUT:
            key = self.get_key()
            if key:
                k = key.lower()
                if k == "y":
                    return True
                if k in ("n", "\r", "\x1b"):
                    return False
            time.sleep(0.1)

        return False

    def handle_stop_experiment(self, exp_id: str):
        """Handle stop experiment action."""
        exp_state = self.experiments[exp_id]
        exp_info = {"exp_id": exp_id, "run_name": self.get_run_name(exp_id)}

        warning = (
            "This will terminate the running process and mark it as stopped.\n"
            "You can resume and continue from the last checkpoint later."
        )

        if self.confirm_action("Stop running experiment", exp_info, warning):
            exp_state.user_stopped = True

            if exp_state.process:
                exp_state.process.terminate()
                exp_state.status = STATUS_STOPPED
                exp_state.stop_time = datetime.now()
                self.display_needs_refresh = True

                run_name = self.get_run_name(exp_id)
                results_dir = (
                    Path(self.server["output_path"])
                    / "runs"
                    / f"{run_name}_{self.server_name}_gpu{self.gpu_rank}"
                )
                (results_dir / "RUNNING").unlink(missing_ok=True)
                (results_dir / "STOPPED").write_text(
                    f"Stopped: {datetime.now().isoformat()}\nBy user request\n"
                )

    def handle_restart_experiment(self, exp_id: str):
        """Handle restart experiment action (delete outputs, reset to pending)."""
        exp_state = self.experiments[exp_id]
        exp_info = {
            "exp_id": exp_id,
            "run_name": self.get_run_name(exp_id),
            "output_path": str(self.get_output_path(exp_id)),
        }

        warning = (
            "This will permanently delete all training progress, checkpoints, and logs!"
        )

        if self.confirm_action("Restart experiment (clean slate)", exp_info, warning):
            output_path = self.get_output_path(exp_id)
            if output_path.exists():
                shutil.rmtree(output_path)

            exp_state.status = STATUS_PENDING
            exp_state.retry_count = 0
            exp_state.oom_wait_start = None
            exp_state.oom_last_retry_time = None
            exp_state.error_wait_start = None
            exp_state.error_last_retry_time = None
            exp_state.last_error = None
            exp_state.last_error_time = None
            exp_state.error_type = None
            exp_state.user_stopped = False
            self.display_needs_refresh = True

    def handle_continue_experiment(self, exp_id: str):
        """Handle continue (resume) experiment action."""
        exp_state = self.experiments[exp_id]
        exp_state.user_stopped = False

        output_path = self.get_output_path(exp_id)
        (output_path / "FAILED").unlink(missing_ok=True)
        (output_path / "STOPPED").unlink(missing_ok=True)

        exp_state.status = STATUS_PENDING
        exp_state.oom_wait_start = None
        exp_state.oom_last_retry_time = None
        exp_state.error_wait_start = None
        exp_state.error_last_retry_time = None
        exp_state.error_type = None
        exp_state.last_error = None
        exp_state.last_error_time = None
        self.display_needs_refresh = True

    # ------------------------------------------------------------------
    # Key handling and navigation
    # ------------------------------------------------------------------
    def handle_key_input(self, key: str):
        """Handle keyboard input."""
        key_lower = key.lower()

        # Navigation: you said you intentionally use [ and ]
        if key == "[":
            self._navigate_up()
            self.display_needs_refresh = True
        elif key == "]":
            self._navigate_down()
            self.display_needs_refresh = True

        # Experiment controls (only when experiment selected)
        elif self.selected_exp_id and key_lower == "s":
            self.handle_stop_experiment(self.selected_exp_id)
        elif self.selected_exp_id and key == "R":
            self.handle_restart_experiment(self.selected_exp_id)
        elif self.selected_exp_id and key_lower == "c":
            self.handle_continue_experiment(self.selected_exp_id)

        # Worker controls
        elif key_lower == "p":
            self.worker_paused = not self.worker_paused
            status = "PAUSED" if self.worker_paused else "RESUMED"
            print(f"Worker {status}")
            self.display_needs_refresh = True
        elif key_lower == "r":
            self.display_needs_refresh = True
        elif key_lower == "q":
            exp_info = {"exp_id": "worker", "run_name": "interactive worker"}
            if self.confirm_action(
                "Quit worker",
                exp_info,
                "This will stop the worker and all running experiments.",
            ):
                self.running = False
                self._cleanup_all_processes()

    def _navigate_up(self):
        """Navigate up through experiments."""
        nav_list = self.get_navigation_list()
        if not nav_list:
            return

        if self.selected_exp_id is None:
            # Default to first item in visual order
            self.selected_exp_id = nav_list[0].exp_id
            return

        current_idx = next(
            (i for i, exp in enumerate(nav_list) if exp.exp_id == self.selected_exp_id),
            0,
        )
        if current_idx > 0:
            self.selected_exp_id = nav_list[current_idx - 1].exp_id

    def _navigate_down(self):
        """Navigate down through experiments."""
        nav_list = self.get_navigation_list()
        if not nav_list:
            return

        if self.selected_exp_id is None:
            self.selected_exp_id = nav_list[0].exp_id
            return

        current_idx = next(
            (i for i, exp in enumerate(nav_list) if exp.exp_id == self.selected_exp_id),
            0,
        )
        if current_idx < len(nav_list) - 1:
            self.selected_exp_id = nav_list[current_idx + 1].exp_id

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run_interactive(self):
        """Main interactive worker loop."""
        # Set up terminal for raw, non-blocking input
        import termios
        import tty
        import select

        old_settings = termios.tcgetattr(sys.stdin)

        # Event timing
        last_refresh = time.time()
        REFRESH_SECS = REFRESH_INTERVAL  # keep your existing value

        try:
            tty.setraw(sys.stdin.fileno())

            # Load experiments BEFORE entering the main loop
            self.load_experiments()
            self.display_needs_refresh = True

            while self.running:
                # -----------------------------------------------
                # 1. Event-driven key input using select()
                # -----------------------------------------------
                rlist, _, _ = select.select(
                    [sys.stdin],  # watch stdin
                    [],           # no write fds
                    [],           # no except fds
                    0.10          # timeout = 100 ms
                )

                if rlist:
                    key = sys.stdin.read(1)
                    if key:
                        self.handle_key_input(key)

                # -----------------------------------------------
                # 2. Periodic tasks (every REFRESH_INTERVAL seconds)
                # -----------------------------------------------
                now = time.time()
                if now - last_refresh >= REFRESH_SECS:
                    last_refresh = now

                    # Refresh experiment configurations
                    self.load_experiments()

                    # Clean up finished experiments
                    self.cleanup_finished_experiments()

                    # Process error-waiting experiments
                    self.handle_error_waiting_experiments()

                    # Process OOM-waiting experiments
                    self.handle_oom_experiments()

                    # Attempt to launch any pending experiments
                    self.try_launch_experiments()

                    # Update GPU memory statistics
                    self._update_gpu_memory()

                    # Parse log updates for running experiments
                    for exp_id, exp_state in self.experiments.items():
                        if exp_state.status == STATUS_RUNNING:
                            self._parse_training_logs(exp_id)

                    # Mark display for repaint
                    self.display_needs_refresh = True

                # -----------------------------------------------
                # 3. Repaint display only when needed
                # -----------------------------------------------
                if self.display_needs_refresh:
                    self.display_status()
                    self.display_needs_refresh = False

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            self.clear_screen()
            print("👋 Exiting interactive worker")


# ----------------------------------------------------------------------
# Git helper
# ----------------------------------------------------------------------
def git_pull() -> None:
    """Pull latest changes from git (master branch)."""
    try:
        result = subprocess.run(
            ["git", "pull", "origin", "master"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("✓ Git pull successful")
        else:
            print(f"WARNING: Git pull failed: {result.stderr.strip()}")
    except Exception as e:
        print(f"WARNING: Git pull failed: {e}")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive distributed experiment worker"
    )
    parser.add_argument(
        "--server",
        required=True,
        choices=["desktop", "bilbo", "frodo"],
        help="Server name (must match conf/servers.yaml)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        required=True,
        help="GPU rank to use (0 or 1 for bilbo, 0 for desktop)",
    )
    args = parser.parse_args()

    # Git pull to get latest experiments
    git_pull()

    # Run interactive worker
    worker = InteractiveWorker(args.server, args.gpu)
    worker.run_interactive()

