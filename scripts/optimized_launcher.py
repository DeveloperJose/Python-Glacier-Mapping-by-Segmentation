#!/usr/bin/env python3
"""Optimized experiment launcher with conservative GPU allocation and OOM protection."""

import argparse
import json
import os
import pathlib
import time
import yaml
from typing import Dict, List, Any, Optional
import subprocess
import sys

import torch


class Experiment:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.name = pathlib.Path(config_path).stem
        
        # Load config to extract memory requirements
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.server = config.get('server', 'unknown')
        self.gpu_rank = config.get('gpu_rank', 0)
        self.memory_req = self._estimate_memory(config)
        
    def _estimate_memory(self, config: Dict) -> float:
        """Conservative memory estimate in GB."""
        loader_opts = config.get('loader_opts', {})
        model_opts = config.get('model_opts', {})
        
        # Base memory from model architecture
        model_args = model_opts.get('args', {})
        depth = model_args.get('net_depth', 4)
        first_output = model_args.get('first_channel_output', 32)
        
        # Memory scaling factors
        depth_factor = 1 + (depth - 4) * 0.3
        width_factor = first_output / 32
        
        # Channel count affects memory
        channels = len(loader_opts.get('use_channels', [0, 1, 2]))
        channel_factor = channels / 3
        
        # Base conservative estimate
        base_memory = 6.0  # GB for standard config
        
        # Apply factors with safety margin
        estimated = base_memory * depth_factor * width_factor * channel_factor
        
        # Add safety margin and cap for smaller GPUs
        return min(estimated * 1.2, 6.0)  # Cap at 6GB for smaller GPUs


class GPUMonitor:
    @staticmethod
    def get_available_memory(gpu_id: int) -> float:
        """Get available GPU memory in GB using PyTorch."""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            torch.cuda.set_device(gpu_id)
            total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
            reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
            allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            available = total - max(reserved, allocated)
            return available
        except:
            return 0.0
    
    @staticmethod
    def get_total_memory(gpu_id: int) -> float:
        """Get total GPU memory in GB."""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            return torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
        except:
            return 0.0


class ExperimentLauncher:
    def __init__(self, server: str, conservative: bool = True):
        self.server = server
        self.conservative = conservative
        self.monitor = GPUMonitor()
        self.state_file = f"launcher_state_{server}.json"
        
        # Load server configuration
        with open('configs/servers.yaml', 'r') as f:
            servers = yaml.safe_load(f)
        self.server_config = servers[server]
        
        # Get available GPUs
        self.available_gpus = self._detect_gpus()
        
        # Load experiments
        self.experiments = self._load_experiments()
        
        # Initialize state
        self.state = self._load_state()
        
    def _detect_gpus(self) -> List[int]:
        """Detect available GPUs."""
        if not torch.cuda.is_available():
            return []
        
        gpu_count = torch.cuda.device_count()
        available = []
        
        for gpu_id in range(gpu_count):
            if self.monitor.get_total_memory(gpu_id) > 0:
                available.append(gpu_id)
        
        return available
    
    def _load_experiments(self) -> List[Experiment]:
        """Load experiments for this server."""
        experiments = []
        exp_dir = pathlib.Path('configs/experiments')
        
        for exp_file in exp_dir.glob('exp_*.yaml'):
            exp = Experiment(str(exp_file))
            if exp.server == self.server:
                experiments.append(exp)
        
        return experiments
    
    def _load_state(self) -> Dict:
        """Load previous state if exists."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'completed': [],
            'failed': [],
            'running': [],
            'pending': [exp.name for exp in self.experiments]
        }
    
    def _save_state(self):
        """Save current state."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def _find_experiment_by_name(self, name: str) -> Optional[Experiment]:
        """Find experiment by name."""
        for exp in self.experiments:
            if exp.name == name:
                return exp
        return None
    
    def _can_run_on_gpu(self, exp: Experiment, gpu_id: int) -> bool:
        """Check if experiment can run on specific GPU."""
        total = self.monitor.get_total_memory(gpu_id)
        required = exp.memory_req
        
        # Simple check: experiment must fit in total GPU memory
        return total > required
    
    def _find_best_gpu(self, exp: Experiment) -> Optional[int]:
        """Find best GPU for experiment."""
        for gpu_id in self.available_gpus:
            if self._can_run_on_gpu(exp, gpu_id):
                return gpu_id
        return None
    
    def _run_experiment(self, exp: Experiment, gpu_id: int) -> bool:
        """Run single experiment with OOM protection."""
        cmd = [
            'uv', 'run', 'python', 'scripts/train.py',
            '--config', exp.config_path,
            '--server', self.server,
            '--gpu', str(gpu_id),
            '--max-epochs', '500',
            '--mlflow-enabled', 'true',
            '--tracking-uri', 'https://mlflow.developerjose.duckdns.org/'
        ]
        
        print(f"Starting {exp.name} on GPU {gpu_id} (est. {exp.memory_req:.1f}GB)")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600*6)  # 6 hour timeout
            
            if result.returncode == 0:
                print(f"✅ {exp.name} completed successfully")
                return True
            else:
                if "out of memory" in result.stderr.lower() or "cuda out of memory" in result.stderr.lower():
                    print(f"💥 {exp.name} OOM on GPU {gpu_id}")
                    return False
                else:
                    print(f"❌ {exp.name} failed: {result.stderr[:200]}")
                    return False
                    
        except subprocess.TimeoutExpired:
            print(f"⏰ {exp.name} timed out")
            return False
        except Exception as e:
            print(f"💥 {exp.name} crashed: {e}")
            return False
    
    def _schedule_experiments(self) -> Dict[int, List[Experiment]]:
        """Schedule experiments across available GPUs."""
        schedule = {gpu_id: [] for gpu_id in self.available_gpus}
        
        # Get pending experiments
        pending = [self._find_experiment_by_name(name) 
                  for name in self.state['pending']]
        pending = [exp for exp in pending if exp is not None]
        
        # Sort by memory requirement (largest first for better packing)
        pending.sort(key=lambda x: x.memory_req, reverse=True)
        
        # Assign to GPUs
        for exp in pending:
            best_gpu = self._find_best_gpu(exp)
            if best_gpu is not None:
                schedule[best_gpu].append(exp)
        
        return schedule
    
    def run(self, dry_run: bool = False):
        """Main execution loop."""
        print(f"🚀 Starting launcher for {self.server}")
        print(f"📊 Available GPUs: {self.available_gpus}")
        for gpu_id in self.available_gpus:
            total = self.monitor.get_total_memory(gpu_id)
            print(f"   GPU {gpu_id}: {total:.1f}GB total")
        print(f"📋 Total experiments: {len(self.experiments)}")
        print(f"⏳ Pending: {len(self.state['pending'])}")
        print(f"✅ Completed: {len(self.state['completed'])}")
        print(f"❌ Failed: {len(self.state['failed'])}")
        
        if dry_run:
            schedule = self._schedule_experiments()
            print("\n📅 DRY RUN SCHEDULE:")
            for gpu_id, exps in schedule.items():
                total_memory = sum(exp.memory_req for exp in exps)
                print(f"  GPU {gpu_id}: {len(exps)} experiments, ~{total_memory:.1f}GB total")
                for exp in exps[:3]:  # Show first 3
                    print(f"    - {exp.name} ({exp.memory_req:.1f}GB)")
                if len(exps) > 3:
                    print(f"    ... and {len(exps)-3} more")
            return
        
        # Main execution loop
        while self.state['pending']:
            schedule = self._schedule_experiments()
            
            # Run experiments on each GPU
            for gpu_id, exps in schedule.items():
                if not exps:
                    continue
                
                exp = exps[0]  # Take first experiment for this GPU
                
                if self._run_experiment(exp, gpu_id):
                    self.state['completed'].append(exp.name)
                    self.state['pending'].remove(exp.name)
                else:
                    # Try alternative GPU
                    alt_gpu = self._find_best_gpu(exp)
                    if alt_gpu is not None and alt_gpu != gpu_id:
                        print(f"🔄 Retrying {exp.name} on GPU {alt_gpu}")
                        if self._run_experiment(exp, alt_gpu):
                            self.state['completed'].append(exp.name)
                            self.state['pending'].remove(exp.name)
                        else:
                            self.state['failed'].append(exp.name)
                            self.state['pending'].remove(exp.name)
                    else:
                        self.state['failed'].append(exp.name)
                        self.state['pending'].remove(exp.name)
                
                self._save_state()
                
                # Small delay between experiments
                time.sleep(5)
            
            # Check if any GPU can take more work
            next_exp = self._find_experiment_by_name(self.state['pending'][0]) if self.state['pending'] else None
            if not any(self._find_best_gpu(next_exp) for _ in [0] if next_exp):
                print("⚠️  No GPU available for remaining experiments. Waiting...")
                time.sleep(60)  # Wait 1 minute before retrying
        
        print(f"\n🎉 All experiments completed for {self.server}!")
        print(f"✅ Successful: {len(self.state['completed'])}")
        print(f"❌ Failed: {len(self.state['failed'])}")
        
        # Clean up state file
        if os.path.exists(self.state_file):
            os.remove(self.state_file)


def main():
    parser = argparse.ArgumentParser(description="Optimized experiment launcher")
    parser.add_argument("--server", type=str, required=True, 
                       choices=["desktop", "bilbo", "frodo"],
                       help="Server name")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show schedule without running")
    parser.add_argument("--conservative", action="store_true", default=True,
                       help="Use conservative memory estimates")
    
    args = parser.parse_args()
    
    # Change to project directory
    os.chdir(pathlib.Path(__file__).parent.parent)
    
    launcher = ExperimentLauncher(args.server, args.conservative)
    launcher.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()