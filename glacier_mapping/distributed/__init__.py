"""
Distributed experiment system for glacier mapping.

Provides tools for:
- Submitting experiments to different servers
- Running experiments on specific GPUs
- Monitoring experiment progress
- Managing experiment configurations

Usage:
    python -m glacier_mapping.distributed.submit --server desktop --gpu 0
    python -m glacier_mapping.distributed.worker --server desktop --gpu 0 --loop 60
    python -m glacier_mapping.distributed.monitor
"""
