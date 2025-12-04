"""Unified configuration loading utilities for glacier mapping project.

This module provides standardized config loading functions to eliminate
duplication across scripts. All functions assume execution from project root.

Usage:
    from glacier_mapping.utils.config import load_config, load_server_config

    # Simple config loading
    config = load_config("configs/my_config.yaml")

    # Load with server paths
    config, server = load_config_with_server("configs/my_config.yaml", "desktop")
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config file (relative to project root)

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_server_config(
    server_name: str, servers_yaml: str = "configs/servers.yaml"
) -> Dict[str, Any]:
    """Load server configuration from servers.yaml.

    Args:
        server_name: Name of server (e.g., 'desktop', 'frodo', 'bilbo')
        servers_yaml: Path to servers.yaml file (relative to project root)

    Returns:
        Dictionary containing server configuration

    Raises:
        FileNotFoundError: If servers.yaml doesn't exist
        ValueError: If server_name not found in servers.yaml
    """
    servers_path = Path(servers_yaml)

    if not servers_path.exists():
        raise FileNotFoundError(f"Servers config not found: {servers_path}")

    with open(servers_path, "r") as f:
        servers = yaml.safe_load(f)

    if server_name not in servers:
        available = ", ".join(servers.keys())
        raise ValueError(
            f"Server '{server_name}' not found in {servers_path}. "
            f"Available servers: {available}"
        )

    return servers[server_name]


def load_config_with_server(
    config_path: str, server_name: str, servers_yaml: str = "configs/servers.yaml"
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load configuration file and server config together.

    This is a convenience function for scripts that need both configs.

    Args:
        config_path: Path to YAML config file
        server_name: Name of server
        servers_yaml: Path to servers.yaml file

    Returns:
        Tuple of (config_dict, server_config_dict)

    Example:
        config, server = load_config_with_server(
            "configs/unet_train.yaml",
            "desktop"
        )
        data_path = server["processed_data_path"]
    """
    config = load_config(config_path)
    server_config = load_server_config(server_name, servers_yaml)

    return config, server_config


def validate_config_keys(config: Dict[str, Any], required_keys: list) -> None:
    """Validate that config contains required keys.

    Args:
        config: Configuration dictionary
        required_keys: List of required key names

    Raises:
        ValueError: If any required keys are missing

    Example:
        validate_config_keys(config, ["model_opts", "training_opts", "loader_opts"])
    """
    missing = [key for key in required_keys if key not in config]

    if missing:
        raise ValueError(
            f"Config missing required keys: {missing}. Required: {required_keys}"
        )
