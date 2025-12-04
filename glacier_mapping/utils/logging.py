#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Logging utilities for glacier mapping project.

This module provides a custom timestamped logger for internal application logging.
Use these functions for library code (glacier_mapping/*) to log processing steps,
warnings, and errors. For CLI scripts (scripts/*), use print() for user-facing output.

Usage:
    import glacier_mapping.utils.logging as log

    log.info("Processing started")
    log.warning("Missing optional parameter, using default")
    log.error("Failed to load file")
    log.debug("Detailed debug information")

Note: The custom log() function adds timestamps and formatting automatically.
"""

import datetime
import logging

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)


def log(level, message):
    """Timestamped logger used throughout the project.

    Args:
        level: Logging level (logging.INFO, logging.WARNING, etc.)
        message: Message string to log

    Example:
        log(logging.INFO, "Starting data preprocessing")
    """
    message = "{}\t{}   {}".format(
        datetime.datetime.now().strftime("%d-%m-%Y, %H:%M:%S"),
        logging._levelToName[level],
        message,
    )
    logging.log(level, "SystemLog: " + message)


def info(message):
    """Log an info message with timestamp."""
    log(logging.INFO, message)


def warning(message):
    """Log a warning message with timestamp."""
    log(logging.WARNING, message)


def error(message):
    """Log an error message with timestamp."""
    log(logging.ERROR, message)


def debug(message):
    """Log a debug message with timestamp."""
    log(logging.DEBUG, message)


def print_conf(conf):
    """Pretty-print config dictionary to logger.

    Args:
        conf: Dictionary of configuration key-value pairs
    """
    for k, v in conf.items():
        log(logging.INFO, f"{k} = {v}")
