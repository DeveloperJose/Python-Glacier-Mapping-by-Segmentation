#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import logging

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)


# ============================================================
# BASIC LOGGING UTILITIES
# ============================================================
def log(level, message):
    """
    Timestamped logger used throughout the project.
    """
    message = "{}\t{}   {}".format(
        datetime.datetime.now().strftime("%d-%m-%Y, %H:%M:%S"),
        logging._levelToName[level],
        message,
    )
    logging.log(level, "SystemLog: " + message)


def print_conf(conf):
    """
    Pretty-print config dictionary.
    """
    for k, v in conf.items():
        log(logging.INFO, f"{k} = {v}")

