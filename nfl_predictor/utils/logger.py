"""
Implements a logging system for the NFL predictor project, providing a unified approach to logging
across the application. This module configures log levels and formats, facilitating debugging and
monitoring by recording operational events and errors.

Features:
- Configurable log levels to control the verbosity of log messages.
- Standardized log format across the application for consistency.
- Integration with Python's built-in logging module for robust log management.

Usage:
This module is used throughout the project to log various events, errors, and informational
messages. It aids in debugging and provides insights into the application's operation.

Dependencies:
- Python's logging module: Utilized for all logging operations.
"""

import logging
from logging.config import dictConfig

# Configure the logging format and handler
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "default": {
            "format": (
                "[%(asctime)s.%(msecs)03d][%(levelname)s]"
                "[%(filename)s:%(funcName)s:%(lineno)s] %(message)s"
            ),
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "class": "coloredlogs.ColoredFormatter",
        },
    },
    "handlers": {
        "default": {
            "level": "DEBUG",
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {
        "root": {
            "handlers": ["default"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}

# Apply the logging configuration
dictConfig(LOGGING_CONFIG)
# Create a logger instance for use throughout the application
log = logging.getLogger()
