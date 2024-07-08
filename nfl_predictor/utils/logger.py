"""
This module configures a logger with a specific format and color support for console output.

The logger is configured to display messages with timestamps, log level, source file name, function
name, and line number.  It uses the `coloredlogs` package to enhance the log output with colors,
making it easier to distinguish between different levels of log messages.  The log messages are
output to stderr.

Attributes:
    LOGGING_CONFIG (dict):  Configuration dictionary for setting up the logging.
    log (logging.Logger):   Configured logger instance for use throughout the application.
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
