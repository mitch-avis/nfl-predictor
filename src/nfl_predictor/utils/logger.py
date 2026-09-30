"""Configure the project logging setup.

This module defines the shared log formatter and stream handler used across ETL, modeling,
and reporting entrypoints.
"""

import logging
from logging.config import dictConfig

# Configure the logging format and handler
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
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
            "level": "INFO",
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "root": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "urllib3": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "urllib3.connectionpool": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "xgboost": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

# Apply the logging configuration
dictConfig(LOGGING_CONFIG)
# Create a logger instance for use throughout the application
log = logging.getLogger()
