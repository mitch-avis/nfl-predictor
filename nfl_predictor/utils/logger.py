import logging
from logging.config import dictConfig

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

dictConfig(LOGGING_CONFIG)
log = logging.getLogger()
