import logging
from logging.config import dictConfig

from definitions import LOGGING_CONFIG

dictConfig(LOGGING_CONFIG)
log = logging.getLogger()
