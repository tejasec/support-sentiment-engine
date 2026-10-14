"""
Centralized logging utility for nlp_engine.
"""

import logging
import sys
from typing import Optional


def get_logger(name: str = "nlp_engine", level: Optional[int] = None) -> logging.Logger:
    """Return a configured logger with console handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level if level is not None else logging.INFO)
    return logger
