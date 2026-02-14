"""
Centralized Logging Configuration

All backend modules use this logger instead of print().
- Logs go to logs/docforge.log (file)
- Console stays clean — only final answers are printed by demo scripts
"""

import os
import logging
from pathlib import Path

# Create logs directory
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "docforge.log"


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger that writes to the log file.
    
    Usage:
        from backend.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Something happened")
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # File handler — all logs go here
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Don't propagate to root logger (prevents console output)
    logger.propagate = False

    return logger
