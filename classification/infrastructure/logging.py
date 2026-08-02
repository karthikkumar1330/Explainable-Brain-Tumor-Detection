import logging
import os
import sys
from typing import Optional


def get_logger(
    name: str = "brain_tumor_classification",
    log_dir: str = "logs",
    log_filename: str = "classification.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configures and returns a logger that outputs to both console and a log file.

    Args:
        name: Name of the logger.
        log_dir: Directory where the log file will be saved.
        log_filename: Name of the log file.
        level: Logger level.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger is already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    try:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_filename)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # Fallback if log directory is unwritable
        print(f"Warning: Could not create log file: {e}", file=sys.stderr)

    return logger
