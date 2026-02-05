"""Logging configuration for pyheatcontrol."""

import logging
import sys

# Create the package logger
logger = logging.getLogger("pyheatcontrol")


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """
    Configure logging for pyheatcontrol.

    Args:
        verbose: If True, show INFO level messages (setup, config, stats)
        debug: If True, show DEBUG level messages (all internal details)

    Levels:
        - Default: WARNING only (errors + essential progress via print)
        - verbose: INFO (setup, configuration, zone statistics)
        - debug: DEBUG (array values, internal checks, everything)
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    # Configure handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Format: simple for INFO, detailed for DEBUG
    if debug:
        fmt = "[%(levelname)s] %(name)s: %(message)s"
    else:
        fmt = "%(message)s"

    handler.setFormatter(logging.Formatter(fmt))

    # Apply to our logger
    logger.setLevel(level)
    logger.handlers.clear()  # Remove any existing handlers
    logger.addHandler(handler)
    logger.propagate = False  # Don't pass to root logger
