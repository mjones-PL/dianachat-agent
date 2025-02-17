"""Logging configuration for DianaChat Agent."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging(log_level=logging.DEBUG):
    """Set up logging configuration."""
    # Create logger
    logger = logging.getLogger("dianachat_agent")
    
    # If logger already has handlers, assume it's configured
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(log_level)
    logger.propagate = False  # Prevent duplicate logs

    # Create logs directory if it doesn't exist
    project_root = Path(__file__).parent.parent.parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Create console handler with a higher log level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # Create rotating file handlers
    # Main log file - includes all levels
    main_log = RotatingFileHandler(
        logs_dir / "dianachat_agent.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    main_log.setLevel(log_level)

    # Error log file - only ERROR and CRITICAL
    error_log = RotatingFileHandler(
        logs_dir / "error.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    error_log.setLevel(logging.ERROR)

    # Create formatters and add them to the handlers
    console_formatter = logging.Formatter(
        "\033[1m%(levelname)s\033[0m [%(name)s] %(message)s"
    )
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
    )

    # Set formatters
    console_handler.setFormatter(console_formatter)
    main_log.setFormatter(file_formatter)
    error_log.setFormatter(file_formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(main_log)
    logger.addHandler(error_log)

    return logger

# Create and configure logger
logger = setup_logging()
