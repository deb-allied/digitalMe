import logging
import sys
from pathlib import Path

import structlog
from structlog.stdlib import LoggerFactory

from .config import settings


def setup_logging():
    """Configure structured logging with lazy formatting."""
    # Create logs directory if it doesn't exist
    log_file = Path(settings.logging.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.logging.format == "json" else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.logging.level.upper()),
    )
    
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, settings.logging.level.upper()))
    logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


# Initialize logging
setup_logging()