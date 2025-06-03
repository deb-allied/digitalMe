import logging
import sys
from typing import Optional

from config.settings import config


class LoggerSetup:
    """Centralized logger setup."""
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get or create a logger with the specified name."""
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, config.logging.log_level))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, config.logging.log_level))
        
        # File handler (optional)
        if config.logging.log_file:
            file_handler = logging.FileHandler(config.logging.log_file)
            file_handler.setLevel(getattr(logging, config.logging.log_level))
            logger.addHandler(file_handler)
        
        # Formatter
        formatter = logging.Formatter(config.logging.log_format)
        console_handler.setFormatter(formatter)
        if config.logging.log_file:
            file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        cls._loggers[name] = logger
        
        return logger