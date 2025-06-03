import logging
import os
from typing import Optional, Any
from config.settings import settings

class Logger:
    """Centralized logging utility with lazy formatting support"""
    
    _instance: Optional['Logger'] = None
    _logger: logging.Logger
    
    def __new__(cls) -> 'Logger':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._logger = cls._instance._setup_logger()
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize logger if not already initialized"""
        pass
    
    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(settings.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Configure logger
        logger = logging.getLogger("personality_qa")
        logger.setLevel(getattr(logging, settings.log_level.upper()))
        
        # Avoid duplicate handlers
        if not logger.handlers:
            # File handler
            file_handler = logging.FileHandler(settings.log_file)
            file_handler.setLevel(getattr(logging, settings.log_level.upper()))
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, settings.log_level.upper()))
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log info message with lazy formatting"""
        if args or kwargs:
            self._logger.info(message, *args, **kwargs)
        else:
            self._logger.info(message)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log error message with lazy formatting"""
        if args or kwargs:
            self._logger.error(message, *args, **kwargs)
        else:
            self._logger.error(message)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log warning message with lazy formatting"""
        if args or kwargs:
            self._logger.warning(message, *args, **kwargs)
        else:
            self._logger.warning(message)

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log debug message with lazy formatting"""
        if args or kwargs:
            self._logger.debug(message, *args, **kwargs)
        else:
            self._logger.debug(message)

# Create singleton instance
logger = Logger()