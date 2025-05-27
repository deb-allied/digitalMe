import logging
import os
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Constants from .env or defaults
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE_PATH = os.getenv("LOG_FILE", "logs/app.log")
LOG_DIR = Path(LOG_FILE_PATH).parent
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class LoggerService:
    def __init__(self, name: str) -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
        self._configure_logger()

    def _configure_logger(self) -> None:
        if not self.logger.handlers:
            LOG_DIR.mkdir(parents=True, exist_ok=True)

            # Console Handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
            self.logger.addHandler(console_handler)

            # Rotating File Handler
            file_handler = RotatingFileHandler(
                LOG_FILE_PATH, maxBytes=5 * 1024 * 1024, backupCount=5
            )
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
            self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        return self.logger
