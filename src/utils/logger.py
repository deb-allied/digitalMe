import logging
import os
import sys
from typing import Optional, Any, Dict
from datetime import datetime
from tqdm import tqdm
import threading
import time

from config.settings import settings

class ProgressAwareLogger:
    """
    Centralized logging utility with progress bar awareness and lazy formatting support
    """
    
    _instance: Optional['ProgressAwareLogger'] = None
    _logger: Optional[logging.Logger] = None
    _log_buffer: list = []
    _buffer_lock = threading.Lock()
    _progress_mode: bool = False
    
    def __new__(cls) -> 'ProgressAwareLogger':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._logger is None:
            self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Setup logging configuration with progress bar compatibility"""
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(settings.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Configure logger
        self._logger = logging.getLogger("personality_qa")
        self._logger.setLevel(getattr(logging, settings.log_level.upper()))
        
        # Avoid duplicate handlers
        if not self._logger.handlers:
            # File handler (always active)
            file_handler = logging.FileHandler(settings.log_file)
            file_handler.setLevel(getattr(logging, settings.log_level.upper()))
            
            # Console handler (progress-aware)
            console_handler = ProgressAwareStreamHandler()
            console_handler.setLevel(getattr(logging, settings.log_level.upper()))
            
            # Formatter with timestamp and progress info
            formatter = ProgressAwareFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self._logger.addHandler(file_handler)
            self._logger.addHandler(console_handler)
    
    def set_progress_mode(self, enabled: bool) -> None:
        """Enable or disable progress-aware logging mode"""
        with self._buffer_lock:
            self._progress_mode = enabled
            
            # Flush buffered logs when exiting progress mode
            if not enabled and self._log_buffer:
                if self._logger is None:
                    self._setup_logger()
                if self._logger is not None:
                    for log_record in self._log_buffer:
                        self._logger.handle(log_record)
                self._log_buffer.clear()
    
    def _log_with_progress_awareness(self, level: int, message: str, *args, **kwargs) -> None:
        """Log with progress bar awareness."""
        # Ensure logger is initialized
        if self._logger is None:
            self._setup_logger()
            if self._logger is None:
                return
            
        record = self._logger.makeRecord(
            name=self._logger.name,
            level=level,
            fn="",
            lno=0,
            msg=message,
            args=args,  # This should be the args tuple, not unpacked
            exc_info=kwargs.get('exc_info'),
            extra=kwargs.get('extra')
        )
        self._logger.handle(record)
        """Log message with progress bar awareness"""
        if args or kwargs:
            formatted_message = message % args if args else message.format(**kwargs)
        else:
            formatted_message = message
        
        # Ensure logger is initialized
        if self._logger is None:
            self._setup_logger()
        if self._logger is None:
            # If logger is still None, skip logging to avoid AttributeError
            return
        # Create log record
        record = self._logger.makeRecord(
            self._logger.name,
            level,
            __file__,
            0,
            formatted_message,
            args,
            None
        )
        
        with self._buffer_lock:
            if self._progress_mode:
                # Buffer the log when progress bars are active
                self._log_buffer.append(record)
                
                # For critical errors, break through progress bars
                if level >= logging.ERROR:
                    tqdm.write(f"🚨 {formatted_message}", file=sys.stderr)
            else:
                # Normal logging
                self._logger.handle(record)
    
    def info(self, message: str, *args, **kwargs) -> None:
        """Log info message with lazy formatting and progress awareness"""
        self._log_with_progress_awareness(logging.INFO, message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs) -> None:
        """Log error message with lazy formatting and progress awareness"""
        self._log_with_progress_awareness(logging.ERROR, message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message with lazy formatting and progress awareness"""
        self._log_with_progress_awareness(logging.WARNING, message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message with lazy formatting and progress awareness"""
        self._log_with_progress_awareness(logging.DEBUG, message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs) -> None:
        """Log critical message with lazy formatting and progress awareness"""
        self._log_with_progress_awareness(logging.CRITICAL, message, *args, **kwargs)
    
    def log_operation_start(self, operation: str, details: Optional[Dict[str, Any]] = None) -> str:
        """Log the start of an operation and return an operation ID"""
        operation_id = f"op_{int(time.time() * 1000)}"
        details_str = f" - {details}" if details else ""
        self.info("🚀 Starting operation: %s [ID: %s]%s", operation, operation_id, details_str)
        return operation_id
    
    def log_operation_end(
        self, 
        operation_id: str, 
        operation: str, 
        success: bool = True, 
        duration: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log the completion of an operation"""
        status = "✅ Completed" if success else "❌ Failed"
        duration_str = f" in {duration:.2f}s" if duration else ""
        details_str = f" - {details}" if details else ""
        
        self.info("%s operation: %s [ID: %s]%s%s", 
                 status, operation, operation_id, duration_str, details_str)
    
    def log_progress_milestone(self, milestone: str, current: int, total: int) -> None:
        """Log a progress milestone"""
        percentage = (current / total * 100) if total > 0 else 0
        self.info("📊 %s: %d/%d (%.1f%%)", milestone, current, total, percentage)
    
    def log_performance_metric(self, metric_name: str, value: float, unit: str = "") -> None:
        """Log a performance metric"""
        unit_str = f" {unit}" if unit else ""
        self.info("⚡ Performance - %s: %.3f%s", metric_name, value, unit_str)
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        with self._buffer_lock:
            return {
                "log_file": str(settings.log_file),
                "log_level": settings.log_level,
                "progress_mode": self._progress_mode,
                "buffered_logs": len(self._log_buffer),
                "log_file_exists": os.path.exists(settings.log_file),
                "log_file_size": os.path.getsize(settings.log_file) if os.path.exists(settings.log_file) else 0
            }

class ProgressAwareStreamHandler(logging.StreamHandler):
    """Custom stream handler that plays nicely with tqdm progress bars"""
    
    def emit(self, record):
        try:
            msg = self.format(record)
            # Use tqdm.write to avoid interfering with progress bars
            tqdm.write(msg, file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)

class ProgressAwareFormatter(logging.Formatter):
    """Custom formatter that adds progress context to log messages"""
    
    def format(self, record):
        # Add progress context if available
        progress_context = getattr(record, 'progress_context', None)
        if progress_context is not None:
            record.msg = f"[{progress_context}] {record.msg}"
        
        return super().format(record)

class LoggingProgressBar(tqdm):
    """Custom progress bar that integrates with logging system"""
    
    def __init__(self, *args, **kwargs):
        # Extract logging-specific parameters
        self.log_milestones = kwargs.pop('log_milestones', True)
        self.milestone_interval = kwargs.pop('milestone_interval', 10)  # Log every 10%
        self.operation_name = kwargs.pop('operation_name', 'Operation')
        
        super().__init__(*args, **kwargs)
        
        self.logger = logger
        self.last_milestone = 0
        
        # Set progress mode on logger
        self.logger.set_progress_mode(True)
        
        # Log operation start
        self.operation_id = self.logger.log_operation_start(
            self.operation_name, 
            {"total": self.total, "unit": self.unit}
        )
    
    def update(self, n=1):
        super().update(n)
        
        if self.log_milestones and self.total:
            progress_percentage = (self.n / self.total) * 100
            milestone = int(progress_percentage // self.milestone_interval) * self.milestone_interval
            
            if milestone > self.last_milestone and milestone <= 100:
                self.logger.log_progress_milestone(
                    self.operation_name, self.n, self.total
                )
                self.last_milestone = milestone
    
    def close(self):
        # Log operation completion
        if hasattr(self, 'operation_id'):
            success = self.n >= self.total if self.total else True
            self.logger.log_operation_end(
                self.operation_id,
                self.operation_name,
                success=success,
                details={"final_count": self.n}
            )
        
        # Restore normal logging mode
        self.logger.set_progress_mode(False)
        
        super().close()

# Progress-aware context manager
class progress_logging_context:
    """Context manager for progress-aware logging operations"""
    
    def __init__(self, operation_name: str, details: Optional[Dict[str, Any]] = None):
        self.operation_name = operation_name
        self.details = details or {}
        self.start_time = None
        self.operation_id = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.operation_id = logger.log_operation_start(self.operation_name, self.details)
        logger.set_progress_mode(True)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else None
        success = exc_type is None
        
        logger.log_operation_end(
            self.operation_id if self.operation_id is not None else "",
            self.operation_name,
            success=success,
            duration=duration,
            details={"exception": str(exc_val) if exc_val else None}
        )
        
        logger.set_progress_mode(False)
        
        # Don't suppress exceptions
        return False

# Initialize the global logger instance
logger = ProgressAwareLogger()
