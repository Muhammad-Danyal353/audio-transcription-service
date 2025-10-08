"""
Structured logging configuration for the audio transcription service.

Provides JSON-formatted logs with request tracing and performance metrics.
"""

import logging
import sys
import json
import time
import functools
from typing import Any, Dict, Optional, Callable
from datetime import datetime
from uuid import uuid4
from contextvars import ContextVar

from ..core.config import settings


# Context variable for request ID propagation
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.

    Outputs logs in JSON format with consistent structure for better observability.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add request ID if available
        request_id = request_id_var.get()
        if request_id:
            log_data["request_id"] = request_id

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                log_data[key] = value

        return json.dumps(log_data, default=str)


class TextFormatter(logging.Formatter):
    """Simple text formatter for development."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as readable text."""
        request_id = request_id_var.get()
        request_str = f"[{request_id[:8]}] " if request_id else ""

        return (
            f"{datetime.utcnow().isoformat()} | "
            f"{record.levelname:8} | "
            f"{request_str}"
            f"{record.name} | "
            f"{record.getMessage()}"
        )


def setup_logging(
    log_level: Optional[str] = None, log_format: Optional[str] = None
) -> None:
    """
    Configure application logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format (json or text)
    """
    log_level = log_level or settings.log_level
    log_format = log_format or settings.log_format

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Set formatter based on format type
    if log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter()

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Set third-party library log levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("whisper").setLevel(logging.WARNING)
    logging.getLogger("demucs").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def generate_request_id() -> str:
    """
    Generate a unique request ID for tracing.

    Returns:
        UUID string for request identification
    """
    return str(uuid4())


def set_request_id(request_id: str) -> None:
    """
    Set the current request ID in context.

    Args:
        request_id: Request ID to set
    """
    request_id_var.set(request_id)


def get_request_id() -> Optional[str]:
    """
    Get the current request ID from context.

    Returns:
        Current request ID or None
    """
    return request_id_var.get()


def log_execution_time(func: Optional[Callable] = None, *, name: Optional[str] = None):
    """
    Decorator to log function execution time.

    Args:
        func: Function to decorate
        name: Optional name for the operation

    Returns:
        Decorated function
    """

    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs) -> Any:
            logger = get_logger(f.__module__)
            operation = name or f.__name__

            start_time = time.perf_counter()
            logger.info(f"Starting {operation}", extra={"operation": operation})

            try:
                result = f(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.info(
                    f"Completed {operation}",
                    extra={
                        "operation": operation,
                        "duration_ms": round(elapsed_ms, 2),
                        "status": "success",
                    },
                )
                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.error(
                    f"Failed {operation}: {str(e)}",
                    extra={
                        "operation": operation,
                        "duration_ms": round(elapsed_ms, 2),
                        "status": "error",
                        "error": str(e),
                    },
                    exc_info=True,
                )
                raise

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


class LogContext:
    """Context manager for adding extra fields to logs."""

    def __init__(self, **kwargs):
        """Initialize with extra log fields."""
        self.extras = kwargs
        self.logger = get_logger(__name__)

    def __enter__(self):
        """Enter context and log with extras."""
        for key, value in self.extras.items():
            self.logger = logging.LoggerAdapter(self.logger, {key: value})
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        pass


# Initialize logging on module import
setup_logging()
