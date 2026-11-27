"""
Advanced logging configuration and utilities for LTL Claims Agent System.

Provides structured logging, performance metrics tracking, and debug logging capabilities.
"""

import logging
import logging.handlers
import sys
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone
from contextlib import contextmanager

import structlog
from structlog.types import FilteringBoundLogger

try:
    from ..config.settings import settings
except ImportError:
    from config.settings import settings


class PerformanceMetrics:
    """Track performance metrics for logging."""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {
            "processing_duration": 0.0,
            "recursion_steps": 0,
            "tool_executions": 0,
            "api_calls": 0,
            "memory_queries": 0,
            "document_downloads": 0,
            "document_extractions": 0,
            "queue_operations": 0
        }
    
    def increment(self, metric_name: str, value: int = 1) -> None:
        """Increment a metric counter."""
        if metric_name in self.metrics:
            self.metrics[metric_name] += value
    
    def set_duration(self) -> None:
        """Calculate and set processing duration."""
        self.metrics["processing_duration"] = time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        self.set_duration()
        return self.metrics.copy()


class StructuredLogger:
    """Enhanced structured logger with context management."""
    
    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
        self.context = {}
    
    def bind(self, **kwargs) -> 'StructuredLogger':
        """Bind context to logger."""
        self.context.update(kwargs)
        return self
    
    def unbind(self, *keys) -> 'StructuredLogger':
        """Remove context from logger."""
        for key in keys:
            self.context.pop(key, None)
        return self
    
    def _log(self, level: str, message: str, **kwargs) -> None:
        """Internal logging method with context."""
        log_data = {**self.context, **kwargs}
        getattr(self.logger, level)(message, **log_data)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log("debug", message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log("info", message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log("warning", message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._log("error", message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._log("critical", message, **kwargs)


def setup_structured_logging() -> FilteringBoundLogger:
    """
    Set up comprehensive structured logging with file output and JSON formatting.
    
    Returns:
        Configured structlog logger
    """
    # Ensure log directory exists
    if settings.log_file_path:
        log_dir = Path(settings.log_file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure log level
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    # Override with debug if enabled
    if settings.enable_debug_logging:
        log_level = logging.DEBUG
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    if settings.log_format.lower() == "json":
        console_formatter = logging.Formatter('%(message)s')
    else:
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if settings.log_file_path:
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                settings.log_file_path,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(log_level)
            
            if settings.log_format.lower() == "json":
                file_formatter = logging.Formatter('%(message)s')
            else:
                file_formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                )
            
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            logging.warning(f"Failed to configure file logging: {e}")
    
    # Configure structlog processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.format_exc_info,
    ]
    
    if settings.log_format.lower() == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    logger = structlog.get_logger()
    logger.info(
        "Structured logging configured",
        log_level=settings.log_level,
        log_format=settings.log_format,
        log_file=settings.log_file_path if settings.log_file_path else "console only",
        debug_enabled=settings.enable_debug_logging
    )
    
    return logger


def get_structured_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger instance with context management.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name)


@contextmanager
def log_operation(logger: FilteringBoundLogger, operation: str, **context):
    """
    Context manager for logging operations with timing.
    
    Args:
        logger: Logger instance
        operation: Operation name
        **context: Additional context to log
    """
    start_time = time.time()
    logger.info(f"Starting {operation}", operation=operation, **context)
    
    try:
        yield
        duration = time.time() - start_time
        logger.info(
            f"Completed {operation}",
            operation=operation,
            duration_seconds=duration,
            status="success",
            **context
        )
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Failed {operation}",
            operation=operation,
            duration_seconds=duration,
            status="failed",
            error_type=type(e).__name__,
            error_message=str(e),
            **context
        )
        raise


def log_configuration_at_startup(logger: FilteringBoundLogger) -> None:
    """
    Log configuration values at startup (excluding sensitive credentials).
    
    Args:
        logger: Logger instance
    """
    if not settings.enable_debug_logging:
        return
    
    config_summary = {
        "uipath_base_url": settings.effective_base_url,
        "uipath_tenant": settings.effective_tenant,
        "uipath_organization": settings.effective_organization,
        "queue_name": settings.effective_queue_name,
        "use_queue_input": settings.use_queue_input,
        "max_recursion_depth": settings.max_recursion_depth,
        "confidence_threshold": settings.confidence_threshold,
        "processing_timeout": settings.processing_timeout,
        "enable_long_term_memory": settings.enable_long_term_memory,
        "memory_store_type": settings.memory_store_type if settings.enable_long_term_memory else "disabled",
        "api_timeout": settings.api_timeout,
        "document_extraction_timeout": settings.document_extraction_timeout,
        "log_level": settings.log_level,
        "log_format": settings.log_format,
        "debug_mode": settings.debug_mode
    }
    
    logger.debug(
        "Configuration loaded",
        event="startup_configuration",
        **config_summary
    )


def log_api_request(
    logger: FilteringBoundLogger,
    service: str,
    operation: str,
    request_data: Optional[Dict[str, Any]] = None,
    **context
) -> None:
    """
    Log API request details when debug logging is enabled.
    
    Args:
        logger: Logger instance
        service: Service name (e.g., "UiPath SDK", "Data Fabric")
        operation: Operation name
        request_data: Request data (will be sanitized)
        **context: Additional context
    """
    if not settings.enable_debug_logging:
        return
    
    # Sanitize sensitive data
    sanitized_data = _sanitize_sensitive_data(request_data) if request_data else None
    
    logger.debug(
        f"API request: {service}.{operation}",
        event="api_request",
        service=service,
        operation=operation,
        request_data=sanitized_data,
        **context
    )


def log_api_response(
    logger: FilteringBoundLogger,
    service: str,
    operation: str,
    response_data: Optional[Dict[str, Any]] = None,
    duration_seconds: Optional[float] = None,
    **context
) -> None:
    """
    Log API response details when debug logging is enabled.
    
    Args:
        logger: Logger instance
        service: Service name
        operation: Operation name
        response_data: Response data (will be sanitized)
        duration_seconds: Request duration
        **context: Additional context
    """
    if not settings.enable_debug_logging:
        return
    
    # Sanitize sensitive data
    sanitized_data = _sanitize_sensitive_data(response_data) if response_data else None
    
    logger.debug(
        f"API response: {service}.{operation}",
        event="api_response",
        service=service,
        operation=operation,
        response_data=sanitized_data,
        duration_seconds=duration_seconds,
        **context
    )


def _sanitize_sensitive_data(data: Any) -> Any:
    """
    Sanitize sensitive data from logs.
    
    Args:
        data: Data to sanitize
        
    Returns:
        Sanitized data
    """
    if isinstance(data, dict):
        sanitized = {}
        sensitive_keys = {
            'password', 'secret', 'token', 'api_key', 'access_token',
            'client_secret', 'authorization', 'credential'
        }
        
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, (dict, list)):
                sanitized[key] = _sanitize_sensitive_data(value)
            else:
                sanitized[key] = value
        
        return sanitized
    elif isinstance(data, list):
        return [_sanitize_sensitive_data(item) for item in data]
    else:
        return data


__all__ = [
    "PerformanceMetrics",
    "StructuredLogger",
    "setup_structured_logging",
    "get_structured_logger",
    "log_operation",
    "log_configuration_at_startup",
    "log_api_request",
    "log_api_response"
]
