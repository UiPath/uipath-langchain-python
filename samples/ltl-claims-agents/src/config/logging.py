"""Logging configuration for LTL Claims Agent System."""

import logging
import logging.handlers
import sys
import os
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from structlog.types import FilteringBoundLogger

from .settings import settings


# Module-level flag to prevent re-initialization
_logging_configured = False


def configure_logging() -> FilteringBoundLogger:
    """
    Configure structured logging for the application.
    
    Sets up both console and file logging with JSON formatting support.
    Configures log rotation for file output.
    
    This function is idempotent - calling it multiple times will return
    the same configured logger without re-initializing handlers.
    
    Returns:
        Configured structlog logger instance
    """
    global _logging_configured
    
    # Return existing logger if already configured
    if _logging_configured:
        return structlog.get_logger()
    
    # Ensure log directory exists
    if settings.log_file_path:
        log_dir: Path = Path(settings.log_file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure standard library logging with handlers
    # Validate and get log level with fallback to INFO
    try:
        log_level: int = getattr(logging, settings.log_level.upper())
    except AttributeError:
        log_level = logging.INFO
        print(f"Warning: Invalid log level '{settings.log_level}', defaulting to INFO", file=sys.stderr)
    
    # Get or create application-specific logger instead of root logger
    # This prevents interference with third-party library logging
    app_logger = logging.getLogger("ltl_claims_agent")
    app_logger.setLevel(log_level)
    app_logger.propagate = False  # Don't propagate to root to avoid duplicates
    
    # Remove existing handlers from app logger only
    app_logger.handlers.clear()
    
    # Add console handler
    console_handler = _create_console_handler(log_level, settings.log_format)
    app_logger.addHandler(console_handler)
    
    # Add file handler if configured
    if settings.log_file_path:
        file_handler = _create_file_handler(
            settings.log_file_path, 
            log_level, 
            settings.log_format
        )
        if file_handler:
            app_logger.addHandler(file_handler)
            app_logger.info(f"File logging configured: {settings.log_file_path}")
        else:
            app_logger.warning(
                f"Continuing with console-only logging due to file handler creation failure"
            )
    
    # Configure structlog processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.format_exc_info,
    ]
    
    # Add JSON renderer for structured logs if configured
    if settings.log_format.lower() == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    logger = structlog.get_logger()
    logger.info(
        "Logging configured",
        log_level=settings.log_level,
        log_format=settings.log_format,
        log_file=settings.log_file_path if settings.log_file_path else "console only",
        debug_logging=settings.enable_debug_logging
    )
    
    # Mark as configured
    _logging_configured = True
    
    return logger


def _create_console_handler(log_level: int, log_format: str) -> logging.StreamHandler:
    """
    Create and configure console handler.
    
    Args:
        log_level: Logging level
        log_format: Format type ('json' or 'text')
        
    Returns:
        Configured console handler
    """
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    if log_format.lower() == "json":
        formatter = logging.Formatter('%(message)s')
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    console_handler.setFormatter(formatter)
    return console_handler


def _create_file_handler(
    log_file_path: str, 
    log_level: int, 
    log_format: str
) -> Optional[logging.handlers.RotatingFileHandler]:
    """
    Create and configure rotating file handler.
    
    Args:
        log_file_path: Path to log file
        log_level: Logging level
        log_format: Format type ('json' or 'text')
        
    Returns:
        Configured file handler or None if creation fails
    """
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        
        if log_format.lower() == "json":
            formatter = logging.Formatter('%(message)s')
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        
        file_handler.setFormatter(formatter)
        return file_handler
    except (OSError, IOError, PermissionError) as e:
        logging.warning(
            f"Failed to create file handler for {log_file_path}: {type(e).__name__}: {e}"
        )
        return None


def get_logger(name: str = __name__) -> FilteringBoundLogger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


def log_claim_processing_start(logger: FilteringBoundLogger, claim_id: str, **kwargs: Any) -> None:
    """Log the start of claim processing with context."""
    logger.info(
        "Starting claim processing",
        claim_id=claim_id,
        event="claim_processing_start",
        **kwargs
    )


def log_claim_processing_complete(
    logger: FilteringBoundLogger, 
    claim_id: str, 
    status: str, 
    duration_seconds: float,
    **kwargs: Any
) -> None:
    """Log the completion of claim processing."""
    logger.info(
        "Claim processing completed",
        claim_id=claim_id,
        final_status=status,
        duration_seconds=duration_seconds,
        event="claim_processing_complete",
        **kwargs
    )


def log_error_with_context(
    logger: FilteringBoundLogger,
    error: Exception,
    claim_id: str = None,
    context: Dict[str, Any] = None,
    **kwargs: Any
) -> None:
    """Log an error with full context information."""
    log_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "event": "error_occurred",
        **kwargs
    }
    
    if claim_id:
        log_data["claim_id"] = claim_id
    
    if context:
        log_data.update(context)
    
    logger.error("Error occurred during processing", **log_data)