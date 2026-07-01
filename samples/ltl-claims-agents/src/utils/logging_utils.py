"""Logging utilities with PII redaction and structured logging support."""

import re
import logging
from typing import Any, Dict, Optional
from datetime import datetime


# Compile regex patterns once at module level for performance
_EMAIL_PATTERN = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
)
_PHONE_PATTERN = re.compile(
    r'\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b'
)
_NAME_FIELD_PATTERN = re.compile(
    r'(["\']?(?:customer_name|CustomerName|name|Name)["\']?\s*[:=]\s*["\'])([^"\']+)(["\'])',
    flags=re.IGNORECASE
)
_EMAIL_FIELD_PATTERN = re.compile(
    r'(["\']?(?:customer_email|CustomerEmail|email|Email)["\']?\s*[:=]\s*["\'])([^"\']+)(["\'])',
    flags=re.IGNORECASE
)
_PHONE_FIELD_PATTERN = re.compile(
    r'(["\']?(?:customer_phone|CustomerPhone|phone|Phone)["\']?\s*[:=]\s*["\'])([^"\']+)(["\'])',
    flags=re.IGNORECASE
)


def redact_pii(text: str) -> str:
    """
    Redact personally identifiable information from text.
    
    Uses pre-compiled regex patterns for better performance.
    
    Args:
        text: Text that may contain PII
        
    Returns:
        Text with PII redacted
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Use pre-compiled patterns for better performance
    text = _EMAIL_PATTERN.sub('[EMAIL_REDACTED]', text)
    text = _PHONE_PATTERN.sub('[PHONE_REDACTED]', text)
    text = _NAME_FIELD_PATTERN.sub(r'\1[NAME_REDACTED]\3', text)
    text = _EMAIL_FIELD_PATTERN.sub(r'\1[EMAIL_REDACTED]\3', text)
    text = _PHONE_FIELD_PATTERN.sub(r'\1[PHONE_REDACTED]\3', text)
    
    return text


class PIIRedactingFormatter(logging.Formatter):
    """Custom logging formatter that redacts PII from log messages."""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with PII redaction.
        
        Simple formatting without complex record manipulation.
        """
        # Redact PII from message (only if it's a string)
        if isinstance(record.msg, str):
            record.msg = redact_pii(record.msg)
        
        # Redact PII from args
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: redact_pii(v) if isinstance(v, str) else v 
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    redact_pii(arg) if isinstance(arg, str) else arg 
                    for arg in record.args
                )
        
        return super().format(record)


def log_sdk_operation_error(
    operation: str,
    error: Exception,
    claim_id: Optional[str] = None,
    entity_key: Optional[str] = None,
    additional_details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Log SDK operation errors with structured context.
    
    Args:
        operation: Name of the operation that failed
        error: The exception that occurred
        claim_id: Optional claim ID for context
        entity_key: Optional entity key for context
        additional_details: Optional additional context
        
    Returns:
        Dictionary with error details
    """
    logger = logging.getLogger(__name__)
    
    error_details = {
        "operation": operation,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": datetime.now().isoformat()
    }
    
    if claim_id:
        error_details["claim_id"] = claim_id
    
    if entity_key:
        error_details["entity_key"] = entity_key
    
    if additional_details:
        error_details["additional_details"] = additional_details
    
    # Log with structured context
    logger.error(
        f"SDK operation failed: {operation}",
        extra={"error_details": error_details},
        exc_info=True
    )
    
    return error_details


def configure_logging_with_pii_redaction(level: int = logging.INFO) -> None:
    """
    Configure basic logging with PII redaction.
    
    Uses simple text-based logging without JSON complexity.
    
    Args:
        level: Logging level (default: INFO)
    """
    # Create simple formatter with PII redaction
    formatter = PIIRedactingFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add simple console handler with PII redaction
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Disable propagation for noisy libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    claim_id: Optional[str] = None,
    **kwargs
) -> None:
    """
    Log message with structured context.
    
    Note: PII redaction is handled by PIIRedactingFormatter if configured.
    This function only structures the context for logging.
    
    Args:
        logger: Logger instance
        level: Logging level
        message: Log message
        claim_id: Optional claim ID for context
        **kwargs: Additional context fields
    """
    context = {}
    
    if claim_id:
        context["claim_id"] = claim_id
    
    context.update(kwargs)
    
    # PII redaction is handled by the formatter, not here
    # This avoids double-redaction and improves performance
    logger.log(level, message, extra=context)
