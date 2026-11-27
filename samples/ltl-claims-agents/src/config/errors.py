"""
Configuration-specific errors.

This module is separate from utils.errors to avoid circular imports,
as settings.py needs to import ConfigurationError.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import traceback


class ConfigurationError(Exception):
    """
    Exception raised for configuration validation errors.
    
    Used when:
    - Required environment variables are missing
    - Configuration values are invalid or out of range
    - Configuration file cannot be loaded
    - Settings validation fails
    
    This error is typically raised during application startup
    and should be treated as a fatal error that prevents the
    agent from running.
    
    Example:
        raise ConfigurationError(
            "UIPATH_PAT_ACCESS_TOKEN is required and cannot be empty",
            context={"env_file": ".env", "missing_vars": ["UIPATH_PAT_ACCESS_TOKEN"]},
            details={"validation_phase": "authentication"}
        )
    """
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        missing_fields: Optional[List[str]] = None
    ):
        """
        Initialize ConfigurationError with configuration-specific information.
        
        Args:
            message: Human-readable error message
            context: Additional context about the configuration error
            details: Detailed error information
            missing_fields: List of missing configuration fields
        """
        super().__init__(message)
        self.message = message
        # Defensive copies to prevent external mutation
        self.context = dict(context) if context else {}
        self.details = dict(details) if details else {}
        self.timestamp = datetime.now(timezone.utc)
        self.missing_fields = list(missing_fields) if missing_fields else []
        if self.missing_fields:
            self.details["missing_fields"] = self.missing_fields
    
    def to_dict(self, include_traceback: bool = False) -> Dict[str, Any]:
        """
        Convert error to dictionary for logging and serialization.
        
        Args:
            include_traceback: Whether to include stack trace information
        
        Returns:
            Dictionary containing all error information
        """
        result = {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "missing_fields": self.missing_fields
        }
        
        if include_traceback:
            result["traceback"] = traceback.format_exc()
        
        return result
    
    def __str__(self) -> str:
        """String representation with context."""
        parts = [self.message]
        if self.context:
            parts.append(f"Context: {self.context}")
        if self.missing_fields:
            parts.append(f"Missing fields: {', '.join(self.missing_fields)}")
        return " ".join(parts)
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"missing_fields={self.missing_fields!r}, "
            f"timestamp={self.timestamp.isoformat()!r})"
        )


__all__ = ["ConfigurationError"]
