"""
Error hierarchy for LTL Claims Agent System.

Defines a comprehensive error hierarchy with context and details for better
error handling, logging, and debugging throughout the agent system.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timezone


class AgentError(Exception):
    """
    Base exception for all agent-related errors.
    
    Provides structured error information including context, details,
    and timestamps for comprehensive error tracking and debugging.
    
    Attributes:
        message: Human-readable error message
        context: Additional context about where/when the error occurred
        details: Detailed error information (stack traces, data, etc.)
        timestamp: When the error occurred
        claim_id: Optional claim ID if error is claim-specific
        operation: Optional operation name that failed
    """
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        claim_id: Optional[str] = None,
        operation: Optional[str] = None
    ):
        """
        Initialize AgentError with comprehensive error information.
        
        Args:
            message: Human-readable error message
            context: Additional context (e.g., {"phase": "document_extraction", "step": 3})
            details: Detailed error information (e.g., {"error_code": "TIMEOUT", "duration": 120})
            claim_id: Optional claim ID if error is claim-specific
            operation: Optional operation name that failed
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.details = details or {}
        self.timestamp = datetime.now(timezone.utc)
        self.claim_id = claim_id
        self.operation = operation
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error to dictionary for logging and serialization.
        
        Returns:
            Dictionary containing all error information
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "claim_id": self.claim_id,
            "operation": self.operation
        }
    
    def __str__(self) -> str:
        """String representation with context."""
        parts = [self.message]
        if self.claim_id:
            parts.append(f"[Claim: {self.claim_id}]")
        if self.operation:
            parts.append(f"[Operation: {self.operation}]")
        if self.context:
            parts.append(f"Context: {self.context}")
        return " ".join(parts)


class InputError(AgentError):
    """
    Exception raised for input data validation or retrieval errors.
    
    Used when:
    - Queue item retrieval fails
    - File input cannot be read or parsed
    - Input data validation fails
    - Required input fields are missing
    - Input data format is invalid
    
    Example:
        raise InputError(
            "Invalid claim input: missing required field 'ClaimAmount'",
            context={"source": "queue", "queue_name": "LTL Claims Processing"},
            details={"missing_fields": ["ClaimAmount"], "received_fields": ["ClaimId", "ClaimType"]},
            claim_id="ABC-123"
        )
    """
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        claim_id: Optional[str] = None,
        operation: Optional[str] = None,
        input_source: Optional[str] = None
    ):
        """
        Initialize InputError with input-specific information.
        
        Args:
            message: Human-readable error message
            context: Additional context about the input error
            details: Detailed error information
            claim_id: Optional claim ID if error is claim-specific
            operation: Optional operation name that failed
            input_source: Source of the input (e.g., "queue", "file", "api")
        """
        super().__init__(message, context, details, claim_id, operation)
        self.input_source = input_source
        if input_source:
            self.context["input_source"] = input_source


class ProcessingError(AgentError):
    """
    Exception raised during claim processing operations.
    
    Used when:
    - Document extraction fails
    - Data validation fails
    - Tool execution fails
    - API calls fail
    - Business logic errors occur
    - Processing cannot continue
    
    Example:
        raise ProcessingError(
            "Document extraction failed: timeout after 120 seconds",
            context={"phase": "document_extraction", "document_type": "BOL"},
            details={"timeout_seconds": 120, "documents_processed": 2, "documents_failed": 1},
            claim_id="ABC-123",
            operation="extract_documents_batch"
        )
    """
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        claim_id: Optional[str] = None,
        operation: Optional[str] = None,
        phase: Optional[str] = None,
        recoverable: bool = False
    ):
        """
        Initialize ProcessingError with processing-specific information.
        
        Args:
            message: Human-readable error message
            context: Additional context about the processing error
            details: Detailed error information
            claim_id: Optional claim ID if error is claim-specific
            operation: Optional operation name that failed
            phase: Processing phase where error occurred (e.g., "initialization", "extraction")
            recoverable: Whether the error is recoverable with retry
        """
        super().__init__(message, context, details, claim_id, operation)
        self.phase = phase
        self.recoverable = recoverable
        if phase:
            self.context["phase"] = phase
        self.context["recoverable"] = recoverable


class RecursionLimitError(AgentError):
    """
    Exception raised when recursion limit is exceeded.
    
    Used when:
    - Agent reasoning cycles exceed max_recursion_depth
    - Infinite loop is detected
    - Processing must be terminated due to step limit
    
    This is a special error that triggers forced finalization
    rather than complete failure.
    
    Example:
        raise RecursionLimitError(
            "Recursion limit reached: 20 steps completed",
            context={"max_depth": 20, "current_step": 20},
            details={
                "reasoning_steps": 20,
                "tool_calls": 15,
                "last_action": "validate_claim_data",
                "confidence": 0.65
            },
            claim_id="ABC-123",
            operation="reasoning_cycle"
        )
    """
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        claim_id: Optional[str] = None,
        operation: Optional[str] = None,
        current_step: Optional[int] = None,
        max_depth: Optional[int] = None
    ):
        """
        Initialize RecursionLimitError with recursion-specific information.
        
        Args:
            message: Human-readable error message
            context: Additional context about the recursion limit
            details: Detailed error information including metrics
            claim_id: Optional claim ID if error is claim-specific
            operation: Optional operation name that failed
            current_step: Current recursion step when limit was reached
            max_depth: Maximum allowed recursion depth
        """
        super().__init__(message, context, details, claim_id, operation)
        self.current_step = current_step
        self.max_depth = max_depth
        if current_step is not None:
            self.context["current_step"] = current_step
        if max_depth is not None:
            self.context["max_depth"] = max_depth


class UiPathServiceError(AgentError):
    """
    Exception raised for UiPath service errors.
    
    Used when:
    - UiPath SDK operations fail
    - Authentication fails
    - API calls to UiPath services fail
    - Connection issues occur
    - Service timeouts occur
    
    Example:
        raise UiPathServiceError(
            "Failed to authenticate with UiPath: Invalid credentials",
            context={"service": "authentication", "base_url": "https://cloud.uipath.com"},
            details={"error_code": "AUTH_FAILED", "status_code": 401},
            operation="authenticate"
        )
    """
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        claim_id: Optional[str] = None,
        operation: Optional[str] = None,
        service_name: Optional[str] = None,
        status_code: Optional[int] = None
    ):
        """
        Initialize UiPathServiceError with service-specific information.
        
        Args:
            message: Human-readable error message
            context: Additional context about the service error
            details: Detailed error information
            claim_id: Optional claim ID if error is claim-specific
            operation: Optional operation name that failed
            service_name: Name of the UiPath service that failed
            status_code: HTTP status code if applicable
        """
        super().__init__(message, context, details, claim_id, operation)
        self.service_name = service_name
        self.status_code = status_code
        if service_name:
            self.context["service_name"] = service_name
        if status_code:
            self.context["status_code"] = status_code


__all__ = [
    "AgentError",
    "InputError",
    "ProcessingError",
    "RecursionLimitError",
    "UiPathServiceError"
]
