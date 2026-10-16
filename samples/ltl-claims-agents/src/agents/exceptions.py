"""
Custom exceptions for agent operations.

This module defines a hierarchy of exceptions used throughout the LTL Claims
processing agent system. All exceptions inherit from OrchestratorError and
provide contextual information for debugging and error handling.

Exception Hierarchy:
    OrchestratorError (base)
    ├── PlanGenerationError
    ├── ReflectionError
    ├── ReplanningError
    └── StepExecutionError

Usage Example:
    ```python
    from src.agents.exceptions import StepExecutionError
    
    try:
        result = await execute_step(state)
    except Exception as e:
        raise StepExecutionError(
            message="Failed to download documents",
            step_name="download_documents",
            claim_id=state.claim_id,
            original_error=e,
            is_recoverable=True,
            context={"document_count": len(state.shipping_documents)}
        ) from e
    ```
"""

from typing import Optional, Dict, Any, List
from datetime import datetime


class OrchestratorError(Exception):
    """
    Base exception for orchestrator errors.
    
    Attributes:
        message: Human-readable error message
        claim_id: Optional claim ID for context
        step_name: Optional step name where error occurred
        context: Additional context data
        timestamp: When the error occurred
        cause: Original exception that caused this error
    """
    
    def __init__(
        self,
        message: str,
        claim_id: Optional[str] = None,
        step_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.message = message
        self.claim_id = claim_id
        self.step_name = step_name
        self.context = context or {}
        self.timestamp = datetime.now()
        self.cause = cause
        
        # Build detailed error message
        error_parts = [message]
        if claim_id:
            error_parts.append(f"Claim ID: {claim_id}")
        if step_name:
            error_parts.append(f"Step: {step_name}")
        
        super().__init__(" | ".join(error_parts))
        
        # Preserve exception chain
        if cause:
            self.__cause__ = cause
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "claim_id": self.claim_id,
            "step_name": self.step_name,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None
        }
    
    def is_critical(self) -> bool:
        """
        Determine if this error is critical and should halt processing.
        Override in subclasses for specific behavior.
        """
        return False
    
    def should_retry(self) -> bool:
        """
        Determine if the operation should be retried.
        Override in subclasses for specific behavior.
        """
        return True
    
    def get_recovery_action(self) -> str:
        """
        Get recommended recovery action.
        Returns: 'retry', 'skip', 'escalate', or 'abort'
        """
        if self.is_critical():
            return 'abort'
        elif self.should_retry():
            return 'retry'
        else:
            return 'skip'


class PlanGenerationError(OrchestratorError):
    """
    Raised when plan generation fails.
    
    Additional Attributes:
        llm_error: Original LLM error if applicable
        retry_count: Number of retries attempted
    """
    
    def __init__(
        self,
        message: str,
        claim_id: Optional[str] = None,
        llm_error: Optional[Exception] = None,
        retry_count: int = 0,
        **kwargs
    ):
        super().__init__(message, claim_id=claim_id, cause=llm_error, **kwargs)
        self.llm_error = llm_error
        self.retry_count = retry_count
    
    def should_retry(self) -> bool:
        """Plan generation can be retried up to 2 times."""
        return self.retry_count < 2


class ReflectionError(OrchestratorError):
    """
    Raised when reflection process fails.
    
    Additional Attributes:
        completed_steps: Steps completed before reflection failed
        observations: Observations gathered before failure
    """
    
    def __init__(
        self,
        message: str,
        claim_id: Optional[str] = None,
        completed_steps: Optional[List[str]] = None,
        observations: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        super().__init__(message, claim_id=claim_id, **kwargs)
        self.completed_steps = completed_steps or []
        self.observations = observations or []
    
    def is_critical(self) -> bool:
        """Reflection errors are not critical - processing can continue."""
        return False


class ReplanningError(OrchestratorError):
    """
    Raised when replanning fails.
    
    Additional Attributes:
        original_plan: The plan that was being replaced
        reflection_data: Reflection results that triggered replanning
    """
    
    def __init__(
        self,
        message: str,
        claim_id: Optional[str] = None,
        original_plan: Optional[List[str]] = None,
        reflection_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(message, claim_id=claim_id, **kwargs)
        self.original_plan = original_plan or []
        self.reflection_data = reflection_data or {}
    
    def should_retry(self) -> bool:
        """Replanning should not be retried - use fallback plan instead."""
        return False


class StepExecutionError(OrchestratorError):
    """
    Raised when a step execution fails.
    
    Additional Attributes:
        original_error: The underlying exception that caused the failure
        is_recoverable: Whether the error can be recovered from
        retry_count: Number of retries attempted
    """
    
    def __init__(
        self,
        message: str,
        step_name: str,
        claim_id: Optional[str] = None,
        original_error: Optional[Exception] = None,
        is_recoverable: bool = True,
        retry_count: int = 0,
        **kwargs
    ):
        super().__init__(
            message,
            claim_id=claim_id,
            step_name=step_name,
            cause=original_error,
            **kwargs
        )
        self.original_error = original_error
        self.is_recoverable = is_recoverable
        self.retry_count = retry_count
    
    def should_retry(self) -> bool:
        """Step execution errors are retryable if marked as recoverable."""
        return self.is_recoverable and self.retry_count < 2
    
    def is_critical(self) -> bool:
        """Critical steps should halt processing if not recoverable."""
        critical_steps = ['validate_data', 'make_decision', 'update_systems']
        return self.step_name in critical_steps and not self.is_recoverable



class DocumentProcessingError(OrchestratorError):
    """
    Raised when document processing (download or extraction) fails.
    
    Additional Attributes:
        step: The specific step that failed ('download' or 'extraction')
        document_count: Number of documents being processed
        failed_documents: List of documents that failed
    """
    
    def __init__(
        self,
        message: str,
        claim_id: Optional[str] = None,
        step: str = "unknown",
        document_count: int = 0,
        failed_documents: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, claim_id=claim_id, step_name=f"document_{step}", **kwargs)
        self.step = step
        self.document_count = document_count
        self.failed_documents = failed_documents or []
    
    def should_retry(self) -> bool:
        """Document processing errors are retryable."""
        return True
    
    def is_critical(self) -> bool:
        """Document processing errors are not critical - can continue without documents."""
        return False
