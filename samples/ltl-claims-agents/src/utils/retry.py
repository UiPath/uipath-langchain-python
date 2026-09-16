"""
Retry utility with exponential backoff for resilient operations.

Provides retry logic for transient failures with configurable backoff
strategy, logging, and error handling.
"""

import asyncio
import logging
import time
from typing import Callable, Any, Optional, Tuple, Type
from functools import wraps


logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 10.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        """
        Initialize retry configuration.
        
        Args:
            max_attempts: Maximum number of retry attempts (default: 3)
            initial_delay: Initial delay in seconds before first retry (default: 1.0)
            max_delay: Maximum delay in seconds between retries (default: 10.0)
            exponential_base: Base for exponential backoff calculation (default: 2.0)
            jitter: Whether to add random jitter to delays (default: True)
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given attempt number.
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        # Calculate exponential backoff
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        # Add jitter to prevent thundering herd
        if self.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay


async def retry_with_backoff(
    func: Callable,
    *args,
    config: Optional[RetryConfig] = None,
    error_types: Tuple[Type[Exception], ...] = (Exception,),
    context: Optional[dict] = None,
    **kwargs
) -> Any:
    """
    Execute an async function with exponential backoff retry logic.
    
    Retries the function on specified error types with exponential backoff
    between attempts. Logs all retry attempts with context for debugging.
    
    Args:
        func: Async function to execute
        *args: Positional arguments to pass to func
        config: Retry configuration (uses defaults if None)
        error_types: Tuple of exception types to retry on
        context: Optional context dict for logging (e.g., {"claim_id": "ABC-123", "operation": "download"})
        **kwargs: Keyword arguments to pass to func
        
    Returns:
        Result from successful function execution
        
    Raises:
        Last exception if all retry attempts fail
        
    Example:
        result = await retry_with_backoff(
            uipath_service.get_claim_by_id,
            claim_id="ABC-123",
            config=RetryConfig(max_attempts=3, initial_delay=1.0),
            error_types=(UiPathServiceError, TimeoutError),
            context={"claim_id": "ABC-123", "operation": "get_claim"}
        )
    """
    # Use default config if none provided
    if config is None:
        config = RetryConfig(
            max_attempts=3,
            initial_delay=1.0,
            max_delay=10.0
        )
    
    # Build context string for logging
    context_str = ""
    if context:
        context_parts = [f"{k}={v}" for k, v in context.items()]
        context_str = f" [{', '.join(context_parts)}]"
    
    last_exception = None
    
    for attempt in range(config.max_attempts):
        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Log success if this was a retry
            if attempt > 0:
                logger.info(
                    f"[SUCCESS] Retry successful on attempt {attempt + 1}/{config.max_attempts}{context_str}"
                )
            
            return result
            
        except error_types as e:
            last_exception = e
            
            # Check if we should retry
            if attempt < config.max_attempts - 1:
                # Calculate delay for next attempt
                delay = config.calculate_delay(attempt)
                
                # Log retry attempt
                logger.warning(
                    f"[RETRY] Attempt {attempt + 1}/{config.max_attempts} failed{context_str}: "
                    f"{type(e).__name__}: {str(e)}. Retrying in {delay:.2f}s..."
                )
                
                # Wait before retrying
                await asyncio.sleep(delay)
            else:
                # Final attempt failed
                logger.error(
                    f"[FAILED] All {config.max_attempts} retry attempts failed{context_str}: "
                    f"{type(e).__name__}: {str(e)}"
                )
                raise
        
        except Exception as e:
            # Non-retryable error - fail immediately
            logger.error(
                f"[ERROR] Non-retryable error{context_str}: {type(e).__name__}: {str(e)}"
            )
            raise
    
    # Should never reach here, but just in case
    if last_exception:
        raise last_exception


def retry_with_backoff_sync(
    func: Callable,
    *args,
    config: Optional[RetryConfig] = None,
    error_types: Tuple[Type[Exception], ...] = (Exception,),
    context: Optional[dict] = None,
    **kwargs
) -> Any:
    """
    Execute a synchronous function with exponential backoff retry logic.
    
    Synchronous version of retry_with_backoff for non-async functions.
    
    Args:
        func: Synchronous function to execute
        *args: Positional arguments to pass to func
        config: Retry configuration (uses defaults if None)
        error_types: Tuple of exception types to retry on
        context: Optional context dict for logging
        **kwargs: Keyword arguments to pass to func
        
    Returns:
        Result from successful function execution
        
    Raises:
        Last exception if all retry attempts fail
    """
    # Use default config if none provided
    if config is None:
        config = RetryConfig(
            max_attempts=3,
            initial_delay=1.0,
            max_delay=10.0
        )
    
    # Build context string for logging
    context_str = ""
    if context:
        context_parts = [f"{k}={v}" for k, v in context.items()]
        context_str = f" [{', '.join(context_parts)}]"
    
    last_exception = None
    
    for attempt in range(config.max_attempts):
        try:
            # Execute the function
            result = func(*args, **kwargs)
            
            # Log success if this was a retry
            if attempt > 0:
                logger.info(
                    f"[SUCCESS] Retry successful on attempt {attempt + 1}/{config.max_attempts}{context_str}"
                )
            
            return result
            
        except error_types as e:
            last_exception = e
            
            # Check if we should retry
            if attempt < config.max_attempts - 1:
                # Calculate delay for next attempt
                delay = config.calculate_delay(attempt)
                
                # Log retry attempt
                logger.warning(
                    f"[RETRY] Attempt {attempt + 1}/{config.max_attempts} failed{context_str}: "
                    f"{type(e).__name__}: {str(e)}. Retrying in {delay:.2f}s..."
                )
                
                # Wait before retrying
                time.sleep(delay)
            else:
                # Final attempt failed
                logger.error(
                    f"[FAILED] All {config.max_attempts} retry attempts failed{context_str}: "
                    f"{type(e).__name__}: {str(e)}"
                )
                raise
        
        except Exception as e:
            # Non-retryable error - fail immediately
            logger.error(
                f"[ERROR] Non-retryable error{context_str}: {type(e).__name__}: {str(e)}"
            )
            raise
    
    # Should never reach here, but just in case
    if last_exception:
        raise last_exception


def with_retry(
    config: Optional[RetryConfig] = None,
    error_types: Tuple[Type[Exception], ...] = (Exception,),
    context_func: Optional[Callable] = None
):
    """
    Decorator to add retry logic to async functions.
    
    Args:
        config: Retry configuration
        error_types: Tuple of exception types to retry on
        context_func: Optional function to extract context from function args
        
    Example:
        @with_retry(
            config=RetryConfig(max_attempts=3),
            error_types=(UiPathServiceError,),
            context_func=lambda self, claim_id: {"claim_id": claim_id}
        )
        async def get_claim(self, claim_id: str):
            return await self.uipath_service.get_claim_by_id(claim_id)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract context if context_func provided
            context = None
            if context_func:
                try:
                    context = context_func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Failed to extract context: {e}")
            
            # Execute with retry
            return await retry_with_backoff(
                func,
                *args,
                config=config,
                error_types=error_types,
                context=context,
                **kwargs
            )
        
        return wrapper
    
    return decorator


__all__ = [
    "RetryConfig",
    "retry_with_backoff",
    "retry_with_backoff_sync",
    "with_retry"
]
