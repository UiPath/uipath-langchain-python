"""Decorators for node functions to reduce boilerplate code."""

import logging
from functools import wraps
from typing import Callable, TypeVar
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')


def node_wrapper(node_name: str, mark_completed: bool = True):
    """
    Decorator to handle common node operations.
    
    Provides:
    - Automatic logging of node start/completion
    - Consistent error handling and error state updates
    - Optional automatic step completion marking
    
    Args:
        node_name: Name of the node for logging and error tracking
        mark_completed: Whether to automatically mark step as completed
        
    Usage:
        @node_wrapper("validate_data")
        async def validate_data_node(state: GraphState) -> GraphState:
            # Core logic only
            return state
    """
    def decorator(func: Callable[[T], T]) -> Callable[[T], T]:
        @wraps(func)
        async def wrapper(state: T) -> T:
            # Get claim_id for logging context
            claim_id = getattr(state, 'claim_id', None) or getattr(state, 'ObjectClaimId', None) or "UNKNOWN"
            
            logger.info(f"[{node_name.upper()}] Starting for claim: {claim_id}")
            
            try:
                # Execute the actual node function
                result = await func(state)
                
                # Mark step as completed if requested
                if mark_completed and hasattr(result, 'completed_steps'):
                    if node_name not in result.completed_steps:
                        result.completed_steps.append(node_name)
                
                logger.info(f"[{node_name.upper()}] Completed for claim: {claim_id}")
                return result
                
            except Exception as e:
                logger.error(f"[{node_name.upper()}] Failed for claim {claim_id}: {e}", exc_info=True)
                
                # Add error to state if possible
                if hasattr(state, 'errors'):
                    state.errors.append({
                        "step": node_name,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Return state even on error to allow graceful degradation
                return state
        
        return wrapper
    return decorator


def requires_uipath_service(func: Callable) -> Callable:
    """
    Decorator to ensure UiPath service is available for node execution.
    
    This is a marker decorator that can be extended to provide
    service injection or validation in the future.
    
    Usage:
        @requires_uipath_service
        async def some_node(state: GraphState) -> GraphState:
            async with UiPathService() as service:
                # Use service
            return state
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)
    
    return wrapper


def log_execution_time(func: Callable) -> Callable:
    """
    Decorator to log execution time of node functions.
    
    Usage:
        @log_execution_time
        async def expensive_node(state: GraphState) -> GraphState:
            # Long-running operation
            return state
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = datetime.now()
        
        result = await func(*args, **kwargs)
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"[TIMING] {func.__name__} completed in {duration:.2f}s")
        
        return result
    
    return wrapper
