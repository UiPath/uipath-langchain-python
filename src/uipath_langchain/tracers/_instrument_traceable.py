import functools
import importlib
import inspect
import logging
import sys
import uuid
from typing import Any, Callable, Dict, List, Literal, Optional

from langchain_core.callbacks import dispatch_custom_event
from uipath.tracing import TracingManager, traced

from ._events import CustomTraceEvents, FunctionCallEventData

# Original module and traceable function references
original_langsmith: Any = None
original_traceable: Any = None

logger = logging.getLogger(__name__)


def dispatch_trace_event(
    func_name,
    inputs: Dict[str, Any],
    event_type="call",
    call_uuid=None,
    result=None,
    exception=None,
    run_type: Optional[
        Literal["tool", "chain", "llm", "retriever", "embedding", "prompt", "parser"]
    ] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Dispatch trace event to our server."""
    try:
        event_data = FunctionCallEventData(
            function_name=func_name,
            event_type=event_type,
            inputs=inputs,
            call_uuid=call_uuid,
            output=result,
            error=str(exception) if exception else None,
            run_type=run_type,
            tags=tags,
            metadata=metadata,
        )

        dispatch_custom_event(CustomTraceEvents.UIPATH_TRACE_FUNCTION_CALL, event_data)
    except Exception as e:
        logger.debug(
            f"Error dispatching trace event: {e}. Function name: {func_name} Event type: {event_type}"
        )


def format_args_for_trace(
    signature: inspect.Signature, *args: Any, **kwargs: Any
) -> Dict[str, Any]:
    try:
        """Return a dictionary of inputs from the function signature."""
        # Create a parameter mapping by partially binding the arguments
        parameter_binding = signature.bind_partial(*args, **kwargs)

        # Fill in default values for any unspecified parameters
        parameter_binding.apply_defaults()

        # Extract the input parameters, skipping special Python parameters
        result = {}
        for name, value in parameter_binding.arguments.items():
            # Skip class and instance references
            if name in ("self", "cls"):
                continue

            # Handle **kwargs parameters specially
            param_info = signature.parameters.get(name)
            if param_info and param_info.kind == inspect.Parameter.VAR_KEYWORD:
                # Flatten nested kwargs directly into the result
                if isinstance(value, dict):
                    result.update(value)
            else:
                # Regular parameter
                result[name] = value

        return result
    except Exception as e:
        logger.warning(
            f"Error formatting arguments for trace: {e}. Using args and kwargs directly."
        )
        return {"args": args, "kwargs": kwargs}


def _create_traced_wrapper(
    func: Callable[..., Any],
    wrapper_func: Optional[Callable[..., Any]] = None,
    func_name: Optional[str] = None,
    run_type: Optional[str] = None,
    span_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    input_processor: Optional[Callable[..., Any]] = None,
    output_processor: Optional[Callable[..., Any]] = None,
):
    """Create a traced wrapper based on function type."""
    # Use function name if not provided
    actual_func_name = func_name or func.__name__
    # Function to actually call (the wrapped function or original)
    target_func = wrapper_func or func
    # Ensure we have metadata
    actual_metadata = metadata or {}

    # Define all wrapper functions

    @functools.wraps(target_func)
    async def async_gen_wrapper(*args, **kwargs):
        try:
            call_uuid = str(uuid.uuid4())

            # Get inputs and process them if needed
            inputs = format_args_for_trace(inspect.signature(func), *args, **kwargs)
            if input_processor is not None:
                inputs = input_processor(inputs)

            # Add span_type to metadata if provided
            if span_type:
                actual_metadata["span_type"] = (
                    span_type or "function_call_generator_async"
                )

            dispatch_trace_event(
                actual_func_name,
                inputs,
                "call",
                call_uuid,
                run_type=run_type,  # type: ignore
                tags=tags,
                metadata=actual_metadata,
            )

            outputs = []
            async_gen = target_func(*args, **kwargs)
            async for item in async_gen:
                outputs.append(item)
                yield item

            # Process output if needed
            output_to_record = outputs
            if output_processor is not None:
                output_to_record = output_processor(outputs)

            dispatch_trace_event(
                actual_func_name,
                inputs,
                "completion",
                call_uuid,
                result=output_to_record,
            )
        except Exception as e:
            dispatch_trace_event(
                actual_func_name, inputs, "completion", call_uuid, exception=e
            )
            raise

    @functools.wraps(target_func)
    def gen_wrapper(*args, **kwargs):
        try:
            call_uuid = str(uuid.uuid4())

            # Get inputs and process them if needed
            inputs = format_args_for_trace(inspect.signature(func), *args, **kwargs)
            if input_processor is not None:
                inputs = input_processor(inputs)

            # Add span_type to metadata if provided
            if span_type:
                actual_metadata["span_type"] = (
                    span_type or "function_call_generator_sync"
                )

            dispatch_trace_event(
                actual_func_name,
                inputs,
                "call",
                call_uuid,
                run_type=run_type,  # type: ignore
                tags=tags,
                metadata=actual_metadata,
            )

            outputs = []
            gen = target_func(*args, **kwargs)
            for item in gen:
                outputs.append(item)
                yield item

            # Process output if needed
            output_to_record = outputs
            if output_processor is not None:
                output_to_record = output_processor(outputs)

            dispatch_trace_event(
                actual_func_name,
                inputs,
                "completion",
                call_uuid,
                result=output_to_record,
            )
        except Exception as e:
            dispatch_trace_event(
                actual_func_name, inputs, "completion", call_uuid, exception=e
            )
            raise

    @functools.wraps(target_func)
    async def async_wrapper(*args, **kwargs):
        try:
            call_uuid = str(uuid.uuid4())

            # Get inputs and process them if needed
            inputs = format_args_for_trace(inspect.signature(func), *args, **kwargs)
            if input_processor is not None:
                inputs = input_processor(inputs)

            # Add span_type to metadata if provided
            if span_type:
                actual_metadata["span_type"] = span_type or "function_call_async"

            dispatch_trace_event(
                actual_func_name,
                inputs,
                "call",
                call_uuid,
                run_type=run_type,  # type: ignore
                tags=tags,
                metadata=actual_metadata,
            )

            result = await target_func(*args, **kwargs)

            # Process output if needed
            output_to_record = result
            if output_processor is not None:
                output_to_record = output_processor(result)

            dispatch_trace_event(
                actual_func_name,
                inputs,
                "completion",
                call_uuid,
                result=output_to_record,
            )

            return result
        except Exception as e:
            dispatch_trace_event(
                actual_func_name, inputs, "completion", call_uuid, exception=e
            )
            raise

    @functools.wraps(target_func)
    def sync_wrapper(*args, **kwargs):
        try:
            call_uuid = str(uuid.uuid4())

            # Get inputs and process them if needed
            inputs = format_args_for_trace(inspect.signature(func), *args, **kwargs)
            if input_processor is not None:
                inputs = input_processor(inputs)

            # Add span_type to metadata if provided
            if span_type:
                actual_metadata["span_type"] = span_type or "function_call_sync"

            dispatch_trace_event(
                actual_func_name,
                inputs,
                "call",
                call_uuid,
                run_type=run_type,  # type: ignore
                tags=tags,
                metadata=actual_metadata,
            )

            result = target_func(*args, **kwargs)

            # Process output if needed
            output_to_record = result
            if output_processor is not None:
                output_to_record = output_processor(result)

            dispatch_trace_event(
                actual_func_name,
                inputs,
                "completion",
                call_uuid,
                result=output_to_record,
            )

            return result
        except Exception as e:
            dispatch_trace_event(
                actual_func_name, inputs, "completion", call_uuid, exception=e
            )
            raise

    # Return the appropriate wrapper based on the function type
    if inspect.isasyncgenfunction(target_func):
        return async_gen_wrapper
    elif inspect.isgeneratorfunction(target_func):
        return gen_wrapper
    elif inspect.iscoroutinefunction(target_func):
        return async_wrapper
    else:
        return sync_wrapper


def _create_appropriate_wrapper(
    original_func: Any, wrapped_func: Any, decorator_kwargs: Dict[str, Any]
):
    """Create the appropriate wrapper based on function type."""

    # Get the function name and tags from decorator arguments
    func_name = decorator_kwargs.get("name", original_func.__name__)
    tags = decorator_kwargs.get("tags", None)
    metadata = decorator_kwargs.get("metadata", None)
    run_type = decorator_kwargs.get("run_type", None)

    return _create_traced_wrapper(
        func=original_func,
        wrapper_func=wrapped_func,
        func_name=func_name,
        run_type=run_type,
        tags=tags,
        metadata=metadata,
    )


def _uipath_traced(
    name: Optional[str] = None,
    run_type: Optional[str] = None,
    span_type: Optional[str] = None,
    input_processor: Optional[Callable[..., Any]] = None,
    output_processor: Optional[Callable[..., Any]] = None,
    *args: Any,
    **kwargs: Any,
):
    """Decorator factory that creates traced functions using dispatch_trace_event."""

    def decorator(func):
        return _create_traced_wrapper(
            func=func,
            func_name=name,
            run_type=run_type,
            span_type=span_type,
            input_processor=input_processor,
            output_processor=output_processor,
        )

    return decorator


# Create patched version of traceable
def patched_traceable(*decorator_args, **decorator_kwargs):
    # Handle the case when @traceable is used directly as decorator without arguments
    if (
        len(decorator_args) == 1
        and callable(decorator_args[0])
        and not decorator_kwargs
    ):
        func = decorator_args[0]
        return _create_appropriate_wrapper(func, original_traceable(func), {})

    # Handle the case when @traceable(args) is used with parameters
    original_decorated = original_traceable(*decorator_args, **decorator_kwargs)

    def uipath_trace_decorator(func):
        # Apply the original decorator with its arguments
        wrapped_func = original_decorated(func)
        return _create_appropriate_wrapper(func, wrapped_func, decorator_kwargs)

    return uipath_trace_decorator


# Register the UIPath traced decorator
def register_uipath_tracing():
    """Register the UIPath tracing decorator with TracedDecoratorRegistry."""
    # Reapply to all previously decorated functions
    TracingManager.reapply_traced_decorator(_uipath_traced)


# Apply the patch
def _map_traceable_to_traced_args(
    run_type: Optional[str] = None,
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Map LangSmith @traceable arguments to UiPath @traced() arguments.
    
    Args:
        run_type: Function type (tool, chain, llm, retriever, etc.)
        name: Custom name for the traced function
        tags: List of tags for categorization
        metadata: Additional metadata dictionary
        **kwargs: Additional arguments (ignored)
    
    Returns:
        Dict containing mapped arguments for @traced()
    """
    traced_args = {}
    
    # Direct mappings
    if name is not None:
        traced_args["name"] = name
    
    # Pass through run_type directly to UiPath @traced()
    if run_type:
        traced_args["run_type"] = run_type
        
    # For span_type, we can derive from run_type or use a default
    if run_type:
        # Map run_type to appropriate span_type for OpenTelemetry
        span_type_mapping = {
            "tool": "tool_call",
            "chain": "chain_execution", 
            "llm": "llm_call",
            "retriever": "retrieval",
            "embedding": "embedding",
            "prompt": "prompt_template",
            "parser": "output_parser"
        }
        traced_args["span_type"] = span_type_mapping.get(run_type, run_type)
    
    # Note: UiPath @traced() doesn't support custom attributes directly
    # Tags and metadata information is lost in the current mapping
    # This could be enhanced in future versions
    
    return traced_args


def otel_traceable_adapter(
    func: Optional[Callable] = None,
    *,
    run_type: Optional[str] = None,
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
):
    """
    OTEL-based adapter that converts LangSmith @traceable decorator calls to UiPath @traced().
    
    This function maintains the same interface as LangSmith's @traceable but uses
    UiPath's OpenTelemetry-based tracing system underneath.
    
    Args:
        func: Function to be decorated (when used without parentheses)
        run_type: Type of function (tool, chain, llm, etc.)
        name: Custom name for tracing
        tags: List of tags for categorization
        metadata: Additional metadata dictionary
        **kwargs: Additional arguments (for future compatibility)
        
    Returns:
        Decorated function or decorator function
    """
    def decorator(f: Callable) -> Callable:
        # Map arguments to @traced() format
        traced_args = _map_traceable_to_traced_args(
            run_type=run_type,
            name=name,
            tags=tags,
            metadata=metadata,
            **kwargs
        )
        
        # Apply UiPath @traced() decorator
        return traced(**traced_args)(f)
    
    # Handle both @traceable and @traceable(...) usage patterns
    if func is None:
        # Called as @traceable(...) - return decorator
        return decorator
    else:
        # Called as @traceable - apply decorator directly
        return decorator(func)


def _instrument_traceable_attributes(useOtel: bool = False):
    """Apply the patch to langsmith module at import time."""
    global original_langsmith, original_traceable

    if not useOtel:
        # Register our custom tracing decorator when not using opentelemetry
        register_uipath_tracing()

    # Import the original module if not already done
    if original_langsmith is None:
        # Temporarily remove our custom module from sys.modules
        if "langsmith" in sys.modules:
            original_langsmith = sys.modules["langsmith"]
            del sys.modules["langsmith"]

        # Import the original module
        original_langsmith = importlib.import_module("langsmith")

        # Store the original traceable
        original_traceable = original_langsmith.traceable

        # Replace the traceable function with our patched version
        if useOtel:
            # Use OTEL-based adapter when OTEL is enabled
            original_langsmith.traceable = otel_traceable_adapter
        else:
            # Use existing dispatch_event-based adapter
            original_langsmith.traceable = patched_traceable

        # Put our modified module back
        sys.modules["langsmith"] = original_langsmith

    return original_langsmith
