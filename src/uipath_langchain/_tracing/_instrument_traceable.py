import importlib
import logging
import sys
from typing import Any, Callable

from uipath.tracing import traced

# Original module and traceable function references
original_langsmith: Any = None
original_traceable: Any = None

logger = logging.getLogger(__name__)


def _map_traceable_to_traced_args(
    run_type: str | None = None,
    name: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
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
            "parser": "output_parser",
        }
        traced_args["span_type"] = span_type_mapping.get(run_type, run_type)

    # Note: UiPath @traced() doesn't support custom attributes directly
    # Tags and metadata information is lost in the current mapping
    # This could be enhanced in future versions

    return traced_args


def otel_traceable_adapter(
    func: Callable[..., Any] | None = None,
    *,
    run_type: str | None = None,
    name: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
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

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        # Map arguments to @traced() format
        traced_args = _map_traceable_to_traced_args(
            run_type=run_type, name=name, tags=tags, metadata=metadata, **kwargs
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


def _instrument_traceable_attributes():
    """Apply the patch to langsmith module at import time."""
    global original_langsmith, original_traceable

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
        original_langsmith.traceable = otel_traceable_adapter

        # Put our modified module back
        sys.modules["langsmith"] = original_langsmith

    return original_langsmith
