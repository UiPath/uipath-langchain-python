"""Runtime requirements for UiPath-created DeepAgents graphs."""

from collections.abc import Mapping
from typing import Any, TypeVar

from langgraph.graph.state import CompiledStateGraph

_UIPATH_RUNTIME_METADATA_KEY = "uipath_runtime"
_REQUIREMENTS_METADATA_KEY = "requirements"
_MANAGED_WORKSPACE_REQUIREMENT = "managed_workspace"

CompiledGraphT = TypeVar(
    "CompiledGraphT",
    bound=CompiledStateGraph[Any, Any, Any, Any],
)


def mark_uipath_deep_agent(
    graph: CompiledGraphT,
) -> CompiledGraphT:
    """Declare the runtime requirement of a top-level UiPath DeepAgent."""
    return with_uipath_managed_workspace(graph)


def with_uipath_managed_workspace(
    graph: CompiledGraphT,
) -> CompiledGraphT:
    """Bind the managed-workspace requirement to a compiled entrypoint graph.

    Call this on a parent graph when a UiPath DeepAgent is embedded in a larger
    LangGraph or runnable composition. Requirements are declared at the
    entrypoint instead of inferred from third-party runnable internals.
    """
    return graph.with_config(
        {
            "metadata": {
                _UIPATH_RUNTIME_METADATA_KEY: {
                    _REQUIREMENTS_METADATA_KEY: [_MANAGED_WORKSPACE_REQUIREMENT],
                },
            }
        }
    )


def get_runtime_requirements(
    graph: CompiledStateGraph[Any, Any, Any, Any],
) -> frozenset[str]:
    """Return the requirements explicitly bound to an entrypoint graph."""
    config = graph.config or {}
    metadata = config.get("metadata", {}) if isinstance(config, Mapping) else {}
    runtime_metadata = (
        metadata.get(_UIPATH_RUNTIME_METADATA_KEY, {})
        if isinstance(metadata, Mapping)
        else {}
    )
    declared_requirements = (
        runtime_metadata.get(_REQUIREMENTS_METADATA_KEY, ())
        if isinstance(runtime_metadata, Mapping)
        else ()
    )
    if isinstance(declared_requirements, str):
        return frozenset({declared_requirements})
    if isinstance(declared_requirements, (list, tuple, set, frozenset)):
        return frozenset(
            requirement
            for requirement in declared_requirements
            if isinstance(requirement, str)
        )
    return frozenset()


def requires_managed_workspace(
    graph: CompiledStateGraph[Any, Any, Any, Any],
) -> bool:
    """Return whether an entrypoint graph needs a managed workspace."""
    return _MANAGED_WORKSPACE_REQUIREMENT in get_runtime_requirements(graph)
