"""DeepAgents graph detection for the UiPath runtime."""

from __future__ import annotations

from typing import Any

_DEEPAGENTS_INTEGRATION_NAME = "deepagents"


def is_deep_agent_graph(graph: Any) -> bool:
    """Return whether a graph-like object carries DeepAgents LangSmith metadata."""
    return _contains_deepagents_metadata(graph, seen=set())


def _contains_deepagents_metadata(graph: Any, seen: set[int]) -> bool:
    graph_id = id(graph)
    if graph_id in seen:
        return False
    seen.add(graph_id)

    if _has_deepagents_metadata(graph):
        return True

    builder = getattr(graph, "builder", None)
    if builder is not None and _contains_deepagents_metadata(builder, seen):
        return True

    nodes = getattr(graph, "nodes", None)
    if isinstance(nodes, dict):
        for node in nodes.values():
            if _contains_deepagents_metadata(node, seen):
                return True
            runnable = getattr(node, "runnable", None)
            if runnable is not None and _contains_deepagents_metadata(runnable, seen):
                return True

    return False


def _has_deepagents_metadata(graph: Any) -> bool:
    config = getattr(graph, "config", None) or {}
    metadata = config.get("metadata", {}) if isinstance(config, dict) else {}
    return metadata.get("ls_integration") == _DEEPAGENTS_INTEGRATION_NAME
