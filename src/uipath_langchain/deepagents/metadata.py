"""Runtime metadata for UiPath DeepAgents graphs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

_RUNTIME_SPEC_ATTR = "__uipath_deep_agent_runtime_spec__"
_DEEPAGENTS_INTEGRATION_NAME = "deepagents"

UiPathDeepAgentHydrationPolicy = Literal[
    "suspend_only",
    "suspend_or_success",
    "always",
]


class UiPathDeepAgentRuntimeSpec(BaseModel):
    """UiPath runtime contract for a DeepAgents-backed graph."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["uipath.deepagents.v1"] = "uipath.deepagents.v1"
    interaction_mode: Literal["task", "conversation"] = "task"
    workspace_scope: Literal["runtime", "conversation"] = "runtime"
    workspace_config_key: str = "uipath_workspace_path"
    persistence: Literal["attachments"] = "attachments"
    hydration_policy: UiPathDeepAgentHydrationPolicy = "suspend_or_success"
    attachment_prefix: str = ".uipath-workspace"


def set_uipath_deep_agent_runtime_spec(
    graph: Any,
    spec: UiPathDeepAgentRuntimeSpec,
) -> Any:
    """Attach UiPath DeepAgents runtime metadata to a graph-like object."""
    setattr(graph, _RUNTIME_SPEC_ATTR, spec)
    builder = getattr(graph, "builder", None)
    if builder is not None:
        setattr(builder, _RUNTIME_SPEC_ATTR, spec)
    return graph


def get_uipath_deep_agent_runtime_spec(
    graph: Any,
) -> UiPathDeepAgentRuntimeSpec | None:
    """Return UiPath DeepAgents runtime metadata from a graph-like object."""
    raw = getattr(graph, _RUNTIME_SPEC_ATTR, None)
    if raw is None:
        builder = getattr(graph, "builder", None)
        raw = (
            getattr(builder, _RUNTIME_SPEC_ATTR, None) if builder is not None else None
        )
    if raw is None:
        return UiPathDeepAgentRuntimeSpec() if is_deep_agent_graph(graph) else None
    if isinstance(raw, UiPathDeepAgentRuntimeSpec):
        return raw
    if isinstance(raw, dict):
        return UiPathDeepAgentRuntimeSpec.model_validate(raw)
    raise TypeError(
        "UiPath DeepAgents runtime spec must be a UiPathDeepAgentRuntimeSpec or dict."
    )


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
