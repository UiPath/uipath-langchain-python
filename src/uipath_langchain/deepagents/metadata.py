"""Runtime metadata for UiPath DeepAgents graphs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

_RUNTIME_SPEC_ATTR = "__uipath_deep_agent_runtime_spec__"

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
        return None
    if isinstance(raw, UiPathDeepAgentRuntimeSpec):
        return raw
    if isinstance(raw, dict):
        return UiPathDeepAgentRuntimeSpec.model_validate(raw)
    raise TypeError(
        "UiPath DeepAgents runtime spec must be a UiPathDeepAgentRuntimeSpec or dict."
    )
