"""UiPath DeepAgents authoring helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from deepagents import CompiledSubAgent, SubAgent
from deepagents.backends import BackendProtocol
from deepagents.backends.protocol import BackendFactory
from langchain.agents.structured_output import ResponseFormat, ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import StateGraph
from pydantic import BaseModel

from uipath_langchain.agent.advanced import create_advanced_agent_graph

from .backend import create_workspace_backend_factory
from .metadata import (
    UiPathDeepAgentHydrationPolicy,
    UiPathDeepAgentRuntimeSpec,
    set_uipath_deep_agent_runtime_spec,
)


@dataclass(frozen=True)
class UiPathDeepAgent:
    """Author-facing result for a UiPath DeepAgents graph."""

    graph: StateGraph[Any, Any, Any, Any]
    runtime_spec: UiPathDeepAgentRuntimeSpec


def create_uipath_deep_agent(
    *,
    model: BaseChatModel,
    input_schema: type[BaseModel] | None = None,
    output_schema: type[BaseModel],
    system_prompt: str = "",
    tools: Sequence[BaseTool] = (),
    subagents: Sequence[SubAgent | CompiledSubAgent] = (),
    backend: BackendProtocol | BackendFactory | None = None,
    response_format: ResponseFormat[Any] | None = None,
    build_user_message: Callable[[dict[str, Any]], str] | None = None,
    interaction_mode: Literal["task", "conversation"] = "task",
    workspace_scope: Literal["runtime", "conversation"] = "runtime",
    workspace_config_key: str = "uipath_workspace_path",
    hydration_policy: UiPathDeepAgentHydrationPolicy = "suspend_or_success",
) -> UiPathDeepAgent:
    """Create a UiPath-runtime-aware DeepAgents graph.

    This PoC fully supports task-mode coded agents. Conversational mode is part
    of the public contract but intentionally rejected until the chat exchange
    adapter can preserve UiPath's conversational suspend/resume loop.
    """
    graph = create_uipath_deep_agent_graph(
        model=model,
        input_schema=input_schema,
        output_schema=output_schema,
        system_prompt=system_prompt,
        tools=tools,
        subagents=subagents,
        backend=backend,
        response_format=response_format,
        build_user_message=build_user_message,
        interaction_mode=interaction_mode,
        workspace_scope=workspace_scope,
        workspace_config_key=workspace_config_key,
        hydration_policy=hydration_policy,
    )
    spec = UiPathDeepAgentRuntimeSpec(
        interaction_mode=interaction_mode,
        workspace_scope=workspace_scope,
        workspace_config_key=workspace_config_key,
        hydration_policy=hydration_policy,
    )
    set_uipath_deep_agent_runtime_spec(graph, spec)
    return UiPathDeepAgent(graph=graph, runtime_spec=spec)


def create_uipath_deep_agent_graph(
    *,
    model: BaseChatModel,
    input_schema: type[BaseModel] | None = None,
    output_schema: type[BaseModel],
    system_prompt: str = "",
    tools: Sequence[BaseTool] = (),
    subagents: Sequence[SubAgent | CompiledSubAgent] = (),
    backend: BackendProtocol | BackendFactory | None = None,
    response_format: ResponseFormat[Any] | None = None,
    build_user_message: Callable[[dict[str, Any]], str] | None = None,
    interaction_mode: Literal["task", "conversation"] = "task",
    workspace_scope: Literal["runtime", "conversation"] = "runtime",
    workspace_config_key: str = "uipath_workspace_path",
    hydration_policy: UiPathDeepAgentHydrationPolicy = "suspend_or_success",
) -> StateGraph[Any, Any, Any, Any]:
    """Create a tagged UiPath DeepAgents graph for coded agents."""
    if interaction_mode == "conversation":
        raise NotImplementedError(
            "Conversational UiPath DeepAgents require a chat-exchange adapter; "
            "task-mode is the only implemented PoC mode."
        )
    if workspace_scope == "conversation":
        raise NotImplementedError(
            "Conversation-scoped DeepAgents workspaces require conversational "
            "runtime support; use workspace_scope='runtime' for task mode."
        )

    backend = backend or create_workspace_backend_factory(
        workspace_config_key=workspace_config_key,
    )
    response_format = response_format or ToolStrategy(output_schema)
    build_user_message = build_user_message or _default_user_message

    graph = create_advanced_agent_graph(
        model=model,
        tools=tools,
        subagents=subagents,
        system_prompt=system_prompt,
        backend=backend,
        response_format=response_format,
        input_schema=input_schema,
        output_schema=output_schema,
        build_user_message=build_user_message,
    )
    set_uipath_deep_agent_runtime_spec(
        graph,
        UiPathDeepAgentRuntimeSpec(
            interaction_mode=interaction_mode,
            workspace_scope=workspace_scope,
            workspace_config_key=workspace_config_key,
            hydration_policy=hydration_policy,
        ),
    )
    return graph


def _default_user_message(args: dict[str, Any]) -> str:
    if not args:
        return ""
    return "\n".join(f"{key}: {value}" for key, value in args.items())
