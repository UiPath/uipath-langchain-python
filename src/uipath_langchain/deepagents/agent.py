"""UiPath DeepAgents authoring helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from deepagents import CompiledSubAgent, SubAgent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import StateGraph
from pydantic import BaseModel

from uipath_langchain.agent.advanced import create_advanced_agent_graph

from .backend import create_workspace_backend_factory
from .metadata import (
    UiPathDeepAgentRuntimeSpec,
    set_uipath_deep_agent_runtime_spec,
)

UiPathDeepAgentPrompt = str | Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class UiPathDeepAgent:
    """Author-facing result for a UiPath DeepAgents graph."""

    graph: StateGraph[Any, Any, Any, Any]


def create_uipath_deep_agent(
    *,
    model: BaseChatModel,
    input_schema: type[BaseModel] | None = None,
    output_schema: type[BaseModel],
    system_prompt: UiPathDeepAgentPrompt = "",
    user_prompt: UiPathDeepAgentPrompt | None = None,
    tools: Sequence[BaseTool] = (),
    subagents: Sequence[SubAgent | CompiledSubAgent] = (),
    build_user_message: Callable[[dict[str, Any]], str] | None = None,
) -> UiPathDeepAgent:
    """Create a UiPath-runtime-aware DeepAgents graph."""
    graph = create_uipath_deep_agent_graph(
        model=model,
        input_schema=input_schema,
        output_schema=output_schema,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tools=tools,
        subagents=subagents,
        build_user_message=build_user_message,
    )
    spec = UiPathDeepAgentRuntimeSpec()
    set_uipath_deep_agent_runtime_spec(graph, spec)
    return UiPathDeepAgent(graph=graph)


def create_uipath_deep_agent_graph(
    *,
    model: BaseChatModel,
    input_schema: type[BaseModel] | None = None,
    output_schema: type[BaseModel],
    system_prompt: UiPathDeepAgentPrompt = "",
    user_prompt: UiPathDeepAgentPrompt | None = None,
    tools: Sequence[BaseTool] = (),
    subagents: Sequence[SubAgent | CompiledSubAgent] = (),
    build_user_message: Callable[[dict[str, Any]], str] | None = None,
) -> StateGraph[Any, Any, Any, Any]:
    """Create a tagged UiPath DeepAgents graph for coded agents."""
    backend = create_workspace_backend_factory()
    response_format = ToolStrategy(output_schema)
    if user_prompt is not None and build_user_message is not None:
        raise ValueError("Use either user_prompt or build_user_message, not both.")

    static_system_prompt, build_system_message = _split_system_prompt(system_prompt)
    build_user_message = (
        _coerce_prompt_renderer(user_prompt)
        if user_prompt is not None
        else build_user_message or _default_user_message
    )

    graph = create_advanced_agent_graph(
        model=model,
        tools=tools,
        subagents=subagents,
        system_prompt=static_system_prompt,
        backend=backend,
        response_format=response_format,
        input_schema=input_schema,
        output_schema=output_schema,
        build_user_message=build_user_message,
        build_system_message=build_system_message,
    )
    set_uipath_deep_agent_runtime_spec(
        graph,
        UiPathDeepAgentRuntimeSpec(),
    )
    return graph


def _split_system_prompt(
    system_prompt: UiPathDeepAgentPrompt,
) -> tuple[str, Callable[[dict[str, Any]], str] | None]:
    if callable(system_prompt):
        return "", system_prompt
    return system_prompt, None


def _coerce_prompt_renderer(
    prompt: UiPathDeepAgentPrompt,
) -> Callable[[dict[str, Any]], str]:
    if callable(prompt):
        return prompt
    return lambda _args: prompt


def _default_user_message(args: dict[str, Any]) -> str:
    if not args:
        return ""
    return "\n".join(f"{key}: {value}" for key, value in args.items())
