"""UiPath-aware DeepAgents authoring API."""

from collections.abc import Callable, Sequence
from typing import Any

from deepagents import (
    AsyncSubAgent,
    CompiledSubAgent,
    FilesystemPermission,
    SubAgent,
)
from deepagents import create_deep_agent as _create_deep_agent
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ResponseT
from langchain.agents.structured_output import ResponseFormat
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.typing import ContextT

from .backend import _UiPathWorkspaceBackendFactory
from .metadata import mark_uipath_deep_agent


# This API intentionally omits runtime-managed DeepAgents arguments. UiPath
# provides the filesystem backend and checkpointer when it compiles the graph.
def create_uipath_deep_agent(
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: Sequence[SubAgent | CompiledSubAgent | AsyncSubAgent] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    permissions: list[FilesystemPermission] | None = None,
    response_format: ResponseFormat[ResponseT]
    | type[ResponseT]
    | dict[str, Any]
    | None = None,
    context_schema: type[ContextT] | None = None,
) -> CompiledStateGraph[AgentState[ResponseT], ContextT, Any, Any]:
    """Create a DeepAgent backed by the UiPath runtime workspace.

    Parameters match ``deepagents.create_deep_agent`` and are forwarded unchanged,
    except that UiPath owns compile-time configuration.
    """
    graph = _create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware,
        subagents=subagents,
        skills=skills,
        memory=memory,
        permissions=permissions,
        backend=_UiPathWorkspaceBackendFactory(),
        response_format=response_format,
        context_schema=context_schema,
    )
    return mark_uipath_deep_agent(graph)
