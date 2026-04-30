"""Deep agent builder.

Thin UiPath wrapper around the `deepagents` library. The deepagents dependency
is optional — install with one of:

    pip install 'uipath-langchain[deep]'
    uv add 'uipath-langchain[deep]'
"""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from langchain.agents.structured_output import ResponseFormat
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph
from pydantic import BaseModel

from .types import DeepAgentGraphState
from .utils import create_state_with_input

if TYPE_CHECKING:
    from deepagents import CompiledSubAgent, SubAgent
    from deepagents.backends import BackendProtocol
    from deepagents.backends.protocol import BackendFactory


_INSTALL_HINT = (
    "deepagents is required for deep agents. Install with: "
    "pip install 'uipath-langchain[deep]' "
    "(or: uv add 'uipath-langchain[deep]')"
)


def _import_create_deep_agent() -> Any:
    try:
        from deepagents import create_deep_agent as _upstream

        return _upstream
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc


def create_deep_agent(
    model: BaseChatModel,
    system_prompt: str = "",
    tools: Sequence[BaseTool] = (),
    subagents: "Sequence[SubAgent | CompiledSubAgent]" = (),
    backend: "BackendProtocol | BackendFactory | None" = None,
    response_format: ResponseFormat[Any] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Create a deep agent.

    Deep agents provide built-in capabilities for:
    - Planning (write_todos, read_todos)
    - Filesystem operations (read_file, write_file, edit_file, ls, glob, grep)
    - Sub-agent delegation (task)
    - Auto-summarization for long conversations

    Args:
        model: A BaseChatModel instance.
        system_prompt: Instructions for the agent.
        tools: Custom tools to provide to the agent.
        subagents: Optional list of subagent configurations. Each entry is a
            ``SubAgent`` (name, description, system_prompt, and optional tools/model/middleware)
            or a ``CompiledSubAgent`` (name, description, and a pre-built runnable).
        backend: Storage backend for filesystem operations. Can be a
            ``BackendProtocol`` instance, a factory callable, or ``None``
            (uses the default in-state backend).
        response_format: Structured output format for the agent response.

    Returns:
        Compiled LangGraph agent ready for execution.

    Raises:
        ImportError: If the ``deepagents`` package is not installed. Install
            with ``pip install 'uipath-langchain[deep]'`` or
            ``uv add 'uipath-langchain[deep]'``.
    """
    upstream_create_deep_agent = _import_create_deep_agent()
    return upstream_create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=list(tools),
        subagents=list(subagents),
        backend=backend,
        response_format=response_format,
    )


def create_deep_agent_graph(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    system_prompt: str,
    backend: "BackendProtocol | BackendFactory | None",
    response_format: ResponseFormat[Any] | None,
    input_schema: type[BaseModel] | None,
    output_schema: type[BaseModel],
    build_user_message: Callable[[dict[str, Any]], str],
) -> StateGraph[Any, Any, Any, Any]:
    """Build a deep agent wrapped in a parent graph that handles I/O transformation.

    The deep agent only understands messages as input and produces
    structured_response as output. The wrapper graph bridges the gap:

        START -> transform_input -> deep_agent -> transform_output -> END

    Args:
        model: Chat model for the deep agent.
        tools: Tools available to the deep agent.
        system_prompt: Combined system + meta prompt.
        backend: Filesystem backend for the deep agent.
        response_format: Structured output format.
        input_schema: Resolved input Pydantic model (or None).
        output_schema: Resolved output Pydantic model.
        build_user_message: Callable that converts input arguments dict to a user message string.

    Raises:
        ImportError: If the ``deepagents`` package is not installed. Install
            with ``pip install 'uipath-langchain[deep]'`` or
            ``uv add 'uipath-langchain[deep]'``.
    """
    inner_graph = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        backend=backend,
        response_format=response_format,
    )

    wrapper_state = create_state_with_input(input_schema)

    internal_fields = set(DeepAgentGraphState.model_fields.keys())

    def transform_input(state: BaseModel) -> dict[str, Any]:
        state_data = state.model_dump()
        input_data = {k: v for k, v in state_data.items() if k not in internal_fields}
        input_args = (
            input_schema.model_validate(input_data).model_dump()
            if input_schema is not None
            else {}
        )
        user_text = build_user_message(input_args)
        return {"messages": [HumanMessage(content=user_text, id="user-input")]}

    def transform_output(state: BaseModel) -> dict[str, Any]:
        structured = getattr(state, "structured_response", {})
        return output_schema.model_validate(structured).model_dump()

    wrapper: StateGraph[Any, Any, Any, Any] = StateGraph(
        wrapper_state, input_schema=input_schema, output_schema=output_schema
    )
    wrapper.add_node("transform_input", transform_input)
    wrapper.add_node("deep_agent", inner_graph)
    wrapper.add_node("transform_output", transform_output)
    wrapper.add_edge(START, "transform_input")
    wrapper.add_edge("transform_input", "deep_agent")
    wrapper.add_edge("deep_agent", "transform_output")
    wrapper.add_edge("transform_output", END)

    return wrapper
