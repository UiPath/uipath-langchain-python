"""Advanced agent builder."""

from collections.abc import Callable, Sequence
from typing import Any

from deepagents import CompiledSubAgent, SubAgent
from deepagents import create_deep_agent as _create_deep_agent
from deepagents.backends import BackendProtocol
from deepagents.backends.protocol import BackendFactory
from langchain.agents.structured_output import ResponseFormat, ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph
from pydantic import BaseModel
from uipath.core.chat import UiPathConversationMessageData

from uipath_langchain.agent.react.job_attachments import get_job_attachment_paths
from uipath_langchain.deepagents.backend import create_workspace_backend_factory

from .types import AdvancedAgentGraphState, ConversationalAdvancedAgentGraphState
from .utils import (
    MEMORY_INDEX_VIRTUAL_PATH,
    create_state_with_input,
    is_workspace_filesystem_backend,
    resolve_input_attachments,
)


def create_advanced_agent(
    model: BaseChatModel,
    system_prompt: str = "",
    tools: Sequence[BaseTool] = (),
    subagents: Sequence[SubAgent | CompiledSubAgent] = (),
    backend: BackendProtocol | BackendFactory | None = None,
    response_format: ResponseFormat[Any] | None = None,
    memory: Sequence[str] = (),
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Create a deepagents agent with planning, filesystem, and sub-agent tools.

    ``memory`` is a list of file paths loaded via deepagents' ``MemoryMiddleware``:
    each is read from ``backend`` and injected into the system prompt every turn,
    and the model maintains them with ``edit_file``. Empty disables the middleware.
    """
    return _create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=list(tools),
        subagents=list(subagents),
        backend=backend,
        response_format=response_format,
        memory=list(memory) or None,
    )


def create_advanced_agent_graph(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    system_prompt: str,
    input_schema: type[BaseModel] | None,
    output_schema: type[BaseModel],
    build_user_message: Callable[[dict[str, Any]], str],
    build_system_message: Callable[[dict[str, Any]], str] | None = None,
    subagents: Sequence[SubAgent | CompiledSubAgent] = (),
) -> StateGraph[Any, Any, Any, Any]:
    """Wrap the advanced agent in a parent graph that maps typed I/O to/from messages.

    The UiPath workspace backend and structured-output strategy are selected by
    the integration. Attachment-shaped inputs are downloaded into the workspace
    and given a ``FilePath`` before the user message is built. DeepAgents'
    ``MemoryMiddleware`` reads ``/memory/MEMORY.md`` from that workspace each
    turn.
    """
    backend = create_workspace_backend_factory()
    response_format = ToolStrategy(output_schema)
    memory_sources = (
        [MEMORY_INDEX_VIRTUAL_PATH] if is_workspace_filesystem_backend(backend) else []
    )

    inner_graph = create_advanced_agent(
        model=model,
        tools=tools,
        subagents=subagents,
        system_prompt=system_prompt,
        backend=backend,
        response_format=response_format,
        memory=memory_sources,
    )

    wrapper_state = create_state_with_input(input_schema)
    internal_fields = set(AdvancedAgentGraphState.model_fields.keys())
    attachment_paths = (
        get_job_attachment_paths(input_schema) if input_schema is not None else []
    )

    async def transform_input_async(
        state: BaseModel,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        state_data = state.model_dump()
        input_data = {k: v for k, v in state_data.items() if k not in internal_fields}
        input_args = (
            input_schema.model_validate(input_data).model_dump(by_alias=True)
            if input_schema is not None
            else {}
        )
        if attachment_paths:
            input_args = await resolve_input_attachments(
                backend,
                attachment_paths,
                input_args,
                state=state,
                config=config,
            )
        messages = []
        if build_system_message is not None:
            system_text = build_system_message(input_args)
            if system_text:
                messages.append(SystemMessage(content=system_text, id="system-input"))
        user_text = build_user_message(input_args)
        messages.append(HumanMessage(content=user_text, id="user-input"))
        return {"messages": messages}

    def transform_output(state: BaseModel) -> dict[str, Any]:
        structured = getattr(state, "structured_response", {})
        return output_schema.model_validate(structured).model_dump()

    wrapper: StateGraph[Any, Any, Any, Any] = StateGraph(
        wrapper_state, input_schema=input_schema, output_schema=output_schema
    )
    wrapper.add_node("transform_input", transform_input_async)
    wrapper.add_node("advanced_agent", inner_graph)
    wrapper.add_node("transform_output", transform_output)
    wrapper.add_edge(START, "transform_input")
    wrapper.add_edge("transform_input", "advanced_agent")
    wrapper.add_edge("advanced_agent", "transform_output")
    wrapper.add_edge("transform_output", END)

    return wrapper


def create_conversational_advanced_agent_graph(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    system_prompt: str,
    backend: BackendProtocol | BackendFactory | None,
    subagents: Sequence[SubAgent | CompiledSubAgent] = (),
) -> StateGraph[Any, Any, Any, Any]:
    """Wrap the advanced agent in a parent graph that speaks the conversational contract.

    Conversational agents receive the full conversation history in the
    ``messages`` input each exchange and must output the newly produced
    messages as ``uipath__agent_response_messages``. The deepagent already
    operates on ``messages``, so the wrapper only records the incoming history
    size and maps the new messages to the conversational output field.
    """
    # deferred: avoids a circular import (runtime.messages imports agent modules)
    from uipath_langchain.runtime.messages import UiPathChatMessagesMapper

    memory_sources = (
        [MEMORY_INDEX_VIRTUAL_PATH] if is_workspace_filesystem_backend(backend) else []
    )

    inner_graph = create_advanced_agent(
        model=model,
        tools=tools,
        subagents=subagents,
        system_prompt=system_prompt,
        backend=backend,
        memory=memory_sources,
    )

    class ConversationalAdvancedAgentOutput(BaseModel):
        uipath__agent_response_messages: list[UiPathConversationMessageData] = []

    def capture_exchange_start(
        state: ConversationalAdvancedAgentGraphState,
    ) -> dict[str, Any]:
        return {"initial_message_count": len(state.messages)}

    def transform_output(
        state: ConversationalAdvancedAgentGraphState,
    ) -> dict[str, Any]:
        initial_count = state.initial_message_count or 0
        new_messages = state.messages[initial_count:]
        converted = (
            UiPathChatMessagesMapper.map_langchain_messages_to_uipath_message_data_list(
                messages=new_messages, include_tool_results=False
            )
            if new_messages
            else []
        )
        return {"uipath__agent_response_messages": converted}

    wrapper: StateGraph[Any, Any, Any, Any] = StateGraph(
        ConversationalAdvancedAgentGraphState,
        output_schema=ConversationalAdvancedAgentOutput,
    )
    wrapper.add_node("capture_exchange_start", capture_exchange_start)
    wrapper.add_node("advanced_agent", inner_graph)
    wrapper.add_node("transform_output", transform_output)
    wrapper.add_edge(START, "capture_exchange_start")
    wrapper.add_edge("capture_exchange_start", "advanced_agent")
    wrapper.add_edge("advanced_agent", "transform_output")
    wrapper.add_edge("transform_output", END)

    return wrapper
