"""Advanced agent builder."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, NotRequired, cast

from deepagents import CompiledSubAgent, SubAgent
from deepagents import create_deep_agent as _create_deep_agent
from deepagents.backends import BackendProtocol, FilesystemBackend
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    TodoListMiddleware,
)
from langchain.agents.structured_output import ResponseFormat
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph
from pydantic import BaseModel, ConfigDict, Field, create_model
from uipath.core.chat import UiPathConversationMessageData

from uipath_langchain._utils import get_unique_model_field_name
from uipath_langchain.agent.attachments.job_attachments import get_job_attachment_paths
from uipath_langchain.runtime.messages import UiPathChatMessagesMapper

from .types import (
    AdvancedAgentGraphState,
    ConversationalAdvancedAgentGraphState,
    _ConversationalAdvancedAgentGraphInput,
)
from .utils import (
    MEMORY_INDEX_VIRTUAL_PATH,
    create_state_with_input,
    resolve_input_attachments,
)


class _RuntimeSystemPromptMiddleware(AgentMiddleware[AgentState[Any], Any]):
    """Attach a once-resolved invocation prompt to every model request."""

    def __init__(self, state_key: str) -> None:
        self.state_key = state_key
        self.state_schema = type(
            "RuntimeSystemPromptState",
            (AgentState,),
            {"__annotations__": {state_key: NotRequired[str | None]}},
        )

    def _prepare_request(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        runtime_prompt = cast("str | None", request.state.get(self.state_key))
        if runtime_prompt is None:
            return request

        if request.system_message is None:
            system_message = SystemMessage(content=runtime_prompt)
        else:
            system_message = request.system_message.model_copy(
                update={
                    "content": [
                        {"type": "text", "text": f"{runtime_prompt}\n\n"},
                        *request.system_message.content_blocks,
                    ]
                }
            )
        return request.override(system_message=system_message)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        return handler(self._prepare_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        return await handler(self._prepare_request(request))


@dataclass(frozen=True)
class _RuntimeSystemPrompt:
    """A system prompt that is either fixed or resolved from each invocation's input."""

    static_prompt: str | None
    build_prompt: Callable[[dict[str, Any]], str] | None
    state_key: str | None

    @property
    def middleware(self) -> list[AgentMiddleware[Any, Any]]:
        if self.state_key is None:
            return []
        return [_RuntimeSystemPromptMiddleware(self.state_key)]

    @property
    def state_fields(self) -> dict[str, Any]:
        if self.state_key is None:
            return {}
        return {self.state_key: (str | None, None)}

    def resolve(self, input_args: dict[str, Any]) -> dict[str, Any]:
        """Build the state update carrying the prompt for this invocation."""
        if self.build_prompt is None or self.state_key is None:
            return {}
        return {self.state_key: self.build_prompt(input_args)}


def _resolve_runtime_system_prompt(
    system_prompt: str | Callable[[dict[str, Any]], str],
    base_state: type[BaseModel],
    input_schema: type[BaseModel] | None,
) -> _RuntimeSystemPrompt:
    if not callable(system_prompt):
        return _RuntimeSystemPrompt(system_prompt, None, None)
    state_key = get_unique_model_field_name(
        "uipath__system_prompt", base_state, input_schema
    )
    return _RuntimeSystemPrompt(None, system_prompt, state_key)


def create_advanced_agent(
    model: BaseChatModel,
    system_prompt: str | SystemMessage | None = "",
    tools: Sequence[BaseTool] = (),
    subagents: Sequence[SubAgent | CompiledSubAgent] = (),
    backend: BackendProtocol | None = None,
    response_format: ResponseFormat[Any] | None = None,
    memory: Sequence[str] = (),
    middleware: Sequence[AgentMiddleware[Any, Any]] = (),
    skills: Sequence[str] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Create a deepagents agent with planning, filesystem, and sub-agent tools.

    ``memory`` is a list of file paths loaded via deepagents' ``MemoryMiddleware``:
    each is read from ``backend`` and injected into the system prompt every turn,
    and the model maintains them with ``edit_file``. Empty disables the middleware.

    ``skills`` is a list of skill source paths for deepagents' ``SkillsMiddleware``;
    ``None`` or empty disables it (mirroring ``_create_deep_agent``'s contract).

    ``TodoListMiddleware`` is prepended so ``write_todos`` stays available: it left
    the deepagents default stack in 0.7.0. Callers may replace it by passing their
    own instance, which deepagents matches on ``.name``.
    """
    return _create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=list(tools),
        subagents=list(subagents),
        backend=backend,
        response_format=response_format,
        memory=list(memory) or None,
        middleware=[TodoListMiddleware(), *middleware],
        skills=list(skills) if skills else None,
    )


def create_advanced_agent_graph(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    system_prompt: str | Callable[[dict[str, Any]], str],
    backend: BackendProtocol | None,
    response_format: ResponseFormat[Any] | None,
    input_schema: type[BaseModel] | None,
    output_schema: type[BaseModel],
    build_user_message: Callable[[dict[str, Any]], str],
    skills: Sequence[str] | None = None,
    middleware: Sequence[AgentMiddleware[Any, Any]] = (),
) -> StateGraph[Any, Any, Any, Any]:
    """Wrap the advanced agent in a parent graph that maps typed I/O to/from messages.

    With a ``FilesystemBackend``, attachment-shaped inputs are downloaded into the
    workspace and given a ``FilePath`` before the user message is built. A
    ``FilesystemBackend`` also enables workspace memory: deepagents'
    ``MemoryMiddleware`` reads ``/memory/MEMORY.md`` from the backend each turn.
    Memory stays disabled for non-filesystem backends, which carry no workspace.
    """
    memory_sources = (
        [MEMORY_INDEX_VIRTUAL_PATH] if isinstance(backend, FilesystemBackend) else []
    )
    runtime_prompt = _resolve_runtime_system_prompt(
        system_prompt, AdvancedAgentGraphState, input_schema
    )

    inner_graph = create_advanced_agent(
        model=model,
        tools=tools,
        system_prompt=runtime_prompt.static_prompt,
        backend=backend,
        response_format=response_format,
        memory=memory_sources,
        middleware=[*runtime_prompt.middleware, *middleware],
        skills=skills,
    )

    wrapper_state = create_state_with_input(input_schema)
    if runtime_prompt.state_fields:
        wrapper_state = create_model(
            "RuntimeAdvancedAgentGraphState",
            __base__=wrapper_state,
            **runtime_prompt.state_fields,
        )
    internal_fields = set(AdvancedAgentGraphState.model_fields) | set(
        runtime_prompt.state_fields
    )
    attachment_paths = (
        get_job_attachment_paths(input_schema) if input_schema is not None else []
    )

    async def transform_input_async(state: BaseModel) -> dict[str, Any]:
        state_data = state.model_dump()
        input_data = {k: v for k, v in state_data.items() if k not in internal_fields}
        input_args = (
            input_schema.model_validate(input_data).model_dump(by_alias=True)
            if input_schema is not None
            else {}
        )
        if attachment_paths:
            input_args = await resolve_input_attachments(
                backend, attachment_paths, input_args
            )
        user_text = build_user_message(input_args)
        update: dict[str, Any] = {
            "messages": [HumanMessage(content=user_text, id="user-input")]
        }
        update.update(runtime_prompt.resolve(input_args))
        return update

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
    system_prompt: str | Callable[[dict[str, Any]], str],
    backend: BackendProtocol | None,
    skills: Sequence[str] | None = None,
    input_schema: type[BaseModel] | None = None,
    middleware: Sequence[AgentMiddleware[Any, Any]] = (),
) -> StateGraph[Any, Any, Any, Any]:
    """Wrap the advanced agent in a parent graph that speaks the conversational contract.

    Conversational agents receive the full conversation history in the
    ``messages`` input each exchange and must output the newly produced
    messages as ``uipath__agent_response_messages``. Callable system prompts
    are resolved once from the exchange input and used by the deep agent for
    that invocation.
    """
    memory_sources = (
        [MEMORY_INDEX_VIRTUAL_PATH] if isinstance(backend, FilesystemBackend) else []
    )
    runtime_prompt = _resolve_runtime_system_prompt(
        system_prompt, _ConversationalAdvancedAgentGraphInput, input_schema
    )
    initial_message_count_key = get_unique_model_field_name(
        "initial_message_count",
        _ConversationalAdvancedAgentGraphInput,
        input_schema,
    )

    inner_graph = create_advanced_agent(
        model=model,
        tools=tools,
        system_prompt=runtime_prompt.static_prompt,
        backend=backend,
        memory=memory_sources,
        middleware=[*runtime_prompt.middleware, *middleware],
        skills=skills,
    )

    class ConversationalAdvancedAgentOutput(BaseModel):
        uipath__agent_response_messages: list[UiPathConversationMessageData] = Field(
            default_factory=list
        )

    graph_input: type[BaseModel] = _ConversationalAdvancedAgentGraphInput
    wrapper_input: type[BaseModel] = _ConversationalAdvancedAgentGraphInput
    if input_schema:
        conflicting_fields = [
            field_name
            for field_name, field in input_schema.model_fields.items()
            if field_name != "messages" and field.alias == "messages"
        ]
        if conflicting_fields:
            raise ValueError(
                "Conversational input fields cannot use the reserved 'messages' alias: "
                + ", ".join(conflicting_fields)
            )
        wrapper_input = create_state_with_input(
            input_schema,
            base=_ConversationalAdvancedAgentGraphInput,
            name="CompleteConversationalAdvancedAgentInput",
            model_config=ConfigDict(validate_by_alias=True, validate_by_name=True),
        )
        graph_input = (
            input_schema if "messages" in input_schema.model_fields else wrapper_input
        )

    state_fields: dict[str, Any] = {
        initial_message_count_key: (int | None, None),
        **runtime_prompt.state_fields,
    }
    wrapper_state = cast(
        type[BaseModel],
        create_model(
            "ConversationalAdvancedAgentGraphState",
            __base__=wrapper_input,
            **state_fields,
        ),
    )

    internal_fields = set(_ConversationalAdvancedAgentGraphInput.model_fields) | set(
        state_fields
    )

    def declared_input(state: BaseModel) -> dict[str, Any]:
        """The exchange input as declared by the agent, without the wrapper's fields."""
        if input_schema is None:
            return {}
        return input_schema.model_construct(
            **{
                field_name: getattr(state, field_name)
                for field_name in input_schema.model_fields
                if field_name not in internal_fields
            }
        ).model_dump(by_alias=True, exclude_unset=True)

    def capture_exchange_start(state: BaseModel) -> dict[str, Any]:
        messages = cast(ConversationalAdvancedAgentGraphState, state).messages
        update: dict[str, Any] = {initial_message_count_key: len(messages)}
        if runtime_prompt.build_prompt is not None:
            update.update(runtime_prompt.resolve(declared_input(state)))
        return update

    def transform_output(state: BaseModel) -> dict[str, Any]:
        initial_count = getattr(state, initial_message_count_key) or 0
        messages = cast(ConversationalAdvancedAgentGraphState, state).messages
        new_messages = messages[initial_count:]
        converted = (
            UiPathChatMessagesMapper.map_langchain_messages_to_uipath_message_data_list(
                messages=new_messages, include_tool_results=False
            )
            if new_messages
            else []
        )
        return {"uipath__agent_response_messages": converted}

    wrapper: StateGraph[Any, Any, Any, Any] = StateGraph(
        wrapper_state,
        input_schema=graph_input,
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
