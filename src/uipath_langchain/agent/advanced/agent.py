"""Advanced agent builder."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, cast

from deepagents import CompiledSubAgent, SubAgent
from deepagents import create_deep_agent as _create_deep_agent
from deepagents.backends import BackendProtocol, FilesystemBackend
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.agents.structured_output import ResponseFormat
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, create_model
from uipath.core.chat import UiPathConversationMessageData
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain._utils import get_unique_model_field_name
from uipath_langchain.agent.attachments.constants import OUTPUT_FILE_TOOL_NAME
from uipath_langchain.agent.attachments.job_attachments import get_job_attachment_paths
from uipath_langchain.agent.attachments.output_files import (
    DEFAULT_MAX_OUTPUT_FILE_RETRIES,
    diagnose_output_files,
    get_output_file_fields,
)
from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
    max_iterations_error,
)
from uipath_langchain.agent.react.conversational_output_node import (
    create_conversational_output_extractor,
)
from uipath_langchain.agent.react.utils import (
    has_custom_conversational_output_fields,
)
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


class _MaxIterationsMiddleware(AgentMiddleware[AgentState[Any], Any]):
    """Stop the loop once it has spent its iteration budget for this turn.

    Counts the AI messages the agent produced since the turn started, the way the
    standard agent's llm node does, and raises the same termination error. Counting
    messages rather than model calls keeps the budget spent across a suspend and
    resume, where any per-run counter starts over.
    """

    def __init__(
        self, max_iterations: int, initial_message_count_key: str | None = None
    ) -> None:
        self.max_iterations = max_iterations
        self.initial_message_count_key = initial_message_count_key
        if initial_message_count_key is not None:
            self.state_schema = type(
                "MaxIterationsState",
                (AgentState,),
                {
                    "__annotations__": {
                        initial_message_count_key: NotRequired[int | None]
                    }
                },
            )

    def _check_budget(self, request: ModelRequest[Any]) -> None:
        initial_count = (
            cast("int | None", request.state.get(self.initial_message_count_key)) or 0
            if self.initial_message_count_key is not None
            else 0
        )
        messages = cast("list[Any]", request.state.get("messages") or [])
        produced = sum(
            1 for message in messages[initial_count:] if isinstance(message, AIMessage)
        )
        if produced >= self.max_iterations:
            raise max_iterations_error(self.max_iterations)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        self._check_budget(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        self._check_budget(request)
        return await handler(request)


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


def _max_iterations_middleware(
    max_iterations: int | None, initial_message_count_key: str | None = None
) -> list[AgentMiddleware[Any, Any]]:
    if max_iterations is None:
        return []
    return [_MaxIterationsMiddleware(max_iterations, initial_message_count_key)]


# A subagent returns only a text report, so a reference it produces never reaches
# the main agent -- the only agent that fills the typed output.
MAIN_AGENT_ONLY_TOOLS: frozenset[str] = frozenset({OUTPUT_FILE_TOOL_NAME})


def _partition_main_agent_tools(
    tools: Sequence[BaseTool],
) -> tuple[list[BaseTool], list[BaseTool]]:
    """Split ``tools`` into (shared with subagents, main agent only)."""
    shared: list[BaseTool] = []
    main_only: list[BaseTool] = []
    for tool in tools:
        (main_only if tool.name in MAIN_AGENT_ONLY_TOOLS else shared).append(tool)
    return shared, main_only


def _subagents_without_main_agent_tools(
    subagents: Sequence[SubAgent | CompiledSubAgent],
    shared_tools: Sequence[BaseTool],
    skills: Sequence[str] | None,
) -> list[SubAgent | CompiledSubAgent]:
    """Give every subagent the shared tool list instead of the parent's.

    deepagents hands a subagent the parent's ``tools`` unless its spec declares its
    own (``graph.py``: ``spec.get("tools") if "tools" in spec else tools``), so
    pinning ``tools`` on each spec is what actually withholds a main-agent-only tool.

    The auto-added ``general-purpose`` subagent is replaced with an explicit spec,
    since it would otherwise inherit the parent list too. Supplying a spec under
    that name suppresses the built-in one. That branch is also the only reader of
    ``profile.general_purpose_subagent``, so its ``enabled`` / ``description`` /
    ``system_prompt`` overrides do not apply here. ``skills`` has to be repeated into the
    spec: the built-in branch reads the top-level ``skills`` argument, while a
    caller-supplied spec reads ``spec["skills"]``, so omitting it silently drops
    skills from that subagent.
    """
    resolved: list[SubAgent | CompiledSubAgent] = []
    for spec in subagents:
        # A CompiledSubAgent brings its own graph and tools; nothing to filter.
        if "runnable" in spec or "tools" in spec:
            resolved.append(spec)
            continue
        resolved.append({**spec, "tools": list(shared_tools)})

    if not any(
        spec.get("name") == GENERAL_PURPOSE_SUBAGENT["name"] for spec in resolved
    ):
        gp: dict[str, Any] = {
            **GENERAL_PURPOSE_SUBAGENT,
            "tools": list(shared_tools),
        }
        if skills:
            gp["skills"] = list(skills)
        resolved.append(gp)  # type: ignore[arg-type]
    return resolved


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

    Tools named in :data:`MAIN_AGENT_ONLY_TOOLS` are withheld from every subagent.
    """
    shared_tools, _ = _partition_main_agent_tools(tools)
    return _create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=list(tools),
        subagents=_subagents_without_main_agent_tools(subagents, shared_tools, skills),
        backend=backend,
        response_format=response_format,
        memory=list(memory) or None,
        middleware=list(middleware),
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
    output_files_enabled: bool = False,
    max_iterations: int | None = None,
    middleware: Sequence[AgentMiddleware[Any, Any]] = (),
) -> StateGraph[Any, Any, Any, Any]:
    """Wrap the advanced agent in a parent graph that maps typed I/O to/from messages.

    With a ``FilesystemBackend``, attachment-shaped inputs are downloaded into the
    workspace and given a ``FilePath`` before the user message is built. A
    ``FilesystemBackend`` also enables workspace memory: deepagents'
    ``MemoryMiddleware`` reads ``/memory/MEMORY.md`` from the backend each turn.
    Memory stays disabled for non-filesystem backends, which carry no workspace.

    With ``output_files_enabled``, a job-attachment field in the output schema
    is gated by a verification node: an unfilled required file field, or a
    reference to an attachment that is not linked to this job, sends the agent
    back for another turn instead of emitting an output it cannot honor.

    ``max_iterations`` caps the model calls the agent loop may make; ``None``
    leaves it uncapped.
    """
    memory_sources = (
        [MEMORY_INDEX_VIRTUAL_PATH] if isinstance(backend, FilesystemBackend) else []
    )
    runtime_prompt = _resolve_runtime_system_prompt(
        system_prompt, AdvancedAgentGraphState, input_schema
    )
    output_file_fields = (
        get_output_file_fields(output_schema) if output_files_enabled else []
    )

    inner_graph = create_advanced_agent(
        model=model,
        tools=tools,
        system_prompt=runtime_prompt.static_prompt,
        backend=backend,
        response_format=response_format,
        memory=memory_sources,
        middleware=[
            *runtime_prompt.middleware,
            *_max_iterations_middleware(max_iterations),
            *middleware,
        ],
        skills=skills,
    )

    output_file_retries_key = get_unique_model_field_name(
        "uipath__output_file_retries", AdvancedAgentGraphState, input_schema
    )
    state_fields: dict[str, Any] = dict(runtime_prompt.state_fields)
    if output_file_fields:
        state_fields[output_file_retries_key] = (int, 0)

    wrapper_state = create_state_with_input(input_schema)
    if state_fields:
        wrapper_state = create_model(
            "RuntimeAdvancedAgentGraphState",
            __base__=wrapper_state,
            **state_fields,
        )
    internal_fields = set(AdvancedAgentGraphState.model_fields) | set(state_fields)
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

    async def verify_output_files(
        state: BaseModel,
    ) -> Command[Literal["advanced_agent", "transform_output"]]:
        structured = getattr(state, "structured_response", {}) or {}
        problem = await diagnose_output_files(output_file_fields, structured)
        if problem is None:
            return Command(goto="transform_output")

        retries = getattr(state, output_file_retries_key, 0) or 0
        if retries >= DEFAULT_MAX_OUTPUT_FILE_RETRIES:
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.OUTPUT_VALIDATION_ERROR,
                title="Agent did not produce the required output file",
                detail=(
                    f"{problem} The agent was given "
                    f"{DEFAULT_MAX_OUTPUT_FILE_RETRIES} chance(s) to correct this "
                    "and did not. Verify the agent's prompt asks for the file, and "
                    "that the output schema's file fields are the ones you intend."
                ),
                category=UiPathErrorCategory.USER,
            )

        # The structured-output tool call is already answered by this point, so the
        # correction goes in as a new user turn rather than a tool result.
        return Command(
            goto="advanced_agent",
            update={
                "messages": [HumanMessage(content=problem)],
                output_file_retries_key: retries + 1,
            },
        )

    wrapper: StateGraph[Any, Any, Any, Any] = StateGraph(
        wrapper_state, input_schema=input_schema, output_schema=output_schema
    )
    wrapper.add_node("transform_input", transform_input_async)
    wrapper.add_node("advanced_agent", inner_graph)
    wrapper.add_node("transform_output", transform_output)
    wrapper.add_edge(START, "transform_input")
    wrapper.add_edge("transform_input", "advanced_agent")
    if output_file_fields:
        wrapper.add_node("verify_output_files", verify_output_files)
        wrapper.add_edge("advanced_agent", "verify_output_files")
    else:
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
    output_schema: type[BaseModel] | None = None,
    max_iterations: int | None = None,
    middleware: Sequence[AgentMiddleware[Any, Any]] = (),
) -> StateGraph[Any, Any, Any, Any]:
    """Wrap the advanced agent in a parent graph that speaks the conversational contract.

    Conversational agents receive the full conversation history in the
    ``messages`` input each exchange and must output the newly produced
    messages as ``uipath__agent_response_messages``. Callable system prompts
    are resolved once from the exchange input and used by the deep agent for
    that invocation.

    When ``output_schema`` declares fields beyond the response messages, they are
    filled the same way the standard conversational agent fills them: a focused
    extraction call over the exchange's messages, after the loop has finished.
    The loop itself produces messages, so nothing in it can produce those fields.

    ``max_iterations`` caps the model calls the agent loop may make per exchange;
    ``None`` leaves it uncapped.
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
        middleware=[
            *runtime_prompt.middleware,
            *_max_iterations_middleware(max_iterations, initial_message_count_key),
            *middleware,
        ],
        skills=skills,
    )

    class ConversationalAdvancedAgentOutput(BaseModel):
        uipath__agent_response_messages: list[UiPathConversationMessageData] = Field(
            default_factory=list
        )

    with_output_extraction = has_custom_conversational_output_fields(output_schema)
    graph_output: type[BaseModel] = (
        output_schema
        if with_output_extraction and output_schema is not None
        else ConversationalAdvancedAgentOutput
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

    conversational_output_key = get_unique_model_field_name(
        "uipath__conversational_output",
        _ConversationalAdvancedAgentGraphInput,
        input_schema,
    )
    state_fields: dict[str, Any] = {
        initial_message_count_key: (int | None, None),
        **runtime_prompt.state_fields,
    }
    if with_output_extraction:
        state_fields[conversational_output_key] = (dict[str, Any] | None, None)
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

    def _new_messages(state: BaseModel) -> list[Any]:
        initial_count = getattr(state, initial_message_count_key) or 0
        messages = cast(ConversationalAdvancedAgentGraphState, state).messages
        return list(messages[initial_count:])

    def transform_output(state: BaseModel) -> dict[str, Any]:
        new_messages = _new_messages(state)
        converted = (
            UiPathChatMessagesMapper.map_langchain_messages_to_uipath_message_data_list(
                messages=new_messages, include_tool_results=False
            )
            if new_messages
            else []
        )
        if not with_output_extraction or output_schema is None:
            return {"uipath__agent_response_messages": converted}

        custom_fields = getattr(state, conversational_output_key, None) or {}
        output = {
            **custom_fields,
            "uipath__agent_response_messages": [
                message.model_dump(by_alias=True) for message in converted
            ],
        }
        return output_schema.model_validate(output).model_dump(
            by_alias=True, exclude_none=True
        )

    extract_output = (
        create_conversational_output_extractor(model, output_schema)
        if with_output_extraction and output_schema is not None
        else None
    )

    async def generate_conversational_output(state: BaseModel) -> dict[str, Any]:
        assert extract_output is not None  # guarded by with_output_extraction
        messages = cast(ConversationalAdvancedAgentGraphState, state).messages
        return {conversational_output_key: await extract_output(messages)}

    wrapper: StateGraph[Any, Any, Any, Any] = StateGraph(
        wrapper_state,
        input_schema=graph_input,
        output_schema=graph_output,
    )
    wrapper.add_node("capture_exchange_start", capture_exchange_start)
    wrapper.add_node("advanced_agent", inner_graph)
    wrapper.add_node("transform_output", transform_output)
    wrapper.add_edge(START, "capture_exchange_start")
    wrapper.add_edge("capture_exchange_start", "advanced_agent")
    if with_output_extraction:
        wrapper.add_node(
            "generate_conversational_output", generate_conversational_output
        )
        wrapper.add_edge("advanced_agent", "generate_conversational_output")
        wrapper.add_edge("generate_conversational_output", "transform_output")
    else:
        wrapper.add_edge("advanced_agent", "transform_output")
    wrapper.add_edge("transform_output", END)

    return wrapper
