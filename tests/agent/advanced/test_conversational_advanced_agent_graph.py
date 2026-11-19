"""Tests for the conversational advanced agent wrapper builder."""

import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from uipath.agent.react import SET_CONVERSATIONAL_OUTPUT_TOOL
from uipath.agent.react.conversational_prompts import get_chat_system_prompt

from uipath_langchain._utils._attachments import render_attachments_block
from uipath_langchain.agent.advanced.agent import (
    _RuntimeSystemPromptMiddleware,
    create_conversational_advanced_agent_graph,
)
from uipath_langchain.agent.advanced.types import (
    ConversationalAdvancedAgentGraphState,
)
from uipath_langchain.agent.react.utils import (
    build_conversational_output_args_schema,
)


class _Input(BaseModel):
    messages: list[Any] = Field(default_factory=list)
    tenant: str = ""
    uipath__user_settings: dict[str, Any] = Field(default_factory=dict)


class _InputWithoutMessages(BaseModel):
    tenant: str


class _AliasedInput(BaseModel):
    messages: list[Any] = Field(default_factory=list)
    tenant_name: str = Field(alias="tenantName")


class _CollidingInput(BaseModel):
    messages: list[Any] = Field(default_factory=list)
    initial_message_count: str
    uipath__system_prompt: str


class _ReservedAliasInput(BaseModel):
    history: list[Any] = Field(alias="messages")


def _mock_model() -> MagicMock:
    model = MagicMock(spec=BaseChatModel)
    model.profile = None
    return model


def _fake_inner_agent() -> Any:
    """A stand-in deepagent that appends one AI message."""

    def respond(state: ConversationalAdvancedAgentGraphState) -> dict[str, Any]:
        return {"messages": [AIMessage(content="here is my plan", id="ai-1")]}

    builder: StateGraph[Any, Any, Any, Any] = StateGraph(
        ConversationalAdvancedAgentGraphState
    )
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    builder.add_edge("respond", END)
    return builder.compile()


def test_wrapper_graph_has_conversational_nodes() -> None:
    graph = create_conversational_advanced_agent_graph(
        model=_mock_model(), tools=[], system_prompt="sys", backend=None
    )
    assert {
        "capture_exchange_start",
        "advanced_agent",
        "transform_output",
    } <= set(graph.nodes)


def _runtime_prompt_middleware(
    middleware: Sequence[Any],
) -> _RuntimeSystemPromptMiddleware | None:
    """The runtime-prompt middleware in the stack handed to deepagents, if any.

    Located by type rather than by index, since callers may append middleware of
    their own and the stack order is not part of the contract.
    """
    found = [m for m in middleware if isinstance(m, _RuntimeSystemPromptMiddleware)]
    assert len(found) <= 1, (
        f"expected at most one runtime-prompt middleware, got {found}"
    )
    return found[0] if found else None


def test_callable_system_prompt_enables_runtime_middleware() -> None:
    with patch(
        "uipath_langchain.agent.advanced.agent._create_deep_agent",
        return_value=MagicMock(),
    ) as create_deep_agent:
        create_conversational_advanced_agent_graph(
            model=_mock_model(),
            tools=[],
            system_prompt=lambda args: f"system:{args}",
            backend=None,
            input_schema=_Input,
        )

    call_kwargs = create_deep_agent.call_args.kwargs
    assert call_kwargs["system_prompt"] is None
    middleware = _runtime_prompt_middleware(call_kwargs["middleware"])
    assert middleware is not None
    assert middleware.state_key == "uipath__system_prompt"


def test_static_system_prompt_skips_runtime_middleware() -> None:
    with patch(
        "uipath_langchain.agent.advanced.agent._create_deep_agent",
        return_value=MagicMock(),
    ) as create_deep_agent:
        create_conversational_advanced_agent_graph(
            model=_mock_model(),
            tools=[],
            system_prompt="sys",
            backend=None,
            input_schema=_Input,
        )

    call_kwargs = create_deep_agent.call_args.kwargs
    assert call_kwargs["system_prompt"] == "sys"
    assert _runtime_prompt_middleware(call_kwargs["middleware"]) is None


@pytest.mark.asyncio
async def test_resolves_system_prompt_from_exchange_input() -> None:
    prompt_inputs: list[dict[str, Any]] = []

    def build_system_prompt(input_arguments: dict[str, Any]) -> str:
        prompt_inputs.append(input_arguments)
        return (
            f"system:{input_arguments['tenant']}:"
            f"{input_arguments['uipath__user_settings']['name']}"
        )

    graph = create_conversational_advanced_agent_graph(
        model=_mock_model(),
        tools=[],
        system_prompt=build_system_prompt,
        backend=None,
        input_schema=_Input,
    )
    state = graph.state_schema(
        messages=[HumanMessage(content="hi")],
        tenant="finance",
        uipath__user_settings={"name": "Ada"},
    )

    capture_exchange_start = cast(Any, graph.nodes["capture_exchange_start"].runnable)
    update = await capture_exchange_start.ainvoke(state)

    assert prompt_inputs == [
        {
            "tenant": "finance",
            "uipath__user_settings": {"name": "Ada"},
        }
    ]
    assert update == {
        "initial_message_count": 1,
        "uipath__system_prompt": "system:finance:Ada",
    }


@pytest.mark.asyncio
async def test_serializes_input_aliases_for_prompt() -> None:
    prompt_inputs: list[dict[str, Any]] = []

    def build_system_prompt(input_arguments: dict[str, Any]) -> str:
        prompt_inputs.append(input_arguments)
        return "system"

    with patch(
        "uipath_langchain.agent.advanced.agent.create_advanced_agent",
        return_value=_fake_inner_agent(),
    ):
        graph = create_conversational_advanced_agent_graph(
            model=_mock_model(),
            tools=[],
            system_prompt=build_system_prompt,
            backend=None,
            input_schema=_AliasedInput,
        ).compile()
        await graph.ainvoke(
            {
                "messages": [HumanMessage(content="hi", id="u1")],
                "tenant_name": "finance",
            }
        )

    assert prompt_inputs == [{"tenantName": "finance"}]


@pytest.mark.asyncio
async def test_custom_input_schema_preserves_conversation_messages() -> None:
    with patch(
        "uipath_langchain.agent.advanced.agent.create_advanced_agent",
        return_value=_fake_inner_agent(),
    ):
        graph = create_conversational_advanced_agent_graph(
            model=_mock_model(),
            tools=[],
            system_prompt=lambda args: f"system:{args['tenant']}",
            backend=None,
            input_schema=_InputWithoutMessages,
        ).compile()
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="hi", id="u1")],
                "tenant": "finance",
            }
        )

    assert len(result["uipath__agent_response_messages"]) == 1


@pytest.mark.asyncio
async def test_internal_state_fields_do_not_collide_with_input_fields() -> None:
    prompt_inputs: list[dict[str, Any]] = []

    def build_system_prompt(input_arguments: dict[str, Any]) -> str:
        prompt_inputs.append(input_arguments)
        return "resolved"

    graph = create_conversational_advanced_agent_graph(
        model=_mock_model(),
        tools=[],
        system_prompt=build_system_prompt,
        backend=None,
        input_schema=_CollidingInput,
    )
    state = graph.state_schema(
        messages=[HumanMessage(content="hi")],
        initial_message_count="custom count",
        uipath__system_prompt="custom prompt",
    )

    capture_exchange_start = cast(Any, graph.nodes["capture_exchange_start"].runnable)
    update = await capture_exchange_start.ainvoke(state)

    assert prompt_inputs == [
        {
            "initial_message_count": "custom count",
            "uipath__system_prompt": "custom prompt",
        }
    ]
    assert update == {
        "initial_message_count_1": 1,
        "uipath__system_prompt_1": "resolved",
    }


def test_rejects_custom_input_alias_that_collides_with_messages() -> None:
    with pytest.raises(ValueError, match="reserved 'messages' alias: history"):
        create_conversational_advanced_agent_graph(
            model=_mock_model(),
            tools=[],
            system_prompt=lambda _: "system",
            backend=None,
            input_schema=_ReservedAliasInput,
        )


@pytest.mark.asyncio
async def test_runtime_prompt_reaches_deep_agent_model_request() -> None:
    captured_requests: list[ModelRequest[Any]] = []

    def create_inner_graph(**kwargs: Any) -> Any:
        middleware = _runtime_prompt_middleware(kwargs["middleware"])
        assert middleware is not None

        def respond(state: BaseModel) -> dict[str, Any]:
            state_data = state.model_dump()
            state_data["messages"] = cast(Any, state).messages
            request = ModelRequest(
                model=_mock_model(),
                messages=state_data["messages"],
                system_message=SystemMessage(content="deepagents prompt"),
                state=cast(Any, state_data),
            )

            def handler(prepared: ModelRequest[Any]) -> ModelResponse[Any]:
                captured_requests.append(prepared)
                return ModelResponse(result=[])

            middleware.wrap_model_call(request, handler)
            return {"messages": [AIMessage(content="done", id="ai-1")]}

        return RunnableLambda(respond)

    with patch(
        "uipath_langchain.agent.advanced.agent._create_deep_agent",
        side_effect=create_inner_graph,
    ):
        graph = create_conversational_advanced_agent_graph(
            model=_mock_model(),
            tools=[],
            system_prompt=lambda args: f"system:{args['tenant']}",
            backend=None,
            input_schema=_Input,
        ).compile()
        await graph.ainvoke(
            {
                "messages": [HumanMessage(content="hi", id="u1")],
                "tenant": "finance",
                "uipath__user_settings": {"name": "Ada"},
            }
        )

    assert len(captured_requests) == 1
    assert captured_requests[0].system_message is not None
    assert (
        captured_requests[0].system_message.text
        == "system:finance\n\ndeepagents prompt"
    )


@pytest.mark.asyncio
async def test_outputs_only_new_messages_as_response_messages() -> None:
    with patch(
        "uipath_langchain.agent.advanced.agent.create_advanced_agent",
        return_value=_fake_inner_agent(),
    ):
        graph = create_conversational_advanced_agent_graph(
            model=_mock_model(), tools=[], system_prompt="sys", backend=None
        ).compile()

    history = [
        HumanMessage(content="hi", id="u1"),
        AIMessage(content="hello", id="a1"),
        HumanMessage(content="make a plan", id="u2"),
    ]
    result = await graph.ainvoke({"messages": history})

    response_messages = result["uipath__agent_response_messages"]
    assert len(response_messages) == 1
    assert response_messages[0].role == "assistant"
    assert response_messages[0].content_parts[0].data.inline == "here is my plan"


@pytest.mark.asyncio
async def test_empty_history_still_produces_response() -> None:
    with patch(
        "uipath_langchain.agent.advanced.agent.create_advanced_agent",
        return_value=_fake_inner_agent(),
    ):
        graph = create_conversational_advanced_agent_graph(
            model=_mock_model(), tools=[], system_prompt="sys", backend=None
        ).compile()

    result = await graph.ainvoke({"messages": [HumanMessage(content="hi", id="u1")]})

    assert len(result["uipath__agent_response_messages"]) == 1


def _conversational_output_model(**properties: dict[str, Any]) -> type[BaseModel]:
    """Build an output model the way the runtime does, from the agent's JSON schema."""
    from uipath_langchain.agent.react.jsonschema_pydantic_converter import (
        create_model as create_model_from_schema,
    )

    return create_model_from_schema(
        {
            "type": "object",
            "properties": {
                "uipath__agent_response_messages": {"type": "array"},
                **properties,
            },
        }
    )


_OutputWithCustomFields = _conversational_output_model(
    ticketId={"type": "string"}, resolved={"type": "boolean"}
)
_OutputMessagesOnly = _conversational_output_model()


class TestCustomOutputFields:
    """Declared output fields are filled by the same extraction call standard
    conversational agents use: the loop produces messages, not fields."""

    def test_custom_fields_insert_the_extraction_node(self) -> None:
        graph = create_conversational_advanced_agent_graph(
            model=_mock_model(),
            tools=[],
            system_prompt="sys",
            backend=None,
            output_schema=_OutputWithCustomFields,
        )

        assert "generate_conversational_output" in set(graph.nodes)

    def test_messages_only_output_skips_the_extraction_node(self) -> None:
        graph = create_conversational_advanced_agent_graph(
            model=_mock_model(),
            tools=[],
            system_prompt="sys",
            backend=None,
            output_schema=_OutputMessagesOnly,
        )

        assert "generate_conversational_output" not in set(graph.nodes)

    def test_no_output_schema_skips_the_extraction_node(self) -> None:
        graph = create_conversational_advanced_agent_graph(
            model=_mock_model(), tools=[], system_prompt="sys", backend=None
        )

        assert "generate_conversational_output" not in set(graph.nodes)

    def test_extraction_state_key_does_not_collide_with_input(self) -> None:
        class _Colliding(BaseModel):
            messages: list[Any] = Field(default_factory=list)
            uipath__conversational_output: str = ""

        graph = create_conversational_advanced_agent_graph(
            model=_mock_model(),
            tools=[],
            system_prompt="sys",
            backend=None,
            input_schema=_Colliding,
            output_schema=_OutputWithCustomFields,
        )

        assert "uipath__conversational_output_1" in graph.state_schema.model_fields

    @pytest.mark.asyncio
    async def test_extracted_fields_are_merged_into_the_output(self) -> None:
        with (
            patch(
                "uipath_langchain.agent.advanced.agent.create_advanced_agent",
                return_value=_fake_inner_agent(),
            ),
            patch(
                "uipath_langchain.agent.advanced.agent.create_conversational_output_extractor",
                return_value=_extractor({"ticketId": "INC-42", "resolved": True}),
            ),
        ):
            graph = create_conversational_advanced_agent_graph(
                model=_mock_model(),
                tools=[],
                system_prompt="sys",
                backend=None,
                output_schema=_OutputWithCustomFields,
            ).compile()
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content="hi", id="u1")]}
            )

        assert result["ticketId"] == "INC-42"
        assert result["resolved"] is True
        assert len(result["uipath__agent_response_messages"]) == 1

    @pytest.mark.asyncio
    async def test_extraction_sees_the_whole_transcript(self) -> None:
        """A declared field's answer often lives in an earlier turn, so the
        extraction gets the full history, as the standard path does."""
        seen: list[list[Any]] = []

        async def record(messages: Any) -> dict[str, Any]:
            seen.append(list(messages))
            return {"ticketId": "INC-1"}

        with (
            patch(
                "uipath_langchain.agent.advanced.agent.create_advanced_agent",
                return_value=_fake_inner_agent(),
            ),
            patch(
                "uipath_langchain.agent.advanced.agent.create_conversational_output_extractor",
                return_value=record,
            ),
        ):
            graph = create_conversational_advanced_agent_graph(
                model=_mock_model(),
                tools=[],
                system_prompt="sys",
                backend=None,
                output_schema=_OutputWithCustomFields,
            ).compile()
            await graph.ainvoke(
                {
                    "messages": [
                        HumanMessage(content="older turn", id="u0"),
                        HumanMessage(content="hi", id="u1"),
                    ]
                }
            )

        # The system prompt leads the list (see TestExtractionSeesTheSystemPrompt);
        # what this test pins is that no part of the transcript is dropped.
        assert [message.id for message in seen[0][1:]] == ["u0", "u1", "ai-1"]


class TestExtractionSeesTheSystemPrompt:
    """The advanced conversational loop keeps its system prompt in state and injects it per
    call, so it never reaches ``state.messages``. The extraction step prepends
    the system prompt to the messages array so the LLM-call for output generation has it
    as context (matching standard loop logic).
    """

    @pytest.mark.asyncio
    async def test_a_static_prompt_is_prepended(self) -> None:
        seen = await _extraction_messages(system_prompt="builder prompt")

        assert isinstance(seen[0], SystemMessage)
        assert seen[0].content == "builder prompt"

    @pytest.mark.asyncio
    async def test_a_resolved_prompt_is_prepended(self) -> None:
        """A callable prompt lives in a state field, not on the closure."""
        seen = await _extraction_messages(
            system_prompt=lambda args: f"system:{args['tenant']}",
            input_schema=_Input,
            invoke_input={
                "messages": [HumanMessage(content="hi", id="u1")],
                "tenant": "finance",
            },
        )

        assert isinstance(seen[0], SystemMessage)
        assert seen[0].content == "system:finance"

    @pytest.mark.asyncio
    async def test_an_empty_prompt_prepends_nothing(self) -> None:
        seen = await _extraction_messages(system_prompt="")

        assert not any(isinstance(message, SystemMessage) for message in seen)

    @pytest.mark.asyncio
    async def test_the_prompt_is_not_written_into_graph_state(self) -> None:
        """``transform_output`` emits ``messages[initial_message_count:]``, so a
        system message persisted into state would surface to the user as a
        response message."""
        seen: list[list[Any]] = []

        async def record(messages: Any) -> dict[str, Any]:
            seen.append(list(messages))
            return {"ticketId": "INC-1"}

        with (
            patch(
                "uipath_langchain.agent.advanced.agent.create_advanced_agent",
                return_value=_fake_inner_agent(),
            ),
            patch(
                "uipath_langchain.agent.advanced.agent.create_conversational_output_extractor",
                return_value=record,
            ),
        ):
            graph = create_conversational_advanced_agent_graph(
                model=_mock_model(),
                tools=[],
                system_prompt="builder prompt",
                backend=None,
                output_schema=_OutputWithCustomFields,
            ).compile()
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content="hi", id="u1")]}
            )

        assert isinstance(seen[0][0], SystemMessage)
        assert len(result["uipath__agent_response_messages"]) == 1

    @pytest.mark.asyncio
    async def test_the_output_hint_section_is_stripped_before_extraction(self) -> None:
        """The prepended prompt carries the loop's output-schema section, which
        tells the model there is no tool for the fields -- the opposite of what
        the extraction call needs. The shared extractor has to strip it, so this
        exercises the real extractor rather than a stand-in."""
        model, bound = _mock_extractor_model()
        loop_prompt = get_chat_system_prompt(
            model="claude-3-sonnet",
            system_message="builder prompt",
            agent_name="Agent",
            output_schema=build_conversational_output_args_schema(
                _OutputWithCustomFields
            ),
        )
        assert "STRUCTURED OUTPUT FOR THIS TURN" in loop_prompt

        with patch(
            "uipath_langchain.agent.advanced.agent.create_advanced_agent",
            return_value=_fake_inner_agent(),
        ):
            graph = create_conversational_advanced_agent_graph(
                model=model,
                tools=[],
                system_prompt=loop_prompt,
                backend=None,
                output_schema=_OutputWithCustomFields,
            ).compile()
            await graph.ainvoke({"messages": [HumanMessage(content="hi", id="u1")]})

        sent = bound.ainvoke.call_args.args[0]
        assert isinstance(sent[0], SystemMessage)
        assert "STRUCTURED OUTPUT FOR THIS TURN" not in sent[0].content
        assert "builder prompt" in sent[0].content


async def _extraction_messages(
    system_prompt: Any,
    input_schema: type[BaseModel] | None = None,
    invoke_input: dict[str, Any] | None = None,
) -> list[Any]:
    """Run the wrapper and return the messages the extraction call received."""
    seen: list[list[Any]] = []

    async def record(messages: Any) -> dict[str, Any]:
        seen.append(list(messages))
        return {"ticketId": "INC-1"}

    with (
        patch(
            "uipath_langchain.agent.advanced.agent.create_advanced_agent",
            return_value=_fake_inner_agent(),
        ),
        patch(
            "uipath_langchain.agent.advanced.agent.create_conversational_output_extractor",
            return_value=record,
        ),
    ):
        graph = create_conversational_advanced_agent_graph(
            model=_mock_model(),
            tools=[],
            system_prompt=system_prompt,
            backend=None,
            input_schema=input_schema,
            output_schema=_OutputWithCustomFields,
        ).compile()
        await graph.ainvoke(
            invoke_input or {"messages": [HumanMessage(content="hi", id="u1")]}
        )

    return seen[0]


def _mock_extractor_model() -> tuple[MagicMock, MagicMock]:
    """A model whose ``model_copy().bind_tools().ainvoke()`` chain records the
    messages and answers with a set_conversational_output call, so the real
    extractor can run against it. Returns (model, bound)."""
    response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": SET_CONVERSATIONAL_OUTPUT_TOOL.name,
                "args": {"ticketId": "INC-1", "resolved": True},
                "id": "call_1",
            }
        ],
    )
    bound = MagicMock()
    bound.ainvoke = AsyncMock(return_value=response)
    non_streaming = MagicMock()
    non_streaming.bind_tools = MagicMock(return_value=bound)
    model = MagicMock(spec=BaseChatModel)
    model.profile = None
    model.model_copy = MagicMock(return_value=non_streaming)
    return model, bound


def _extractor(args: dict[str, Any]) -> Any:
    """An extraction callable that always returns ``args``."""

    async def extract(messages: Any) -> dict[str, Any]:
        return args

    return extract


def _recording_inner_agent(seen: list[Any]) -> Any:
    """A stand-in deepagent that records the messages the wrapper handed it."""

    def respond(state: ConversationalAdvancedAgentGraphState) -> dict[str, Any]:
        seen.extend(state.messages)
        return {"messages": [AIMessage(content="here is my plan", id="ai-1")]}

    builder: StateGraph[Any, Any, Any, Any] = StateGraph(
        ConversationalAdvancedAgentGraphState
    )
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    builder.add_edge("respond", END)
    return builder.compile()


def _attachment_message(attachment_id: uuid.UUID) -> HumanMessage:
    attachments = [
        {
            "id": str(attachment_id),
            "full_name": "uipath_company_report.md",
            "mime_type": "text/markdown",
        }
    ]
    return HumanMessage(
        id="u1",
        content_blocks=[
            {"type": "text", "text": "can you read this file?"},
            {"type": "text", "text": render_attachments_block(attachments)},
        ],
        additional_kwargs={"attachments": attachments},
    )


@pytest.mark.asyncio
async def test_chat_attachments_are_downloaded_and_pathed(tmp_path: Path) -> None:
    """A file attached in the chat reaches the workspace and the model sees its path."""
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    attachment_id = uuid.uuid4()
    seen: list[Any] = []

    mock_client = MagicMock()
    mock_client.attachments.download_async = AsyncMock()
    with (
        patch(
            "uipath_langchain.agent.advanced.agent.create_advanced_agent",
            return_value=_recording_inner_agent(seen),
        ),
        patch(
            "uipath_langchain.agent.advanced.utils.UiPath",
            return_value=mock_client,
        ),
    ):
        graph = create_conversational_advanced_agent_graph(
            model=_mock_model(), tools=[], system_prompt="sys", backend=backend
        ).compile()
        result = await graph.ainvoke({"messages": [_attachment_message(attachment_id)]})

    expected_name = f"{attachment_id}_uipath_company_report.md"
    assert mock_client.attachments.download_async.call_args.kwargs[
        "destination_path"
    ] == str(backend.cwd / expected_name)

    hydrated = next(message for message in seen if message.id == "u1")
    assert hydrated.additional_kwargs["attachments"][0]["file_path"] == (
        f"/{expected_name}"
    )
    assert f"/{expected_name}" in hydrated.content[1]["text"]
    assert len(result["uipath__agent_response_messages"]) == 1


@pytest.mark.asyncio
async def test_chat_attachments_need_a_filesystem_backend() -> None:
    """Without a workspace the attachment block is passed through unchanged."""
    attachment_id = uuid.uuid4()
    message = _attachment_message(attachment_id)
    seen: list[Any] = []

    with patch(
        "uipath_langchain.agent.advanced.agent.create_advanced_agent",
        return_value=_recording_inner_agent(seen),
    ):
        graph = create_conversational_advanced_agent_graph(
            model=_mock_model(), tools=[], system_prompt="sys", backend=None
        ).compile()
        await graph.ainvoke({"messages": [message]})

    unchanged = next(seen_message for seen_message in seen if seen_message.id == "u1")
    assert unchanged.content == message.content
    assert "file_path" not in unchanged.additional_kwargs["attachments"][0]
