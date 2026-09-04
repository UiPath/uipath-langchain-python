"""Tests for the create_advanced_agent_graph wrapper builder."""

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, ConfigDict, Field

from uipath_langchain.agent.advanced.agent import (
    _RuntimeSystemPromptMiddleware,
    create_advanced_agent_graph,
)
from uipath_langchain.agent.advanced.types import AdvancedAgentGraphState
from uipath_langchain.agent.advanced.utils import create_state_with_input


class _Output(BaseModel):
    result: str = ""


class _Input(BaseModel):
    book: dict[str, Any] = {}
    question: str = ""


class _AliasedInput(BaseModel):
    model_config = ConfigDict(validate_by_alias=True, validate_by_name=True)

    schema_: str = Field(alias="schema")
    question: str = ""


class _PromptNamedInput(BaseModel):
    uipath__system_prompt: str
    uipath__system_prompt_1: str


def _mock_model() -> MagicMock:
    model = MagicMock(spec=BaseChatModel)
    model.profile = None
    return model


def _build(**overrides: Any) -> Any:
    kwargs: dict[str, Any] = dict(
        model=_mock_model(),
        tools=[],
        system_prompt="sys",
        backend=None,
        response_format=None,
        input_schema=None,
        output_schema=_Output,
        build_user_message=lambda args: "hello",
    )
    kwargs.update(overrides)
    return create_advanced_agent_graph(**kwargs)


def test_wrapper_graph_has_io_nodes() -> None:
    """The wrapper wires transform_input -> advanced_agent -> transform_output."""
    graph = _build()
    assert {"transform_input", "advanced_agent", "transform_output"} <= set(graph.nodes)


def test_callable_system_prompt_enables_runtime_middleware() -> None:
    """The inner deep agent keeps its base prompt and receives runtime middleware."""
    with patch(
        "uipath_langchain.agent.advanced.agent._create_deep_agent",
        return_value=MagicMock(),
    ) as mock_create:
        _build(system_prompt=lambda args: f"system:{args}")

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["system_prompt"] is None
    assert len(call_kwargs["middleware"]) == 1
    assert isinstance(call_kwargs["middleware"][0], _RuntimeSystemPromptMiddleware)
    assert call_kwargs["middleware"][0].state_key == "uipath__system_prompt"


def test_static_system_prompt_skips_runtime_middleware() -> None:
    """A plain string prompt reaches the deep agent unchanged, with no middleware."""
    with patch(
        "uipath_langchain.agent.advanced.agent._create_deep_agent",
        return_value=MagicMock(),
    ) as mock_create:
        _build(system_prompt="sys")

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["system_prompt"] == "sys"
    assert call_kwargs["middleware"] == []


@pytest.mark.asyncio
async def test_transform_input_without_schema_builds_single_user_message() -> None:
    """With no input schema, the built message comes straight from build_user_message."""
    graph = _build(build_user_message=lambda args: "hi there")
    out = await graph.nodes["transform_input"].runnable.ainvoke(
        AdvancedAgentGraphState()
    )
    message = out["messages"][0]
    assert isinstance(message, HumanMessage)
    assert message.content == "hi there"
    assert message.id == "user-input"
    assert "uipath__system_prompt" not in out


@pytest.mark.asyncio
async def test_transform_input_builds_runtime_system_prompt_once() -> None:
    """A callable system prompt is resolved from invocation input by the pre-node."""
    calls: list[dict[str, Any]] = []

    def build_system_prompt(args: dict[str, Any]) -> str:
        calls.append(args)
        return f"system:{args['question']}"

    graph = _build(
        input_schema=_Input,
        system_prompt=build_system_prompt,
    )
    state_cls = create_state_with_input(_Input)
    state = state_cls(question="runtime value")

    out = await graph.nodes["transform_input"].runnable.ainvoke(state)

    assert calls == [{"book": {}, "question": "runtime value"}]
    assert out["uipath__system_prompt"] == "system:runtime value"


@pytest.mark.asyncio
async def test_internal_prompt_state_does_not_shadow_user_input() -> None:
    """A generated collision is rejected in favor of a fresh internal state key."""
    captured_args: list[dict[str, Any]] = []
    colliding_key = "uipath__system_prompt"
    fresh_key = "uipath__system_prompt_2"

    def build_user_message(args: dict[str, Any]) -> str:
        captured_args.append(args)
        return "user"

    graph = _build(
        input_schema=_PromptNamedInput,
        system_prompt=lambda args: f"system:{args[colliding_key]}",
        build_user_message=build_user_message,
    )
    state_cls = create_state_with_input(_PromptNamedInput)
    state = state_cls(
        uipath__system_prompt="user value",
        uipath__system_prompt_1="another user value",
    )

    out = await graph.nodes["transform_input"].runnable.ainvoke(state)

    assert captured_args == [
        {
            colliding_key: "user value",
            "uipath__system_prompt_1": "another user value",
        }
    ]
    assert out[fresh_key] == "system:user value"


@pytest.mark.asyncio
async def test_runtime_system_prompt_crosses_into_deep_agent_once() -> None:
    """The wrapper carries one resolved prompt into the deep-agent state schema."""
    resolver_calls: list[dict[str, Any]] = []
    captured_requests: list[ModelRequest[Any]] = []
    prepared_blocks = [
        {
            "type": "text",
            "text": "deepagents prompt",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "non_standard", "value": {"custom": "value"}},
    ]

    def build_system_prompt(args: dict[str, Any]) -> str:
        resolver_calls.append(args)
        return f"runtime:{args['question']}"

    def create_inner_graph(**kwargs: Any) -> Any:
        middleware = kwargs["middleware"][0]
        runtime_key = middleware.state_key

        def capture_model_request(state: BaseModel) -> dict[str, Any]:
            state_data = state.model_dump()
            assert runtime_key in state_data
            state_data["messages"] = cast(Any, state).messages
            request = ModelRequest(
                model=_mock_model(),
                messages=state_data["messages"],
                system_message=SystemMessage(content_blocks=cast(Any, prepared_blocks)),
                state=cast(Any, state_data),
            )

            def handler(prepared: ModelRequest[Any]) -> ModelResponse[Any]:
                captured_requests.append(prepared)
                return ModelResponse(result=[])

            middleware.wrap_model_call(request, handler)
            return {"structured_response": {"result": "done"}}

        return RunnableLambda(capture_model_request)

    with patch(
        "uipath_langchain.agent.advanced.agent._create_deep_agent",
        side_effect=create_inner_graph,
    ):
        wrapper = create_advanced_agent_graph(
            model=_mock_model(),
            tools=[],
            system_prompt=build_system_prompt,
            backend=None,
            response_format=None,
            input_schema=_Input,
            output_schema=_Output,
            build_user_message=lambda args: f"user:{args['question']}",
        )
        initial_state = wrapper.state_schema(question="value")
        transform_node = cast(Any, wrapper.nodes["transform_input"].runnable)
        transform_update = await transform_node.ainvoke(initial_state)
        inner_input = wrapper.state_schema(question="value", **transform_update)
        inner_node = cast(Any, wrapper.nodes["advanced_agent"].runnable)

        result = inner_node.invoke(inner_input)

    assert result == {"structured_response": {"result": "done"}}
    assert resolver_calls == [{"book": {}, "question": "value"}]
    assert len(captured_requests) == 1
    request = captured_requests[0]
    assert request.system_message is not None
    assert request.system_message.text == "runtime:value\n\ndeepagents prompt"
    assert request.system_message.content_blocks[1:] == prepared_blocks
    assert isinstance(request.messages[0], HumanMessage)
    assert request.messages[0].content == "user:value"


@pytest.mark.asyncio
async def test_transform_input_resolves_attachments_when_present() -> None:
    """With an input schema and attachment paths, input attachments are resolved first."""
    with (
        patch(
            "uipath_langchain.agent.advanced.agent.get_job_attachment_paths",
            return_value=["$.book"],
        ),
        patch(
            "uipath_langchain.agent.advanced.agent.resolve_input_attachments",
            new_callable=AsyncMock,
        ) as mock_resolve,
    ):
        mock_resolve.return_value = {"book": {"FilePath": "/x"}, "question": "q"}
        graph = _build(
            input_schema=_Input,
            build_user_message=lambda args: f"msg:{args['question']}",
        )
        state_cls = create_state_with_input(_Input)
        state = state_cls(book={"ID": "1"}, question="q")
        out = await graph.nodes["transform_input"].runnable.ainvoke(state)

    mock_resolve.assert_awaited_once()
    assert out["messages"][0].content == "msg:q"


@pytest.mark.asyncio
async def test_transform_input_passes_alias_keyed_args_to_message_builder() -> None:
    """Input args use JSON/schema field names, including aliases."""
    captured_args: dict[str, Any] = {}

    def build_user_message(args: dict[str, Any]) -> str:
        captured_args.update(args)
        return f"schema:{args['schema']}"

    graph = _build(
        input_schema=_AliasedInput,
        build_user_message=build_user_message,
    )
    state_cls = create_state_with_input(_AliasedInput)
    state = state_cls(schema="invoice", question="q")

    out = await graph.nodes["transform_input"].runnable.ainvoke(state)

    assert captured_args == {"schema": "invoice", "question": "q"}
    assert out["messages"][0].content == "schema:invoice"


def test_transform_output_validates_structured_response() -> None:
    """transform_output coerces the agent's structured_response into the output schema."""
    graph = _build()
    out = graph.nodes["transform_output"].runnable.invoke(
        AdvancedAgentGraphState(structured_response={"result": "done"})
    )
    assert out == {"result": "done"}


def test_runtime_system_prompt_middleware_preserves_prepared_prompt() -> None:
    """The runtime prompt is prepended without discarding deepagents' prompt."""
    middleware = _RuntimeSystemPromptMiddleware("runtime_prompt")
    prepared_blocks = [
        {
            "type": "text",
            "text": "deepagents prompt",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "non_standard", "value": {"custom": "value"}},
    ]
    request = ModelRequest(
        model=_mock_model(),
        messages=[],
        system_message=SystemMessage(
            content_blocks=cast(Any, prepared_blocks),
            id="prepared-system-message",
            name="prepared-system",
            additional_kwargs={"provider": "value"},
            response_metadata={"response": "value"},
        ),
        state=cast(
            Any,
            {
                "messages": [],
                middleware.state_key: "runtime prompt",
            },
        ),
    )
    captured: list[ModelRequest[Any]] = []

    def handler(prepared: ModelRequest[Any]) -> ModelResponse[Any]:
        captured.append(prepared)
        return ModelResponse(result=[])

    middleware.wrap_model_call(request, handler)

    assert captured[0].system_message is not None
    assert captured[0].system_message.text == ("runtime prompt\n\ndeepagents prompt")
    assert captured[0].system_message.content_blocks[1:] == prepared_blocks
    assert captured[0].system_message.id == "prepared-system-message"
    assert captured[0].system_message.name == "prepared-system"
    assert captured[0].system_message.additional_kwargs == {"provider": "value"}
    assert captured[0].system_message.response_metadata == {"response": "value"}


@pytest.mark.asyncio
async def test_runtime_system_prompt_middleware_supports_async_model_calls() -> None:
    middleware = _RuntimeSystemPromptMiddleware("runtime_prompt")
    request = ModelRequest(
        model=_mock_model(),
        messages=[],
        system_message=SystemMessage(content="deepagents prompt"),
        state=cast(
            Any,
            {
                "messages": [],
                middleware.state_key: "runtime prompt",
            },
        ),
    )
    captured: list[ModelRequest[Any]] = []

    async def handler(prepared: ModelRequest[Any]) -> ModelResponse[Any]:
        captured.append(prepared)
        return ModelResponse(result=[])

    await middleware.awrap_model_call(request, handler)

    assert captured[0].system_message is not None
    assert captured[0].system_message.text == ("runtime prompt\n\ndeepagents prompt")


class TestOutputFileVerification:
    """The wrapper gates typed output on the declared output file fields."""

    ATTACHMENT_ID = "11111111-1111-1111-1111-111111111111"

    @staticmethod
    def _output_model(required: bool = True) -> type[BaseModel]:
        from uipath_langchain.agent.react.jsonschema_pydantic_converter import (
            create_model as create_model_from_schema,
        )
        from uipath_langchain.agent.tools.internal_tools.schema_utils import (
            JOB_ATTACHMENT_DEFINITION,
        )

        return create_model_from_schema(
            {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "report": {"$ref": "#/definitions/job-attachment"},
                },
                "required": ["report"] if required else [],
                "definitions": {"job-attachment": JOB_ATTACHMENT_DEFINITION},
            }
        )

    @staticmethod
    def _tools() -> list[Any]:
        from uipath_langchain.agent.tools.internal_tools.output_file_tool import (
            create_output_file_tool,
        )

        return [create_output_file_tool()]

    def test_file_output_inserts_the_verification_node(self) -> None:
        graph = _build(output_schema=self._output_model(), tools=self._tools())

        assert "verify_output_files" in set(graph.nodes)

    def test_no_file_output_keeps_the_direct_edge(self) -> None:
        graph = _build(output_schema=_Output, tools=self._tools())

        assert "verify_output_files" not in set(graph.nodes)

    def test_file_output_without_the_tool_is_not_verified(self) -> None:
        """With the feature off the tool is absent, so nothing gates the output."""
        graph = _build(output_schema=self._output_model(), tools=[])

        assert "verify_output_files" not in set(graph.nodes)

    def test_retry_and_problem_state_fields_are_added(self) -> None:
        graph = _build(output_schema=self._output_model(), tools=self._tools())
        fields = set(graph.state_schema.model_fields)

        assert "uipath__output_file_retries" in fields
        assert "uipath__output_file_problem" in fields

    def test_no_file_output_adds_no_verification_state(self) -> None:
        graph = _build(output_schema=_Output, tools=self._tools())
        fields = set(graph.state_schema.model_fields)

        assert "uipath__output_file_retries" not in fields
        assert "uipath__output_file_problem" not in fields

    async def test_verification_state_is_not_forwarded_as_agent_input(self) -> None:
        """The keys are internal, so transform_input must not treat them as inputs."""
        graph = _build(
            input_schema=_Input,
            output_schema=self._output_model(),
            tools=self._tools(),
        )
        state = graph.state_schema(book={"title": "x"}, question="q")

        update = await graph.nodes["transform_input"].runnable.ainvoke(state)

        assert "messages" in update
        assert "uipath__output_file_retries" not in update
