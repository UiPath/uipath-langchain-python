"""Tests for the conversational advanced agent wrapper builder."""

from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from uipath_langchain.agent.advanced.agent import (
    _RuntimeSystemPromptMiddleware,
    create_conversational_advanced_agent_graph,
)
from uipath_langchain.agent.advanced.types import (
    ConversationalAdvancedAgentGraphState,
)


class _Input(BaseModel):
    messages: list[Any] = []
    tenant: str = ""
    uipath__user_settings: dict[str, Any] = {}


class _InputWithoutMessages(BaseModel):
    tenant: str


class _AliasedInput(BaseModel):
    messages: list[Any] = []
    tenant_name: str = Field(alias="tenantName")


class _CollidingInput(BaseModel):
    messages: list[Any] = []
    initial_message_count: str
    uipath__system_prompt: str


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
    assert len(call_kwargs["middleware"]) == 1
    middleware = call_kwargs["middleware"][0]
    assert isinstance(middleware, _RuntimeSystemPromptMiddleware)
    assert middleware.state_key == "uipath__system_prompt"


def test_static_system_prompt_keeps_existing_deep_agent_configuration() -> None:
    with patch(
        "uipath_langchain.agent.advanced.agent._create_deep_agent",
        return_value=MagicMock(),
    ) as create_deep_agent:
        create_conversational_advanced_agent_graph(
            model=_mock_model(),
            tools=[],
            system_prompt="system",
            backend=None,
        )

    call_kwargs = create_deep_agent.call_args.kwargs
    assert call_kwargs["system_prompt"] == "system"
    assert call_kwargs["middleware"] == []


@pytest.mark.asyncio
async def test_resolves_system_prompt_from_exchange_input() -> None:
    calls: list[dict[str, Any]] = []

    def build_system_prompt(args: dict[str, Any]) -> str:
        calls.append(args)
        return f"system:{args['tenant']}:{args['uipath__user_settings']['name']}"

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

    assert calls == [
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
async def test_compiled_graph_serializes_input_aliases_for_prompt() -> None:
    calls: list[dict[str, Any]] = []

    def build_system_prompt(args: dict[str, Any]) -> str:
        calls.append(args)
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
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="hi", id="u1")],
                "tenant_name": "finance",
            }
        )

    assert calls == [{"tenantName": "finance"}]
    assert len(result["uipath__agent_response_messages"]) == 1


@pytest.mark.asyncio
async def test_custom_input_schema_does_not_drop_conversation_messages() -> None:
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
    calls: list[dict[str, Any]] = []

    def build_system_prompt(args: dict[str, Any]) -> str:
        calls.append(args)
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

    assert calls == [
        {
            "initial_message_count": "custom count",
            "uipath__system_prompt": "custom prompt",
        }
    ]
    assert update == {
        "initial_message_count_1": 1,
        "uipath__system_prompt_1": "resolved",
    }


@pytest.mark.asyncio
async def test_runtime_prompt_crosses_into_deep_agent() -> None:
    resolver_calls: list[dict[str, Any]] = []
    captured: list[ModelRequest[Any]] = []

    def build_system_prompt(args: dict[str, Any]) -> str:
        resolver_calls.append(args)
        return f"system:{args['tenant']}"

    def create_inner_graph(**kwargs: Any) -> Any:
        middleware = kwargs["middleware"][0]

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
                captured.append(prepared)
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
            system_prompt=build_system_prompt,
            backend=None,
            input_schema=_Input,
        ).compile()
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="hi", id="u1")],
                "tenant": "finance",
                "uipath__user_settings": {"name": "Ada"},
            }
        )

    assert resolver_calls == [
        {
            "tenant": "finance",
            "uipath__user_settings": {"name": "Ada"},
        }
    ]
    assert len(captured) == 1
    assert captured[0].system_message is not None
    assert captured[0].system_message.text == "system:finance\n\ndeepagents prompt"
    assert len(result["uipath__agent_response_messages"]) == 1


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
