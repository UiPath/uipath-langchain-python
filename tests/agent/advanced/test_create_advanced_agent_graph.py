"""Tests for the create_advanced_agent_graph wrapper builder."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from uipath_langchain.agent.advanced.agent import create_advanced_agent_graph
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


def test_subagents_are_passed_to_inner_advanced_agent() -> None:
    """The typed I/O wrapper preserves DeepAgents sub-agent delegation."""
    subagent = {
        "name": "researcher",
        "description": "Researches one part of the task",
        "system_prompt": "Research carefully.",
    }

    with patch(
        "uipath_langchain.agent.advanced.agent.create_advanced_agent",
        return_value=MagicMock(),
    ) as mock_create:
        _build(subagents=[subagent])

    assert mock_create.call_args.kwargs["subagents"] == [subagent]


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
