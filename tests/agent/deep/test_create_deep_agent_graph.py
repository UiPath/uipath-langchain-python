"""Tests for create_deep_agent_graph wrapper I/O transformations."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import StateGraph
from pydantic import BaseModel

pytest.importorskip("deepagents")

from uipath_langchain.agent.deep import create_deep_agent_graph  # noqa: E402


class _Input(BaseModel):
    topic: str = ""


class _Output(BaseModel):
    answer: str = ""


def _build_user_message(args: dict[str, Any]) -> str:
    return f"Research: {args.get('topic', '')}"


def test_create_deep_agent_graph_returns_state_graph() -> None:
    model = MagicMock(spec=BaseChatModel)

    with patch(
        "uipath_langchain.agent.deep.agent.create_deep_agent",
        return_value=MagicMock(),
    ):
        wrapper = create_deep_agent_graph(
            model=model,
            tools=[],
            system_prompt="hi",
            backend=None,
            response_format=None,
            input_schema=_Input,
            output_schema=_Output,
            build_user_message=_build_user_message,
        )

    assert isinstance(wrapper, StateGraph)
    assert "transform_input" in wrapper.nodes
    assert "deep_agent" in wrapper.nodes
    assert "transform_output" in wrapper.nodes
