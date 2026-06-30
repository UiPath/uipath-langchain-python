"""Tests for create_agent function in agent.py module."""

from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.runnables.graph import Edge
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph

from uipath_langchain.agent.react.agent import create_agent
from uipath_langchain.agent.react.init_node import create_init_node
from uipath_langchain.agent.react.router import create_route_agent
from uipath_langchain.agent.react.router_conversational import (
    create_route_agent_conversational,
)
from uipath_langchain.agent.react.terminate_node import create_terminate_node
from uipath_langchain.agent.react.tools.tools import create_flow_control_tools
from uipath_langchain.agent.react.types import (
    AgentGraphConfig,
    AgentGraphNode,
)


def _make_mock_model() -> MagicMock:
    """Create a mock chat model."""
    model = MagicMock(spec=BaseChatModel)
    return model


def _make_mock_tool(name: str = "test_tool") -> Mock:
    """Create a mock BaseTool."""
    tool = Mock(spec=BaseTool)
    tool.name = name
    return tool


# Patch targets (all internal functions called by create_agent)
_PATCH_BASE = "uipath_langchain.agent.react.agent"
_PATCHES = {
    "create_init_node": {
        "target": f"{_PATCH_BASE}.create_init_node",
        "wraps": create_init_node,
    },
    "create_terminate_node": {
        "target": f"{_PATCH_BASE}.create_terminate_node",
        "wraps": create_terminate_node,
    },
    "create_flow_control_tools": {
        "target": f"{_PATCH_BASE}.create_flow_control_tools",
        "wraps": create_flow_control_tools,
    },
    "create_route_agent": {
        "target": f"{_PATCH_BASE}.create_route_agent",
        "wraps": create_route_agent,
    },
    "create_route_agent_conversational": {
        "target": f"{_PATCH_BASE}.create_route_agent_conversational",
        "wraps": create_route_agent_conversational,
    },
}


def _patch(name, **overrides):
    kwargs = {**_PATCHES[name], **overrides}
    target = kwargs.pop("target")
    return patch(target, **kwargs)


mock_tool_a_name = "mock_tool_a"
mock_tool_b_name = "mock_tool_b"


class TestCreateAgent:
    """Test that create_agent wires up nodes/edges correctly with default config."""

    @pytest.fixture
    def mock_model(self):
        return _make_mock_model()

    @pytest.fixture
    def mock_tool_a(self):
        return _make_mock_tool(mock_tool_a_name)

    @pytest.fixture
    def mock_tool_b(self):
        return _make_mock_tool(mock_tool_b_name)

    @pytest.fixture
    def messages(self):
        return [SystemMessage(content="You are a helpful assistant.")]

    @_patch("create_route_agent_conversational")
    @_patch("create_route_agent")
    @_patch("create_flow_control_tools")
    @_patch("create_init_node")
    @_patch("create_terminate_node")
    def test_autonomous_agent_with_tools(
        self,
        mock_create_terminate_node,
        mock_create_init_node,
        mock_create_flow_control_tools,
        mock_route_agent,
        mock_route_agent_conversational,
        mock_model,
        mock_tool_a,
        mock_tool_b,
        messages,
    ):
        """Should return a StateGraph instance."""
        result: StateGraph[Any] = create_agent(
            mock_model, [mock_tool_a, mock_tool_b], messages
        )
        graph = result.compile().get_graph()
        assert set(graph.nodes.keys()) == set(
            [
                "__start__",
                "__end__",
                mock_tool_a_name,
                mock_tool_b_name,
                AgentGraphNode.TERMINATE,
                AgentGraphNode.AGENT,
                AgentGraphNode.INIT,
            ]
        )
        assert set(graph.edges) == set(
            [
                Edge("__start__", AgentGraphNode.INIT),
                Edge(AgentGraphNode.INIT, AgentGraphNode.AGENT),
                Edge(AgentGraphNode.TERMINATE, "__end__"),
                Edge(AgentGraphNode.AGENT, mock_tool_a_name, conditional=True),
                Edge(AgentGraphNode.AGENT, mock_tool_b_name, conditional=True),
                Edge(AgentGraphNode.AGENT, AgentGraphNode.TERMINATE, conditional=True),
                Edge(AgentGraphNode.AGENT, AgentGraphNode.AGENT, conditional=True),
                Edge(mock_tool_a_name, AgentGraphNode.TERMINATE, conditional=True),
                Edge(mock_tool_a_name, AgentGraphNode.AGENT, conditional=True),
                Edge(mock_tool_a_name, mock_tool_a_name, conditional=True),
                Edge(mock_tool_a_name, mock_tool_b_name, conditional=True),
                Edge(mock_tool_b_name, AgentGraphNode.TERMINATE, conditional=True),
                Edge(mock_tool_b_name, AgentGraphNode.AGENT, conditional=True),
                Edge(mock_tool_b_name, mock_tool_a_name, conditional=True),
                Edge(mock_tool_b_name, mock_tool_b_name, conditional=True),
            ]
        )
        mock_route_agent.assert_called_once()
        mock_route_agent_conversational.assert_not_called()
        mock_create_flow_control_tools.assert_called_once()
        mock_create_init_node.assert_called_once_with(
            messages,
            None,  # input schema
            False,  # is_conversational
            None,  # client_side_tools
        )
        mock_create_terminate_node.assert_called_once_with(
            None,  # output schema
            False,  # is_conversational
        )

    @_patch("create_route_agent_conversational")
    @_patch("create_route_agent")
    @_patch("create_flow_control_tools")
    def test_autonomous_agent_without_tools(
        self,
        mock_create_flow_control_tools,
        mock_route_agent,
        mock_route_agent_conversational,
        mock_model,
        messages,
    ):
        """Should return a StateGraph instance."""
        result: StateGraph[Any] = create_agent(mock_model, [], messages)
        graph = result.compile().get_graph()
        assert set(graph.nodes.keys()) == set(
            [
                "__start__",
                "__end__",
                AgentGraphNode.TERMINATE,
                AgentGraphNode.AGENT,
                AgentGraphNode.INIT,
            ]
        )
        assert set(graph.edges) == set(
            [
                Edge("__start__", AgentGraphNode.INIT),
                Edge(AgentGraphNode.INIT, AgentGraphNode.AGENT),
                Edge(AgentGraphNode.TERMINATE, "__end__"),
                Edge(AgentGraphNode.AGENT, AgentGraphNode.TERMINATE, conditional=True),
                Edge(AgentGraphNode.AGENT, AgentGraphNode.AGENT, conditional=True),
            ]
        )
        mock_route_agent.assert_called_once()
        mock_route_agent_conversational.assert_not_called()
        mock_create_flow_control_tools.assert_called_once()

    @_patch("create_route_agent_conversational")
    @_patch("create_route_agent")
    @_patch("create_flow_control_tools")
    @_patch("create_init_node")
    @_patch("create_terminate_node")
    def test_conversational_agent_with_tools(
        self,
        mock_create_terminate_node,
        mock_create_init_node,
        mock_create_flow_control_tools,
        mock_route_agent,
        mock_route_agent_conversational,
        mock_model,
        mock_tool_a,
        messages,
    ):
        """Should return a StateGraph instance."""
        result: StateGraph[Any] = create_agent(
            mock_model,
            [mock_tool_a],
            messages,
            config=AgentGraphConfig(is_conversational=True),
        )
        graph = result.compile().get_graph()
        assert set(graph.nodes.keys()) == set(
            [
                "__start__",
                "__end__",
                mock_tool_a_name,
                AgentGraphNode.TERMINATE,
                AgentGraphNode.AGENT,
                AgentGraphNode.INIT,
            ]
        )
        assert set(graph.edges) == set(
            [
                Edge("__start__", AgentGraphNode.INIT),
                Edge(AgentGraphNode.INIT, AgentGraphNode.AGENT),
                Edge(AgentGraphNode.TERMINATE, "__end__"),
                Edge(AgentGraphNode.AGENT, mock_tool_a_name, conditional=True),
                Edge(AgentGraphNode.AGENT, AgentGraphNode.TERMINATE, conditional=True),
                Edge(mock_tool_a_name, AgentGraphNode.TERMINATE, conditional=True),
                Edge(mock_tool_a_name, mock_tool_a_name, conditional=True),
                Edge(mock_tool_a_name, AgentGraphNode.AGENT, conditional=True),
            ]
        )
        mock_route_agent.assert_not_called()
        mock_route_agent_conversational.assert_called_once()
        # Always invoked now; returns [] for conversational + no custom output schema,
        # preserving the no-flow-control-tool behavior.
        mock_create_flow_control_tools.assert_called_once_with(
            None, is_conversational=True
        )
        mock_create_init_node.assert_called_once_with(
            messages,
            None,  # input schema
            True,  # is_conversational
            None,  # client_side_tools
        )
        mock_create_terminate_node.assert_called_once_with(
            None,  # output schema
            True,  # is_conversational
        )

    @_patch("create_route_agent_conversational")
    @_patch("create_route_agent")
    @_patch("create_flow_control_tools")
    def test_conversational_agent_without_tools(
        self,
        mock_create_flow_control_tools,
        mock_route_agent,
        mock_route_agent_conversational,
        mock_model,
        messages,
    ):
        """Should return a StateGraph instance."""
        result: StateGraph[Any] = create_agent(
            mock_model,
            [],
            messages,
            config=AgentGraphConfig(is_conversational=True),
        )
        graph = result.compile().get_graph()
        assert set(graph.nodes.keys()) == set(
            [
                "__start__",
                "__end__",
                AgentGraphNode.TERMINATE,
                AgentGraphNode.AGENT,
                AgentGraphNode.INIT,
            ]
        )
        assert set(graph.edges) == set(
            [
                Edge("__start__", AgentGraphNode.INIT),
                Edge(AgentGraphNode.INIT, AgentGraphNode.AGENT),
                Edge(AgentGraphNode.TERMINATE, "__end__"),
                Edge(AgentGraphNode.AGENT, AgentGraphNode.TERMINATE, conditional=True),
            ]
        )
        mock_route_agent.assert_not_called()
        mock_route_agent_conversational.assert_called_once()
        mock_create_flow_control_tools.assert_called_once_with(
            None, is_conversational=True
        )


class TestCreateAgentConversationalCustomOutputSchema:
    """Conversational agents with a custom output schema bind end_execution with
    args_schema = output_schema minus uipath__agent_response_messages."""

    @pytest.fixture
    def mock_model(self):
        return _make_mock_model()

    @pytest.fixture
    def messages(self):
        return [SystemMessage(content="You are a helpful assistant.")]

    def test_conversational_no_custom_schema_binds_no_flow_control_tools(
        self, mock_model, messages
    ):
        """Conversational + no output_schema → llm sees zero flow-control tools."""
        from pydantic import BaseModel as _BM
        from uipath.core.chat import UiPathConversationMessageData

        class DefaultSchema(_BM):
            uipath__agent_response_messages: list[UiPathConversationMessageData]

        with patch(
            "uipath_langchain.agent.react.agent.create_llm_node"
        ) as mock_create_llm_node:
            mock_create_llm_node.return_value = lambda state: state

            create_agent(
                mock_model,
                [],
                messages,
                output_schema=DefaultSchema,
                config=AgentGraphConfig(is_conversational=True),
            )

            llm_tools = mock_create_llm_node.call_args.args[1]
            tool_names = [t.name for t in llm_tools]
            assert "end_execution" not in tool_names
            assert "raise_error" not in tool_names

    def test_conversational_custom_schema_binds_end_execution_with_trimmed_args(
        self, mock_model, messages
    ):
        """Conversational + output_schema with custom fields → end_execution bound
        with args_schema containing only the custom fields."""
        from pydantic import BaseModel as _BM
        from uipath.core.chat import UiPathConversationMessageData

        class ResponseSchema(_BM):
            uipath__agent_response_messages: list[UiPathConversationMessageData]
            sentiment: str
            score: float

        with patch(
            "uipath_langchain.agent.react.agent.create_llm_node"
        ) as mock_create_llm_node:
            mock_create_llm_node.return_value = lambda state: state

            create_agent(
                mock_model,
                [],
                messages,
                output_schema=ResponseSchema,
                config=AgentGraphConfig(is_conversational=True),
            )

            llm_tools = mock_create_llm_node.call_args.args[1]
            end_execution_tools = [t for t in llm_tools if t.name == "end_execution"]
            assert len(end_execution_tools) == 1
            assert not any(t.name == "raise_error" for t in llm_tools)

            args_schema_fields = set(end_execution_tools[0].args_schema.model_fields)
            assert args_schema_fields == {"sentiment", "score"}
            assert "uipath__agent_response_messages" not in args_schema_fields
