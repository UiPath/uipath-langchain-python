"""Tests for LLM node tool call filtering functionality."""

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.content import create_text_block, create_tool_call
from langchain_core.tools import BaseTool
from langchain_openai import AzureChatOpenAI
from uipath.agent.react import END_EXECUTION_TOOL, RAISE_ERROR_TOOL

from uipath_langchain.agent.react.llm_node import create_llm_node
from uipath_langchain.agent.react.types import AgentGraphState


class _StubAzureChatOpenAI(AzureChatOpenAI):
    """AzureChatOpenAI subclass that bypasses Pydantic validation for testing.

    Keeps AzureChatOpenAI in the MRO for handler resolution while allowing
    arbitrary attribute assignment (e.g. setting Mock methods).
    """

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)


class TestLLMNodeParallelToolCalls:
    """Test that parallel_tool_calls parameter flows to model.bind_tools()."""

    mock_model: Any

    def setup_method(self):
        self.mock_model = _StubAzureChatOpenAI.model_construct()
        self.mock_model.bind_tools = Mock(return_value=self.mock_model)
        self.mock_model.bind = Mock(return_value=self.mock_model)

        self.regular_tool = Mock(spec=BaseTool)
        self.regular_tool.name = "regular_tool"

        self.test_state = AgentGraphState(messages=[HumanMessage(content="Test")])

    @pytest.mark.asyncio
    async def test_parallel_true_passes_kwarg(self):
        mock_response = AIMessage(content="done", tool_calls=[])
        self.mock_model.ainvoke = AsyncMock(return_value=mock_response)

        llm_node = create_llm_node(
            self.mock_model, [self.regular_tool], parallel_tool_calls=True
        )
        await llm_node(self.test_state)

        self.mock_model.bind_tools.assert_called_once()
        _, kwargs = self.mock_model.bind_tools.call_args
        assert kwargs["parallel_tool_calls"] is True

    @pytest.mark.asyncio
    async def test_parallel_false_passes_kwarg(self):
        mock_response = AIMessage(content="done", tool_calls=[])
        self.mock_model.ainvoke = AsyncMock(return_value=mock_response)

        llm_node = create_llm_node(
            self.mock_model,
            [self.regular_tool],
            parallel_tool_calls=False,
        )
        await llm_node(self.test_state)

        self.mock_model.bind_tools.assert_called_once()
        _, kwargs = self.mock_model.bind_tools.call_args
        assert kwargs["parallel_tool_calls"] is False

    @pytest.mark.asyncio
    async def test_default_is_true(self):
        mock_response = AIMessage(content="done", tool_calls=[])
        self.mock_model.ainvoke = AsyncMock(return_value=mock_response)

        llm_node = create_llm_node(self.mock_model, [self.regular_tool])
        await llm_node(self.test_state)

        self.mock_model.bind_tools.assert_called_once()
        _, kwargs = self.mock_model.bind_tools.call_args
        assert kwargs["parallel_tool_calls"] is True

    @pytest.mark.asyncio
    async def test_non_openai_provider_no_parallel_kwarg(self):
        """Default handler does not include parallel_tool_calls in binding kwargs."""
        mock_model = Mock(spec=BaseChatModel)
        mock_model.bind_tools.return_value = mock_model
        mock_model.bind.return_value = mock_model

        mock_response = AIMessage(content="done", tool_calls=[])
        mock_model.ainvoke = AsyncMock(return_value=mock_response)

        llm_node = create_llm_node(
            mock_model,
            [self.regular_tool],
            parallel_tool_calls=False,
        )
        await llm_node(self.test_state)

        mock_model.bind_tools.assert_called_once()
        _, kwargs = mock_model.bind_tools.call_args
        assert "parallel_tool_calls" not in kwargs


class TestLLMNodeToolCallFiltering:
    """Test cases for LLM node tool call filtering integration."""

    mock_model: Any

    def setup_method(self):
        """Set up test fixtures."""
        self.regular_tool = Mock(spec=BaseTool)
        self.regular_tool.name = "regular_tool"

        self.mock_model = _StubAzureChatOpenAI.model_construct()
        self.mock_model.bind_tools = Mock(return_value=self.mock_model)
        self.mock_model.bind = Mock(return_value=self.mock_model)

        self.test_state = AgentGraphState(messages=[HumanMessage(content="Test query")])

    @pytest.mark.asyncio
    async def test_single_flow_control_call_not_filtered(self):
        """Single flow control calls should not be filtered by LLM node."""
        # Mock response with single flow control call
        mock_response = AIMessage(
            content_blocks=[
                create_text_block("I need to end execution"),
                create_tool_call(name=END_EXECUTION_TOOL.name, args={}, id="call_1"),
            ],
            tool_calls=[
                {
                    "name": END_EXECUTION_TOOL.name,
                    "args": {},
                    "id": "call_1",
                }
            ],
        )
        self.mock_model.ainvoke = AsyncMock(return_value=mock_response)

        # Create LLM node
        llm_node = create_llm_node(self.mock_model, [self.regular_tool])

        # Execute node
        result = await llm_node(self.test_state)

        # Verify single flow control call is not filtered
        response_message = result["messages"][0]
        assert len(response_message.tool_calls) == 1
        assert response_message.tool_calls[0]["name"] == END_EXECUTION_TOOL.name

    @pytest.mark.asyncio
    async def test_parallel_flow_control_calls_filtered(self):
        """Flow control calls in parallel should be filtered by LLM node."""
        # Mock response with parallel calls including flow control
        mock_response = AIMessage(
            content_blocks=[
                create_text_block("Using multiple tools"),
                create_tool_call(name="regular_tool", args={}, id="call_1"),
                create_tool_call(name=END_EXECUTION_TOOL.name, args={}, id="call_2"),
            ],
            tool_calls=[
                {"name": "regular_tool", "args": {}, "id": "call_1"},
                {
                    "name": END_EXECUTION_TOOL.name,
                    "args": {},
                    "id": "call_2",
                },
            ],
        )
        self.mock_model.ainvoke = AsyncMock(return_value=mock_response)

        # Create LLM node
        llm_node = create_llm_node(self.mock_model, [self.regular_tool])

        # Execute node
        result = await llm_node(self.test_state)

        # Verify flow control call was filtered out
        response_message = result["messages"][0]
        assert len(response_message.tool_calls) == 1
        assert response_message.tool_calls[0]["name"] == "regular_tool"

        # Verify content blocks were also updated
        tool_call_blocks = [
            block
            for block in response_message.content_blocks
            if block["type"] == "tool_call"
        ]
        assert len(tool_call_blocks) == 1
        assert tool_call_blocks[0]["name"] == "regular_tool"

    @pytest.mark.asyncio
    async def test_no_flow_control_calls_unchanged(self):
        """Regular tool calls without flow control should remain unchanged."""
        # Mock response with only regular calls
        mock_response = AIMessage(
            content_blocks=[
                create_text_block("Using regular tools"),
                create_tool_call(name="regular_tool", args={}, id="call_1"),
                create_tool_call(name="another_tool", args={}, id="call_2"),
            ],
            tool_calls=[
                {"name": "regular_tool", "args": {}, "id": "call_1"},
                {"name": "another_tool", "args": {}, "id": "call_2"},
            ],
        )
        self.mock_model.ainvoke = AsyncMock(return_value=mock_response)

        # Create LLM node
        llm_node = create_llm_node(self.mock_model, [self.regular_tool])

        # Execute node
        result = await llm_node(self.test_state)

        # Verify no filtering occurred
        response_message = result["messages"][0]
        assert len(response_message.tool_calls) == 2
        assert response_message.tool_calls[0]["name"] == "regular_tool"
        assert response_message.tool_calls[1]["name"] == "another_tool"

    @pytest.mark.asyncio
    async def test_multiple_flow_control_calls_all_filtered(self):
        """Multiple flow control calls in parallel should all be filtered."""
        # Mock response with regular and multiple flow control calls
        mock_response = AIMessage(
            content_blocks=[
                create_text_block("Complex scenario"),
                create_tool_call(name="regular_tool", args={}, id="call_1"),
                create_tool_call(name=END_EXECUTION_TOOL.name, args={}, id="call_2"),
                create_tool_call(name=RAISE_ERROR_TOOL.name, args={}, id="call_3"),
            ],
            tool_calls=[
                {"name": "regular_tool", "args": {}, "id": "call_1"},
                {
                    "name": END_EXECUTION_TOOL.name,
                    "args": {},
                    "id": "call_2",
                },
                {"name": RAISE_ERROR_TOOL.name, "args": {}, "id": "call_3"},
            ],
        )
        self.mock_model.ainvoke = AsyncMock(return_value=mock_response)

        # Create LLM node
        llm_node = create_llm_node(self.mock_model, [self.regular_tool])

        # Execute node
        result = await llm_node(self.test_state)

        # Verify only regular tool call remains
        response_message = result["messages"][0]
        assert len(response_message.tool_calls) == 1
        assert response_message.tool_calls[0]["name"] == "regular_tool"

        # Verify content blocks were updated accordingly
        tool_call_blocks = [
            block
            for block in response_message.content_blocks
            if block["type"] == "tool_call"
        ]
        assert len(tool_call_blocks) == 1
        assert tool_call_blocks[0]["name"] == "regular_tool"
