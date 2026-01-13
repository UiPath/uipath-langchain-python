"""Tests for LLM utilities."""

import pytest
from langchain_core.language_models import BaseChatModel

from uipath_agents.agent_graph_builder.config import AgentExecutionType
from uipath_agents.agent_graph_builder.llm_utils import (
    LLMProvider,
    create_llm,
    detect_provider,
)


class TestDetectProvider:
    """Test provider detection from model names."""

    def test_detect_openai_gpt(self):
        """Test detecting OpenAI provider from gpt model names."""
        assert detect_provider("gpt-4") == LLMProvider.OPENAI
        assert detect_provider("gpt-4o-2024-08-06") == LLMProvider.OPENAI
        assert detect_provider("gpt-5-mini-2025-08-07") == LLMProvider.OPENAI
        assert detect_provider("GPT-4") == LLMProvider.OPENAI

    def test_detect_bedrock_claude(self):
        """Test detecting Bedrock provider from Claude model names."""
        assert detect_provider("anthropic.claude-haiku-4-5") == LLMProvider.BEDROCK
        assert detect_provider("claude-3-sonnet") == LLMProvider.BEDROCK
        assert detect_provider("ANTHROPIC.CLAUDE-OPUS") == LLMProvider.BEDROCK

    def test_detect_vertex_gemini(self):
        """Test detecting Vertex provider from Gemini model names."""
        assert detect_provider("gemini-2.5-flash") == LLMProvider.VERTEX
        assert detect_provider("gemini-pro") == LLMProvider.VERTEX
        assert detect_provider("GEMINI-2.5-FLASH") == LLMProvider.VERTEX

    def test_detect_unknown_provider(self):
        """Test that unknown model names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model provider"):
            detect_provider("unknown-model")


class TestCreateLLM:
    """Test LLM creation and configuration."""

    def test_create_openai_llm(self):
        """Test creating OpenAI LLM."""
        llm = create_llm(
            model="gpt-4",
            temperature=0.5,
            max_tokens=1024,
            execution_type=AgentExecutionType.RUNTIME,
        )
        assert isinstance(llm, BaseChatModel)

    def test_create_bedrock_llm(self):
        """Test creating Bedrock LLM."""
        llm = create_llm(
            model="anthropic.claude-haiku-4-5",
            temperature=0.7,
            max_tokens=2048,
            execution_type=AgentExecutionType.PLAYGROUND,
        )
        assert isinstance(llm, BaseChatModel)

    def test_create_vertex_llm(self):
        """Test creating Vertex LLM."""
        llm = create_llm(
            model="gemini-2.5-flash",
            temperature=0.3,
            max_tokens=4096,
            execution_type=AgentExecutionType.RUNTIME,
        )
        assert isinstance(llm, BaseChatModel)

    def test_create_with_different_parameters(self):
        """Test creating LLMs with different parameter values."""
        llm1 = create_llm(
            model="gpt-4",
            temperature=0.0,
            max_tokens=512,
            execution_type=AgentExecutionType.RUNTIME,
        )
        llm2 = create_llm(
            model="gpt-4",
            temperature=1.0,
            max_tokens=16384,
            execution_type=AgentExecutionType.RUNTIME,
        )

        assert isinstance(llm1, BaseChatModel)
        assert isinstance(llm2, BaseChatModel)

    def test_create_with_unknown_model(self):
        """Test that unknown model names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model provider"):
            create_llm(
                model="unknown-model",
                temperature=0.5,
                max_tokens=1024,
                execution_type=AgentExecutionType.RUNTIME,
            )
