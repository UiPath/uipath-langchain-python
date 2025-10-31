"""Tests for LLM utilities."""

from uipath_lowcode.agent_graph_builder.llm_utils import (
    LLM_MAX_RETRIES,
    create_llm,
)


class TestCreateLLM:
    """Test LLM creation and configuration."""

    def test_create_with_defaults(self):
        """Test creating LLM with default parameters."""
        llm = create_llm(model="gpt-4")

        assert llm.model_name == "gpt-4"
        assert llm.temperature == 0
        assert llm.max_tokens == 16_384
        assert llm.max_retries == LLM_MAX_RETRIES
        assert llm.disable_streaming is True

    def test_create_with_custom_temperature(self):
        """Test creating LLM with custom temperature."""
        llm = create_llm(model="gpt-4", temperature=0.7)
        assert llm.temperature == 0.7

    def test_create_with_custom_max_tokens(self):
        """Test creating LLM with custom max_tokens."""
        llm = create_llm(model="gpt-4", max_tokens=8192)
        assert llm.max_tokens == 8192

    def test_create_with_custom_timeout(self):
        """Test creating LLM with custom timeout."""
        llm = create_llm(model="gpt-4", timeout=600)
        assert llm.request_timeout == 600

    def test_create_with_custom_retries(self):
        """Test creating LLM with custom max retries."""
        llm = create_llm(model="gpt-4", max_retries=5)
        assert llm.max_retries == 5

    def test_create_with_streaming_enabled(self):
        """Test creating LLM with streaming enabled."""
        llm = create_llm(model="gpt-4", disable_streaming=False)
        assert llm.disable_streaming is False

    def test_create_with_parallel_tool_calls(self):
        """Test creating LLM with parallel tool calls enabled."""
        llm = create_llm(model="gpt-4", parallel_tool_calls=True)
        assert llm.model_kwargs.get("parallel_tool_calls") is True

    def test_create_with_all_custom_params(self):
        """Test creating LLM with all custom parameters."""
        llm = create_llm(
            model="gpt-3.5-turbo",
            temperature=0.9,
            max_tokens=4096,
            timeout=120,
            max_retries=3,
            disable_streaming=False,
            parallel_tool_calls=True,
        )

        assert llm.model_name == "gpt-3.5-turbo"
        assert llm.temperature == 0.9
        assert llm.max_tokens == 4096
        assert llm.request_timeout == 120
        assert llm.max_retries == 3
        assert llm.disable_streaming is False
        assert llm.model_kwargs.get("parallel_tool_calls") is True
