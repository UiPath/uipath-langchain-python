"""Tests for message utility functions."""

from uipath.agent.models.agent import (
    AgentMessage,
    AgentMessageRole,
    TextToken,
    TextTokenType,
)

from uipath_agents.agent_graph_builder.message_utils import (
    build_agent_messages,
    interpolate_legacy_message,
)


class TestInterpolateLegacyMessage:
    """Test legacy message interpolation."""

    def test_simple_placeholder(self):
        """Test simple placeholder replacement."""
        result = interpolate_legacy_message("Hello {{name}}", {"name": "Alice"})
        assert result == "Hello Alice"

    def test_nested_placeholder(self):
        """Test nested placeholder replacement."""
        data = {"user": {"name": "Alice"}}
        result = interpolate_legacy_message("Hello {{user.name}}", data)
        assert result == "Hello Alice"

    def test_multiple_placeholders(self):
        """Test multiple placeholder replacement."""
        data = {"first": "Alice", "last": "Smith"}
        result = interpolate_legacy_message("{{first}} {{last}}", data)
        assert result == "Alice Smith"

    def test_missing_placeholder(self):
        """Test missing placeholder is left unchanged."""
        result = interpolate_legacy_message("Hello {{name}}", {})
        assert result == "Hello {{name}}"

    def test_json_serialization(self):
        """Test JSON serialization of arrays."""
        data = {"items": [1, 2, 3]}
        result = interpolate_legacy_message("Items: {{items}}", data)
        assert result == "Items: [1, 2, 3]"

    def test_dict_serialization(self):
        """Test JSON serialization of dicts."""
        data = {"config": {"timeout": 30}}
        result = interpolate_legacy_message("Config: {{config}}", data)
        assert '"timeout"' in result
        assert "30" in result

    def test_unsafe_field_path_skipped(self):
        """Test that unsafe field paths are skipped."""
        data = {"name": "Alice"}
        # This should skip the unsafe placeholder
        result = interpolate_legacy_message("Hello {{__import__('os')}}", data)
        assert result == "Hello {{__import__('os')}}"

    def test_safe_underscore_field(self):
        """Test that safe underscore fields work."""
        data = {"user_name": "Alice"}
        result = interpolate_legacy_message("Hello {{user_name}}", data)
        assert result == "Hello Alice"


class TestBuildAgentMessages:
    """Test building agent messages from definitions."""

    def test_system_and_user_messages(self):
        """Test creating system and user messages."""
        from langchain_core.messages import HumanMessage, SystemMessage
        from uipath.agent.models.agent import AgentMessage

        messages = [
            AgentMessage(role=AgentMessageRole.SYSTEM, content="You are helpful"),
            AgentMessage(role=AgentMessageRole.USER, content="Hello"),
        ]
        result = build_agent_messages(messages, {}, "TestAgent")

        assert len(result) == 2
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], HumanMessage)
        assert "You are helpful" in result[0].content
        assert "TestAgent" in result[0].content
        assert result[1].content == "Hello"

    def test_interpolation_in_messages(self):
        """Test variable interpolation in messages."""
        from uipath.agent.models.agent import AgentMessage

        messages = [
            AgentMessage(
                role=AgentMessageRole.SYSTEM,
                content="Process {{task}} for {{user.name}}",
            ),
            AgentMessage(role=AgentMessageRole.USER, content="Start processing"),
        ]
        input_data = {"task": "analysis", "user": {"name": "Alice"}}
        result = build_agent_messages(messages, input_data, "TestAgent")

        assert "Process analysis for Alice" in result[0].content
        assert result[1].content == "Start processing"

    def test_multiple_messages(self):
        """Test building multiple messages."""
        from uipath.agent.models.agent import AgentMessage

        messages = [
            AgentMessage(role=AgentMessageRole.SYSTEM, content="System prompt"),
            AgentMessage(role=AgentMessageRole.USER, content="User input"),
        ]
        result = build_agent_messages(messages, {}, "TestAgent")

        assert len(result) == 2
        assert "System prompt" in result[0].content
        assert result[1].content == "User input"

    def test_with_content_tokens_in_system_message(self):
        """Test content_tokens in system message."""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            AgentMessage(
                role=AgentMessageRole.SYSTEM,
                content="",
                content_tokens=[
                    TextToken(
                        type=TextTokenType.SIMPLE_TEXT, raw_string="You help with "
                    ),
                    TextToken(type=TextTokenType.VARIABLE, raw_string="input.task"),
                ],
            ),
            AgentMessage(role=AgentMessageRole.USER, content="Do it"),
        ]
        result = build_agent_messages(messages, {"task": "coding"}, "TestAgent")

        assert len(result) == 2
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], HumanMessage)
        assert "You help with coding" in result[0].content
        assert result[1].content == "Do it"

    def test_with_content_tokens_in_user_message(self):
        """Test content_tokens in user message."""
        messages = [
            AgentMessage(role=AgentMessageRole.SYSTEM, content="You are helpful"),
            AgentMessage(
                role=AgentMessageRole.USER,
                content="",
                content_tokens=[
                    TextToken(type=TextTokenType.SIMPLE_TEXT, raw_string="Process "),
                    TextToken(type=TextTokenType.VARIABLE, raw_string="input.item"),
                ],
            ),
        ]
        result = build_agent_messages(messages, {"item": "request"}, "TestAgent")

        assert result[1].content == "Process request"

    def test_mixed_legacy_and_tokens(self):
        """Test mixing legacy content and content_tokens in different messages."""
        messages = [
            AgentMessage(
                role=AgentMessageRole.SYSTEM,
                content="Legacy: {{task}}",
                content_tokens=None,
            ),
            AgentMessage(
                role=AgentMessageRole.USER,
                content="",
                content_tokens=[
                    TextToken(type=TextTokenType.SIMPLE_TEXT, raw_string="Token: "),
                    TextToken(type=TextTokenType.VARIABLE, raw_string="input.task"),
                ],
            ),
        ]
        result = build_agent_messages(messages, {"task": "analysis"}, "TestAgent")

        assert "Legacy: analysis" in result[0].content
        assert result[1].content == "Token: analysis"

    def test_tokens_with_tool_names(self):
        """Test build_agent_messages with tool names."""
        messages = [
            AgentMessage(role=AgentMessageRole.SYSTEM, content="You are helpful"),
            AgentMessage(
                role=AgentMessageRole.USER,
                content="",
                content_tokens=[
                    TextToken(type=TextTokenType.SIMPLE_TEXT, raw_string="Use "),
                    TextToken(type=TextTokenType.VARIABLE, raw_string="tools.weather"),
                    TextToken(type=TextTokenType.SIMPLE_TEXT, raw_string=" to check "),
                    TextToken(type=TextTokenType.VARIABLE, raw_string="input.city"),
                ],
            ),
        ]
        result = build_agent_messages(
            messages,
            {"city": "London"},
            "TestAgent",
            tool_names=["weather", "calculator"],
        )

        assert result[1].content == "Use weather to check London"

    def test_tokens_with_escalation_and_context_names(self):
        """Test build_agent_messages with escalation and context names."""
        messages = [
            AgentMessage(role=AgentMessageRole.SYSTEM, content="You are helpful"),
            AgentMessage(
                role=AgentMessageRole.USER,
                content="",
                content_tokens=[
                    TextToken(type=TextTokenType.SIMPLE_TEXT, raw_string="Search "),
                    TextToken(type=TextTokenType.VARIABLE, raw_string="contexts.docs"),
                    TextToken(
                        type=TextTokenType.SIMPLE_TEXT, raw_string=" or escalate to "
                    ),
                    TextToken(
                        type=TextTokenType.VARIABLE, raw_string="escalations.support"
                    ),
                ],
            ),
        ]
        result = build_agent_messages(
            messages,
            {},
            "TestAgent",
            escalation_names=["support"],
            context_names=["docs"],
        )

        assert result[1].content == "Search docs or escalate to support"
