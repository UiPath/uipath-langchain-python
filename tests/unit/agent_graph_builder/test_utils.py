"""Tests for message utility functions."""

from uipath_lowcode.agent_graph_builder.message_utils import (
    build_agent_messages,
    interpolate_message,
    safe_get_nested,
    serialize_argument,
)


class TestSafeGetNested:
    """Test nested dictionary access."""

    def test_simple_key(self):
        """Test accessing simple top-level key."""
        data = {"name": "Alice"}
        assert safe_get_nested(data, "name") == "Alice"

    def test_nested_key(self):
        """Test accessing nested keys with dot notation."""
        data = {"user": {"name": "Alice", "age": 30}}
        assert safe_get_nested(data, "user.name") == "Alice"
        assert safe_get_nested(data, "user.age") == 30

    def test_missing_key(self):
        """Test accessing missing keys returns None."""
        data = {"user": {"name": "Alice"}}
        assert safe_get_nested(data, "user.missing") is None
        assert safe_get_nested(data, "missing.key") is None

    def test_array_value(self):
        """Test accessing array values."""
        data = {"items": [1, 2, 3]}
        assert safe_get_nested(data, "items") == [1, 2, 3]

    def test_deeply_nested(self):
        """Test deeply nested access."""
        data = {"level1": {"level2": {"level3": {"value": "deep"}}}}
        assert safe_get_nested(data, "level1.level2.level3.value") == "deep"

    def test_null_intermediate_value(self):
        """Test access through null intermediate value."""
        data = {"user": None}
        assert safe_get_nested(data, "user.name") is None


class TestSerializeValue:
    """Test value serialization."""

    def test_string(self):
        """Test serializing strings."""
        assert serialize_argument("hello") == "hello"

    def test_number(self):
        """Test serializing numbers."""
        assert serialize_argument(42) == "42"
        assert serialize_argument(3.14) == "3.14"

    def test_list(self):
        """Test serializing lists."""
        assert serialize_argument([1, 2, 3]) == "[1, 2, 3]"

    def test_dict(self):
        """Test serializing dictionaries."""
        result = serialize_argument({"key": "value"})
        assert '"key"' in result
        assert '"value"' in result

    def test_none(self):
        """Test serializing None returns empty string."""
        assert serialize_argument(None) == ""

    def test_boolean(self):
        """Test serializing booleans (JSON-style lowercase)."""
        assert serialize_argument(True) == "true"
        assert serialize_argument(False) == "false"


class TestInterpolateMessage:
    """Test template interpolation."""

    def test_simple_placeholder(self):
        """Test simple placeholder replacement."""
        result = interpolate_message("Hello {{name}}", {"name": "Alice"})
        assert result == "Hello Alice"

    def test_nested_placeholder(self):
        """Test nested placeholder replacement."""
        data = {"user": {"name": "Alice"}}
        result = interpolate_message("Hello {{user.name}}", data)
        assert result == "Hello Alice"

    def test_multiple_placeholders(self):
        """Test multiple placeholder replacement."""
        data = {"first": "Alice", "last": "Smith"}
        result = interpolate_message("{{first}} {{last}}", data)
        assert result == "Alice Smith"

    def test_missing_placeholder(self):
        """Test missing placeholder is left unchanged."""
        result = interpolate_message("Hello {{name}}", {})
        assert result == "Hello {{name}}"

    def test_json_serialization(self):
        """Test JSON serialization of arrays."""
        data = {"items": [1, 2, 3]}
        result = interpolate_message("Items: {{items}}", data)
        assert result == "Items: [1, 2, 3]"

    def test_dict_serialization(self):
        """Test JSON serialization of dicts."""
        data = {"config": {"timeout": 30}}
        result = interpolate_message("Config: {{config}}", data)
        assert '"timeout"' in result
        assert "30" in result

    def test_unsafe_field_path_skipped(self):
        """Test that unsafe field paths are skipped."""
        data = {"name": "Alice"}
        # This should skip the unsafe placeholder
        result = interpolate_message("Hello {{__import__('os')}}", data)
        assert result == "Hello {{__import__('os')}}"

    def test_safe_underscore_field(self):
        """Test that safe underscore fields work."""
        data = {"user_name": "Alice"}
        result = interpolate_message("Hello {{user_name}}", data)
        assert result == "Hello Alice"


class TestBuildAgentMessages:
    """Test building agent messages from definitions."""

    def test_system_and_user_messages(self):
        """Test creating system and user messages."""
        from langchain_core.messages import HumanMessage, SystemMessage
        from uipath.agent.models.agent import AgentMessage

        messages = [
            AgentMessage(role="system", content="You are helpful"),
            AgentMessage(role="user", content="Hello"),
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
            AgentMessage(role="system", content="Process {{task}} for {{user.name}}"),
            AgentMessage(role="user", content="Start processing"),
        ]
        input_data = {"task": "analysis", "user": {"name": "Alice"}}
        result = build_agent_messages(messages, input_data, "TestAgent")

        assert "Process analysis for Alice" in result[0].content
        assert result[1].content == "Start processing"

    def test_multiple_messages(self):
        """Test building multiple messages."""
        from uipath.agent.models.agent import AgentMessage

        messages = [
            AgentMessage(role="system", content="System prompt"),
            AgentMessage(role="user", content="User input"),
        ]
        result = build_agent_messages(messages, {}, "TestAgent")

        assert len(result) == 2
        assert "System prompt" in result[0].content
        assert result[1].content == "User input"
