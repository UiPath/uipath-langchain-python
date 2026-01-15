"""Tests for conversational agent message utility functions."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from uipath.agent.models.agent import AgentMessage, AgentMessageRole
from uipath.agent.react import PromptUserSettings

from uipath_agents.agent_graph_builder.message_utils import (
    build_conversational_agent_messages,
    create_message_factory,
    extract_user_settings,
)


class TestExtractUserSettings:
    """Test cases for extract_user_settings function."""

    def test_returns_none_for_none_input(self):
        """Should return None when input_data is None."""
        result = extract_user_settings(None)

        assert result is None

    def test_returns_none_for_empty_dict(self):
        """Should return None when input_data is empty dict."""
        result = extract_user_settings({})

        assert result is None

    def test_returns_none_for_non_dict_input(self):
        """Should return None when input_data is not a dict."""
        result = extract_user_settings("not a dict")  # type: ignore

        assert result is None

    def test_returns_none_when_user_settings_missing(self):
        """Should return None when userSettings key is not present."""
        result = extract_user_settings({"other_key": "value"})

        assert result is None

    def test_returns_none_when_user_settings_is_none(self):
        """Should return None when userSettings value is None."""
        result = extract_user_settings({"userSettings": None})

        assert result is None

    def test_returns_none_when_user_settings_is_not_dict(self):
        """Should return None when userSettings is not a dict."""
        result = extract_user_settings({"userSettings": "not a dict"})

        assert result is None

    def test_extracts_all_user_settings_fields(self):
        """Should extract all PromptUserSettings fields when present."""
        input_data = {
            "userSettings": {
                "name": "John Doe",
                "email": "john@example.com",
                "role": "Developer",
                "department": "Engineering",
                "company": "Acme Corp",
                "country": "USA",
                "timezone": "America/New_York",
            }
        }

        result = extract_user_settings(input_data)

        assert result is not None
        assert isinstance(result, PromptUserSettings)
        assert result.name == "John Doe"
        assert result.email == "john@example.com"
        assert result.role == "Developer"
        assert result.department == "Engineering"
        assert result.company == "Acme Corp"
        assert result.country == "USA"
        assert result.timezone == "America/New_York"

    def test_extracts_partial_user_settings(self):
        """Should handle partial userSettings with some fields None."""
        input_data = {
            "userSettings": {
                "name": "Jane Smith",
                "email": "jane@example.com",
                # Other fields not provided
            }
        }

        result = extract_user_settings(input_data)

        assert result is not None
        assert result.name == "Jane Smith"
        assert result.email == "jane@example.com"
        assert result.role is None
        assert result.department is None
        assert result.company is None
        assert result.country is None
        assert result.timezone is None

    def test_handles_empty_user_settings_dict(self):
        """Should return PromptUserSettings with all None for empty dict."""
        input_data: dict[str, Any] = {"userSettings": {}}

        result = extract_user_settings(input_data)

        # Returns None because empty dict is falsy
        assert result is None

    def test_extracts_only_known_fields(self):
        """Should only extract known fields, ignoring extras."""
        input_data = {
            "userSettings": {
                "name": "Test User",
                "unknown_field": "should be ignored",
                "another_unknown": 123,
            }
        }

        result = extract_user_settings(input_data)

        assert result is not None
        assert result.name == "Test User"
        # Extra fields should not cause errors


class TestBuildConversationalAgentMessages:
    """Test cases for build_conversational_agent_messages function."""

    @pytest.fixture
    def mock_agent_definition(self):
        """Fixture for a mock agent definition."""
        mock_def = MagicMock()
        mock_def.name = "Test Conversational Agent"
        mock_def.messages = [
            AgentMessage(role=AgentMessageRole.SYSTEM, content="System message")
        ]
        return mock_def

    def test_returns_system_message_only(self, mock_agent_definition):
        """Should return only a SystemMessage for conversational agents."""
        with patch(
            "uipath_agents.agent_graph_builder.message_utils.get_chat_system_prompt",
            return_value="Generated system prompt",
        ):
            result = build_conversational_agent_messages(
                mock_agent_definition, {"userSettings": {"name": "Test"}}
            )

        assert len(result) == 1
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "Generated system prompt"

    def test_passes_user_settings_to_prompt_generator(self, mock_agent_definition):
        """Should extract and pass user settings to prompt generator."""
        with patch(
            "uipath_agents.agent_graph_builder.message_utils.get_chat_system_prompt",
            return_value="System prompt",
        ) as mock_generate:
            input_args = {
                "userSettings": {
                    "name": "Alice",
                    "email": "alice@example.com",
                }
            }
            build_conversational_agent_messages(mock_agent_definition, input_args)

            mock_generate.assert_called_once()
            call_args = mock_generate.call_args.kwargs
            user_settings = call_args.get("user_settings")
            assert user_settings is not None
            assert user_settings.name == "Alice"
            assert user_settings.email == "alice@example.com"

    def test_passes_none_when_user_settings_missing(self, mock_agent_definition):
        """Should pass None for user_settings when not in input."""
        with patch(
            "uipath_agents.agent_graph_builder.message_utils.get_chat_system_prompt",
            return_value="System prompt",
        ) as mock_generate:
            build_conversational_agent_messages(mock_agent_definition, {})

            mock_generate.assert_called_once()
            call_args = mock_generate.call_args.kwargs
            user_settings = call_args.get("user_settings")
            assert user_settings is None

    def test_logs_warning_when_user_settings_missing(self, mock_agent_definition):
        """Should log a warning when userSettings is not provided."""
        with (
            patch(
                "uipath_agents.agent_graph_builder.message_utils.get_chat_system_prompt",
                return_value="System prompt",
            ),
            patch(
                "uipath_agents.agent_graph_builder.message_utils.logger"
            ) as mock_logger,
        ):
            build_conversational_agent_messages(mock_agent_definition, {})

            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert (
                "user_settings" in warning_msg.lower() or "userSettings" in warning_msg
            )

    def test_ignores_user_messages_in_agent_definition(self, mock_agent_definition):
        """Should ignore any user messages in the agent definition."""
        mock_agent_definition.messages = [
            AgentMessage(role=AgentMessageRole.SYSTEM, content="System message"),
            AgentMessage(
                role=AgentMessageRole.USER, content="What is the current date?"
            ),
        ]

        with patch(
            "uipath_agents.agent_graph_builder.message_utils.get_chat_system_prompt",
            return_value="Generated prompt",
        ):
            result = build_conversational_agent_messages(
                mock_agent_definition, {"userSettings": {"name": "Test"}}
            )

        # Should only return the generated system message, not the user message
        assert len(result) == 1
        assert isinstance(result[0], SystemMessage)
        # The vestigial user message should NOT be included
        assert not any(isinstance(msg, HumanMessage) for msg in result)


class TestCreateMessageFactoryConversational:
    """Test cases for create_message_factory with conversational agents."""

    @pytest.fixture
    def mock_conversational_agent_definition(self):
        """Fixture for a conversational agent definition."""
        mock_def = MagicMock()
        mock_def.is_conversational = True
        mock_def.name = "Conversational Agent"
        mock_def.messages = []
        return mock_def

    @pytest.fixture
    def mock_regular_agent_definition(self):
        """Fixture for a regular (non-conversational) agent definition."""
        from uipath.agent.models.agent import AgentMessage, AgentMessageRole

        mock_def = MagicMock()
        mock_def.is_conversational = False
        mock_def.name = "Regular Agent"
        mock_def.messages = [
            AgentMessage(
                role=AgentMessageRole.SYSTEM, content="You are a helpful assistant"
            ),
            AgentMessage(role=AgentMessageRole.USER, content="Process {{task}}"),
        ]
        return mock_def

    def test_returns_conversational_factory_when_is_conversational(
        self, mock_conversational_agent_definition
    ):
        """Should return conversational message factory when is_conversational=True."""

        class InputModel(BaseModel):
            pass

        with patch(
            "uipath_agents.agent_graph_builder.message_utils.build_conversational_agent_messages",
            return_value=[SystemMessage(content="Test")],
        ) as mock_build:
            factory = create_message_factory(
                mock_conversational_agent_definition, InputModel
            )

            # Call the factory
            factory({})

            # Should call the conversational builder
            mock_build.assert_called_once()

    def test_returns_regular_factory_when_not_conversational(
        self, mock_regular_agent_definition
    ):
        """Should return regular message factory when is_conversational=False."""

        class InputModel(BaseModel):
            task: str = ""

        with patch(
            "uipath_agents.agent_graph_builder.message_utils.build_agent_messages",
            return_value=[
                SystemMessage(content="System"),
                HumanMessage(content="User"),
            ],
        ) as mock_build:
            factory = create_message_factory(mock_regular_agent_definition, InputModel)

            # Call the factory
            factory({"task": "test"})

            # Should call the regular builder
            mock_build.assert_called_once()

    def test_conversational_factory_extracts_input_from_state(
        self, mock_conversational_agent_definition
    ):
        """Should extract input data from state before building messages."""

        class InputModel(BaseModel):
            userSettings: dict[str, Any] = {}

        with patch(
            "uipath_agents.agent_graph_builder.message_utils.build_conversational_agent_messages",
            return_value=[SystemMessage(content="Test")],
        ) as mock_build:
            factory = create_message_factory(
                mock_conversational_agent_definition, InputModel
            )

            # State with both input fields and internal fields
            state: dict[str, Any] = {
                "userSettings": {"name": "Test User"},
                "messages": [],  # Internal field
                "termination": None,  # Internal field
            }

            factory(state)

            # The extracted input should only contain userSettings
            call_args = mock_build.call_args[0]
            input_arguments = call_args[1]
            assert "userSettings" in input_arguments
            # Internal fields should be filtered out
            assert "messages" not in input_arguments
            assert "termination" not in input_arguments
