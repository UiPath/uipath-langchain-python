"""Tests for AgentsLangGraphRuntime."""

from unittest.mock import MagicMock

import pytest
from uipath.agent.models.agent import AgentDefinition, AgentMetadata, AgentSettings

from uipath_agents._cli.runtime.runtime import AgentsLangGraphRuntime


@pytest.fixture
def agent_definition():
    """Create agent span info with telemetry properties."""
    return AgentDefinition(
        name="test-agent",
        input_schema={"type": "object"},
        output_schema={"type": "string"},
        messages=[],
        settings=AgentSettings(
            model="gpt-4",
            max_tokens=1000,
            temperature=0.7,
            engine="openai",
        ),
        metadata=AgentMetadata(
            is_conversational=True,
            storage_version="v1",
        ),
    )


@pytest.fixture
def mock_storage():
    """Create a mock storage object."""
    return MagicMock()


class TestAgentsLangGraphRuntimeGetAgentModel:
    """Tests for AgentsLangGraphRuntime.get_agent_model()."""

    def test_get_agent_model_returns_model_from_agent_config(
        self, tmp_path, mock_storage
    ):
        """Test get_agent_model returns the model from agent.json config."""
        mock_graph = MagicMock()
        mock_agent_definition = MagicMock()
        mock_agent_definition.settings.model = "gpt-4o-2024-11-20"

        runtime = AgentsLangGraphRuntime(
            graph=mock_graph,
            runtime_id="test-id",
            entrypoint="agent.json",
            agent_definition=mock_agent_definition,
            storage=mock_storage,
        )

        model = runtime.get_agent_model()

        assert model == "gpt-4o-2024-11-20"

    def test_get_agent_model_returns_none_when_no_entrypoint(self, mock_storage):
        """Test get_agent_model returns None when entrypoint is None."""
        mock_graph = MagicMock()
        mock_agent_definition = MagicMock()
        mock_agent_definition.settings = None
        runtime = AgentsLangGraphRuntime(
            graph=mock_graph,
            runtime_id="test-id",
            entrypoint=None,
            agent_definition=mock_agent_definition,
            storage=mock_storage,
        )

        model = runtime.get_agent_model()

        assert model is None

    def test_get_agent_model_returns_none_when_model_not_set(
        self, tmp_path, mock_storage
    ):
        """Test get_agent_model returns None when model is not set in config."""
        mock_graph = MagicMock()
        mock_agent_definition = MagicMock()
        mock_agent_definition.settings.model = None
        runtime = AgentsLangGraphRuntime(
            graph=mock_graph,
            runtime_id="test-id",
            entrypoint="agent.json",
            agent_definition=mock_agent_definition,
            storage=mock_storage,
        )

        model = runtime.get_agent_model()

        assert model is None


class TestAgentsLangGraphRuntimeStorage:
    """Tests for AgentsLangGraphRuntime storage parameter."""

    def test_storage_parameter_passed_to_base_class(self, mock_storage):
        """Test that storage parameter is passed through to the base class."""
        mock_graph = MagicMock()
        mock_agent_definition = MagicMock()
        mock_agent_definition.settings = None

        runtime = AgentsLangGraphRuntime(
            graph=mock_graph,
            runtime_id="test-id",
            entrypoint=None,
            agent_definition=mock_agent_definition,
            storage=mock_storage,
        )

        # Verify storage was passed to base class by checking the chat mapper
        # The UiPathLangGraphRuntime passes storage to UiPathChatMessagesMapper
        assert runtime.chat.storage is mock_storage
