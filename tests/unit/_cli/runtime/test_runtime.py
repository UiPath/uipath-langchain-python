"""Tests for AgentsLangGraphRuntime."""

from unittest.mock import MagicMock, patch

from uipath_agents._cli.runtime.runtime import AgentsLangGraphRuntime


class TestAgentsLangGraphRuntimeGetAgentModel:
    """Tests for AgentsLangGraphRuntime.get_agent_model()."""

    def test_get_agent_model_returns_model_from_agent_config(self, tmp_path):
        """Test get_agent_model returns the model from agent.json config."""
        mock_graph = MagicMock()
        runtime = AgentsLangGraphRuntime(
            graph=mock_graph,
            runtime_id="test-id",
            entrypoint="agent.json",
        )

        mock_agent_definition = MagicMock()
        mock_agent_definition.settings.model = "gpt-4o-2024-11-20"

        with patch(
            "uipath_agents._cli.runtime.runtime.load_agent_configuration",
            return_value=mock_agent_definition,
        ):
            with patch(
                "uipath_agents._cli.runtime.runtime.Path.cwd", return_value=tmp_path
            ):
                model = runtime.get_agent_model()

        assert model == "gpt-4o-2024-11-20"

    def test_get_agent_model_caches_result(self, tmp_path):
        """Test get_agent_model caches the result after first call."""
        mock_graph = MagicMock()
        runtime = AgentsLangGraphRuntime(
            graph=mock_graph,
            runtime_id="test-id",
            entrypoint="agent.json",
        )

        mock_agent_definition = MagicMock()
        mock_agent_definition.settings.model = "gpt-4o"

        with patch(
            "uipath_agents._cli.runtime.runtime.load_agent_configuration",
            return_value=mock_agent_definition,
        ) as mock_load:
            with patch(
                "uipath_agents._cli.runtime.runtime.Path.cwd", return_value=tmp_path
            ):
                model1 = runtime.get_agent_model()
                model2 = runtime.get_agent_model()

        assert model1 == "gpt-4o"
        assert model2 == "gpt-4o"
        # Should only be called once due to caching
        assert mock_load.call_count == 1

    def test_get_agent_model_returns_none_when_no_entrypoint(self):
        """Test get_agent_model returns None when entrypoint is None."""
        mock_graph = MagicMock()
        runtime = AgentsLangGraphRuntime(
            graph=mock_graph,
            runtime_id="test-id",
            entrypoint=None,
        )

        model = runtime.get_agent_model()

        assert model is None

    def test_get_agent_model_returns_none_on_exception(self, tmp_path):
        """Test get_agent_model returns None when config loading fails."""
        mock_graph = MagicMock()
        runtime = AgentsLangGraphRuntime(
            graph=mock_graph,
            runtime_id="test-id",
            entrypoint="agent.json",
        )

        with patch(
            "uipath_agents._cli.runtime.runtime.load_agent_configuration",
            side_effect=FileNotFoundError("agent.json not found"),
        ):
            with patch(
                "uipath_agents._cli.runtime.runtime.Path.cwd", return_value=tmp_path
            ):
                model = runtime.get_agent_model()

        assert model is None

    def test_get_agent_model_returns_none_when_model_not_set(self, tmp_path):
        """Test get_agent_model returns None when model is not set in config."""
        mock_graph = MagicMock()
        runtime = AgentsLangGraphRuntime(
            graph=mock_graph,
            runtime_id="test-id",
            entrypoint="agent.json",
        )

        mock_agent_definition = MagicMock()
        mock_agent_definition.settings.model = None

        with patch(
            "uipath_agents._cli.runtime.runtime.load_agent_configuration",
            return_value=mock_agent_definition,
        ):
            with patch(
                "uipath_agents._cli.runtime.runtime.Path.cwd", return_value=tmp_path
            ):
                model = runtime.get_agent_model()

        assert model is None
