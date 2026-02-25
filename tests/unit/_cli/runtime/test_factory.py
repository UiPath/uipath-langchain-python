"""Tests for AgentsRuntimeFactory startup error handling."""

from unittest.mock import MagicMock, patch

import pytest

from uipath_agents._cli.runtime.factory import AgentsRuntimeFactory
from uipath_agents._cli.runtime.reporter import ReporterRuntime
from uipath_agents._observability.instrumented_runtime import InstrumentedRuntime


@pytest.fixture
def factory() -> AgentsRuntimeFactory:

    mock_context = MagicMock()
    mock_context.resume = False
    mock_context.trace_manager = None
    mock_context.command = "debug"
    with patch("uipath_agents._cli.runtime.factory._prepare_agent_execution_contract"):
        return AgentsRuntimeFactory(mock_context)


class TestFactoryStartupErrorReturnsReporter:
    """new_runtime() returns InstrumentedRuntime wrapping ReporterRuntime on startup failure."""

    @pytest.mark.asyncio
    async def test_returns_instrumented_reporter_on_load_failure(
        self, factory: AgentsRuntimeFactory
    ) -> None:
        with patch.object(
            factory,
            "_load_agent_definition",
            side_effect=FileNotFoundError("agent.json not found"),
        ):
            runtime = await factory.new_runtime("agent.json", "test-id")

        assert isinstance(runtime, InstrumentedRuntime)
        assert isinstance(runtime._delegate, ReporterRuntime)

    @pytest.mark.asyncio
    async def test_reporter_preserves_agent_definition_on_late_failure(
        self, factory: AgentsRuntimeFactory
    ) -> None:
        mock_agent_def = MagicMock()
        mock_agent_def.name = "test-agent"

        with (
            patch.object(
                factory,
                "_load_agent_definition",
                return_value=mock_agent_def,
            ),
            patch.object(
                factory,
                "_get_memory",
                side_effect=RuntimeError("db error"),
            ),
        ):
            runtime = await factory.new_runtime("agent.json", "test-id")

        assert isinstance(runtime, InstrumentedRuntime)
        reporter = runtime._delegate
        assert isinstance(reporter, ReporterRuntime)
        assert reporter.agent_definition is mock_agent_def
