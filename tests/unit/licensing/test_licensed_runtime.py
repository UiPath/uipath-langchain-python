"""Tests for LicensedRuntime and ToolCallTracker."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from uipath.agent.models.agent import AgentDefinition, AgentSettings
from uipath.runtime import UiPathRuntimeStatus

from uipath_agents._licensing.licensed_runtime import (
    LicensedRuntime,
    ToolCallTracker,
    _had_tool_calls_from_result,
)
from uipath_agents.agent_graph_builder.config import AgentExecutionType

# --- ToolCallTracker ---


class TestToolCallTracker:
    """Tests for the lightweight tool call tracking callback."""

    def test_initial_state_is_false(self) -> None:
        tracker = ToolCallTracker()
        assert tracker.had_tool_calls is False

    def test_on_tool_start_sets_flag(self) -> None:
        tracker = ToolCallTracker()
        tracker.on_tool_start(serialized={}, input_str="test")
        assert tracker.had_tool_calls is True

    def test_reset_clears_flag(self) -> None:
        tracker = ToolCallTracker()
        tracker.on_tool_start(serialized={}, input_str="test")
        assert tracker.had_tool_calls is True
        tracker.reset()
        assert tracker.had_tool_calls is False

    def test_multiple_tool_starts(self) -> None:
        tracker = ToolCallTracker()
        tracker.on_tool_start(serialized={}, input_str="first")
        tracker.on_tool_start(serialized={}, input_str="second")
        assert tracker.had_tool_calls is True


# --- _had_tool_calls_from_result ---


class TestHadToolCallsFromResult:
    """Tests for the result-inspection-based tool call detection."""

    def _make_result(self, output: dict[str, object] | str | None = None) -> MagicMock:
        result = MagicMock()
        result.output = output
        return result

    def test_detects_tool_messages(self) -> None:
        result = self._make_result(
            {"messages": [{"type": "human"}, {"type": "tool"}, {"type": "ai"}]}
        )
        assert _had_tool_calls_from_result(result) is True

    def test_no_tool_messages(self) -> None:
        result = self._make_result({"messages": [{"type": "human"}, {"type": "ai"}]})
        assert _had_tool_calls_from_result(result) is False

    def test_empty_messages(self) -> None:
        result = self._make_result({"messages": []})
        assert _had_tool_calls_from_result(result) is False

    def test_no_messages_key(self) -> None:
        result = self._make_result({"output": "hello"})
        assert _had_tool_calls_from_result(result) is False

    def test_non_dict_output(self) -> None:
        result = self._make_result("just a string")
        assert _had_tool_calls_from_result(result) is False

    def test_none_output(self) -> None:
        result = self._make_result(None)
        assert _had_tool_calls_from_result(result) is False

    def test_non_list_messages(self) -> None:
        result = self._make_result({"messages": "not a list"})
        assert _had_tool_calls_from_result(result) is False


# --- LicensedRuntime ---


@pytest.fixture
def mock_delegate() -> AsyncMock:
    delegate = AsyncMock()
    delegate.entrypoint = "test-agent"
    delegate.runtime_id = "test-id"
    return delegate


@pytest.fixture
def agent_definition() -> AgentDefinition:
    return AgentDefinition(
        name="test-agent",
        messages=[],
        settings=AgentSettings(
            model="gpt-4o",
            engine="v1",
            max_tokens=1000,
            temperature=0.7,
        ),
        input_schema={"type": "object"},
        output_schema={"type": "string"},
    )


@pytest.fixture
def conversational_agent_definition() -> MagicMock:
    agent_def = MagicMock()
    agent_def.name = "conv-agent"
    agent_def.is_conversational = True
    agent_def.settings.model = "gpt-4o"
    agent_def.settings.byom_properties = None
    return agent_def


def _make_successful_result(output: dict[str, object] | None = None) -> MagicMock:
    result = MagicMock()
    result.status = UiPathRuntimeStatus.SUCCESSFUL
    result.output = output or {}
    return result


def _make_suspended_result() -> MagicMock:
    result = MagicMock()
    result.status = UiPathRuntimeStatus.SUSPENDED
    result.output = {}
    return result


class TestLicensedRuntimeStartupLicensing:
    """Tests for one-time startup licensing registration."""

    @pytest.mark.asyncio
    async def test_registers_licensing_on_first_execute(
        self, mock_delegate: AsyncMock, agent_definition: AgentDefinition
    ) -> None:
        mock_delegate.execute.return_value = _make_successful_result()

        runtime = LicensedRuntime(
            mock_delegate,
            agent_definition=agent_definition,
            execution_type=AgentExecutionType.RUNTIME,
        )

        with patch(
            "uipath_agents._licensing.licensed_runtime.register_licensing_async"
        ) as mock_license:
            await runtime.execute({"input": "test"})
            if runtime._licensing_task:
                await runtime._licensing_task
            mock_license.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_does_not_re_register_on_second_execute(
        self, mock_delegate: AsyncMock, agent_definition: AgentDefinition
    ) -> None:
        mock_delegate.execute.return_value = _make_successful_result()

        runtime = LicensedRuntime(
            mock_delegate,
            agent_definition=agent_definition,
            execution_type=AgentExecutionType.RUNTIME,
        )

        with patch(
            "uipath_agents._licensing.licensed_runtime.register_licensing_async"
        ) as mock_license:
            await runtime.execute({"input": "first"})
            if runtime._licensing_task:
                await runtime._licensing_task
            await runtime.execute({"input": "second"})
            mock_license.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_licensing_on_resume(
        self, mock_delegate: AsyncMock, agent_definition: AgentDefinition
    ) -> None:
        mock_delegate.execute.return_value = _make_successful_result()

        runtime = LicensedRuntime(
            mock_delegate,
            agent_definition=agent_definition,
            execution_type=AgentExecutionType.RUNTIME,
            is_resume=True,
        )

        with patch(
            "uipath_agents._licensing.licensed_runtime.register_licensing_async"
        ) as mock_license:
            await runtime.execute({"input": "test"})
            mock_license.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_registers_licensing_on_first_stream(
        self, mock_delegate: AsyncMock, agent_definition: AgentDefinition
    ) -> None:
        result = _make_successful_result()

        async def fake_stream(*args: object, **kwargs: object):
            yield result

        mock_delegate.stream = fake_stream

        runtime = LicensedRuntime(
            mock_delegate,
            agent_definition=agent_definition,
            execution_type=AgentExecutionType.RUNTIME,
        )

        with patch(
            "uipath_agents._licensing.licensed_runtime.register_licensing_async"
        ) as mock_license:
            async for _ in runtime.stream({"input": "test"}):
                pass
            if runtime._licensing_task:
                await runtime._licensing_task
            mock_license.assert_awaited_once()


class TestLicensedRuntimeConversationalConsumption:
    """Tests for per-exchange conversational consumption."""

    @pytest.mark.asyncio
    async def test_registers_consumption_for_conversational_agent_with_tools(
        self,
        mock_delegate: AsyncMock,
        conversational_agent_definition: AgentDefinition,
    ) -> None:
        result = _make_successful_result({"messages": [{"type": "tool"}]})
        mock_delegate.execute.return_value = result

        tracker = ToolCallTracker()
        tracker.on_tool_start(serialized={}, input_str="test")

        runtime = LicensedRuntime(
            mock_delegate,
            agent_definition=conversational_agent_definition,
            tool_call_tracker=tracker,
            execution_type=AgentExecutionType.RUNTIME,
        )

        with (
            patch("uipath_agents._licensing.licensed_runtime.register_licensing_async"),
            patch(
                "uipath_agents._licensing.consumption.register_conversational_licensing_async"
            ) as mock_conv_license,
        ):
            await runtime.execute({"messages": [{"type": "human", "content": "hi"}]})
            mock_conv_license.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_consumption_for_non_conversational_agent(
        self,
        mock_delegate: AsyncMock,
        agent_definition: AgentDefinition,
    ) -> None:
        mock_delegate.execute.return_value = _make_successful_result()

        runtime = LicensedRuntime(
            mock_delegate,
            agent_definition=agent_definition,
            execution_type=AgentExecutionType.RUNTIME,
        )

        with (
            patch("uipath_agents._licensing.licensed_runtime.register_licensing_async"),
            patch(
                "uipath_agents._licensing.consumption.register_conversational_licensing_async"
            ) as mock_conv_license,
        ):
            await runtime.execute({"input": "test"})
            mock_conv_license.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_consumption_on_suspended_status(
        self,
        mock_delegate: AsyncMock,
        conversational_agent_definition: AgentDefinition,
    ) -> None:
        mock_delegate.execute.return_value = _make_suspended_result()

        runtime = LicensedRuntime(
            mock_delegate,
            agent_definition=conversational_agent_definition,
            execution_type=AgentExecutionType.RUNTIME,
        )

        with (
            patch("uipath_agents._licensing.licensed_runtime.register_licensing_async"),
            patch(
                "uipath_agents._licensing.consumption.register_conversational_licensing_async"
            ) as mock_conv_license,
        ):
            await runtime.execute({"input": "test"})
            mock_conv_license.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_resets_tracker_between_executions(
        self,
        mock_delegate: AsyncMock,
        conversational_agent_definition: AgentDefinition,
    ) -> None:
        tracker = ToolCallTracker()
        runtime = LicensedRuntime(
            mock_delegate,
            agent_definition=conversational_agent_definition,
            tool_call_tracker=tracker,
            execution_type=AgentExecutionType.RUNTIME,
        )

        # First execution: tracker has tool calls
        tracker.on_tool_start(serialized={}, input_str="test")
        mock_delegate.execute.return_value = _make_successful_result()

        with patch(
            "uipath_agents._licensing.licensed_runtime.register_licensing_async"
        ):
            await runtime.execute({"messages": [{"type": "human", "content": "hi"}]})

        # After execute, tracker should have been reset before execution
        # (reset happens at start of execute, not end)
        # Simulate: tracker is reset, then delegate sets it during execution
        assert tracker.had_tool_calls is False


class TestLicensedRuntimeToolCallDetection:
    """Tests for dual tool call detection strategies."""

    @pytest.mark.asyncio
    async def test_tracker_takes_precedence(
        self,
        mock_delegate: AsyncMock,
        conversational_agent_definition: AgentDefinition,
    ) -> None:
        """Tracker detects tool calls even if result has no tool messages."""
        result = _make_successful_result({"messages": [{"type": "ai"}]})
        mock_delegate.execute.return_value = result

        tracker = ToolCallTracker()

        runtime = LicensedRuntime(
            mock_delegate,
            agent_definition=conversational_agent_definition,
            tool_call_tracker=tracker,
            execution_type=AgentExecutionType.RUNTIME,
        )

        # Simulate tool call during execution
        def side_effect(*args: object, **kwargs: object) -> MagicMock:
            tracker.on_tool_start(serialized={}, input_str="test")
            return result

        mock_delegate.execute.side_effect = side_effect

        with (
            patch("uipath_agents._licensing.licensed_runtime.register_licensing_async"),
            patch(
                "uipath_agents._licensing.consumption.register_conversational_licensing_async"
            ) as mock_conv_license,
        ):
            await runtime.execute(
                {"messages": [{"type": "human", "content": "hello there"}]}
            )
            mock_conv_license.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_result_fallback_detects_tools(
        self,
        mock_delegate: AsyncMock,
        conversational_agent_definition: AgentDefinition,
    ) -> None:
        """Result inspection detects tool calls when no tracker is provided."""
        result = _make_successful_result(
            {"messages": [{"type": "tool"}, {"type": "ai"}]}
        )
        mock_delegate.execute.return_value = result

        runtime = LicensedRuntime(
            mock_delegate,
            agent_definition=conversational_agent_definition,
            tool_call_tracker=None,
            execution_type=AgentExecutionType.RUNTIME,
        )

        with (
            patch("uipath_agents._licensing.licensed_runtime.register_licensing_async"),
            patch(
                "uipath_agents._licensing.consumption.register_conversational_licensing_async"
            ) as mock_conv_license,
        ):
            await runtime.execute(
                {"messages": [{"type": "human", "content": "hello there"}]}
            )
            mock_conv_license.assert_awaited_once()


class TestLicensedRuntimeDelegation:
    """Tests that LicensedRuntime properly delegates to the wrapped runtime."""

    @pytest.mark.asyncio
    async def test_delegates_execute(self, mock_delegate: AsyncMock) -> None:
        mock_delegate.execute.return_value = _make_successful_result()

        runtime = LicensedRuntime(
            mock_delegate,
            execution_type=AgentExecutionType.RUNTIME,
        )

        with patch(
            "uipath_agents._licensing.licensed_runtime.register_licensing_async"
        ):
            input_data = {"test": "data"}
            await runtime.execute(input_data)
            mock_delegate.execute.assert_awaited_once_with(input_data, None)

    @pytest.mark.asyncio
    async def test_delegates_get_schema(self, mock_delegate: AsyncMock) -> None:
        runtime = LicensedRuntime(
            mock_delegate,
            execution_type=AgentExecutionType.RUNTIME,
        )
        await runtime.get_schema()
        mock_delegate.get_schema.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delegates_dispose(self, mock_delegate: AsyncMock) -> None:
        runtime = LicensedRuntime(
            mock_delegate,
            execution_type=AgentExecutionType.RUNTIME,
        )
        await runtime.dispose()
        mock_delegate.dispose.assert_awaited_once()

    def test_delegates_get_agent_model(self) -> None:
        delegate = MagicMock()
        delegate.get_agent_model.return_value = "gpt-4o"
        runtime = LicensedRuntime(
            delegate,
            execution_type=AgentExecutionType.RUNTIME,
        )
        assert runtime.get_agent_model() == "gpt-4o"
