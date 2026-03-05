"""Tests for VoiceLangGraphRuntime input handling."""

from typing import Any

import pytest
from langgraph.types import Command
from uipath.runtime import UiPathExecuteOptions

from uipath_langchain.runtime.runtime import VoiceLangGraphRuntime


@pytest.mark.asyncio
class TestVoiceLangGraphRuntimeInput:
    """Verify _get_graph_input skips chat message mapping."""

    @staticmethod
    def _make_runtime() -> VoiceLangGraphRuntime:
        return VoiceLangGraphRuntime.__new__(VoiceLangGraphRuntime)

    async def test_normal_input_passes_through(self) -> None:
        runtime = self._make_runtime()
        input_state = {"messages": [{"type": "ai", "content": "hello"}]}
        options = UiPathExecuteOptions(resume=False)

        result = await runtime._get_graph_input(input_state, options)

        assert result == input_state

    async def test_resume_wraps_in_command(self) -> None:
        runtime = self._make_runtime()
        input_state = {"messages": [{"type": "ai", "content": "hello"}]}
        options = UiPathExecuteOptions(resume=True)

        result = await runtime._get_graph_input(input_state, options)

        assert isinstance(result, Command)

    async def test_none_input_becomes_empty_dict(self) -> None:
        runtime = self._make_runtime()

        result = await runtime._get_graph_input(None, None)

        assert result == {}

    async def test_none_options_no_resume(self) -> None:
        runtime = self._make_runtime()
        input_state: dict[str, Any] = {"messages": []}

        result = await runtime._get_graph_input(input_state, None)

        assert result == input_state
        assert not isinstance(result, Command)
