"""Wiring test: the wrapper enables DeepAgents workspace memory.

Phase 1 workspace memory is delegated to deepagents' ``MemoryMiddleware``, which
``create_deep_agent`` builds when a ``memory=`` source list is supplied. The
wrapper's responsibility is therefore to pass ``memory=["/memory/MEMORY.md"]``
through to ``_create_deep_agent``. The actual loading and system-prompt
injection are DeepAgents' concern and covered by their own tests.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from uipath_langchain.agent.advanced.agent import (
    create_advanced_agent_graph,
)
from uipath_langchain.agent.advanced.utils import (
    MEMORY_INDEX_VIRTUAL_PATH,
)


class _Input(BaseModel):
    """Minimal input schema for the test."""

    task: str


class _Output(BaseModel):
    """Minimal output schema for the test."""

    result: str = ""


def _build_user_message(args: dict[str, Any]) -> str:
    return args.get("task", "")


def _memory_kwarg() -> Any:
    """Build the wrapper graph and return the ``memory`` kwarg handed to deepagents."""
    with patch(
        "uipath_langchain.agent.advanced.agent._create_deep_agent",
        return_value=MagicMock(),
    ) as mock_create:
        create_advanced_agent_graph(
            model=MagicMock(spec=BaseChatModel),
            tools=[],
            system_prompt="",
            input_schema=_Input,
            output_schema=_Output,
            build_user_message=_build_user_message,
        )
    return mock_create.call_args.kwargs["memory"]


class TestWorkspaceMemoryWiring:
    """The wrapper turns DeepAgents memory on for the UiPath workspace."""

    def test_enables_memory_for_runtime_workspace(self) -> None:
        assert _memory_kwarg() == [MEMORY_INDEX_VIRTUAL_PATH]


@pytest.mark.asyncio
class TestWrapperInputUnchanged:
    """transform_input no longer hand-rolls a memory SystemMessage."""

    async def test_transform_input_emits_only_user_message(self, tmp_path: Any) -> None:
        # Even with a populated MEMORY.md on disk, the wrapper passes just the user
        # message; memory injection is now MemoryMiddleware's job inside the inner
        # agent, not the wrapper's.
        from langchain_core.messages import HumanMessage

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "MEMORY.md").write_text("- entry: x", encoding="utf-8")

        with patch(
            "uipath_langchain.agent.advanced.agent._create_deep_agent",
            return_value=MagicMock(),
        ):
            wrapper = create_advanced_agent_graph(
                model=MagicMock(spec=BaseChatModel),
                tools=[],
                system_prompt="",
                input_schema=_Input,
                output_schema=_Output,
                build_user_message=_build_user_message,
            )
            out = await wrapper.nodes["transform_input"].runnable.ainvoke(
                _Input(task="do something")
            )

        messages = out["messages"]
        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "do something"
