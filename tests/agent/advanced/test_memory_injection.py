"""Wiring test: the wrapper enables deepagents memory based on the backend.

Phase 1 workspace memory is delegated to deepagents' ``MemoryMiddleware``, which
``create_deep_agent`` builds when a ``memory=`` source list is supplied. The
wrapper's responsibility is therefore to pass ``memory=["/memory/MEMORY.md"]``
through to ``_create_deep_agent`` when (and only when) the backend is a
``FilesystemBackend`` that carries a durable workspace. The actual loading and
system-prompt injection are deepagents' concern and covered by their own tests.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from deepagents.backends.filesystem import FilesystemBackend
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


def _memory_kwarg(
    backend: Any,
) -> Any:
    """Build the wrapper graph and return the ``memory`` kwarg handed to deepagents."""
    with patch(
        "uipath_langchain.agent.advanced.agent._create_deep_agent",
        return_value=MagicMock(),
    ) as mock_create:
        create_advanced_agent_graph(
            model=MagicMock(spec=BaseChatModel),
            tools=[],
            system_prompt="",
            backend=backend,
            response_format=None,
            input_schema=_Input,
            output_schema=_Output,
            build_user_message=_build_user_message,
        )
    return mock_create.call_args.kwargs["memory"]


class TestWorkspaceMemoryWiring:
    """The wrapper turns deepagents memory on for filesystem-backed workspaces."""

    def test_enables_memory_for_filesystem_backend(self, tmp_path: Any) -> None:
        backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
        assert _memory_kwarg(backend) == [MEMORY_INDEX_VIRTUAL_PATH]

    def test_enables_memory_for_marked_runtime_workspace_factory(
        self, tmp_path: Any
    ) -> None:
        def backend_factory(runtime: Any) -> FilesystemBackend:
            return FilesystemBackend(root_dir=tmp_path, virtual_mode=True)

        backend_factory.is_uipath_workspace_filesystem_backend = True  # type: ignore[attr-defined]

        assert _memory_kwarg(backend_factory) == [MEMORY_INDEX_VIRTUAL_PATH]

    def test_disables_memory_for_non_filesystem_backend(self) -> None:
        # The default in-state backend (None) carries no durable workspace, so
        # passing memory=None leaves MemoryMiddleware out of the stack entirely.
        assert _memory_kwarg(None) is None


@pytest.mark.asyncio
class TestWrapperInputUnchanged:
    """transform_input no longer hand-rolls a memory SystemMessage."""

    async def test_transform_input_emits_only_user_message(self, tmp_path: Any) -> None:
        # Even with a populated MEMORY.md on disk, the wrapper passes just the user
        # message; memory injection is now MemoryMiddleware's job inside the inner
        # agent, not the wrapper's.
        from langchain_core.messages import HumanMessage
        from langgraph.graph import END, START, StateGraph

        from uipath_langchain.agent.advanced.types import (
            AdvancedAgentGraphState,
        )

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "MEMORY.md").write_text("- entry: x", encoding="utf-8")

        captured: list[list[Any]] = []

        def _make_inner() -> Any:
            def _capture(state: AdvancedAgentGraphState) -> dict[str, Any]:
                captured.append(list(state.messages))
                return {"structured_response": {"result": "ok"}}

            inner = StateGraph(AdvancedAgentGraphState)
            inner.add_node("capture", _capture)
            inner.add_edge(START, "capture")
            inner.add_edge("capture", END)
            return inner.compile()

        with patch(
            "uipath_langchain.agent.advanced.agent._create_deep_agent",
            return_value=_make_inner(),
        ):
            wrapper = create_advanced_agent_graph(
                model=MagicMock(spec=BaseChatModel),
                tools=[],
                system_prompt="",
                backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
                response_format=None,
                input_schema=_Input,
                output_schema=_Output,
                build_user_message=_build_user_message,
            ).compile()
            await wrapper.ainvoke({"task": "do something"})

        assert len(captured) == 1
        messages = captured[0]
        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "do something"
