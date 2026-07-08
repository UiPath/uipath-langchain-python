"""Tests for deepagents skills based on the backend wrapper."""

from typing import Any
from unittest.mock import MagicMock, patch

from deepagents.backends.filesystem import FilesystemBackend
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from uipath_langchain.agent.advanced.agent import (
    create_advanced_agent_graph,
)
from uipath_langchain.agent.advanced.utils import (
    SKILLS_VIRTUAL_PATH,
)


class _Input(BaseModel):
    """Minimal input schema for the test."""

    task: str


class _Output(BaseModel):
    """Minimal output schema for the test."""

    result: str = ""


def _build_user_message(args: dict[str, Any]) -> str:
    return args.get("task", "")


def _skills_kwarg(backend: Any, **overrides: Any) -> Any:
    """Build the wrapper graph and return the ``skills`` kwarg handed to deepagents."""
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
            **overrides,
        )
    return mock_create.call_args.kwargs["skills"]


class TestWorkspaceSkillsWiring:
    """Skills are opt-in: enabled only when the caller declares them."""

    def test_disabled_by_default_on_filesystem_backend(self, tmp_path: Any) -> None:
        backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
        assert _skills_kwarg(backend) is None

    def test_empty_skills_sequence_disables(self, tmp_path: Any) -> None:
        # An empty sequence disables skills, matching the ``memory`` parameter.
        backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
        assert _skills_kwarg(backend, skills=[]) is None

    def test_declared_skills_enable_and_prepend_workspace(self, tmp_path: Any) -> None:
        backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
        assert _skills_kwarg(backend, skills=["/extra/"]) == [
            SKILLS_VIRTUAL_PATH,
            "/extra/",
        ]

    def test_no_duplicate_workspace_source(self, tmp_path: Any) -> None:
        backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
        assert _skills_kwarg(backend, skills=[SKILLS_VIRTUAL_PATH]) == [
            SKILLS_VIRTUAL_PATH
        ]

    def test_explicit_sources_only_for_non_filesystem_backend(self) -> None:
        assert _skills_kwarg(None, skills=["/extra/"]) == ["/extra/"]
