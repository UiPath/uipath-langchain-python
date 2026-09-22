"""deepagents filesystem permissions reach the deep agent and its subagents."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from deepagents import FilesystemPermission
from deepagents.backends import CompositeBackend, FilesystemBackend
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import BaseModel

from uipath_langchain.agent.advanced.agent import (
    create_advanced_agent,
    create_advanced_agent_graph,
    create_conversational_advanced_agent_graph,
)

READ_ONLY_SKILLS = [
    FilesystemPermission(operations=["write"], paths=["/skills/**"], mode="deny")
]


class _Input(BaseModel):
    task: str = ""


class _Output(BaseModel):
    result: str = ""


class _ScriptedModel(GenericFakeChatModel):
    """Replays ``messages`` and records every prompt it is sent."""

    seen: list[list[BaseMessage]] = []

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "_ScriptedModel":
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def _deep_agent_kwargs(build: Any) -> dict[str, Any]:
    with patch(
        "uipath_langchain.agent.advanced.agent._create_deep_agent",
        return_value=MagicMock(),
    ) as mock_create:
        build()
    return dict(mock_create.call_args.kwargs)


# --- Forwarding ---


def test_autonomous_graph_forwards_permissions() -> None:
    kwargs = _deep_agent_kwargs(
        lambda: create_advanced_agent_graph(
            model=MagicMock(spec=BaseChatModel),
            tools=[],
            system_prompt="",
            backend=None,
            response_format=None,
            input_schema=_Input,
            output_schema=_Output,
            build_user_message=lambda args: "",
            permissions=READ_ONLY_SKILLS,
        )
    )

    assert kwargs["permissions"] == READ_ONLY_SKILLS


def test_conversational_graph_forwards_permissions() -> None:
    kwargs = _deep_agent_kwargs(
        lambda: create_conversational_advanced_agent_graph(
            model=MagicMock(spec=BaseChatModel),
            tools=[],
            system_prompt="",
            backend=None,
            permissions=READ_ONLY_SKILLS,
        )
    )

    assert kwargs["permissions"] == READ_ONLY_SKILLS


def test_no_permissions_by_default() -> None:
    kwargs = _deep_agent_kwargs(
        lambda: create_conversational_advanced_agent_graph(
            model=MagicMock(spec=BaseChatModel),
            tools=[],
            system_prompt="",
            backend=None,
        )
    )

    assert kwargs["permissions"] is None


# --- Enforcement on a real deep agent ---


def _mounted(tmp_path: Path) -> tuple[CompositeBackend, Path, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skills = tmp_path / "skills"
    (skills / "alpha").mkdir(parents=True)
    (skills / "alpha" / "SKILL.md").write_text("Step one.\n")
    backend = CompositeBackend(
        default=FilesystemBackend(root_dir=workspace, virtual_mode=True),
        routes={"/skills/": FilesystemBackend(root_dir=skills, virtual_mode=True)},
    )
    return backend, workspace, skills


def _tool_results(model: _ScriptedModel) -> dict[str, str]:
    return {
        str(message.tool_call_id): str(message.content)
        for message in model.seen[-1]
        if isinstance(message, ToolMessage)
    }


def test_deny_rule_makes_a_routed_path_read_only(tmp_path: Path) -> None:
    backend, workspace, skills = _mounted(tmp_path)
    model = _ScriptedModel(
        seen=[],
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "args": {"file_path": "/skills/alpha/SKILL.md"},
                            "id": "read",
                        },
                        {
                            "name": "write_file",
                            "args": {"file_path": "/skills/alpha/x.md", "content": "x"},
                            "id": "write",
                        },
                        {
                            "name": "edit_file",
                            "args": {
                                "file_path": "/skills/alpha/SKILL.md",
                                "old_string": "Step one.",
                                "new_string": "Changed.",
                            },
                            "id": "edit",
                        },
                        {
                            "name": "write_file",
                            "args": {"file_path": "/notes.md", "content": "ok"},
                            "id": "workspace",
                        },
                    ],
                ),
                AIMessage(content="done"),
            ]
        ),
    )

    agent = create_advanced_agent(
        model=model, backend=backend, permissions=READ_ONLY_SKILLS
    )
    agent.invoke({"messages": [{"role": "user", "content": "go"}]})

    results = _tool_results(model)
    assert "Step one." in results["read"]
    assert "permission denied" in results["write"]
    assert "permission denied" in results["edit"]
    assert (skills / "alpha" / "SKILL.md").read_text() == "Step one.\n"
    assert not (skills / "alpha" / "x.md").exists()
    assert (workspace / "notes.md").read_text() == "ok"


def test_subagents_inherit_the_deny_rule(tmp_path: Path) -> None:
    backend, _, skills = _mounted(tmp_path)
    # The model is shared, so one script plays the main agent and the subagent.
    model = _ScriptedModel(
        seen=[],
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {
                                "description": "write a file",
                                "subagent_type": "general-purpose",
                            },
                            "id": "dispatch",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {"file_path": "/skills/alpha/x.md", "content": "x"},
                            "id": "sub-write",
                        }
                    ],
                ),
                AIMessage(content="subagent done"),
                AIMessage(content="done"),
            ]
        ),
    )

    agent = create_advanced_agent(
        model=model, backend=backend, permissions=READ_ONLY_SKILLS
    )
    agent.invoke({"messages": [{"role": "user", "content": "go"}]})

    subagent_results = {
        str(message.tool_call_id): str(message.content)
        for prompt in model.seen
        for message in prompt
        if isinstance(message, ToolMessage)
    }
    assert "permission denied" in subagent_results["sub-write"]
    assert not (skills / "alpha" / "x.md").exists()
