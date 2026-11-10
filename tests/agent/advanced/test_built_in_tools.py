"""Built-in tool settings: how ``settings.advancedAgent.builtInTools`` shapes the agent.

The resolution tests are pure. The build tests run a real deep agent and read
what reaches the model, because deepagents decides at build and call time which
tools exist, and only a real graph catches an upstream change to that.
"""

import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from deepagents import HarnessProfile, register_harness_profile
from deepagents.backends import FilesystemBackend
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel
from uipath.agent.models.agent import AdvancedAgentBuiltInToolSettings

from uipath_langchain.agent.advanced import (
    FILE_TOOL_NAMES,
    ResolvedBuiltInTools,
    build_code_interpreter_middleware,
    create_advanced_agent,
    create_advanced_agent_graph,
    create_conversational_advanced_agent_graph,
    resolve_built_in_tools,
)

ALL_FILE_TOOLS = (
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "delete",
    "glob",
    "grep",
)


def _settings(**tools: dict[str, Any]) -> dict[str, AdvancedAgentBuiltInToolSettings]:
    return {
        name: AdvancedAgentBuiltInToolSettings.model_validate(entry)
        for name, entry in tools.items()
    }


def _all_on(**overrides: dict[str, Any]) -> dict[str, AdvancedAgentBuiltInToolSettings]:
    entries: dict[str, dict[str, Any]] = {
        name: {"enabled": True} for name in FILE_TOOL_NAMES
    }
    entries.update(overrides)
    return _settings(**entries)


def test_the_file_tools_match_deepagents() -> None:
    """A new deepagents file tool must be added to the designer's catalog too."""
    assert FILE_TOOL_NAMES == ALL_FILE_TOOLS


def test_every_tool_on_resolves_every_tool() -> None:
    resolved = resolve_built_in_tools(_all_on())
    assert resolved.enabled_file_tools == ALL_FILE_TOOLS


def test_a_tool_missing_from_the_settings_is_off() -> None:
    resolved = resolve_built_in_tools(_settings(read_file={"enabled": True}))
    assert resolved.enabled_file_tools == ("read_file",)
    assert resolved.eval_requested is False


@pytest.mark.parametrize(
    "settings",
    [_settings(read_file={"enabled": False}), _settings()],
    ids=["turned-off", "missing"],
)
def test_read_file_stays_on(
    settings: dict[str, AdvancedAgentBuiltInToolSettings],
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        resolved = resolve_built_in_tools(settings)
    assert "read_file" in resolved.enabled_file_tools
    assert "read_file" in caplog.text


def test_valid_settings_log_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    """A setting the runtime applies must not also be reported as ignored."""
    with caplog.at_level(logging.WARNING):
        resolve_built_in_tools(_all_on(eval={"enabled": True}))
    assert caplog.messages == []


def test_config_on_a_file_tool_is_ignored_and_the_tool_stays_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        resolved = resolve_built_in_tools(
            _all_on(grep={"enabled": True, "config": {"maxCount": 5}})
        )
    assert "grep" in resolved.enabled_file_tools
    assert "maxCount" in caplog.text


@pytest.mark.parametrize("name", ["execute", "move_file"])
def test_an_unknown_tool_is_ignored(
    name: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        resolved = resolve_built_in_tools(_all_on(**{name: {"enabled": True}}))
    assert name not in resolved.enabled_file_tools
    assert name in caplog.text


@pytest.mark.parametrize(
    ("eval_entry", "expected"),
    [({"enabled": True}, True), ({"enabled": False}, False)],
    ids=["on", "off"],
)
def test_the_eval_setting_is_recorded(
    eval_entry: dict[str, Any], expected: bool, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        resolved = resolve_built_in_tools(
            _all_on(eval=eval_entry, move_file={"enabled": True})
        )
    assert resolved.eval_requested is expected
    assert "Ignoring unknown built-in tool 'move_file'" in caplog.messages
    assert "Ignoring unknown built-in tool 'eval'" not in caplog.messages


def test_config_on_eval_is_ignored_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        resolved = resolve_built_in_tools(
            _all_on(eval={"enabled": True, "config": {"timeout": 9}})
        )
    assert resolved.eval_requested is True
    assert "Ignoring unknown setting 'timeout' of built-in tool 'eval'" in (
        caplog.messages
    )


@pytest.mark.parametrize(
    ("write_file", "edit_file", "memory"),
    [(True, True, True), (True, False, False), (False, True, False)],
    ids=["both", "no-edit", "no-write"],
)
def test_memory_needs_both_write_and_edit(
    write_file: bool, edit_file: bool, memory: bool
) -> None:
    resolved = resolve_built_in_tools(
        _all_on(write_file={"enabled": write_file}, edit_file={"enabled": edit_file})
    )
    assert resolved.supports_memory is memory


class _RecordingModel(GenericFakeChatModel):
    """Records the tools of every binding and the messages of every call.

    A subagent without its own ``model`` reuses this instance, so the main agent
    and the general-purpose subagent record into the same lists, in call order.
    """

    model_name: str = "test-model-built-in-tools"
    bindings: list[dict[str, str]]
    calls: list[list[BaseMessage]]

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> Any:
        return {"ls_provider": "openai", "ls_model_name": self.model_name}

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "_RecordingModel":
        self.bindings.append({t.name: t.description for t in tools})
        return self

    def _generate(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> ChatResult:
        self.calls.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def _model(name: str, script: Sequence[AIMessage]) -> _RecordingModel:
    return _RecordingModel(
        model_name=name,
        messages=iter([*script, *[AIMessage(content="done")] * 20]),
        bindings=[],
        calls=[],
    )


def _task_call(description: str = "go") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "task",
                "args": {
                    "description": description,
                    "subagent_type": GENERAL_PURPOSE_SUBAGENT["name"],
                },
                "id": "c1",
            }
        ],
    )


def _tool(name: str) -> BaseTool:
    return StructuredTool.from_function(
        func=lambda value="": value, name=name, description=f"tool {name}"
    )


def _run(
    model: _RecordingModel,
    workspace: Path,
    resolved: ResolvedBuiltInTools | None,
    prompt: str = "hi",
    **kwargs: Any,
) -> None:
    graph = create_advanced_agent(
        model=model,
        tools=[_tool("read_invoice")],
        backend=FilesystemBackend(root_dir=workspace, virtual_mode=True),
        built_in_tools=resolved,
        **kwargs,
    )
    asyncio.run(graph.ainvoke({"messages": [{"role": "user", "content": prompt}]}))


def _no_writes(**overrides: dict[str, Any]) -> ResolvedBuiltInTools:
    return resolve_built_in_tools(
        _all_on(
            write_file={"enabled": False},
            edit_file={"enabled": False},
            delete={"enabled": False},
            **overrides,
        )
    )


READ_ONLY_TOOLS = {"ls", "read_file", "glob", "grep", "read_invoice"}


def test_without_tool_settings_the_agent_keeps_todays_tools(tmp_path: Path) -> None:
    model = _model("test-model-no-tool-settings", [])
    _run(model, tmp_path, None)
    assert set(model.bindings[0]) == {*ALL_FILE_TOOLS, "task", "read_invoice"}


def test_turned_off_file_tools_are_not_offered(tmp_path: Path) -> None:
    model = _model("test-model-no-writes", [])
    _run(model, tmp_path, _no_writes())
    assert set(model.bindings[0]) == {*READ_ONLY_TOOLS, "task"}


def test_the_subagent_does_not_get_turned_off_file_tools(tmp_path: Path) -> None:
    model = _model("test-model-no-writes-subagent", [_task_call("inspect-invoice-42")])
    _run(model, tmp_path, _no_writes())
    assert len(model.bindings) == 3, f"main, subagent, main: {model.bindings}"
    assert "inspect-invoice-42" in str(model.calls[1][-1].content)
    for binding in model.bindings:
        assert set(binding) - {"task"} == READ_ONLY_TOOLS


def test_the_subagent_starts_without_the_parents_conversation(tmp_path: Path) -> None:
    """The meta prompt tells the model its subagents can't see the conversation."""
    model = _model("test-model-isolated-subagent", [_task_call("inspect-invoice-42")])
    _run(model, tmp_path, _no_writes(), prompt="parent-question-7")
    subagent_messages = [str(message.content) for message in model.calls[1]]
    assert "inspect-invoice-42" in subagent_messages[-1]
    assert not [text for text in subagent_messages if "parent-question-7" in text]


@pytest.mark.parametrize(
    ("write_file", "written"), [(True, True), (False, False)], ids=["on", "off"]
)
def test_eval_reaches_write_file_only_while_it_is_on(
    tmp_path: Path, write_file: bool, written: bool
) -> None:
    pytest.importorskip("langchain_quickjs", reason="needs the code-interpreter extra")
    code = 'await tools.writeFile({ file_path: "/note.txt", content: "x" })'
    model = _model(
        f"test-model-eval-write-{write_file}",
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "eval", "args": {"code": code}, "id": "c1"}],
            )
        ],
    )
    resolved = resolve_built_in_tools(_all_on(write_file={"enabled": write_file}))
    _run(model, tmp_path, resolved, middleware=build_code_interpreter_middleware([]))
    assert (tmp_path / "note.txt").exists() is written


def test_a_profile_description_override_survives_restricted_file_tools(
    tmp_path: Path,
) -> None:
    name = "test-model-description-override"
    register_harness_profile(
        f"openai:{name}",
        HarnessProfile(tool_description_overrides={"read_file": "UIPATH READ"}),
    )
    model = _model(name, [])
    _run(model, tmp_path, _no_writes())
    assert model.bindings[0]["read_file"] == "UIPATH READ"


class _Input(BaseModel):
    task: str


class _Output(BaseModel):
    result: str = ""


def _workspace_with_memory(tmp_path: Path) -> FilesystemBackend:
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "MEMORY.md").write_text("- REMEMBER-ME", encoding="utf-8")
    return FilesystemBackend(root_dir=tmp_path, virtual_mode=True)


@pytest.mark.parametrize(
    ("edit_file", "remembers"), [(True, True), (False, False)], ids=["on", "off"]
)
def test_memory_is_left_out_of_the_prompt_without_edit_file(
    tmp_path: Path, edit_file: bool, remembers: bool
) -> None:
    model = _model(f"test-model-memory-{edit_file}", [])
    graph = create_advanced_agent_graph(
        model=model,
        tools=[],
        system_prompt="",
        backend=_workspace_with_memory(tmp_path),
        response_format=None,
        input_schema=_Input,
        output_schema=_Output,
        build_user_message=lambda args: args["task"],
        built_in_tools=resolve_built_in_tools(
            _all_on(edit_file={"enabled": edit_file})
        ),
    ).compile()
    asyncio.run(graph.ainvoke({"task": "go"}))
    prompt = " ".join(str(m.content) for m in model.calls[0])
    assert ("REMEMBER-ME" in prompt) is remembers


def test_the_conversational_graph_applies_the_resolved_tools(tmp_path: Path) -> None:
    model = _model("test-model-conversational-tools", [])
    graph = create_conversational_advanced_agent_graph(
        model=model,
        tools=[],
        system_prompt="",
        backend=_workspace_with_memory(tmp_path),
        built_in_tools=_no_writes(),
    ).compile()
    asyncio.run(graph.ainvoke({"messages": [{"role": "user", "content": "hi"}]}))
    assert set(model.bindings[0]) == {*READ_ONLY_TOOLS - {"read_invoice"}, "task"}
    prompt = " ".join(str(m.content) for m in model.calls[0])
    assert "REMEMBER-ME" not in prompt


@pytest.mark.parametrize("task_entry", [{"enabled": False}, {"enabled": True}])
def test_a_task_entry_is_ignored_and_task_stays_on(
    tmp_path: Path, task_entry: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """``task`` isn't a capability: sub-agents get their own settings later."""
    model = _model(f"test-model-task-entry-{task_entry['enabled']}", [])
    with caplog.at_level(logging.WARNING):
        resolved = resolve_built_in_tools(_all_on(task=task_entry))
    _run(model, tmp_path, resolved)
    assert "Ignoring unknown built-in tool 'task'" in caplog.messages
    assert "task" in model.bindings[0]
