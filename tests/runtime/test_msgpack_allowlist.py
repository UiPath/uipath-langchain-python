"""Verify the msgpack allowlist plumbing from langgraph.json -> saver.serde."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from uipath.platform.common.interrupt_models import CreateTask, InvokeProcess

from uipath_langchain.runtime.config import LangGraphConfig
from uipath_langchain.runtime.factory import _collect_sdk_interrupt_modules


def _write_config(path: Path, payload: dict[str, object]) -> LangGraphConfig:
    path.write_text(json.dumps(payload))
    return LangGraphConfig(str(path))


def test_no_serde_block_means_no_opt_in(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "langgraph.json", {"graphs": {"a": "x.py:g"}})
    assert config.allowed_msgpack_modules is None


def test_explicit_user_list_is_parsed(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "langgraph.json",
        {
            "graphs": {"a": "x.py:g"},
            "checkpointer": {
                "serde": {
                    "allowed_msgpack_modules": [
                        ["my_app.state", "MyState"],
                        ["my_app.tools", "ToolResult"],
                    ]
                }
            },
        },
    )
    assert config.allowed_msgpack_modules == [
        ("my_app.state", "MyState"),
        ("my_app.tools", "ToolResult"),
    ]


def test_top_level_serde_block_is_ignored(tmp_path: Path) -> None:
    """Only `checkpointer.serde` is read — top-level `serde` is not a recognized location."""
    config = _write_config(
        tmp_path / "langgraph.json",
        {
            "graphs": {"a": "x.py:g"},
            "serde": {
                "allowed_msgpack_modules": [["ignored", "Type"]],
            },
        },
    )
    assert config.allowed_msgpack_modules is None


def test_malformed_entry_raises(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "langgraph.json",
        {
            "graphs": {"a": "x.py:g"},
            "checkpointer": {
                "serde": {"allowed_msgpack_modules": [["only_module_no_class"]]},
            },
        },
    )
    with pytest.raises(ValueError, match="Invalid entry"):
        _ = config.allowed_msgpack_modules


def test_sdk_interrupt_modules_include_create_task_and_invoke_process() -> None:
    sdk_modules = _collect_sdk_interrupt_modules()
    assert (CreateTask.__module__, "CreateTask") in sdk_modules
    assert (InvokeProcess.__module__, "InvokeProcess") in sdk_modules


def test_sdk_types_round_trip_when_user_opts_in() -> None:
    """Strict-mode serde must still reconstruct CreateTask without manual config."""
    sdk_modules = _collect_sdk_interrupt_modules()
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[*sdk_modules, ("my_app", "MyState")]
    )

    task = CreateTask(title="hi", data={})
    type_, payload = serde.dumps_typed(task)
    restored = serde.loads_typed((type_, payload))

    assert isinstance(restored, CreateTask)
    assert restored.title == "hi"


def test_unlisted_user_type_is_blocked_in_strict_mode() -> None:
    """Sanity: types absent from the allowlist degrade to dict (langgraph behaviour)."""
    from pydantic import BaseModel

    class UnlistedState(BaseModel):
        x: int = 0

    serde = JsonPlusSerializer(allowed_msgpack_modules=[("my_app", "MyState")])
    type_, payload = serde.dumps_typed(UnlistedState(x=7))
    restored = serde.loads_typed((type_, payload))
    assert isinstance(restored, dict)
    assert restored == {"x": 7}
