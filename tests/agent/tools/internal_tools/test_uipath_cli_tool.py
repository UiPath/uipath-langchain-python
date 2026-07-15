"""Tests for the sandboxed ``uipath_cli`` internal tool."""

import asyncio
import shutil

import pytest
from pydantic import ValidationError
from uipath.runtime import Workspace

from uipath_langchain.agent.tools.internal_tools.uipath_cli_tool import (
    _REJECTED_EXIT_CODE,
    UiPathCliInput,
    _parse_uip_command,
    create_uipath_cli_tool,
)

_TOOL_MODULE = "uipath_langchain.agent.tools.internal_tools.uipath_cli_tool"

# --- _parse_uip_command: the security boundary -----------------------------


@pytest.mark.parametrize(
    "command,expected",
    [
        ("pack", ["pack"]),
        ("solution publish", ["solution", "publish"]),
        ("init my-agent", ["init", "my-agent"]),
        # explicit binary prefix is tolerated and stripped
        ("uip pack", ["pack"]),
        ("uipath solution pack", ["solution", "pack"]),
    ],
)
def test_parse_accepts_valid_commands(command: str, expected: list[str]) -> None:
    assert _parse_uip_command(command) == expected


@pytest.mark.parametrize("command", ["", "   ", "uip", "uipath"])
def test_parse_rejects_empty_or_binary_only(command: str) -> None:
    with pytest.raises(ValueError):
        _parse_uip_command(command)


@pytest.mark.parametrize(
    "command", ["pack && rm -rf /", "pack | grep x", "a ; b", "a || b"]
)
def test_parse_rejects_shell_operators(command: str) -> None:
    with pytest.raises(ValueError, match="Shell operator"):
        _parse_uip_command(command)


@pytest.mark.parametrize(
    "command", ["solution delete x", "assets REMOVE y", "destroy", "purge all"]
)
def test_parse_rejects_destructive_subcommands(command: str) -> None:
    with pytest.raises(ValueError, match="Destructive subcommand"):
        _parse_uip_command(command)


# --- UiPathCliInput / tool-level validation --------------------------------


def test_input_model_rejects_bad_command() -> None:
    with pytest.raises(ValidationError):
        UiPathCliInput(command="solution delete x")


def test_input_model_accepts_good_command() -> None:
    assert UiPathCliInput(command="pack").command == "pack"


async def test_tool_returns_recoverable_error_on_bad_command(tmp_path) -> None:
    tool = create_uipath_cli_tool(Workspace(tmp_path))
    msg = await tool.ainvoke(
        {
            "name": "uipath_cli",
            "args": {"command": "solution delete x"},
            "id": "1",
            "type": "tool_call",
        }
    )
    assert msg.status == "error"
    assert "Destructive subcommand" in msg.content


# --- execution mapping (mocked subprocess) ---------------------------------


class _FakeProc:
    def __init__(self, returncode: int, stdout: bytes, stderr: bytes) -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        return self.returncode


@pytest.fixture
def fake_uip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make binary resolution deterministic as ``uip``."""
    monkeypatch.setattr(
        shutil, "which", lambda name: "/usr/bin/uip" if name == "uip" else None
    )


@pytest.mark.parametrize("returncode", [0, 1])
async def test_run_maps_process_result(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_uip: None, returncode: int
) -> None:
    async def fake_exec(*args, **kwargs):  # noqa: ANN002, ANN003
        return _FakeProc(returncode, b"out\n", b"err\n")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    tool = create_uipath_cli_tool(Workspace(tmp_path))

    assert tool.coroutine is not None
    result = await tool.coroutine("pack")

    assert result == {
        "command": "uip pack",
        "exit_code": returncode,
        "stdout": "out\n",
        "stderr": "err\n",
    }


async def test_run_returns_failure_on_oserror(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_uip: None
) -> None:
    async def boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise FileNotFoundError("no such binary")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", boom)
    tool = create_uipath_cli_tool(Workspace(tmp_path))

    assert tool.coroutine is not None
    result = await tool.coroutine("pack")

    assert result["exit_code"] == _REJECTED_EXIT_CODE
    assert "no such binary" in result["stderr"]


async def test_run_times_out_and_kills(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_uip: None
) -> None:
    proc = _FakeProc(0, b"", b"")

    async def hang() -> tuple[bytes, bytes]:
        await asyncio.sleep(3600)
        return b"", b""

    proc.communicate = hang  # type: ignore[method-assign]

    async def fake_exec(*args, **kwargs):  # noqa: ANN002, ANN003
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(f"{_TOOL_MODULE}._COMMAND_TIMEOUT_SECONDS", 0.01)
    tool = create_uipath_cli_tool(Workspace(tmp_path))

    assert tool.coroutine is not None
    result = await tool.coroutine("pack")

    assert result["exit_code"] == _REJECTED_EXIT_CODE
    assert "timed out" in result["stderr"]
    assert proc.killed is True


@pytest.mark.parametrize("subdir", ["../outside", "../../etc", "/tmp"])
async def test_run_rejects_subdir_escape(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_uip: None, subdir: str
) -> None:
    async def fake_exec(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("Subprocess must not start for an escaping subdir.")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    tool = create_uipath_cli_tool(Workspace(tmp_path))

    assert tool.coroutine is not None
    result = await tool.coroutine("pack", subdir=subdir)

    assert result["exit_code"] == _REJECTED_EXIT_CODE
    assert "escapes the workspace" in result["stderr"]


async def test_run_rejects_when_no_binary(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)

    async def fake_exec(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("Subprocess must not start when no binary is found.")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    tool = create_uipath_cli_tool(Workspace(tmp_path))

    assert tool.coroutine is not None
    result = await tool.coroutine("pack")

    assert result["exit_code"] == _REJECTED_EXIT_CODE
    assert "binary" in result["stderr"]
