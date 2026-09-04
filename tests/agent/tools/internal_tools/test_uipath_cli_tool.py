"""Tests for the sandboxed ``uipath_cli`` internal tool."""

import asyncio
import json
import shutil
import subprocess
from typing import Any

import pytest
from uipath.runtime import Workspace

from uipath_langchain.agent.tools.internal_tools.uipath_cli_tool import (
    _COMMAND_EXAMPLES,
    _REJECTED_EXIT_CODE,
    UiPathCliInput,
    _parse_uip_command,
    create_uipath_cli_tool,
)

_TOOL_MODULE = "uipath_langchain.agent.tools.internal_tools.uipath_cli_tool"


def _schema_command_examples() -> list[str]:
    """Read the examples back out of the schema the model actually receives."""
    return UiPathCliInput.model_json_schema()["properties"]["command"]["examples"]


# --- _parse_uip_command: the security boundary -----------------------------


@pytest.mark.parametrize(
    "command,expected",
    [
        ("pack", ["pack"]),
        ("solution publish", ["solution", "publish"]),
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


# --- command examples stay aligned with the real CLI -----------------------


def test_schema_examples_come_from_the_module_constant() -> None:
    assert _schema_command_examples() == list(_COMMAND_EXAMPLES)


@pytest.mark.parametrize("example", _COMMAND_EXAMPLES)
def test_schema_examples_are_accepted_by_the_parser(example: str) -> None:
    assert _parse_uip_command(example)


@pytest.mark.skipif(shutil.which("uip") is None, reason="uip CLI not on PATH")
@pytest.mark.parametrize("example", _COMMAND_EXAMPLES)
def test_schema_examples_exist_in_the_cli(example: str) -> None:
    """Guard against examples drifting from the installed CLI's vocabulary.

    Matches the ``unknown command`` message rather than the exit status: an
    unknown command exits non-zero, but so do other failures (auth, network),
    which would make an exit-code assertion flaky rather than precise.
    """
    proc = subprocess.run(
        ["uip", *_parse_uip_command(example), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "unknown command" not in (proc.stdout + proc.stderr).lower()


# --- UiPathCliInput / tool-level validation --------------------------------


def test_input_model_does_not_validate_command() -> None:
    """Command safety is enforced at run time, not by a field validator.

    A field validator would surface a refusal as AgentRuntimeError and abort the
    run; refusals must stay recoverable via the _REJECTED_EXIT_CODE contract.
    """
    assert UiPathCliInput(command="pack && rm -rf /").command == "pack && rm -rf /"


def test_input_model_accepts_good_command() -> None:
    assert UiPathCliInput(command="pack").command == "pack"


async def test_tool_returns_recoverable_error_on_bad_command(tmp_path) -> None:
    tool = create_uipath_cli_tool(Workspace(tmp_path))
    msg = await tool.ainvoke(
        {
            "name": "uipath_cli",
            "args": {"command": "pack && rm -rf /"},
            "id": "1",
            "type": "tool_call",
        }
    )
    payload = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
    assert payload["exit_code"] == _REJECTED_EXIT_CODE
    assert "Shell operator" in payload["stderr"]


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


async def test_run_times_out_when_child_exits_during_kill(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_uip: None
) -> None:
    """The child can exit in the window between the timeout and ``kill()``.

    ``Process.kill()`` then raises ProcessLookupError -- either from the
    transport's ``_check_proc`` once ``_proc`` has been cleared, or from
    ``os.kill`` losing the race in ``Popen.send_signal``. The tool must still
    return the recoverable timeout payload rather than propagating.
    """
    proc = _FakeProc(0, b"", b"")
    waited = False

    async def hang() -> tuple[bytes, bytes]:
        await asyncio.sleep(3600)
        return b"", b""

    def kill_races() -> None:
        raise ProcessLookupError()

    async def wait() -> int:
        nonlocal waited
        waited = True
        return 0

    proc.communicate = hang  # type: ignore[method-assign]
    proc.kill = kill_races  # type: ignore[method-assign]
    proc.wait = wait  # type: ignore[method-assign]

    async def fake_exec(*args, **kwargs):  # noqa: ANN002, ANN003
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(f"{_TOOL_MODULE}._COMMAND_TIMEOUT_SECONDS", 0.01)
    tool = create_uipath_cli_tool(Workspace(tmp_path))

    assert tool.coroutine is not None
    result = await tool.coroutine("pack")

    assert result["exit_code"] == _REJECTED_EXIT_CODE
    assert "timed out" in result["stderr"]
    assert waited is True


async def test_run_kills_child_when_task_is_cancelled(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_uip: None
) -> None:
    """Cancelling the waiter must not leave the child OS process running.

    Cancelling ``proc.communicate()`` only stops the coroutine draining the
    pipes; the child keeps running. The tool must kill and reap it, and let the
    original CancelledError propagate so the task still reports as cancelled.
    """
    proc = _FakeProc(0, b"", b"")
    waited = False

    async def hang() -> tuple[bytes, bytes]:
        await asyncio.sleep(3600)
        return b"", b""

    async def wait() -> int:
        nonlocal waited
        waited = True
        return 0

    proc.communicate = hang  # type: ignore[method-assign]
    proc.wait = wait  # type: ignore[method-assign]

    async def fake_exec(*args, **kwargs):  # noqa: ANN002, ANN003
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    tool = create_uipath_cli_tool(Workspace(tmp_path))

    assert tool.coroutine is not None

    async def run() -> dict[str, Any]:
        assert tool.coroutine is not None
        return await tool.coroutine("pack")

    task: asyncio.Task[dict[str, Any]] = asyncio.create_task(run())
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert proc.killed is True
    assert waited is True


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
