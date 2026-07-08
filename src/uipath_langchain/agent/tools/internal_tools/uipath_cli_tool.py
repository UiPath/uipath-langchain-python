"""Workspace CLI tool injected when skills are active."""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Iterable
from pathlib import Path

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from uipath.runtime import Workspace

from uipath_langchain.agent.tools.base_uipath_structured_tool import (
    BaseUiPathStructuredTool,
)

ALLOWED_BINARIES = {"uipath", "uip"}
MAX_LOG_BYTES = 4000

# Classify as "internal" so traces label it (untyped default is "Integration").
UIPATH_CLI_TOOL_NAME = "uipath_cli"
UIPATH_CLI_TOOL_TYPE = "internal"

# Destructive verbs blocked by default.
DEFAULT_DENIED_SUBCOMMANDS = frozenset(
    {
        "delete",
        "remove",
        "cancel",
        "uninstall",
        "unassign",
        "unlink",
        "unset",
        "clear",
        "rm",
        "del",
        "destroy",
        "purge",
        "drop",
        "revoke",
        "deactivate",
        "undeploy",
        "unpublish",
        "unregister",
        "unimport",
        "prune",
        "wipe",
        "reset",
    }
)

__all__ = [
    "create_uipath_cli_tool",
    "DEFAULT_DENIED_SUBCOMMANDS",
    "UIPATH_CLI_TOOL_NAME",
    "UIPATH_CLI_TOOL_TYPE",
]

_TOOL_DESCRIPTION = (
    "Run a `uipath` / `uip` command in a workspace subdirectory. Returns combined "
    "stdout/stderr. Only `uipath` and `uip` are allowed. Inherits the agent process "
    "environment, including `UIPATH_ACCESS_TOKEN` / `UIPATH_BASE_URL` for tenant auth."
)


class UiPathCliArgs(BaseModel):
    """Arguments for ``uipath_cli``."""

    command: str = Field(
        description=(
            "Full CLI command starting with `uipath` or `uip`, e.g. 'uipath pack' "
            "or 'uip solution publish dev'."
        ),
    )
    subdir: str = Field(
        default="",
        description=(
            "Workspace-relative subdirectory to run in. Give each scaffolded "
            "project its own subdir. Defaults to the workspace root."
        ),
    )


def _resolve_run_dir(workspace_path: Path, subdir: str) -> Path | str:
    """Resolve ``subdir`` under the workspace"""
    if not subdir:
        return workspace_path
    candidate = (workspace_path / subdir).resolve()
    if not candidate.is_relative_to(workspace_path):
        return f"ERROR: subdir '{subdir}' escapes the workspace directory."
    return candidate


def _is_denied_token(tok: str, denied: frozenset[str]) -> bool:
    low = tok.lower()
    return any(low == d or low.startswith(f"{d}-") for d in denied)


def _validate_command(argv: list[str], denied: frozenset[str]) -> str | None:
    """Return an ERROR string if the command is empty, uses a disallowed binary, or
    contains a blocked destructive subcommand; else ``None``."""
    if not argv:
        return "ERROR: empty command"

    binary = argv[0]
    # Reject any path-qualified binary (e.g. ./uip, /tmp/uip) so only the
    # UiPath CLI resolved from PATH runs, not a lookalike in the workspace.
    if os.path.basename(binary) != binary or binary not in ALLOWED_BINARIES:
        return (
            f"ERROR: '{binary}' is not allowed. Only these binaries are "
            f"permitted (by name, without a path): {sorted(ALLOWED_BINARIES)}"
        )

    blocked = next(
        (
            tok
            for tok in argv[1:]
            if not tok.startswith("-") and _is_denied_token(tok, denied)
        ),
        None,
    )
    if blocked is not None:
        return (
            f"ERROR: subcommand '{blocked}' is not allowed. These destructive "
            f"subcommands are blocked: {sorted(denied)}"
        )
    return None


def _execute_cli(argv: list[str], run_dir: Path, timeout_sec: int) -> str:
    """Run ``argv`` in ``run_dir`` and return combined exit code / stdout / stderr."""
    try:
        proc = subprocess.run(
            argv,
            cwd=str(run_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {timeout_sec}s"
    except FileNotFoundError:
        return f"ERROR: '{argv[0]}' not found on PATH. Install the UiPath CLI."

    return (
        f"exit_code: {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout[:MAX_LOG_BYTES]}\n"
        f"--- stderr ---\n{proc.stderr[:MAX_LOG_BYTES]}"
    )


def _run_uipath_cli(
    command: str,
    subdir: str,
    workspace_path: Path,
    denied: frozenset[str],
    timeout_sec: int,
) -> str:
    """Validate and run a single ``uipath`` / ``uip`` command, jailed to the workspace."""
    run_dir = _resolve_run_dir(workspace_path, subdir)
    if isinstance(run_dir, str):  # ERROR string
        return run_dir

    try:
        argv = shlex.split(command)
    except ValueError as e:
        return f"ERROR: could not parse command: {e}"

    error = _validate_command(argv, denied)
    if error is not None:
        return error

    return _execute_cli(argv, run_dir, timeout_sec)


def create_uipath_cli_tool(
    workspace: Workspace,
    timeout_sec: int = 120,
    denied_subcommands: Iterable[str] | None = None,
) -> BaseTool:
    """Build a ``uipath_cli`` tool jailed to ``workspace`` for advanced agents with skills."""
    workspace_path = Path(workspace.path).expanduser().resolve()
    denied = frozenset(
        s.lower()
        for s in (
            DEFAULT_DENIED_SUBCOMMANDS
            if denied_subcommands is None
            else denied_subcommands
        )
    )

    def run_uipath_cli(command: str, subdir: str = "") -> str:
        return _run_uipath_cli(command, subdir, workspace_path, denied, timeout_sec)

    return BaseUiPathStructuredTool(
        name=UIPATH_CLI_TOOL_NAME,
        description=_TOOL_DESCRIPTION,
        args_schema=UiPathCliArgs,
        func=run_uipath_cli,
        metadata={
            "tool_type": UIPATH_CLI_TOOL_TYPE,
            "display_name": UIPATH_CLI_TOOL_NAME,
            "args_schema": UiPathCliArgs,
        },
    )
