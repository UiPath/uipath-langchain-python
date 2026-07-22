"""Sandboxed ``uip`` CLI runner injected into advanced agents.

Unlike the resource-selected internal tools built by ``create_internal_tool``, this
tool is injected programmatically by the agent graph builder when UiPath skills are
active — the agent invokes it to run the ``uip`` commands a skill prescribes. It runs
exactly one ``uip`` invocation per call.

"""

import asyncio
import logging
import os
import shlex
import shutil
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator
from uipath.eval.mocks import mockable
from uipath.runtime import Workspace

from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)

logger = logging.getLogger("uipath")

_TOOL_NAME = "uipath_cli"

_TOOL_DESCRIPTION = (
    "Run a single UiPath `uip` command in the agent workspace; returns "
    "exit_code/stdout/stderr. Only the uip/uipath binaries, one command per call; "
    "destructive subcommands and shell chaining are refused. A negative exit_code "
    "means the command was refused before it ran and stderr explains why. Use "
    "`subdir` to target a scaffolded project folder."
)

_ALLOWED_BINARIES: tuple[str, ...] = ("uip", "uipath")

# Distinct from any real process exit status: normal exits are 0-255 and
# signal-terminated processes surface as -1..-64, so a value well below -255
# unambiguously marks a command rejected before it ever ran.
_REJECTED_EXIT_CODE = -1000

_COMMAND_TIMEOUT_SECONDS = 600

_SHELL_OPERATORS: frozenset[str] = frozenset({"&&", "||", ";", "|"})

# Adding this list for the first iteration of the tool, but we may not need it in the future.
_DESTRUCTIVE_SUBCOMMANDS: frozenset[str] = frozenset(
    {"delete", "remove", "destroy", "uninstall", "purge"}
)


def _parse_uip_command(command: str) -> list[str]:
    """Validate a single ``uip`` command and return its argv (binary excluded).

    Args:
        command: The command to run, e.g. ``"pack"`` or ``"solution publish"``.
            An explicit ``uip``/``uipath`` prefix is tolerated and stripped.

    Returns:
        The argument tokens to pass after the resolved binary.

    Raises:
        ValueError: If the command is empty, chains multiple commands via a shell
            operator, or invokes a destructive subcommand.
    """
    try:
        lexer = shlex.shlex(command, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        lexer.escape = ""
        tokens = list(lexer)
    except ValueError as exc:
        raise ValueError(f"Could not parse command: {exc}") from exc

    if not tokens:
        raise ValueError("No command provided.")

    if tokens[0] in _ALLOWED_BINARIES:
        tokens = tokens[1:]
    if not tokens:
        raise ValueError("No `uip` subcommand provided.")

    # One pass rejects both shell operators and destructive subcommands. The operator
    # check is defense-in-depth: execution uses ``create_subprocess_exec`` (never a
    # shell), so chaining cannot happen regardless — this just returns a clear error
    # instead of handing a nonsense token to ``uip``.
    for token in tokens:
        if token in _SHELL_OPERATORS:
            raise ValueError(
                f"Shell operator '{token}' is not allowed; run one command at a time."
            )
        if token.lower() in _DESTRUCTIVE_SUBCOMMANDS:
            raise ValueError(f"Destructive subcommand '{token}' is not allowed.")

    return tokens


class UiPathCliInput(BaseModel):
    """Input schema for the ``uipath_cli`` tool."""

    command: str = Field(
        description=(
            "A single `uip` command without the binary prefix, e.g. `pack`, "
            "`solution publish`. One command only; no shell operators."
        ),
        examples=["pack", "solution publish", "init my-agent"],
    )
    subdir: str = Field(
        default="",
        description="Workspace-relative directory to run in; defaults to the root.",
    )

    @field_validator("command")
    @classmethod
    def _validate_command(cls, value: str) -> str:
        """Reject unsafe commands before execution (see ``_parse_uip_command``)."""
        _parse_uip_command(value)
        return value


class UiPathCliOutput(BaseModel):
    """Output schema for the ``uipath_cli`` tool."""

    command: str = Field(description="The argv actually executed (or attempted).")
    exit_code: int = Field(
        description=(
            f"Process exit code, or {_REJECTED_EXIT_CODE} when the command was "
            "rejected before running (see stderr for the reason)."
        )
    )
    stdout: str = Field(description="Captured standard output.")
    stderr: str = Field(description="Captured standard error, or the failure reason.")


def _rejected(command: str, reason: str) -> dict[str, Any]:
    """Build a 'rejected before running' result carrying the sentinel and reason."""
    logger.warning("uipath_cli rejected command %r: %s", command, reason)
    return UiPathCliOutput(
        command=command,
        exit_code=_REJECTED_EXIT_CODE,
        stdout="",
        stderr=reason,
    ).model_dump()


def _resolve_run_dir_within_workspace(workspace_root: Path, subdir: str) -> Path | None:
    """Resolve ``subdir`` under the workspace, or ``None`` if it escapes.

    ``.resolve()`` collapses ``..`` and follows symlinks, so absolute paths, parent
    traversal, and symlink escapes are all rejected.
    """
    run_dir = (workspace_root / subdir).resolve()
    if run_dir == workspace_root or workspace_root in run_dir.parents:
        return run_dir
    return None


async def _run_uip_subprocess(
    binary: str, args: list[str], run_dir: Path, echoed: str
) -> dict[str, Any]:
    """Run the resolved ``uip`` command in ``run_dir`` and map it to output."""
    logger.info("uipath_cli running %s (cwd=%s)", echoed, run_dir)
    try:
        proc = await asyncio.create_subprocess_exec(
            binary,
            *args,
            cwd=run_dir,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        return _rejected(echoed, str(exc))

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_COMMAND_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return _rejected(
            echoed, f"Command timed out after {_COMMAND_TIMEOUT_SECONDS}s."
        )

    logger.info("uipath_cli command %s exited with code %s", echoed, proc.returncode)
    exit_code = proc.returncode if proc.returncode is not None else _REJECTED_EXIT_CODE
    return UiPathCliOutput(
        command=echoed,
        exit_code=exit_code,
        stdout=stdout.decode(errors="replace"),
        stderr=stderr.decode(errors="replace"),
    ).model_dump()


def create_uipath_cli_tool(workspace: Workspace) -> StructuredTool:
    """Create the sandboxed ``uipath_cli`` tool bound to ``workspace``.

    The returned tool runs one ``uip`` command per call inside the workspace
    (optionally under ``subdir``) and returns a :class:`UiPathCliOutput` dict. It is
    injected by the agent graph builder rather than selected as a resource, and
    enforces that only the ``uip``/``uipath`` binaries run, one command at a time,
    excluding destructive subcommands.

    Args:
        workspace: The agent's workspace; commands run within its directory. Must be a
            real on-disk workspace (inject only for a ``FilesystemBackend``).

    Returns:
        A ``StructuredTool`` named ``uipath_cli``.
    """

    async def run_uipath_command(command: str, subdir: str = "") -> dict[str, Any]:
        # Parse again here to obtain the argv we exec (and to stay correct if this
        # coroutine is invoked directly, bypassing the model validator). Pure and
        # deterministic, so it stays outside the mockable body below.
        try:
            args = _parse_uip_command(command)
        except ValueError as exc:
            return _rejected(command.strip() or "uip", str(exc))

        @mockable(
            name=_TOOL_NAME,
            description=_TOOL_DESCRIPTION,
            input_schema=UiPathCliInput.model_json_schema(),
            output_schema=UiPathCliOutput.model_json_schema(),
            example_calls=[],  # Examples cannot be provided for internal tools
        )
        async def _execute(**_tool_kwargs: Any) -> dict[str, Any]:
            # Wrapped by @mockable so eval runs get a deterministic result without a
            # real binary/workspace — hence binary resolution and the workspace guard
            # live here, above the subprocess, so the mock intercepts before them.
            binary = shutil.which("uip") or shutil.which("uipath")
            if binary is None:
                return _rejected(
                    shlex.join(["uip", *args]),
                    "No `uip` or `uipath` binary found on PATH.",
                )
            echoed = shlex.join([os.path.basename(binary), *args])

            run_dir = _resolve_run_dir_within_workspace(
                workspace.path.resolve(), subdir
            )
            if run_dir is None:
                return _rejected(
                    echoed, f"Invalid subdir '{subdir}': escapes the workspace."
                )

            return await _run_uip_subprocess(binary, args, run_dir, echoed)

        return await _execute(command=command, subdir=subdir)

    return StructuredToolWithArgumentProperties(
        name=_TOOL_NAME,
        description=_TOOL_DESCRIPTION,
        args_schema=UiPathCliInput,
        coroutine=run_uipath_command,
        output_type=UiPathCliOutput,
        argument_properties={},
        metadata={
            "tool_type": _TOOL_NAME,
            "display_name": _TOOL_NAME,
            "args_schema": UiPathCliInput,
            "output_schema": UiPathCliOutput,
        },
        # Validator errors become a recoverable ToolMessage in all modes (the repo
        # only auto-wraps tool errors in conversational mode).
        handle_validation_error=lambda e: str(e),
    )
