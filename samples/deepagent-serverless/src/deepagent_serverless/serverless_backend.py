"""BackendProtocol implementation that delegates to the serverless sandbox.

Operations are sent as interrupt(InvokeProcess(...)) to the sandbox entry point.
Write/edit operations are accumulated and flushed as a batch together with
read or execute operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileInfo,
    GrepMatch,
    SandboxBackendProtocol,
    WriteResult,
)
from langgraph.types import interrupt
from uipath.platform.common import InvokeProcess


@dataclass
class ServerlessConfig:
    sandbox_process_name: str = "sandbox-deepagent"


class ServerlessBackend(SandboxBackendProtocol):
    """Routes BackendProtocol calls to the sandbox via InvokeProcess interrupt.

    When a read-type operation (read, ls, grep, glob) or execute is called,
    all pending writes are flushed together with the new operation in a single InvokeProcess call.
    """

    def __init__(self, config: ServerlessConfig | None = None) -> None:
        self._config = config or ServerlessConfig()
        self._pending: list[dict[str, Any]] = []

    def _call_sandbox_batch(self, operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Send a batch of operations to the sandbox in a single InvokeProcess."""
        response = interrupt(
            InvokeProcess(
                name=self._config.sandbox_process_name,
                input_arguments={"operations": operations},
            )
        )
        results = response.get("results", [])
        for r in results:
            if r.get("error"):
                raise RuntimeError(r["error"])
        return results

    def _flush_with(self, operation: str, args: dict[str, Any]) -> Any:
        """Flush pending writes + this operation as a single batch. Return last result."""
        ops = self._pending + [{"operation": operation, "args": args}]
        self._pending = []
        results = self._call_sandbox_batch(ops)
        return results[-1]["result"]

    def _enqueue_write(self, operation: str, args: dict[str, Any]) -> None:
        """Buffer a write/edit operation for later flushing."""
        self._pending.append({"operation": operation, "args": args})

    # --- Write operations (buffered) ---

    def write(self, file_path: str, content: str) -> WriteResult:
        self._enqueue_write("write", {"file_path": file_path, "content": content})
        return WriteResult(path=file_path, files_update=None)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        self._enqueue_write("edit", {
            "file_path": file_path,
            "old_string": old_string,
            "new_string": new_string,
            "replace_all": replace_all,
        })
        return EditResult(path=file_path, files_update=None)

    # --- Read operations (flush pending first) ---

    def ls_info(self, path: str) -> list[FileInfo]:
        return self._flush_with("ls", {"path": path})

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        return self._flush_with("read", {
            "file_path": file_path,
            "offset": offset,
            "limit": limit,
        })

    def grep_raw(
        self, pattern: str, path: str | None = None, glob: str | None = None
    ) -> list[GrepMatch] | str:
        return self._flush_with("grep", {
            "pattern": pattern,
            "path": path,
            "glob": glob,
        })

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        return self._flush_with("glob", {
            "pattern": pattern,
            "path": path,
        })

    # --- Execute (flush pending writes + run command on same pod) ---

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        result = self._flush_with("execute", {
            "command": command,
            "timeout": timeout,
        })
        return ExecuteResponse(
            output=result.get("output", ""),
            exit_code=result.get("exit_code"),
            truncated=result.get("truncated", False),
        )
