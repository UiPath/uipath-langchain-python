"""Serverless sandbox - a non-LLM graph that executes batched operations.

Receives a list of operations (file ops + shell execution), dispatches them
sequentially to FilesystemBackend / subprocess on the same pod, and returns
all results. This ensures writes and executes happen on the same filesystem.
"""

import dataclasses
import subprocess
from enum import Enum
from typing import Any

from deepagents.backends.filesystem import FilesystemBackend
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

_DEFAULT_TIMEOUT = 120
_MAX_OUTPUT_BYTES = 100_000
_ROOT_DIR = "/tmp/workspace"


def _serialize(obj: Any) -> Any:
    """Convert protocol objects (dataclasses, lists of dataclasses) to JSON-safe dicts."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, list):
        return [_serialize(item) for item in obj]
    return obj


class Operation(str, Enum):
    LS = "ls"
    READ = "read"
    WRITE = "write"
    EDIT = "edit"
    GREP = "grep"
    GLOB = "glob"
    EXECUTE = "execute"


class OperationRequest(BaseModel):
    operation: Operation
    args: dict[str, Any] = Field(default_factory=dict)


class OperationResult(BaseModel):
    result: Any = None
    error: str | None = None


class SandboxInput(BaseModel):
    operations: list[OperationRequest]


class SandboxOutput(BaseModel):
    results: list[OperationResult]


_backend = FilesystemBackend(root_dir=_ROOT_DIR)


def _execute_shell(args: dict[str, Any]) -> dict[str, Any]:
    """Run a shell command and return output, exit_code, truncated."""
    command = args["command"]
    timeout = args.get("timeout") or _DEFAULT_TIMEOUT
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            timeout=timeout,
            cwd=_ROOT_DIR,
        )
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        parts = []
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.extend(f"[stderr] {line}" for line in stderr.splitlines())
        output = "\n".join(parts)
        truncated = len(output.encode()) > _MAX_OUTPUT_BYTES
        if truncated:
            output = output.encode()[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
        return {"output": output, "exit_code": proc.returncode, "truncated": truncated}
    except subprocess.TimeoutExpired:
        return {"output": f"Command timed out after {timeout}s", "exit_code": 124, "truncated": False}
    except Exception as e:
        return {"output": str(e), "exit_code": 1, "truncated": False}


def _dispatch_one(op: OperationRequest) -> OperationResult:
    """Dispatch a single operation."""
    try:
        match op.operation:
            case Operation.LS:
                result = _backend.ls_info(op.args["path"])
            case Operation.READ:
                result = _backend.read(
                    op.args["file_path"],
                    offset=op.args.get("offset", 0),
                    limit=op.args.get("limit", 2000),
                )
            case Operation.WRITE:
                result = _backend.write(
                    op.args["file_path"],
                    op.args["content"],
                )
            case Operation.EDIT:
                result = _backend.edit(
                    op.args["file_path"],
                    op.args["old_string"],
                    op.args["new_string"],
                    replace_all=op.args.get("replace_all", False),
                )
            case Operation.GREP:
                result = _backend.grep_raw(
                    op.args["pattern"],
                    path=op.args.get("path"),
                    glob=op.args.get("glob"),
                )
            case Operation.GLOB:
                result = _backend.glob_info(
                    op.args["pattern"],
                    path=op.args.get("path", "/"),
                )
            case Operation.EXECUTE:
                result = _execute_shell(op.args)
        return OperationResult(result=_serialize(result))
    except Exception as e:
        return OperationResult(error=str(e))


async def execute(input: SandboxInput) -> SandboxOutput:
    """Execute all operations sequentially on the same pod."""
    results = [_dispatch_one(op) for op in input.operations]
    return SandboxOutput(results=results)


builder = StateGraph(
    state_schema=SandboxInput,
    input=SandboxInput,
    output=SandboxOutput,
)
builder.add_node("execute", execute)
builder.add_edge(START, "execute")
builder.add_edge("execute", END)

graph = builder.compile()
