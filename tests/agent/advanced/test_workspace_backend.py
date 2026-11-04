import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Awaitable, Callable

import pytest
from langchain_core.tools import tool
from uipath.runtime import (
    HydrationRuntime,
    UiPathExecuteOptions,
    UiPathRuntimeResult,
    UiPathRuntimeSchema,
    UiPathRuntimeStatus,
    UiPathStreamOptions,
    Workspace,
    WorkspaceHydrator,
    WorkspaceRegistryStore,
    get_workspace_path,
)
from uipath.runtime.events import UiPathRuntimeEvent

from uipath_langchain.agent.advanced import UiPathWorkspaceBackend


class MemoryStorage:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str, str], Any] = {}

    async def set_value(
        self, runtime_id: str, namespace: str, key: str, value: Any
    ) -> None:
        self.values[(runtime_id, namespace, key)] = value

    async def get_value(self, runtime_id: str, namespace: str, key: str) -> Any:
        return self.values.get((runtime_id, namespace, key))


class NoAttachments:
    async def upload_async(self, **kwargs: Any) -> Any:
        raise AssertionError("a successful run must not upload")

    async def download_async(self, **kwargs: Any) -> str:
        raise AssertionError("an empty registry must not download")


class StoredAttachments:
    def __init__(self) -> None:
        self.blobs: dict[uuid.UUID, bytes] = {}

    async def upload_async(self, *, source_path: str, **kwargs: Any) -> uuid.UUID:
        key = uuid.uuid4()
        self.blobs[key] = Path(source_path).read_bytes()
        return key

    async def download_async(
        self, *, key: uuid.UUID, destination_path: str, **kwargs: Any
    ) -> str:
        Path(destination_path).write_bytes(self.blobs[key])
        return destination_path


class StepRuntime:
    """Runs one step inside the managed run, like a deepagents file tool would."""

    def __init__(
        self,
        step: Callable[[], Awaitable[Any]],
        status: UiPathRuntimeStatus = UiPathRuntimeStatus.SUCCESSFUL,
    ) -> None:
        self.step = step
        self.status = status

    async def execute(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathExecuteOptions | None = None,
    ) -> UiPathRuntimeResult:
        return UiPathRuntimeResult(
            status=self.status, output={"value": await self.step()}
        )

    async def stream(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathStreamOptions | None = None,
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        yield await self.execute(input, options)

    async def get_schema(self) -> UiPathRuntimeSchema:
        raise NotImplementedError

    async def dispose(self) -> None:
        pass


async def _run_managed(
    workspace: Workspace,
    step: Callable[[], Awaitable[Any]],
    *,
    status: UiPathRuntimeStatus = UiPathRuntimeStatus.SUCCESSFUL,
    attachments: Any = None,
    storage: MemoryStorage | None = None,
) -> Any:
    runtime = HydrationRuntime(
        StepRuntime(step, status),
        workspace=workspace,
        hydrator=WorkspaceHydrator(
            workspace_path=workspace.path,
            attachments=attachments or NoAttachments(),
        ),
        registry_store=WorkspaceRegistryStore(storage or MemoryStorage(), "runtime-1"),
    )
    result = await runtime.execute()
    assert isinstance(result.output, dict)
    return result.output["value"]


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace.create(tmp_path / "managed")


async def test_files_land_in_the_runs_managed_workspace(
    workspace: Workspace,
) -> None:
    backend = UiPathWorkspaceBackend()

    async def write() -> str | None:
        return backend.write("/notes/plan.md", "step 1").error

    assert await _run_managed(workspace, write) is None
    assert (workspace.path / "notes" / "plan.md").read_text() == "step 1"


async def test_async_file_tools_see_the_managed_workspace(
    workspace: Workspace,
) -> None:
    backend = UiPathWorkspaceBackend()

    async def awrite() -> str | None:
        return (await backend.awrite("/plan.md", "from a thread")).error

    assert await _run_managed(workspace, awrite) is None
    assert (workspace.path / "plan.md").read_text() == "from a thread"


async def test_one_backend_follows_each_runs_workspace(tmp_path: Path) -> None:
    backend = UiPathWorkspaceBackend()
    first = Workspace.create(tmp_path / "first")
    second = Workspace.create(tmp_path / "second")

    async def write_first() -> None:
        backend.write("/first.md", "first run")

    async def use_second() -> dict[str, Any]:
        backend.write("/second.md", "second run")
        read = backend.read("/second.md")
        return {
            "read": read.file_data["content"] if read.file_data else None,
            "ls": [e["path"] for e in backend.ls("/").entries or []],
            "glob": [m["path"] for m in backend.glob("*.md", "/").matches or []],
            "grep": [m["path"] for m in backend.grep("run", "/").matches or []],
        }

    await _run_managed(first, write_first)
    seen = await _run_managed(second, use_second)

    assert (first.path / "first.md").read_text() == "first run"
    assert seen == {
        "read": "second run",
        "ls": ["/second.md"],
        "glob": ["/second.md"],
        "grep": ["/second.md"],
    }


async def test_files_written_before_a_suspend_are_readable_after_resume(
    tmp_path: Path,
) -> None:
    backend = UiPathWorkspaceBackend()
    attachments, storage = StoredAttachments(), MemoryStorage()

    async def write() -> None:
        backend.write("/plan.md", "step 1")

    async def read() -> str | None:
        result = backend.read("/plan.md")
        return result.file_data["content"] if result.file_data else None

    await _run_managed(
        Workspace.create(tmp_path / "before"),
        write,
        status=UiPathRuntimeStatus.SUSPENDED,
        attachments=attachments,
        storage=storage,
    )
    restored = await _run_managed(
        Workspace.create(tmp_path / "after"),
        read,
        attachments=attachments,
        storage=storage,
    )

    assert restored == "step 1"


async def test_paths_cannot_leave_the_managed_workspace(
    workspace: Workspace,
) -> None:
    backend = UiPathWorkspaceBackend()

    async def escape() -> None:
        backend.write("/../escaped.md", "outside")

    with pytest.raises(ValueError, match="Path traversal not allowed"):
        await _run_managed(workspace, escape)
    assert not (workspace.path.parent / "escaped.md").exists()


def test_outside_a_managed_run_files_go_to_one_temp_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.chdir(tmp_path)
    backend = UiPathWorkspaceBackend()

    with caplog.at_level(logging.WARNING):
        backend.write("/a.md", "one")
        backend.write("/b.md", "two")

    assert list(tmp_path.iterdir()) == []
    assert backend.cwd.is_relative_to(Path(tempfile.gettempdir()).resolve())
    assert (backend.cwd / "a.md").read_text() == "one"
    assert (backend.cwd / "b.md").read_text() == "two"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert str(backend.cwd) in warnings[0].getMessage()


async def test_sync_tools_run_by_langchain_can_use_the_workspace(
    workspace: Workspace,
) -> None:
    @tool
    def save_report(text: str) -> str:
        """Save a report into the workspace."""
        path = get_workspace_path() / "report.md"
        path.write_text(text)
        return str(path)

    async def call_tool() -> str:
        return await save_report.ainvoke({"text": "from a tool"})

    await _run_managed(workspace, call_tool)

    assert (workspace.path / "report.md").read_text() == "from a tool"
