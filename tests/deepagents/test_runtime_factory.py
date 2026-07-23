import asyncio
import hashlib
import shutil
import time
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict
from uipath.core.feature_flags import FeatureFlags
from uipath.runtime import (
    HydrationPolicy,
    HydrationRuntime,
    UiPathResumableRuntime,
    UiPathRuntimeContext,
    UiPathRuntimeResult,
    WorkspaceHydrator,
    WorkspaceRegistryStore,
)

from uipath_langchain.deepagents.metadata import (
    mark_uipath_deep_agent,
    requires_managed_workspace,
    with_uipath_managed_workspace,
)
from uipath_langchain.runtime.factory import (
    _MANAGED_WORKSPACE_HYDRATION_FEATURE_FLAG,
    UiPathLangGraphRuntimeFactory,
)
from uipath_langchain.runtime.runtime import UiPathLangGraphRuntime


class _State(TypedDict):
    value: str


class _AttachmentStore:
    def __init__(self) -> None:
        self.files: dict[UUID, bytes] = {}

    async def upload_async(
        self,
        *,
        name: str,
        content: str | bytes | None = None,
        source_path: str | None = None,
        folder_key: str | None = None,
        folder_path: str | None = None,
    ) -> UUID:
        key = uuid4()
        if source_path is not None:
            payload = Path(source_path).read_bytes()
        elif isinstance(content, str):
            payload = content.encode()
        elif content is not None:
            payload = content
        else:
            raise ValueError("Expected attachment content or source path")
        self.files[key] = payload
        return key

    async def download_async(
        self,
        *,
        key: UUID,
        destination_path: str,
        folder_key: str | None = None,
        folder_path: str | None = None,
    ) -> str:
        Path(destination_path).write_bytes(self.files[key])
        return destination_path


class _RegistryStore:
    def __init__(self) -> None:
        self.value: dict[str, dict[str, Any]] = {}

    async def load(self) -> dict[str, dict[str, Any]]:
        return self.value.copy()

    async def save(self, registry: dict[str, dict[str, Any]]) -> None:
        self.value = registry.copy()


def _build_graph() -> StateGraph[Any, Any, Any]:
    graph = StateGraph(_State)
    graph.add_node("noop", lambda state: state)
    graph.add_edge(START, "noop")
    graph.add_edge("noop", END)
    return graph


def _test_checkpointer() -> AsyncSqliteSaver:
    return cast(AsyncSqliteSaver, InMemorySaver())


@pytest.fixture(autouse=True)
def _configure_managed_workspace_feature_flag():
    FeatureFlags.reset_flags()
    FeatureFlags.configure_flags({_MANAGED_WORKSPACE_HYDRATION_FEATURE_FLAG: True})
    yield
    FeatureFlags.reset_flags()


async def test_runtime_recompilation_preserves_bound_graph_config(tmp_path) -> None:
    context = UiPathRuntimeContext(runtime_dir=str(tmp_path), state_file="state.db")
    factory = UiPathLangGraphRuntimeFactory(context)
    loaded = mark_uipath_deep_agent(
        _build_graph()
        .compile()
        .with_config(
            {
                "metadata": {"ls_integration": "deepagents"},
                "recursion_limit": 9999,
                "tags": ["bound-tag"],
            }
        )
    )

    result = await factory._compile_graph(loaded, _test_checkpointer())

    assert requires_managed_workspace(result)
    config = result.config
    assert config is not None
    assert config["recursion_limit"] == 9999
    assert config["tags"] == ["bound-tag"]
    assert config["metadata"] == {
        "ls_integration": "deepagents",
        "uipath_runtime": {"requirements": ["managed_workspace"]},
    }


async def test_resolved_deep_agent_graph_keeps_marker_when_cached(tmp_path) -> None:
    context = UiPathRuntimeContext(runtime_dir=str(tmp_path), state_file="state.db")
    factory = UiPathLangGraphRuntimeFactory(context)
    loaded = mark_uipath_deep_agent(_build_graph().compile())

    with patch.object(factory, "_load_graph", AsyncMock(return_value=loaded)) as load:
        first = await factory._resolve_and_compile_graph("agent", _test_checkpointer())
        second = await factory._resolve_and_compile_graph("agent", _test_checkpointer())

    assert second is first
    assert requires_managed_workspace(second)
    load.assert_awaited_once()


async def test_marked_deep_agent_runtime_uses_hydrated_workspace(tmp_path) -> None:
    context = UiPathRuntimeContext(
        runtime_dir=str(tmp_path),
        state_file="state.db",
        job_id="job-1",
        folder_key="folder-1",
    )
    factory = UiPathLangGraphRuntimeFactory(context)
    compiled = mark_uipath_deep_agent(_build_graph().compile())
    sdk = MagicMock()
    sdk.attachments.download_async = AsyncMock(return_value="downloaded")
    sdk.attachments.aclose = AsyncMock()
    sdk.jobs.aclose = AsyncMock()

    with (
        patch(
            "uipath_langchain.runtime.factory.UiPath", return_value=sdk
        ) as sdk_constructor,
        patch.object(factory, "_get_memory", AsyncMock(return_value=MagicMock())),
    ):
        runtime = await factory._create_runtime_instance(
            compiled_graph=compiled,
            runtime_id="runtime-1",
            entrypoint="agent",
        )
        assert isinstance(runtime, HydrationRuntime)
        await runtime.get_schema()
        sdk_constructor.assert_not_called()
        assert runtime.hydrator is None
        hydrator = runtime._get_hydrator()
        sdk_constructor.assert_called_once_with()

    assert runtime.policy == HydrationPolicy.SUSPEND_OR_SUCCESS
    assert runtime.workspace.path.parent == tmp_path / "workspaces"
    assert runtime.workspace.path.name != "runtime-1"

    resumable_runtime = runtime.delegate
    assert isinstance(resumable_runtime, UiPathResumableRuntime)
    langgraph_runtime = resumable_runtime.delegate
    assert isinstance(langgraph_runtime, UiPathLangGraphRuntime)
    assert langgraph_runtime.workspace_path == runtime.workspace.path

    assert hydrator.workspace_path == runtime.workspace.path
    assert hydrator.attachments is sdk.attachments
    assert hydrator.jobs is sdk.jobs
    assert hydrator.current_job_key == "job-1"
    assert hydrator.folder_key == "folder-1"
    assert hydrator.attachment_prefix == ".uipath-workspace"
    assert runtime.registry_store.runtime_id == "runtime-1"
    await runtime.dispose()


async def test_nested_deep_agent_runtime_uses_hydrated_workspace(tmp_path) -> None:
    context = UiPathRuntimeContext(runtime_dir=str(tmp_path), state_file="state.db")
    factory = UiPathLangGraphRuntimeFactory(context)
    child = mark_uipath_deep_agent(_build_graph().compile())
    parent = StateGraph(_State)
    parent.add_node("deep_agent", child)
    parent.add_edge(START, "deep_agent")
    parent.add_edge("deep_agent", END)

    with patch.object(factory, "_get_memory", AsyncMock(return_value=MagicMock())):
        runtime = await factory._create_runtime_instance(
            compiled_graph=with_uipath_managed_workspace(parent.compile()),
            runtime_id="runtime-1",
            entrypoint="agent",
        )

    assert isinstance(runtime, HydrationRuntime)
    await runtime.dispose()


async def test_workspace_is_persisted_and_rehydrated_before_resumed_execution(
    tmp_path,
) -> None:
    context = UiPathRuntimeContext(runtime_dir=str(tmp_path), state_file="state.db")
    factory = UiPathLangGraphRuntimeFactory(context)
    compiled = mark_uipath_deep_agent(_build_graph().compile())
    attachments = _AttachmentStore()
    registry = _RegistryStore()

    with patch.object(factory, "_get_memory", AsyncMock(return_value=MagicMock())):
        first = await factory._create_runtime_instance(
            compiled_graph=compiled,
            runtime_id="runtime-1",
            entrypoint="agent",
        )
        assert isinstance(first, HydrationRuntime)
        first.hydrator = WorkspaceHydrator(
            workspace_path=first.workspace.path,
            attachments=attachments,
        )
        first.registry_store = cast(WorkspaceRegistryStore, registry)
        workspace_file = first.workspace.path / "notes" / "plan.txt"

        async def write_workspace(*args, **kwargs) -> UiPathRuntimeResult:
            workspace_file.parent.mkdir(parents=True)
            workspace_file.write_text("persisted plan")
            return UiPathRuntimeResult()

        with patch.object(first.delegate, "execute", side_effect=write_workspace):
            await first.execute({})

        assert list(attachments.files.values()) == [b"persisted plan"]

        resumed = await factory._create_runtime_instance(
            compiled_graph=compiled,
            runtime_id="runtime-1",
            entrypoint="agent",
        )
        assert isinstance(resumed, HydrationRuntime)
        resumed.hydrator = WorkspaceHydrator(
            workspace_path=resumed.workspace.path,
            attachments=attachments,
        )
        resumed.registry_store = cast(WorkspaceRegistryStore, registry)
        resumed_file = resumed.workspace.path / "notes" / "plan.txt"
        assert not resumed_file.exists()

        async def assert_hydrated(*args, **kwargs) -> UiPathRuntimeResult:
            assert resumed_file.read_text() == "persisted plan"
            return UiPathRuntimeResult()

        with patch.object(resumed.delegate, "execute", side_effect=assert_hydrated):
            await resumed.execute({})

        await resumed.workspace.dispose()


async def test_plain_graph_runtime_is_not_hydrated(tmp_path) -> None:
    context = UiPathRuntimeContext(runtime_dir=str(tmp_path), state_file="state.db")
    factory = UiPathLangGraphRuntimeFactory(context)

    with (
        patch(
            "uipath_langchain.runtime.factory.UiPath",
            side_effect=AssertionError("UiPath SDK must remain lazy"),
        ),
        patch.object(factory, "_get_memory", AsyncMock(return_value=MagicMock())),
    ):
        runtime = await factory._create_runtime_instance(
            compiled_graph=_build_graph().compile(),
            runtime_id="runtime-1",
            entrypoint="agent",
        )

    assert not isinstance(runtime, HydrationRuntime)
    assert isinstance(runtime, UiPathResumableRuntime)
    langgraph_runtime = runtime.delegate
    assert isinstance(langgraph_runtime, UiPathLangGraphRuntime)
    assert langgraph_runtime.workspace_path is None
    assert not (tmp_path / "workspaces").exists()


async def test_marked_graph_does_not_hydrate_when_feature_is_disabled(tmp_path) -> None:
    FeatureFlags.configure_flags({_MANAGED_WORKSPACE_HYDRATION_FEATURE_FLAG: False})
    context = UiPathRuntimeContext(runtime_dir=str(tmp_path), state_file="state.db")
    factory = UiPathLangGraphRuntimeFactory(context)

    with patch.object(factory, "_get_memory", AsyncMock(return_value=MagicMock())):
        runtime = await factory._create_runtime_instance(
            compiled_graph=mark_uipath_deep_agent(_build_graph().compile()),
            runtime_id="runtime-1",
            entrypoint="agent",
        )

    assert isinstance(runtime, UiPathResumableRuntime)
    assert not isinstance(runtime, HydrationRuntime)


async def test_hydration_runtime_closes_platform_services(tmp_path) -> None:
    context = UiPathRuntimeContext(runtime_dir=str(tmp_path), state_file="state.db")
    factory = UiPathLangGraphRuntimeFactory(context)
    attachments = MagicMock()
    attachments.aclose = AsyncMock()
    jobs = MagicMock()
    jobs.aclose = AsyncMock()
    sdk = MagicMock(attachments=attachments, jobs=jobs)

    with (
        patch("uipath_langchain.runtime.factory.UiPath", return_value=sdk),
        patch.object(factory, "_get_memory", AsyncMock(return_value=MagicMock())),
    ):
        runtime = await factory._create_runtime_instance(
            compiled_graph=mark_uipath_deep_agent(_build_graph().compile()),
            runtime_id="runtime-1",
            entrypoint="agent",
        )
        cast(HydrationRuntime, runtime)._get_hydrator()
        await runtime.dispose()

    attachments.aclose.assert_awaited_once()
    jobs.aclose.assert_awaited_once()


async def test_managed_workspace_is_deterministic_and_path_safe(tmp_path) -> None:
    context = UiPathRuntimeContext(runtime_dir=str(tmp_path), state_file="state.db")
    factory = UiPathLangGraphRuntimeFactory(context)

    workspace = await factory._create_managed_workspace("../../outside")

    assert workspace.path.parent == tmp_path / "workspaces"
    assert workspace.path.name == hashlib.sha256(b"../../outside").hexdigest()


async def test_managed_workspace_clears_files_left_by_previous_process(
    tmp_path,
) -> None:
    context = UiPathRuntimeContext(runtime_dir=str(tmp_path), state_file="state.db")
    factory = UiPathLangGraphRuntimeFactory(context)
    workspace = await factory._create_managed_workspace("runtime-1")
    stale_file = workspace.path / "stale.txt"
    stale_file.write_text("stale")

    recreated = await factory._create_managed_workspace("runtime-1")

    assert recreated.path == workspace.path
    assert not stale_file.exists()


async def test_managed_workspace_cleanup_does_not_block_event_loop(
    tmp_path, monkeypatch
) -> None:
    context = UiPathRuntimeContext(runtime_dir=str(tmp_path), state_file="state.db")
    factory = UiPathLangGraphRuntimeFactory(context)
    workspace = await factory._create_managed_workspace("runtime-1")
    (workspace.path / "stale.txt").write_text("stale")
    original_rmtree = shutil.rmtree

    def slow_rmtree(path: Path) -> None:
        time.sleep(0.2)
        original_rmtree(path)

    monkeypatch.setattr("uipath_langchain.runtime.factory.shutil.rmtree", slow_rmtree)

    async def heartbeat() -> float:
        await asyncio.sleep(0.01)
        return asyncio.get_running_loop().time()

    started_at = asyncio.get_running_loop().time()
    recreate = asyncio.create_task(factory._create_managed_workspace("runtime-1"))
    heartbeat_at = await heartbeat()
    await recreate

    assert heartbeat_at - started_at < 0.1
