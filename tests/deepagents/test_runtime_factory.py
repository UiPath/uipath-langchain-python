from typing import Any, TypedDict
from unittest.mock import MagicMock, patch

from langgraph.graph import END, START, StateGraph
from uipath.runtime import HydrationPolicy, HydrationRuntime, UiPathRuntimeContext

from uipath_langchain.deepagents import (
    UiPathDeepAgentRuntimeSpec,
    set_uipath_deep_agent_runtime_spec,
)
from uipath_langchain.runtime.factory import UiPathLangGraphRuntimeFactory


class _State(TypedDict):
    value: str


def _build_graph() -> StateGraph[Any, Any, Any]:
    graph = StateGraph(_State)
    graph.add_node("noop", lambda state: state)
    graph.add_edge(START, "noop")
    graph.add_edge("noop", END)
    return graph


async def test_tagged_deep_agent_runtime_uses_hydrated_workspace(tmp_path) -> None:
    context = UiPathRuntimeContext(
        runtime_dir=str(tmp_path),
        state_file="state.db",
        job_id="job-1",
        folder_key="folder-1",
    )
    factory = UiPathLangGraphRuntimeFactory(context)
    compiled = _build_graph().compile()
    spec = UiPathDeepAgentRuntimeSpec(
        workspace_config_key="workspace",
        hydration_policy="always",
        attachment_prefix=".deepagent-workspace",
    )
    set_uipath_deep_agent_runtime_spec(compiled, spec)

    sdk = MagicMock()
    with patch("uipath_langchain.runtime.factory.UiPath", return_value=sdk):
        runtime = await factory._create_runtime_instance(
            compiled_graph=compiled,
            runtime_id="runtime-1",
            entrypoint="agent",
        )

    assert isinstance(runtime, HydrationRuntime)
    assert runtime.policy == HydrationPolicy.ALWAYS
    assert runtime.workspace.path.parent == tmp_path / "workspaces"

    resumable_runtime = runtime.delegate
    langgraph_runtime = resumable_runtime.delegate
    assert langgraph_runtime.configurable == {
        "workspace": str(runtime.workspace.path),
    }

    assert runtime.hydrator.workspace_path == runtime.workspace.path
    assert runtime.hydrator.attachments is sdk.attachments
    assert runtime.hydrator.jobs is sdk.jobs
    assert runtime.hydrator.current_job_key == "job-1"
    assert runtime.hydrator.folder_key == "folder-1"
    assert runtime.hydrator.attachment_prefix == ".deepagent-workspace"
    assert runtime.registry_store.runtime_id == "runtime-1"

    await factory.dispose()
