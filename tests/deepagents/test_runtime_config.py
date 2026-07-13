from unittest.mock import MagicMock

from uipath_langchain.runtime.runtime import UiPathLangGraphRuntime


def test_runtime_merges_deep_agent_configurable_values() -> None:
    runtime = UiPathLangGraphRuntime(
        graph=MagicMock(),
        runtime_id="runtime-1",
        configurable={"uipath_workspace_path": "/tmp/workspace"},
    )

    config = runtime._get_graph_config()

    assert config["configurable"]["thread_id"] == "runtime-1"
    assert config["configurable"]["uipath_workspace_path"] == "/tmp/workspace"
