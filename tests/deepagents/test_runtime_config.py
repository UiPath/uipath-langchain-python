from pathlib import Path
from unittest.mock import MagicMock

from uipath_langchain.runtime._workspace import WORKSPACE_PATH_CONFIG_KEY
from uipath_langchain.runtime.runtime import UiPathLangGraphRuntime


def test_runtime_adds_workspace_path_to_graph_config(tmp_path) -> None:
    runtime = UiPathLangGraphRuntime(
        graph=MagicMock(),
        runtime_id="runtime-1",
        workspace_path=Path(tmp_path),
    )

    config = runtime._get_graph_config()

    assert config["configurable"]["thread_id"] == "runtime-1"
    assert config["configurable"][WORKSPACE_PATH_CONFIG_KEY] == str(tmp_path)


def test_runtime_without_workspace_has_only_thread_config() -> None:
    runtime = UiPathLangGraphRuntime(
        graph=MagicMock(),
        runtime_id="runtime-1",
    )

    assert runtime._get_graph_config()["configurable"] == {"thread_id": "runtime-1"}
