from types import SimpleNamespace
from unittest.mock import patch

import pytest
from deepagents.backends import FilesystemBackend
from langchain.tools import ToolRuntime
from langchain_core.runnables import RunnableConfig

from uipath_langchain.deepagents.backend import _UiPathWorkspaceBackendFactory
from uipath_langchain.runtime._workspace import WORKSPACE_PATH_CONFIG_KEY


def _tool_runtime(config: RunnableConfig) -> ToolRuntime:
    return ToolRuntime(
        state={},
        context=None,
        config=config,
        stream_writer=lambda _: None,
        tool_call_id=None,
        store=None,
    )


def test_workspace_backend_factory_resolves_configured_workspace(tmp_path) -> None:
    factory = _UiPathWorkspaceBackendFactory()

    backend = factory(
        _tool_runtime({"configurable": {WORKSPACE_PATH_CONFIG_KEY: str(tmp_path)}})
    )

    assert isinstance(backend, FilesystemBackend)
    assert backend.cwd == tmp_path.resolve()
    assert backend.virtual_mode is True


def test_workspace_backend_confines_file_access_to_workspace(tmp_path) -> None:
    backend = _UiPathWorkspaceBackendFactory()(
        _tool_runtime({"configurable": {WORKSPACE_PATH_CONFIG_KEY: str(tmp_path)}})
    )

    result = backend.write("/launch/brief.md", "launch plan")

    assert result.error is None
    assert (tmp_path / "launch" / "brief.md").read_text() == "launch plan"
    with pytest.raises(ValueError, match="Path traversal not allowed"):
        backend.write("../escape.md", "outside")
    assert not (tmp_path.parent / "escape.md").exists()


def test_workspace_backend_factory_raises_when_config_missing() -> None:
    factory = _UiPathWorkspaceBackendFactory()

    with pytest.raises(RuntimeError, match="workspace path is unavailable"):
        factory(_tool_runtime({"configurable": {}}))


@pytest.mark.parametrize("workspace_path", [None, 42, ""])
def test_workspace_backend_factory_rejects_invalid_workspace_path(
    workspace_path: object,
) -> None:
    factory = _UiPathWorkspaceBackendFactory()

    with pytest.raises(RuntimeError, match="workspace path is unavailable"):
        factory(
            _tool_runtime({"configurable": {WORKSPACE_PATH_CONFIG_KEY: workspace_path}})
        )


def test_workspace_backend_factory_uses_active_config_for_current_runtime(
    tmp_path,
) -> None:
    factory = _UiPathWorkspaceBackendFactory()

    with patch(
        "uipath_langchain.deepagents.backend.get_config",
        return_value={"configurable": {WORKSPACE_PATH_CONFIG_KEY: str(tmp_path)}},
    ):
        backend = factory(SimpleNamespace())  # type: ignore[arg-type]

    assert backend.cwd == tmp_path.resolve()
