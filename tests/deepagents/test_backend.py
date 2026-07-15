from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from deepagents.backends.filesystem import FilesystemBackend
from langchain.tools import ToolRuntime

from uipath_langchain.deepagents.backend import (
    WORKSPACE_FILESYSTEM_BACKEND_ATTR,
    create_workspace_backend_factory,
)


def _tool_runtime(config: dict[str, Any]) -> ToolRuntime:
    return ToolRuntime(
        state=None,
        context=None,
        config=config,
        stream_writer=lambda _: None,
        tool_call_id=None,
        store=None,
    )


def test_workspace_backend_factory_resolves_configured_workspace(tmp_path) -> None:
    factory = create_workspace_backend_factory(workspace_config_key="workspace")

    assert getattr(factory, WORKSPACE_FILESYSTEM_BACKEND_ATTR) is True

    backend = factory(_tool_runtime({"configurable": {"workspace": str(tmp_path)}}))

    assert isinstance(backend, FilesystemBackend)
    assert backend.cwd == tmp_path.resolve()
    assert backend.virtual_mode is True


def test_workspace_backend_factory_raises_when_config_missing() -> None:
    factory = create_workspace_backend_factory(workspace_config_key="workspace")

    with pytest.raises(RuntimeError, match="workspace path missing"):
        factory(_tool_runtime({"configurable": {}}))


def test_workspace_backend_factory_uses_active_config_for_current_runtime(
    tmp_path,
) -> None:
    factory = create_workspace_backend_factory(workspace_config_key="workspace")

    with patch(
        "uipath_langchain.deepagents.backend.get_config",
        return_value={"configurable": {"workspace": str(tmp_path)}},
    ):
        backend = factory(SimpleNamespace())  # type: ignore[arg-type]

    assert backend.cwd == tmp_path.resolve()
