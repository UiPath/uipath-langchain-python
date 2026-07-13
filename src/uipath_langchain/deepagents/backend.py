"""Workspace backend helpers for UiPath DeepAgents."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import BackendFactory
from langchain.tools import ToolRuntime

WORKSPACE_FILESYSTEM_BACKEND_ATTR = "is_uipath_workspace_filesystem_backend"


@dataclass(frozen=True)
class UiPathWorkspaceBackendFactory:
    """DeepAgents backend factory resolved from UiPath runtime config."""

    workspace_config_key: str = "uipath_workspace_path"
    virtual_mode: bool = True
    is_uipath_workspace_filesystem_backend: bool = field(default=True, init=False)

    def __call__(self, runtime: ToolRuntime) -> FilesystemBackend:
        config = getattr(runtime, "config", None) or {}
        configurable = (
            config.get("configurable", {}) if isinstance(config, dict) else {}
        )
        workspace_path = configurable.get(self.workspace_config_key)
        if not workspace_path:
            raise RuntimeError(
                f"UiPath DeepAgents workspace path missing from config key "
                f"'{self.workspace_config_key}'."
            )
        return FilesystemBackend(
            root_dir=Path(workspace_path), virtual_mode=self.virtual_mode
        )


def create_workspace_backend_factory(
    *,
    workspace_config_key: str = "uipath_workspace_path",
    virtual_mode: bool = True,
) -> BackendFactory:
    """Create a DeepAgents backend factory bound to the runtime workspace path."""
    return UiPathWorkspaceBackendFactory(
        workspace_config_key=workspace_config_key,
        virtual_mode=virtual_mode,
    )
