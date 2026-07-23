"""Workspace backend helpers for UiPath DeepAgents."""

from deepagents.backends import FilesystemBackend
from langchain.tools import ToolRuntime
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_config

from uipath_langchain.runtime._workspace import get_workspace_path


class _UiPathWorkspaceBackendFactory:
    """DeepAgents backend factory resolved from UiPath runtime config."""

    def __call__(self, runtime: ToolRuntime) -> FilesystemBackend:
        config: RunnableConfig | None = getattr(runtime, "config", None)
        if config is None:
            try:
                config = get_config()
            except RuntimeError:
                config = {}
        return FilesystemBackend(
            root_dir=get_workspace_path(config),
            virtual_mode=True,
        )
