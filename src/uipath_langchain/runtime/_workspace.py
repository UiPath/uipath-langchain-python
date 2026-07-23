"""Internal workspace integration shared by runtime-backed agents."""

from pathlib import Path
from typing import Any, NotRequired, TypedDict

from langchain_core.runnables import RunnableConfig

WORKSPACE_PATH_CONFIG_KEY = "uipath_workspace_path"


class _UiPathGraphConfigurable(TypedDict):
    """UiPath-owned values injected into LangGraph's configurable payload."""

    thread_id: str
    uipath_workspace_path: NotRequired[str]


def create_graph_configurable(
    *,
    thread_id: str,
    workspace_path: Path | None,
) -> dict[str, Any]:
    """Create the UiPath-owned portion of LangGraph's configurable payload."""
    configurable: _UiPathGraphConfigurable = {"thread_id": thread_id}
    if workspace_path is not None:
        configurable["uipath_workspace_path"] = str(workspace_path)
    return dict(configurable)


def get_workspace_path(config: RunnableConfig) -> Path:
    """Decode and validate the UiPath-managed workspace path from graph config."""
    configurable = config.get("configurable")
    workspace_path = (
        configurable.get(WORKSPACE_PATH_CONFIG_KEY)
        if isinstance(configurable, dict)
        else None
    )
    if not isinstance(workspace_path, str) or not workspace_path:
        raise RuntimeError(
            "UiPath DeepAgents workspace path is unavailable. Run graphs created "
            "by create_uipath_deep_agent through the UiPath runtime."
        )
    return Path(workspace_path)
