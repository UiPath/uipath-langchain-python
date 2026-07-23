"""UiPath DeepAgents integration."""

from typing import Any

__all__ = ["create_uipath_deep_agent", "with_uipath_managed_workspace"]


def __getattr__(name: str) -> Any:
    if name == "create_uipath_deep_agent":
        from .agent import create_uipath_deep_agent

        return create_uipath_deep_agent
    if name == "with_uipath_managed_workspace":
        from .metadata import with_uipath_managed_workspace

        return with_uipath_managed_workspace
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
