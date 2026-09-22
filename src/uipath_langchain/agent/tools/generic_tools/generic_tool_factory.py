"""Factory for generic agent tools.

A generic tool (``type: "generic"``) carries a free-form ``properties.settings``
object and is dispatched on ``properties.sub_type``. Supporting a new subtype
only requires registering a handler here; the agent definition models stay
unchanged.
"""

from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from uipath.agent.models.agent import AgentGenericToolResourceConfig
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import AgentStartupError, AgentStartupErrorCode

from .jev import JEV_SUB_TYPE, create_jev_tool

_GENERIC_TOOL_HANDLERS: dict[
    str,
    Callable[[AgentGenericToolResourceConfig, BaseChatModel], BaseTool],
] = {
    JEV_SUB_TYPE: create_jev_tool,
}


def create_generic_tool(
    resource: AgentGenericToolResourceConfig, llm: BaseChatModel
) -> BaseTool:
    """Create a generic tool based on its ``sub_type``.

    Raises:
        AgentStartupError: If no handler is registered for the subtype.
    """
    sub_type = resource.properties.sub_type.lower()
    handler = _GENERIC_TOOL_HANDLERS.get(sub_type)
    if handler is None:
        raise AgentStartupError(
            code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
            title="Unsupported generic tool subtype",
            detail=f"Tool '{resource.name}' has unsupported subtype "
            f"'{resource.properties.sub_type}'. "
            f"Supported subtypes: {sorted(_GENERIC_TOOL_HANDLERS)}.",
            category=UiPathErrorCategory.USER,
        )
    return handler(resource, llm)
