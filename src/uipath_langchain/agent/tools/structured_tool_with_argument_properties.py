from uipath.agent.models.agent import AgentToolArgumentProperties

from uipath_langchain.agent.tools.structured_tool_with_output_type import (
    StructuredToolWithOutputType,
)
from uipath_langchain.agent.tools.tool_node import ToolWrapperMixin


class StructuredToolWithArgumentProperties(
    StructuredToolWithOutputType, ToolWrapperMixin
):
    """A structured tool with static arguments."""

    argument_properties: dict[str, AgentToolArgumentProperties]
