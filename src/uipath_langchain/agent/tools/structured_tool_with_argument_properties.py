from uipath_langchain.agent.tools.static_args import ArgumentPropertiesMixin
from uipath_langchain.agent.tools.structured_tool_with_output_type import (
    StructuredToolWithOutputType,
)
from uipath_langchain.agent.tools.tool_node import ToolWrapperMixin


class StructuredToolWithArgumentProperties(
    StructuredToolWithOutputType, ToolWrapperMixin, ArgumentPropertiesMixin
):
    """A structured tool with static arguments."""

    pass
