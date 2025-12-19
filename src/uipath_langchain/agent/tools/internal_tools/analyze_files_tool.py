from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import StructuredTool
from uipath.agent.models.agent import (
    AgentInternalToolResourceConfig,
)
from uipath.eval.mocks import mockable

from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.tools.structured_tool_with_output_type import (
    StructuredToolWithOutputType,
)
from uipath_langchain.agent.tools.utils import sanitize_tool_name


def create_analyze_file_tool(
    resource: AgentInternalToolResourceConfig,
) -> StructuredTool:
    """
    Creates an internal tool based on the resource configuration.

    Routes to the appropriate handler based on the tool_type specified in
    the resource properties.

    Args:
        resource: Internal tool resource configuration

    Returns:
        A structured tool that can be used by LangChain agents

    Raises:
        ValueError: If schema creation fails or tool_type is not supported
    """
    tool_name = sanitize_tool_name(resource.name)
    input_model = create_model(resource.input_schema)
    output_model = create_model(resource.output_schema)

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=input_model.model_json_schema(),
        output_schema=output_model.model_json_schema(),
    )
    async def tool_fn(runtime: ToolRuntime, **kwargs: Any):
        return "Tool result message."

    return StructuredToolWithOutputType(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=tool_fn,
        output_type=output_model,
    )
