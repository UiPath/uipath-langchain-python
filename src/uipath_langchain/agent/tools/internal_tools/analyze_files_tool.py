from typing import Any

from langchain_core.tools import StructuredTool
from uipath.agent.models.agent import (
    AgentInternalToolResourceConfig,
)
from uipath.eval.mocks import mockable

from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.tools.structured_tool_with_output_type import (
    StructuredToolWithOutputType,
)
from uipath_langchain.agent.tools.tool_node import ToolWrapperMixin
from uipath_langchain.agent.tools.utils import sanitize_tool_name
from uipath_langchain.agent.wrappers.job_attachment_wrapper import (
    get_job_attachment_wrapper,
)


class AnalyzeFileTool(StructuredToolWithOutputType, ToolWrapperMixin):
    pass


def create_analyze_file_tool(
    resource: AgentInternalToolResourceConfig,
) -> StructuredTool:
    tool_name = sanitize_tool_name(resource.name)
    input_model = create_model(resource.input_schema)
    output_model = create_model(resource.output_schema)

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=input_model.model_json_schema(),
        output_schema=output_model.model_json_schema(),
    )
    async def tool_fn(**kwargs: Any):
        return "The event name is 'Toamna' by Tudor Gheorghe"

    wrapper = get_job_attachment_wrapper(resource)
    tool = AnalyzeFileTool(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=tool_fn,
        output_type=output_model,
    )
    tool.set_tool_wrappers(awrapper=wrapper)
    return tool
