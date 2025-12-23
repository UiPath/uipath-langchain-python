"""Process tool creation for UiPath process execution."""

from typing import Any

from langchain_core.tools import StructuredTool
from langgraph.types import interrupt
from uipath.agent.models.agent import AgentProcessToolResourceConfig
from uipath.eval.mocks import mockable
from uipath.platform.common import InvokeProcess

from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model

from .structured_tool_with_output_type import StructuredToolWithOutputType
from .utils import sanitize_tool_name


def create_process_tool(resource: AgentProcessToolResourceConfig) -> StructuredTool:
    """Uses interrupt() to suspend graph execution until process completes (handled by runtime)."""
    tool_name: str = sanitize_tool_name(resource.name)
    process_name = resource.properties.process_name
    folder_path = resource.properties.folder_path

    input_model: Any = create_model(resource.input_schema)
    output_model: Any = create_model(resource.output_schema)

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=input_model.model_json_schema(),
        output_schema=output_model.model_json_schema(),
        example_calls=resource.properties.example_calls,
    )
    async def process_tool_fn(**kwargs: Any):
        return interrupt(
            InvokeProcess(
                name=process_name,
                input_arguments=kwargs,
                process_folder_path=folder_path,
                process_folder_key=None,
            )
        )

    tool = StructuredToolWithOutputType(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=process_tool_fn,
        output_type=output_model,
    )

    return tool
