"""Process tool creation for UiPath process execution."""

from typing import Any

from jsonschema_pydantic_converter import transform as create_model
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import interrupt
from uipath.agent.models.agent import AgentProcessToolResourceConfig
from uipath.eval.mocks import mockable
from uipath.platform.common import InvokeProcess

from .structured_tool_with_output_type import StructuredToolWithOutputType
from .utils import sanitize_tool_name


def create_process_tool(resource: AgentProcessToolResourceConfig) -> StructuredTool:
    """Uses interrupt() to suspend graph execution until process completes (handled by runtime)."""
    tool_name: str = sanitize_tool_name(resource.name)
    process_name = resource.properties.process_name
    folder_path = resource.properties.folder_path

    input_model: Any = create_model(resource.input_schema)
    output_model: Any = create_model(resource.output_schema)

    async def process_tool_fn(runtime: ToolRuntime, **kwargs: Any):
        @mockable(
            name=resource.name,
            description=resource.description,
            input_schema=input_model.model_json_schema(),
            output_schema=output_model.model_json_schema(),
            example_calls=[],  # TODO: pass these in from runtime.
        )
        async def process_tool_impl(**inner_kwargs: Any):
            return interrupt(
                InvokeProcess(
                    name=process_name,
                    input_arguments=inner_kwargs,
                    process_folder_path=folder_path,
                    process_folder_key=None,
                )
            )

        return await process_tool_impl(**kwargs)

    tool = StructuredToolWithOutputType(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=process_tool_fn,
        output_type=output_model,
    )

    return tool
