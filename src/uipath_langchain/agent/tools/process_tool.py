"""Process tool creation for UiPath process execution."""

from __future__ import annotations

from typing import Any

from jsonschema_pydantic_converter import transform as create_model
from langchain_core.tools import StructuredTool
from langgraph.types import interrupt
from pydantic import TypeAdapter
from uipath.agent.models.agent import AgentProcessToolResourceConfig
from uipath.eval.mocks import mockable
from uipath.platform.common import InvokeProcess

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
    )
    async def process_tool_fn(**kwargs: Any) -> Any:
        result = interrupt(
            InvokeProcess(
                name=process_name,
                input_arguments=kwargs,
                process_folder_path=folder_path,
                process_folder_key=None,
            )
        )

        return TypeAdapter(output_model).validate_python(result)

    process_tool_fn.__annotations__["return"] = output_model

    tool = StructuredTool(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=process_tool_fn,
    )

    tool.__dict__["OutputType"] = output_model

    return tool
