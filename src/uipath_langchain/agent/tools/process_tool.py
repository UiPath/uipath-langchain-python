"""Process tool creation for UiPath process execution."""

from __future__ import annotations

from typing import Any, Type

from jsonschema_pydantic import jsonschema_to_pydantic  # type: ignore[import-untyped]
from langchain_core.tools import StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel
from uipath.agent.models.agent import AgentProcessToolResourceConfig
from uipath.models import InvokeProcess

from .utils import sanitize_tool_name


def create_process_tool(resource: AgentProcessToolResourceConfig) -> StructuredTool:
    """Uses interrupt() to suspend graph execution until process completes (handled by runtime)."""
    tool_name: str = sanitize_tool_name(resource.name)
    process_name = resource.properties.process_name
    folder_path = resource.properties.folder_path

    input_model: Type[BaseModel] = jsonschema_to_pydantic(resource.input_schema)
    output_model: Type[BaseModel] = jsonschema_to_pydantic(resource.output_schema)

    async def process_tool_fn(**kwargs: Any):
        try:
            result = interrupt(
                InvokeProcess(
                    name=process_name,
                    input_arguments=kwargs,
                    process_folder_path=folder_path,
                    process_folder_key=None,
                )
            )
        except Exception:
            raise

        return result

    tool = StructuredTool(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=process_tool_fn,
    )

    tool.__dict__["OutputType"] = output_model

    return tool
