"""DeepRAG tool creation for enhanced knowledge base retrieval."""

import uuid
from typing import Any

from langchain_core.tools import StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from uipath.agent.models.agent import AgentContextResourceConfig
from uipath.eval.mocks import mockable
from uipath.platform.common import CreateDeepRag

from .structured_tool_with_output_type import StructuredToolWithOutputType
from .utils import sanitize_tool_name


def create_deeprag_tool(resource: AgentContextResourceConfig) -> StructuredTool:
    """Uses interrupt() to create a DeepRAG request for enhanced knowledge retrieval."""
    tool_name = sanitize_tool_name(resource.name)
    index_name = resource.index_name
    folder_path = resource.folder_path

    class DeepRagInputSchemaModel(BaseModel):
        prompt: str = Field(
            ..., description="The prompt/query to search for in the knowledge base using DeepRAG"
        )

    class DeepRagOutputSchemaModel(BaseModel):
        text: str = Field(
            ..., description="The DeepRAG response text based on the knowledge base"
        )

    input_model = DeepRagInputSchemaModel
    output_model = DeepRagOutputSchemaModel

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=input_model.model_json_schema(),
        output_schema=output_model.model_json_schema(),
    )
    async def deeprag_tool_fn(prompt: str) -> dict[str, Any]:
        result = interrupt(
            CreateDeepRag(
                name=str(uuid.uuid4()),
                index_name=index_name,
                index_folder_path=folder_path,
                prompt=prompt,
            )
        )
        return {"text": result["text"]}

    return StructuredToolWithOutputType(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=deeprag_tool_fn,
        output_type=output_model,
    )
