"""Context tool creation for semantic index retrieval."""

from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from uipath.agent.models.agent import AgentContextResourceConfig
from uipath.eval.mocks import mockable

from uipath_langchain.retrievers import ContextGroundingRetriever

from .structured_tool_with_output_type import StructuredToolWithOutputType
from .utils import sanitize_tool_name


def create_context_tool(resource: AgentContextResourceConfig) -> StructuredTool:
    tool_name = sanitize_tool_name(resource.name)
    retriever = ContextGroundingRetriever(
        index_name=resource.index_name,
        folder_path=resource.folder_path,
        number_of_results=resource.settings.result_count,
    )

    class ContextInputSchemaModel(BaseModel):
        query: str = Field(
            ..., description="The query to search for in the knowledge base"
        )

    class ContextOutputSchemaModel(BaseModel):
        documents: list[Document] = Field(
            ..., description="List of retrieved documents."
        )

    input_model = ContextInputSchemaModel
    output_model = ContextOutputSchemaModel

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=input_model.model_json_schema(),
        output_schema=output_model.model_json_schema(),
    )
    async def context_tool_fn(query: str) -> dict[str, Any]:
        return {"documents": await retriever.ainvoke(query)}

    return StructuredToolWithOutputType(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=context_tool_fn,
        output_type=output_model,
    )
