"""Context tool creation for semantic index retrieval."""

from __future__ import annotations

import json

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from uipath.agent.models.agent import AgentContextResourceConfig

from uipath_langchain.retrievers import ContextGroundingRetriever

from .utils import sanitize_tool_name


def create_context_tool(resource: AgentContextResourceConfig) -> StructuredTool:
    tool_name = sanitize_tool_name(resource.name)
    retriever = ContextGroundingRetriever(
        index_name=resource.index_name,
        folder_path=resource.folder_path,
        number_of_results=resource.settings.result_count,
    )

    async def context_tool_fn(query: str) -> str:
        docs = await retriever.ainvoke(query)

        if not docs:
            return ""

        return json.dumps([doc.model_dump() for doc in docs], indent=2)

    class ContextInputSchemaModel(BaseModel):
        query: str = Field(
            ..., description="The query to search for in the knowledge base"
        )

    return StructuredTool(
        name=tool_name,
        description=resource.description,
        args_schema=ContextInputSchemaModel,
        coroutine=context_tool_fn,
    )
