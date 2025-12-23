"""Context tool creation for semantic index retrieval."""

import uuid
from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from uipath.agent.models.agent import (
    AgentContextResourceConfig,
    AgentContextRetrievalMode,
)
from uipath.eval.mocks import mockable
from uipath.platform.common import CreateBatchTransform, CreateDeepRag
from uipath.platform.context_grounding import (
    BatchTransformOutputColumn,
    BatchTransformResponse,
    CitationMode,
    DeepRagResponse,
)

from uipath_langchain.retrievers import ContextGroundingRetriever

from .structured_tool_with_output_type import StructuredToolWithOutputType
from .utils import sanitize_tool_name


def create_context_tool(resource: AgentContextResourceConfig) -> StructuredTool:
    tool_name = sanitize_tool_name(resource.name)
    retrieval_mode = resource.settings.retrieval_mode.lower()
    if retrieval_mode == AgentContextRetrievalMode.DEEP_RAG.value.lower():
        return handle_deep_rag(tool_name, resource)
    elif retrieval_mode == AgentContextRetrievalMode.BATCH_TRANSFORM.value.lower():
        return handle_batch_transform(tool_name, resource)
    else:
        return handle_semantic_search(tool_name, resource)


def handle_semantic_search(
    tool_name: str, resource: AgentContextResourceConfig
) -> StructuredTool:
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
        example_calls=[],  # Examples cannot be provided for context.
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


def handle_deep_rag(
    tool_name: str, resource: AgentContextResourceConfig
) -> StructuredTool:
    ensure_valid_fields(resource)
    # needed for type checking
    assert resource.settings.query is not None
    assert resource.settings.query.value is not None

    index_name = resource.index_name
    prompt = resource.settings.query.value
    if not resource.settings.citation_mode:
        raise ValueError("Citation mode is required for Deep RAG")
    citation_mode = CitationMode(resource.settings.citation_mode.value)

    input_model = None
    output_model = DeepRagResponse

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=input_model,
        output_schema=output_model.model_json_schema(),
        example_calls=[],  # Examples cannot be provided for context.
    )
    async def context_tool_fn() -> dict[str, Any]:
        # TODO: add glob pattern support
        return interrupt(
            CreateDeepRag(
                name=f"task-{uuid.uuid4()}",
                index_name=index_name,
                prompt=prompt,
                citation_mode=citation_mode,
            )
        )

    return StructuredToolWithOutputType(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=context_tool_fn,
        output_type=output_model,
    )


def handle_batch_transform(
    tool_name: str, resource: AgentContextResourceConfig
) -> StructuredTool:
    ensure_valid_fields(resource)

    # needed for type checking
    assert resource.settings.query is not None
    assert resource.settings.query.value is not None

    index_name = resource.index_name
    prompt = resource.settings.query.value

    index_folder_path = resource.folder_path
    if not resource.settings.web_search_grounding:
        raise ValueError("Web search grounding field is required for Batch Transform")
    enable_web_search_grounding = (
        resource.settings.web_search_grounding.value.lower() == "enabled"
    )

    batch_transform_output_columns: list[BatchTransformOutputColumn] = []
    if (output_columns := resource.settings.output_columns) is None or not len(
        output_columns
    ):
        raise ValueError(
            "Batch transform requires at least one output column to be specified in settings.output_columns"
        )

    for column in output_columns:
        batch_transform_output_columns.append(
            BatchTransformOutputColumn(
                name=column.name,
                description=column.description,
            )
        )

    class BatchTransformSchemaModel(BaseModel):
        destination_path: str = Field(
            ...,
            description="The relative file path destination for the modified csv file",
        )

    input_model = BatchTransformSchemaModel
    output_model = BatchTransformResponse

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=input_model.model_json_schema(),
        output_schema=output_model.model_json_schema(),
        example_calls=[],  # Examples cannot be provided for context.
    )
    async def context_tool_fn(destination_path: str) -> dict[str, Any]:
        # TODO: storage_bucket_folder_path_prefix  support
        return interrupt(
            CreateBatchTransform(
                name=f"task-{uuid.uuid4()}",
                index_name=index_name,
                prompt=prompt,
                destination_path=destination_path,
                index_folder_path=index_folder_path,
                enable_web_search_grounding=enable_web_search_grounding,
                output_columns=batch_transform_output_columns,
            )
        )

    return StructuredToolWithOutputType(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=context_tool_fn,
        output_type=output_model,
    )


def ensure_valid_fields(resource_config: AgentContextResourceConfig):
    if not resource_config.settings.query:
        raise ValueError("Query object is required")
    if not resource_config.settings.query.value:
        raise ValueError("Query prompt is required")
