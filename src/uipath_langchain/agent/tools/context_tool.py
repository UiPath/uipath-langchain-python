"""Context tool creation for semantic index retrieval."""

import uuid
from typing import Any, Optional, Type

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field, create_model
from uipath.agent.models.agent import (
    AgentContextResourceConfig,
    AgentContextRetrievalMode,
)
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.platform.common import CreateBatchTransform, CreateDeepRag, UiPathConfig
from uipath.platform.context_grounding import (
    BatchTransformOutputColumn,
    CitationMode,
    DeepRagContent,
)
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import AgentStartupError, AgentStartupErrorCode
from uipath_langchain.retrievers import ContextGroundingRetriever

from .durable_interrupt import durable_interrupt
from .structured_tool_with_output_type import StructuredToolWithOutputType
from .utils import sanitize_tool_name


def is_static_query(resource: AgentContextResourceConfig) -> bool:
    """Check if the resource configuration uses a static query variant."""
    if resource.settings.query is None or resource.settings.query.variant is None:
        return False
    return resource.settings.query.variant.lower() == "static"


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
    ensure_valid_fields(resource)

    # needed for type checking
    assert resource.settings.query is not None
    assert resource.settings.query.variant is not None

    retriever = ContextGroundingRetriever(
        index_name=resource.index_name,
        folder_path=resource.folder_path,
        number_of_results=resource.settings.result_count,
    )

    class ContextOutputSchemaModel(BaseModel):
        documents: list[Document] = Field(
            ..., description="List of retrieved documents."
        )

    output_model = ContextOutputSchemaModel
    input_model: Type[BaseModel]

    if is_static_query(resource):
        static_query_value = resource.settings.query.value
        assert static_query_value is not None

        class SemanticSearchStaticInputModel(BaseModel):
            pass

        input_model = SemanticSearchStaticInputModel

        @mockable(
            name=resource.name,
            description=resource.description,
            input_schema=input_model.model_json_schema(),
            output_schema=output_model.model_json_schema(),
            example_calls=[],  # Examples cannot be provided for context.
        )
        async def context_tool_fn() -> dict[str, Any]:
            docs = await retriever.ainvoke(static_query_value)
            return {
                "documents": [
                    {"metadata": doc.metadata, "page_content": doc.page_content}
                    for doc in docs
                ]
            }

    else:
        # Dynamic query - requires query parameter
        class ContextInputSchemaModel(BaseModel):
            query: str = Field(
                ..., description="The query to search for in the knowledge base"
            )

        input_model = ContextInputSchemaModel

        @mockable(
            name=resource.name,
            description=resource.description,
            input_schema=input_model.model_json_schema(),
            output_schema=output_model.model_json_schema(),
            example_calls=[],  # Examples cannot be provided for context.
        )
        async def context_tool_fn(query: str) -> dict[str, Any]:
            docs = await retriever.ainvoke(query)
            return {
                "documents": [
                    {"metadata": doc.metadata, "page_content": doc.page_content}
                    for doc in docs
                ]
            }

    return StructuredToolWithOutputType(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=context_tool_fn,
        output_type=output_model,
        metadata={
            "tool_type": "context",
            "display_name": resource.name,
        },
    )


def handle_deep_rag(
    tool_name: str, resource: AgentContextResourceConfig
) -> StructuredTool:
    ensure_valid_fields(resource)

    # needed for type checking
    assert resource.settings.query is not None
    assert resource.settings.query.variant is not None

    index_name = resource.index_name
    if not resource.settings.citation_mode:
        raise AgentStartupError(
            code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
            title="Missing citation mode",
            detail="Citation mode is required for Deep RAG. Please set the citation_mode field in context settings.",
            category=UiPathErrorCategory.USER,
        )
    citation_mode = CitationMode(resource.settings.citation_mode.value)

    output_model = create_model(
        "DeepRagOutputModel",
        __base__=DeepRagContent,
        deep_rag_id=(str, Field(alias="deepRagId")),
    )
    input_model: Type[BaseModel]

    if is_static_query(resource):
        # Static query - no input parameter needed
        static_prompt = resource.settings.query.value
        assert static_prompt is not None

        class DeepRagStaticInputModel(BaseModel):
            pass

        input_model = DeepRagStaticInputModel

        @mockable(
            name=resource.name,
            description=resource.description,
            input_schema=input_model.model_json_schema(),
            output_schema=output_model.model_json_schema(),
            example_calls=[],  # Examples cannot be provided for context.
        )
        async def context_tool_fn() -> dict[str, Any]:
            # TODO: add glob pattern support
            return interrupt(
                CreateDeepRag(
                    name=f"task-{uuid.uuid4()}",
                    index_name=index_name,
                    prompt=static_prompt,
                    citation_mode=citation_mode,
                )
            )

    else:
        # Dynamic query - requires query parameter
        class DeepRagInputSchemaModel(BaseModel):
            query: str = Field(
                ...,
                description="Describe the task: what to research across documents, what to synthesize, and how to cite sources",
            )

        input_model = DeepRagInputSchemaModel

        @mockable(
            name=resource.name,
            description=resource.description,
            input_schema=input_model.model_json_schema(),
            output_schema=output_model.model_json_schema(),
            example_calls=[],  # Examples cannot be provided for context.
        )
        async def context_tool_fn(query: str) -> dict[str, Any]:
            # TODO: add glob pattern support
            return interrupt(
                CreateDeepRag(
                    name=f"task-{uuid.uuid4()}",
                    index_name=index_name,
                    prompt=query,
                    citation_mode=citation_mode,
                )
            )

    return StructuredToolWithOutputType(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=context_tool_fn,
        output_type=output_model,
        metadata={
            "tool_type": "context",
            "display_name": resource.name,
        },
    )


def handle_batch_transform(
    tool_name: str, resource: AgentContextResourceConfig
) -> StructuredTool:
    ensure_valid_fields(resource)

    # needed for type checking
    assert resource.settings.query is not None
    assert resource.settings.query.variant is not None

    index_name = resource.index_name
    index_folder_path = resource.folder_path
    if not resource.settings.web_search_grounding:
        raise AgentStartupError(
            code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
            title="Missing web search grounding",
            detail="Web search grounding field is required for Batch Transform. Please set the web_search_grounding field in context settings.",
            category=UiPathErrorCategory.USER,
        )
    enable_web_search_grounding = (
        resource.settings.web_search_grounding.value.lower() == "enabled"
    )

    batch_transform_output_columns: list[BatchTransformOutputColumn] = []
    if (output_columns := resource.settings.output_columns) is None or not len(
        output_columns
    ):
        raise AgentStartupError(
            code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
            title="Missing output columns",
            detail="Batch transform requires at least one output column to be specified in settings.output_columns. Please add output columns to the context configuration.",
            category=UiPathErrorCategory.USER,
        )

    for column in output_columns:
        batch_transform_output_columns.append(
            BatchTransformOutputColumn(
                name=column.name,
                description=column.description,
            )
        )

    class JobAttachmentModel(BaseModel):
        ID: str = Field(description="Orchestrator attachment key")
        FullName: str = Field(description="File name")
        MimeType: str = Field(description="The MIME type of the content")

    class BatchTransformAttachmentOutput(BaseModel):
        result: JobAttachmentModel = Field(
            description="The transformed result file as an attachment"
        )

    output_model = BatchTransformAttachmentOutput

    input_model: Optional[Type[BaseModel]]

    if is_static_query(resource):
        # Static query - only destination_path parameter needed
        static_prompt = resource.settings.query.value
        assert static_prompt is not None

        class StaticBatchTransformSchemaModel(BaseModel):
            destination_path: str = Field(
                default="output.csv",
                description="The relative file path destination for the modified csv file",
            )

        input_model = StaticBatchTransformSchemaModel

        @mockable(
            name=resource.name,
            description=resource.description,
            input_schema=input_model.model_json_schema(),
            output_schema=output_model.model_json_schema(),
            example_calls=[],  # Examples cannot be provided for context.
        )
        async def context_tool_fn(
            destination_path: str = "output.csv",
        ) -> dict[str, Any]:
            # TODO: storage_bucket_folder_path_prefix  support
            @durable_interrupt
            async def create_batch_transform():
                return CreateBatchTransform(
                    name=f"task-{uuid.uuid4()}",
                    index_name=index_name,
                    prompt=static_prompt,
                    destination_path=destination_path,
                    index_folder_path=index_folder_path,
                    enable_web_search_grounding=enable_web_search_grounding,
                    output_columns=batch_transform_output_columns,
                )

            await create_batch_transform()

            uipath = UiPath()
            result_attachment_id = await uipath.jobs.create_attachment_async(
                name=destination_path,
                source_path=destination_path,
                job_key=UiPathConfig.job_key,
            )

            return {
                "result": {
                    "ID": str(result_attachment_id),
                    "FullName": destination_path,
                    "MimeType": "text/csv",
                }
            }

    else:
        # Dynamic query - requires both query and destination_path parameters
        class DynamicBatchTransformSchemaModel(BaseModel):
            query: str = Field(
                ...,
                description="Describe the task for each row: what to analyze, what to extract, and how to populate the output columns",
            )
            destination_path: str = Field(
                default="output.csv",
                description="The relative file path destination for the modified csv file",
            )

        input_model = DynamicBatchTransformSchemaModel

        @mockable(
            name=resource.name,
            description=resource.description,
            input_schema=input_model.model_json_schema(),
            output_schema=output_model.model_json_schema(),
            example_calls=[],  # Examples cannot be provided for context.
        )
        async def context_tool_fn(
            query: str, destination_path: str = "output.csv"
        ) -> dict[str, Any]:
            # TODO: storage_bucket_folder_path_prefix  support
            @durable_interrupt
            async def create_batch_transform():
                return CreateBatchTransform(
                    name=f"task-{uuid.uuid4()}",
                    index_name=index_name,
                    prompt=query,
                    destination_path=destination_path,
                    index_folder_path=index_folder_path,
                    enable_web_search_grounding=enable_web_search_grounding,
                    output_columns=batch_transform_output_columns,
                )

            await create_batch_transform()

            uipath = UiPath()
            result_attachment_id = await uipath.jobs.create_attachment_async(
                name=destination_path,
                source_path=destination_path,
                job_key=UiPathConfig.job_key,
            )

            return {
                "result": {
                    "ID": str(result_attachment_id),
                    "FullName": destination_path,
                    "MimeType": "text/csv",
                }
            }

    return StructuredToolWithOutputType(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=context_tool_fn,
        output_type=output_model,
        metadata={
            "tool_type": "context",
            "display_name": resource.name,
        },
    )


def ensure_valid_fields(resource_config: AgentContextResourceConfig):
    if not resource_config.settings.query:
        raise AgentStartupError(
            code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
            title="Missing query object",
            detail="Query object is required. Please set the query field in context settings.",
            category=UiPathErrorCategory.USER,
        )

    if not resource_config.settings.query.variant:
        raise AgentStartupError(
            code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
            title="Missing query variant",
            detail="Query variant is required. Please set the query variant in context settings.",
            category=UiPathErrorCategory.USER,
        )

    if is_static_query(resource_config) and not resource_config.settings.query.value:
        raise AgentStartupError(
            code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
            title="Missing static query value",
            detail="Static query requires a query value to be set. Please provide a value for the static query in context settings.",
            category=UiPathErrorCategory.USER,
        )
