"""Batch Transform tool for creation and retrieval of batch transforms."""

import uuid
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from uipath.agent.models.agent import (
    AgentContextResourceConfig,
    AgentInternalBatchTransformSettings,
    AgentInternalBatchTransformToolProperties,
    AgentInternalToolResourceConfig,
    AgentInternalToolType,
    AgentResourceType,
    AgentToolType,
)
from uipath.eval.mocks import mockable
from uipath.platform.common import CreateBatchTransform
from uipath.platform.context_grounding import (
    BatchTransformOutputColumn,
    BatchTransformResponse,
)

from uipath_langchain.agent.tools.structured_tool_with_output_type import (
    StructuredToolWithOutputType,
)
from uipath_langchain.agent.tools.tool_node import ToolWrapperMixin
from uipath_langchain.agent.tools.utils import sanitize_tool_name


class BatchTransformTool(StructuredToolWithOutputType, ToolWrapperMixin):
    pass


def convert_context_to_internal_batch_transform(
    resource: AgentContextResourceConfig,
) -> AgentInternalToolResourceConfig:
    """Convert AgentContextResourceConfig to AgentInternalToolResourceConfig for BatchTransform."""
    # Validate required fields
    if not resource.settings.query or not resource.settings.query.value:
        raise ValueError("Query prompt is required for BatchTransform")
    if not resource.settings.web_search_grounding:
        raise ValueError("Web search grounding field is required for Batch Transform")
    if not resource.settings.output_columns or not len(
        resource.settings.output_columns
    ):
        raise ValueError(
            "Batch transform requires at least one output column to be specified in settings.output_columns"
        )

    # Create the internal BatchTransform settings
    settings = AgentInternalBatchTransformSettings(
        contextType="attachments",  # Default context type
        indexName=resource.index_name,
        folderPath=resource.folder_path,
        query=resource.settings.query.value,
        folderPathPrefix=(
            resource.settings.folder_path_prefix.value
            if resource.settings.folder_path_prefix
            else None
        ),
        fileExtension=(
            resource.settings.file_extension.value
            if resource.settings.file_extension
            else None
        ),
        useWebSearchGrounding=(
            resource.settings.web_search_grounding.value.lower() == "enabled"
        ),
        outputColumns=resource.settings.output_columns,
    )

    # Create the internal BatchTransform properties
    properties = AgentInternalBatchTransformToolProperties(
        toolType=AgentInternalToolType.BATCH_TRANSFORM,
        settings=settings,
    )

    # Create the internal tool resource config
    return AgentInternalToolResourceConfig(
        name=resource.name,
        description=resource.description,
        resource_type=AgentResourceType.TOOL,
        type=AgentToolType.INTERNAL,
        input_schema={
            "type": "object",
            "properties": {
                "destination_path": {
                    "type": "string",
                    "description": "The relative file path destination for the modified csv file",
                }
            },
            "required": ["destination_path"],
        },
        output_schema=BatchTransformResponse.model_json_schema(),
        properties=properties,
        is_enabled=resource.is_enabled,
    )


def create_batch_transform_tool(
    resource: AgentInternalToolResourceConfig, llm: BaseChatModel
) -> StructuredTool:
    """Create a BatchTransform internal tool from resource configuration."""
    if not isinstance(resource.properties, AgentInternalBatchTransformToolProperties):
        raise ValueError(
            f"Expected AgentInternalBatchTransformToolProperties, got {type(resource.properties)}"
        )

    tool_name = sanitize_tool_name(resource.name)
    properties = resource.properties
    settings = properties.settings

    # Extract settings
    context_type = settings.context_type
    index_name = settings.index_name
    folder_path = settings.folder_path or "Shared"
    query = settings.query
    folder_path_prefix = settings.folder_path_prefix
    file_extension = settings.file_extension
    use_web_search_grounding = settings.use_web_search_grounding
    output_columns = settings.output_columns

    # Validate required fields
    if not query:
        raise ValueError("Query is required for BatchTransform tool")
    if not index_name:
        raise ValueError("Index name is required for BatchTransform tool")
    if not output_columns or not len(output_columns):
        raise ValueError(
            "At least one output column is required for BatchTransform tool"
        )

    # Convert output columns to BatchTransformOutputColumn
    batch_transform_output_columns = [
        BatchTransformOutputColumn(
            name=column.name,
            description=column.description,
        )
        for column in output_columns
    ]

    # Create input model
    class BatchTransformInputModel(BaseModel):
        destination_path: str = Field(
            ...,
            description="The relative file path destination for the modified csv file",
        )

    input_model = BatchTransformInputModel
    output_model = BatchTransformResponse

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=input_model.model_json_schema(),
        output_schema=output_model.model_json_schema(),
        example_calls=[],  # Examples cannot be provided for internal tools
    )
    async def batch_transform_tool_fn(destination_path: str) -> dict[str, Any]:
        # Create BatchTransform request using interrupt
        return interrupt(
            CreateBatchTransform(
                name=f"task-{uuid.uuid4()}",
                index_name=index_name,
                prompt=query,
                destination_path=destination_path,
                index_folder_path=folder_path,
                enable_web_search_grounding=use_web_search_grounding,
                output_columns=batch_transform_output_columns,
            )
        )

    return BatchTransformTool(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=batch_transform_tool_fn,
        output_type=output_model,
    )
