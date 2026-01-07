"""Deeprag tool for creation and retrieval of deeprags."""

import uuid
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from uipath.agent.models.agent import (
    AgentContextResourceConfig,
    AgentInternalDeepRagSettings,
    AgentInternalDeepRagToolProperties,
    AgentInternalToolResourceConfig,
    AgentInternalToolType,
    AgentResourceType,
    AgentToolType,
)
from uipath.eval.mocks import mockable
from uipath.platform.common import CreateDeepRag
from uipath.platform.context_grounding import (
    CitationMode,
    DeepRagResponse,
)

from uipath_langchain.agent.tools.structured_tool_with_output_type import (
    StructuredToolWithOutputType,
)
from uipath_langchain.agent.tools.tool_node import ToolWrapperMixin
from uipath_langchain.agent.tools.utils import sanitize_tool_name


class DeepRagTool(StructuredToolWithOutputType, ToolWrapperMixin):
    pass


def convert_context_to_internal_deeprag(
    resource: AgentContextResourceConfig,
) -> AgentInternalToolResourceConfig:
    """Convert AgentContextResourceConfig to AgentInternalToolResourceConfig for DeepRAG."""
    # Validate required fields
    if not resource.settings.query or not resource.settings.query.value:
        raise ValueError("Query prompt is required for DeepRAG")
    if not resource.settings.citation_mode:
        raise ValueError("Citation mode is required for DeepRAG")

    # Create the internal DeepRAG settings
    settings = AgentInternalDeepRagSettings(
        contextType="attachments",  # Default context type
        indexName=resource.index_name,
        folderPath=resource.folder_path,
        query=resource.settings.query,
        folderPathPrefix=resource.settings.folder_path_prefix,
        citationMode=resource.settings.citation_mode,
        fileExtension=resource.settings.file_extension,
    )

    # Create the internal DeepRAG properties
    properties = AgentInternalDeepRagToolProperties(
        toolType=AgentInternalToolType.DEEP_RAG,
        settings=settings,
    )

    # Create the internal tool resource config
    return AgentInternalToolResourceConfig(
        name=resource.name,
        description=resource.description,
        resource_type=AgentResourceType.TOOL,
        type=AgentToolType.INTERNAL,
        input_schema={"type": "object", "properties": {}},
        output_schema=DeepRagResponse.model_json_schema(),
        properties=properties,
        is_enabled=resource.is_enabled,
    )


def create_deeprag_tool(
    resource: AgentInternalToolResourceConfig, llm: BaseChatModel
) -> StructuredTool:
    """Create a DeepRAG internal tool from resource configuration."""
    if not isinstance(resource.properties, AgentInternalDeepRagToolProperties):
        raise ValueError(
            f"Expected AgentInternalDeepRagToolProperties, got {type(resource.properties)}"
        )

    tool_name = sanitize_tool_name(resource.name)
    properties = resource.properties
    settings = properties.settings

    # Extract settings
    context_type = settings.context_type
    index_name = settings.index_name
    folder_path = settings.folder_path or "Shared"
    query_setting = settings.query
    folder_path_prefix = settings.folder_path_prefix
    citation_mode_setting = settings.citation_mode
    file_extension_setting = settings.file_extension

    # Determine citation mode
    citation_mode = (
        CitationMode(citation_mode_setting.value)
        if citation_mode_setting
        else CitationMode.INLINE
    )

    # Check if query is dynamic or static
    is_query_dynamic = query_setting and query_setting.variant == "dynamic"
    static_query = (
        query_setting.value if query_setting and not is_query_dynamic else None
    )

    # Create input model based on whether query is dynamic
    if is_query_dynamic:

        class DynamicQueryInput(BaseModel):
            query: str = Field(..., description="The query to create a deeprag off of")

        input_model = DynamicQueryInput
    else:
        # No input when query is static
        input_model = None

    output_model = DeepRagResponse

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=input_model.model_json_schema() if input_model else None,
        output_schema=output_model.model_json_schema(),
        example_calls=[],  # Examples cannot be provided for internal tools
    )
    async def deeprag_tool_fn(**kwargs: Any) -> dict[str, Any]:
        # Get query - dynamic from kwargs or static from settings
        query = kwargs.get("query") if is_query_dynamic else static_query
        if not query:
            raise ValueError("Query is required for DeepRAG tool")

        # Get index name from settings
        if not index_name:
            raise ValueError("Index name is required for DeepRAG tool")

        # Create DeepRAG request using interrupt
        return interrupt(
            CreateDeepRag(
                name=f"task-{uuid.uuid4()}",
                index_name=index_name,
                prompt=query,
                citation_mode=citation_mode,
                index_folder_path=folder_path,
            )
        )

    return DeepRagTool(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=deeprag_tool_fn,
        output_type=output_model,
    )
