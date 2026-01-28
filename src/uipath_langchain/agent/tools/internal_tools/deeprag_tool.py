"""Deeprag tool for creation and retrieval of deeprags."""

import uuid
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import interrupt
from uipath.agent.models.agent import (
    AgentInternalDeepRagToolProperties,
    AgentInternalToolResourceConfig,
)
from uipath.eval.mocks import mockable
from uipath.platform.common import CreateDeepRag
from uipath.platform.common.interrupt_models import CreateEphemeralIndex
from uipath.platform.context_grounding import (
    CitationMode,
)

from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.static_args import handle_static_args
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)
from uipath_langchain.agent.tools.tool_node import ToolWrapperReturnType
from uipath_langchain.agent.tools.utils import sanitize_tool_name
from uipath_langchain.agent.wrappers import get_job_attachment_wrapper


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
    query_setting = settings.query
    citation_mode_setting = settings.citation_mode

    # Determine citation mode
    citation_mode = (
        CitationMode(citation_mode_setting.value)
        if citation_mode_setting
        else CitationMode.INLINE
    )

    # Check if query is dynamic or static
    is_query_static = query_setting and query_setting.variant == "static"
    static_query = query_setting.value if is_query_static else None

    # Use resource input schema and add query field if dynamic
    input_schema = dict(resource.input_schema)
    if not is_query_static:
        # Add query field to the schema
        if "properties" not in input_schema:
            input_schema["properties"] = {}
        input_schema["properties"]["query"] = {
            "type": "string",
            "description": query_setting.description
            if query_setting and query_setting.description
            else "The query to create a deeprag off of",
        }
        # Add query to required fields
        if "required" not in input_schema:
            input_schema["required"] = []
        if "query" not in input_schema["required"]:
            input_schema["required"].append("query")

    # Create input model from modified schema
    input_model = create_model(input_schema)
    output_model = create_model(resource.output_schema)

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=input_model.model_json_schema() if input_model else None,
        output_schema=output_model.model_json_schema(),
        example_calls=[],  # Examples cannot be provided for internal tools
    )
    async def deeprag_tool_fn(**kwargs: Any) -> dict[str, Any]:
        # Get query - dynamic from kwargs or static from settings
        query = kwargs.get("query") if not is_query_static else static_query
        if not query:
            raise ValueError("Query is required for DeepRAG tool")

        if "attachment" not in kwargs:
            raise ValueError("Argument 'attachment' is not available")

        attachment = kwargs.get("attachment")
        if not attachment:
            raise ValueError("Attachment is required for DeepRAG tool")

        # Extract attachment ID using getattr (works for Pydantic models)
        attachment_id = getattr(attachment, "ID", None)
        if not attachment_id:
            raise ValueError("Attachment ID is required")

        ephemeral_index = interrupt(
            CreateEphemeralIndex(usage="DeepRAG", attachments=[attachment_id])
        )

        # Create DeepRAG request using interrupt
        return interrupt(
            CreateDeepRag(
                name=f"task-{uuid.uuid4()}",
                index_name=ephemeral_index.name,
                index_id=ephemeral_index.id,
                prompt=query,
                citation_mode=citation_mode,
            )
        )

    job_attachment_wrapper = get_job_attachment_wrapper(output_type=output_model)

    async def deeprag_tool_wrapper(
        tool: BaseTool,
        call: ToolCall,
        state: AgentGraphState,
    ) -> ToolWrapperReturnType:
        call["args"] = handle_static_args(resource, state, call["args"])
        return await job_attachment_wrapper(tool, call, state)

    tool = StructuredToolWithArgumentProperties(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=deeprag_tool_fn,
        output_type=output_model,
        argument_properties=resource.argument_properties,
    )
    tool.set_tool_wrappers(awrapper=deeprag_tool_wrapper)
    return tool
