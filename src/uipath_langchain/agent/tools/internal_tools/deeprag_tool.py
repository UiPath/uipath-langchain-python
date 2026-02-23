"""Deeprag tool for creation and retrieval of deeprags."""

import uuid
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool, StructuredTool
from uipath.agent.models.agent import (
    AgentInternalDeepRagToolProperties,
    AgentInternalToolResourceConfig,
)
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.platform.common import CreateDeepRag
from uipath.platform.common.interrupt_models import WaitEphemeralIndex
from uipath.platform.context_grounding import (
    CitationMode,
    EphemeralIndexUsage,
)
from uipath.platform.context_grounding.context_grounding_index import (
    ContextGroundingIndex,
)
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import AgentStartupError, AgentStartupErrorCode
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.durable_interrupt import durable_interrupt
from uipath_langchain.agent.tools.internal_tools.schema_utils import (
    add_query_field_to_schema,
)
from uipath_langchain.agent.tools.static_args import handle_static_args
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)
from uipath_langchain.agent.tools.tool_node import ToolWrapperReturnType
from uipath_langchain.agent.tools.utils import sanitize_tool_name


def create_deeprag_tool(
    resource: AgentInternalToolResourceConfig, llm: BaseChatModel
) -> StructuredTool:
    """Create a DeepRAG internal tool from resource configuration."""
    if not isinstance(resource.properties, AgentInternalDeepRagToolProperties):
        raise AgentStartupError(
            code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
            title="Invalid DeepRAG tool properties",
            detail=f"Expected AgentInternalDeepRagToolProperties, got {type(resource.properties)}.",
            category=UiPathErrorCategory.SYSTEM,
        )

    tool_name = sanitize_tool_name(resource.name)
    properties = resource.properties
    settings = properties.settings

    # Extract settings
    query_setting = settings.query
    citation_mode_setting = settings.citation_mode

    citation_mode = (
        CitationMode(citation_mode_setting.value)
        if citation_mode_setting
        else CitationMode.INLINE
    )

    is_query_static = query_setting and query_setting.variant == "static"
    static_query = query_setting.value if is_query_static else None

    input_schema = dict(resource.input_schema)
    if not is_query_static:
        add_query_field_to_schema(
            input_schema,
            query_description=query_setting.description if query_setting else None,
            default_description="Describe the task: what to research across documents, what to synthesize and how to cite sources.",
        )

    input_model = create_model(input_schema)
    output_model = create_model(resource.output_schema)

    async def deeprag_tool_fn(**kwargs: Any) -> dict[str, Any]:
        query = kwargs.get("query") if not is_query_static else static_query
        if not query:
            raise ValueError("Query is required for DeepRAG tool")

        if "attachment" not in kwargs:
            raise ValueError("Argument 'attachment' is not available")

        attachment = kwargs.get("attachment")
        if not attachment:
            raise ValueError("Attachment is required for DeepRAG tool")

        attachment_id = getattr(attachment, "ID", None)
        if not attachment_id:
            raise ValueError("Attachment ID is required")

        @mockable(
            name=resource.name,
            description=resource.description,
            input_schema=input_model.model_json_schema() if input_model else None,
            output_schema=output_model.model_json_schema(),
            example_calls=[],  # Examples cannot be provided for internal tools
        )
        async def invoke_deeprag(**_tool_kwargs: Any):
            async def create_ephemeral_index():
                uipath = UiPath()
                ephemeral_index = (
                    await uipath.context_grounding.create_ephemeral_index_async(
                        usage=EphemeralIndexUsage.DEEP_RAG,
                        attachments=[attachment_id],
                    )
                )

                # TODO this will not resume on concurrent runs for the same attachment
                if ephemeral_index.in_progress_ingestion():

                    @durable_interrupt
                    async def wait_for_ephemeral_index():
                        return WaitEphemeralIndex(index=ephemeral_index)

                    index_result = await wait_for_ephemeral_index()
                    if isinstance(index_result, dict):
                        ephemeral_index = ContextGroundingIndex(**index_result)
                    else:
                        ephemeral_index = index_result

                return ephemeral_index

            index_result = await create_ephemeral_index()
            if isinstance(index_result, dict):
                ephemeral_index = ContextGroundingIndex(**index_result)
            else:
                ephemeral_index = index_result

            @durable_interrupt
            async def create_deeprag():
                return CreateDeepRag(
                    name=f"task-{uuid.uuid4()}",
                    index_name=ephemeral_index.name,
                    index_id=ephemeral_index.id,
                    prompt=query,
                    citation_mode=citation_mode,
                    is_ephemeral_index=True,
                )

            return await create_deeprag()

        return await invoke_deeprag(**kwargs)

    # Import here to avoid circular dependency
    from uipath_langchain.agent.wrappers import get_job_attachment_wrapper

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
        metadata={
            "tool_type": resource.type.lower(),
            "display_name": tool_name,
            "args_schema": input_model,
            "output_schema": output_model,
        },
    )
    tool.set_tool_wrappers(awrapper=deeprag_tool_wrapper)
    return tool
