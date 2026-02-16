"""Batch Transform tool for creating and retrieving batch transformations."""

import uuid
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.func import task
from langgraph.types import interrupt
from uipath.agent.models.agent import (
    AgentInternalBatchTransformToolProperties,
    AgentInternalToolResourceConfig,
)
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.platform.common import CreateBatchTransform
from uipath.platform.common.interrupt_models import WaitEphemeralIndex
from uipath.platform.context_grounding import (
    BatchTransformOutputColumn,
    EphemeralIndexUsage,
)
from uipath.platform.context_grounding.context_grounding_index import (
    ContextGroundingIndex,
)

from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.internal_tools.schema_utils import (
    add_query_field_to_schema,
)
from uipath_langchain.agent.tools.static_args import handle_static_args
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)
from uipath_langchain.agent.tools.tool_node import ToolWrapperReturnType
from uipath_langchain.agent.tools.utils import sanitize_tool_name


def create_batch_transform_tool(
    resource: AgentInternalToolResourceConfig, llm: BaseChatModel
) -> StructuredTool:
    """Create a Batch Transform internal tool from resource configuration."""
    if not isinstance(resource.properties, AgentInternalBatchTransformToolProperties):
        raise ValueError(
            f"Expected AgentInternalBatchTransformToolProperties, got {type(resource.properties)}"
        )

    tool_name = sanitize_tool_name(resource.name)
    properties = resource.properties
    settings = properties.settings

    # Extract settings
    query_setting = settings.query
    folder_path_prefix_setting = settings.folder_path_prefix
    output_columns_setting = settings.output_columns
    web_search_grounding_setting = settings.web_search_grounding

    is_query_static = query_setting and query_setting.variant == "static"
    static_query = query_setting.value if is_query_static else None

    static_folder_path_prefix = None
    if folder_path_prefix_setting:
        static_folder_path_prefix = getattr(folder_path_prefix_setting, "value", None)

    static_web_search = False
    if web_search_grounding_setting:
        value = getattr(web_search_grounding_setting, "value", None)
        static_web_search = value == "Enabled" if value else False

    batch_transform_output_columns = [
        BatchTransformOutputColumn(name=col.name, description=col.description or "")
        for col in output_columns_setting
    ]

    # Use resource input schema and add query field if dynamic
    input_schema = dict(resource.input_schema)
    if not is_query_static:
        add_query_field_to_schema(
            input_schema,
            query_description=query_setting.description if query_setting else None,
            default_description="Describe the task: what to research, what to synthesize.",
        )

    # Create input model from modified schema
    input_model = create_model(input_schema)
    output_model = create_model(resource.output_schema)

    async def batch_transform_tool_fn(**kwargs: Any) -> dict[str, Any]:
        query = kwargs.get("query") if not is_query_static else static_query
        if not query:
            raise ValueError("Query is required for Batch Transform tool")

        if "attachment" not in kwargs:
            raise ValueError("Argument 'attachment' is not available")

        attachment = kwargs.get("attachment")
        if not attachment:
            raise ValueError("Attachment is required for Batch Transform tool")

        attachment_id = getattr(attachment, "ID", None)
        if not attachment_id:
            raise ValueError("Attachment ID is required")

        destination_path = kwargs.get("destination_path", "output.csv")

        @mockable(
            name=resource.name,
            description=resource.description,
            input_schema=input_model.model_json_schema() if input_model else None,
            output_schema=output_model.model_json_schema(),
            example_calls=[],  # Examples cannot be provided for internal tools
        )
        async def invoke_batch_transform():
            @task
            async def create_ephemeral_index():
                uipath = UiPath()
                ephemeral_index = (
                    await uipath.context_grounding.create_ephemeral_index_async(
                        usage=EphemeralIndexUsage.BATCH_RAG,
                        attachments=[attachment_id],
                    )
                )
                if ephemeral_index.in_progress_ingestion():
                    ephemeral_index_dict = interrupt(
                        WaitEphemeralIndex(index=ephemeral_index)
                    )
                    return ContextGroundingIndex(**ephemeral_index_dict)
                return ephemeral_index

            ephemeral_index = await create_ephemeral_index()

            return interrupt(
                CreateBatchTransform(
                    name=f"task-{uuid.uuid4()}",
                    index_name=ephemeral_index.name,
                    index_id=ephemeral_index.id,
                    prompt=query,
                    output_columns=batch_transform_output_columns,
                    storage_bucket_folder_path_prefix=static_folder_path_prefix,
                    enable_web_search_grounding=static_web_search,
                    destination_path=destination_path,
                    is_ephemeral_index=True,
                )
            )

        return await invoke_batch_transform()

    # Import here to avoid circular dependency
    from uipath_langchain.agent.wrappers import get_job_attachment_wrapper

    job_attachment_wrapper = get_job_attachment_wrapper(output_type=output_model)

    async def batch_transform_tool_wrapper(
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
        coroutine=batch_transform_tool_fn,
        output_type=output_model,
        argument_properties=resource.argument_properties,
        metadata={
            "tool_type": resource.type.lower(),
            "display_name": tool_name,
            "args_schema": input_model,
            "output_schema": output_model,
        },
    )
    tool.set_tool_wrappers(awrapper=batch_transform_tool_wrapper)
    return tool
