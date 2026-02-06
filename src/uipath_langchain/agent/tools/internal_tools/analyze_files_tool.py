import asyncio
import uuid
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
    ContentBlock,
    DataContentBlock,
    HumanMessage,
    SystemMessage,
)
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool, StructuredTool
from uipath.agent.models.agent import (
    AgentInternalToolResourceConfig,
)
from uipath.eval.mocks import mockable
from uipath.platform import UiPath

from uipath_langchain.agent.multimodal import FileInfo, build_file_content_block
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.static_args import handle_static_args
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)
from uipath_langchain.agent.tools.tool_node import ToolWrapperReturnType
from uipath_langchain.agent.tools.utils import sanitize_tool_name
from uipath_langchain.chat.helpers import (
    append_content_blocks_to_message,
    extract_text_content,
)

ANALYZE_FILES_SYSTEM_MESSAGE = (
    "Process the provided files to complete the given task. "
    "Analyze the files contents thoroughly to deliver an accurate response "
    "based on the extracted information."
)


def create_analyze_file_tool(
    resource: AgentInternalToolResourceConfig, llm: BaseChatModel
) -> StructuredTool:
    # Import here to avoid circular dependency
    from uipath_langchain.agent.wrappers import get_job_attachment_wrapper

    tool_name = sanitize_tool_name(resource.name)
    input_model = create_model(resource.input_schema)
    output_model = create_model(resource.output_schema)

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=input_model.model_json_schema(),
        output_schema=output_model.model_json_schema(),
    )
    async def tool_fn(**kwargs: Any):
        if "analysisTask" not in kwargs:
            raise ValueError("Argument 'analysisTask' is not available")
        if "attachments" not in kwargs:
            raise ValueError("Argument 'attachments' is not available")

        analysis_task = kwargs["analysisTask"]
        if not analysis_task:
            raise ValueError("Argument 'analysisTask' is not available")

        attachments = kwargs["attachments"]

        files = await _resolve_job_attachment_arguments(attachments)
        if not files:
            return {"analysisResult": "No attachments provided to analyze."}

        human_message = HumanMessage(content=analysis_task)
        human_message_with_files = await add_files_to_message(human_message, files)

        messages: list[AnyMessage] = [
            SystemMessage(content=ANALYZE_FILES_SYSTEM_MESSAGE),
            cast(AnyMessage, human_message_with_files),
        ]
        result = await llm.ainvoke(messages)

        analysis_result = extract_text_content(result)
        return analysis_result

    job_attachment_wrapper = get_job_attachment_wrapper(output_type=output_model)

    async def analyze_file_tool_wrapper(
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
        coroutine=tool_fn,
        output_type=output_model,
        argument_properties=resource.argument_properties,
        metadata={
            "tool_type": resource.type.lower(),
            "display_name": tool_name,
            "args_schema": input_model,
            "output_schema": output_model,
        },
    )
    tool.set_tool_wrappers(awrapper=analyze_file_tool_wrapper)
    return tool


async def _resolve_job_attachment_arguments(
    attachments: list[Any],
) -> list[FileInfo]:
    """Resolve job attachments to FileInfo objects.

    Args:
        attachments: List of job attachment objects (dynamically typed from schema)

    Returns:
        List of FileInfo objects with blob URIs for each attachment
    """
    client = UiPath()
    file_infos: list[FileInfo] = []

    for attachment in attachments:
        # Access using Pydantic field aliases (ID, FullName, MimeType)
        # These are dynamically created from the JSON schema
        attachment_id_value = getattr(attachment, "ID", None)
        if attachment_id_value is None:
            continue

        attachment_id = uuid.UUID(attachment_id_value)
        mime_type = getattr(attachment, "MimeType", "")

        blob_info = await client.attachments.get_blob_file_access_uri_async(
            key=attachment_id
        )

        file_info = FileInfo(
            url=blob_info.uri,
            name=blob_info.name,
            mime_type=mime_type,
        )
        file_infos.append(file_info)

    return file_infos


async def add_files_to_message(
    message: BaseMessage,
    files: list[FileInfo],
) -> BaseMessage:
    """Add file attachments to a message.

    Args:
        message: The message to add files to (any BaseMessage subclass)
        files: List of file attachments to add

    Returns:
        New message of the same type with file content blocks appended
    """
    if not files:
        return message

    file_content_blocks: list[DataContentBlock] = await asyncio.gather(
        *[build_file_content_block(file) for file in files]
    )
    return append_content_blocks_to_message(
        message, cast(list[ContentBlock], file_content_blocks)
    )
