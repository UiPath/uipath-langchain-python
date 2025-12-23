import uuid
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from uipath.agent.models.agent import (
    AgentInternalToolResourceConfig,
)
from uipath.eval.mocks import mockable
from uipath.platform import UiPath

from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.react.llm_with_files import FileInfo, llm_call_with_files
from uipath_langchain.agent.tools.structured_tool_with_output_type import (
    StructuredToolWithOutputType,
)
from uipath_langchain.agent.tools.tool_node import ToolWrapperMixin
from uipath_langchain.agent.tools.utils import sanitize_tool_name
from uipath_langchain.agent.wrappers.job_attachment_wrapper import (
    get_job_attachment_wrapper,
)

ANALYZE_FILES_SYSTEM_MESSAGE = (
    "Process the provided files to complete the given task. "
    "Analyze the files contents thoroughly to deliver an accurate response "
    "based on the extracted information."
)


class AnalyzeFileTool(StructuredToolWithOutputType, ToolWrapperMixin):
    pass


def create_analyze_file_tool(
    resource: AgentInternalToolResourceConfig, llm: BaseChatModel
) -> StructuredTool:
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

        attachments = kwargs["attachments"]
        analysisTask = kwargs["analysisTask"]

        files = await _resolve_job_attachment_arguments(attachments)
        messages: list[AnyMessage] = [
            SystemMessage(content=ANALYZE_FILES_SYSTEM_MESSAGE),
            HumanMessage(content=analysisTask),
        ]
        result = await llm_call_with_files(messages, files, llm)
        return result

    wrapper = get_job_attachment_wrapper()
    tool = AnalyzeFileTool(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=tool_fn,
        output_type=output_model,
    )
    tool.set_tool_wrappers(awrapper=wrapper)
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
