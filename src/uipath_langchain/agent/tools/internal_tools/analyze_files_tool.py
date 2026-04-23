import logging
import mimetypes
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
from langchain_core.runnables.config import var_child_runnable_config
from langchain_core.tools import StructuredTool
from uipath.agent.models.agent import (
    AgentInternalToolResourceConfig,
)
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)
from uipath_langchain.agent.multimodal import (
    FileInfo,
    build_file_content_blocks_for,
)
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.tools.internal_tools.pii_masker import PiiMasker
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)
from uipath_langchain.agent.tools.utils import sanitize_tool_name
from uipath_langchain.chat.helpers import (
    append_content_blocks_to_message,
    extract_text_content,
)

logger = logging.getLogger("uipath")

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

    # Disable streaming so for conversational loops, the internal LLM call doesn't leak
    # AIMessageChunk events into the graph stream.
    non_streaming_llm = llm.model_copy(update={"disable_streaming": True})

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

        client: UiPath | None = None
        policy: dict[str, Any] | None = None
        try:
            client = UiPath()
            policy = await client.automation_ops.get_deployed_policy_async()
        except Exception:
            logger.exception("Failed to fetch deployed policy")

        masker: PiiMasker | None = None
        if client is not None and PiiMasker.is_policy_enabled(policy):
            masker = PiiMasker(client, policy)
            try:
                analysis_task, files = await masker.apply(analysis_task, files)
            except Exception as exc:
                raise AgentRuntimeError(
                    code=AgentRuntimeErrorCode.UNEXPECTED_ERROR,
                    title="PII masking failed",
                    detail=f"PII detection raised: {exc!r}",
                    category=UiPathErrorCategory.SYSTEM,
                ) from exc

        try:
            human_message = HumanMessage(content=analysis_task)
            human_message_with_files = await add_files_to_message(human_message, files)
        except ValueError as exc:
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.FILE_ERROR,
                title="File attachment too large",
                detail=str(exc),
                category=UiPathErrorCategory.USER,
            ) from exc

        messages: list[AnyMessage] = [
            SystemMessage(content=ANALYZE_FILES_SYSTEM_MESSAGE),
            cast(AnyMessage, human_message_with_files),
        ]
        config = var_child_runnable_config.get(None)
        result = await non_streaming_llm.ainvoke(messages, config=config)

        del messages, human_message_with_files, files

        analysis_result = extract_text_content(result)

        if masker is not None:
            try:
                analysis_result = masker.rehydrate(analysis_result)
            except Exception as exc:
                raise AgentRuntimeError(
                    code=AgentRuntimeErrorCode.UNEXPECTED_ERROR,
                    title="PII rehydration failed",
                    detail=f"Failed to rehydrate LLM response: {exc!r}",
                    category=UiPathErrorCategory.SYSTEM,
                ) from exc

        return {"analysisResult": analysis_result}

    job_attachment_wrapper = get_job_attachment_wrapper(output_type=output_model)

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
    tool.set_tool_wrappers(awrapper=job_attachment_wrapper)
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
        blob_info = await client.attachments.get_blob_file_access_uri_async(
            key=attachment_id
        )

        input_mime_type = getattr(attachment, "MimeType", None)
        mime_type = (
            input_mime_type
            if input_mime_type
            else (mimetypes.guess_type(blob_info.name)[0] or "")
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

    file_content_blocks: list[DataContentBlock] = []
    for file in files:
        blocks = await build_file_content_blocks_for(file)
        file_content_blocks.extend(blocks)
    return append_content_blocks_to_message(
        message, cast(list[ContentBlock], file_content_blocks)
    )
