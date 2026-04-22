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
from uipath.core.feature_flags import FeatureFlags
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.platform.semantic_proxy import (
    PiiDetectionRequest,
    PiiDetectionResponse,
    PiiDocument,
    PiiEntityThreshold,
    PiiFile,
    rehydrate_from_pii_response,
)
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

_PII_MASKING_FEATURE_FLAG = "File-Pii-Masking-Enabled"


def is_pii_policy_enabled(policy: dict[str, Any] | None) -> bool:
    """Determine whether PII detection should run.

    Two gates (both must allow):

    1. Local kill-switch — the ``File-Pii-Masking-Enabled`` feature flag
       (defaults to ``True``; override via ``FeatureFlags.configure_flags``
       or ``UIPATH_FEATURE_File-Pii-Masking-Enabled`` env var).
    2. Platform policy — ``data.container.pii-in-flight-agents`` from the
       AutomationOps deployed-policy response.
    """
    if not FeatureFlags.is_flag_enabled(_PII_MASKING_FEATURE_FLAG, default=True):
        return False
    if not policy:
        return False
    container = policy.get("data", {}).get("container", {})
    return bool(container.get("pii-in-flight-agents", False))


def _build_entity_thresholds_from_policy(
    policy: dict[str, Any] | None,
) -> list[PiiEntityThreshold]:
    """Extract enabled entity thresholds from the deployed policy.

    Filters ``data.pii-entity-table`` to entries where ``pii-entity-is-enabled``
    is true and maps them to ``PiiEntityThreshold`` objects.
    """
    if not policy:
        return []
    table = policy.get("data", {}).get("pii-entity-table", [])
    thresholds: list[PiiEntityThreshold] = []
    for entry in table:
        if not entry.get("pii-entity-is-enabled", False):
            continue
        category = entry.get("pii-entity-category")
        confidence = entry.get("pii-entity-confidence-threshold")
        if category is None or confidence is None:
            continue
        thresholds.append(
            PiiEntityThreshold(
                category=category,
                confidence_threshold=confidence,
            )
        )
    return thresholds


def _rename_for_masking(file: FileInfo, redacted_url: str) -> FileInfo:
    """Return a new ``FileInfo`` pointing at the redacted URL with a ``pii_masked_`` name prefix."""
    if "." in file.name:
        base, ext = file.name.rsplit(".", 1)
        new_name = f"pii_masked_{base}.{ext}"
    else:
        new_name = f"pii_masked_{file.name}"
    return FileInfo(url=redacted_url, name=new_name, mime_type=file.mime_type)


async def _apply_pii_masking(
    client: UiPath,
    policy: dict[str, Any] | None,
    analysis_task: str,
    files: list[FileInfo],
) -> tuple[str, list[FileInfo], PiiDetectionResponse]:
    """Run PII detection and return a masked prompt, redacted files, and the raw response.

    The returned response is retained so the LLM output can be rehydrated via
    :func:`rehydrate_from_pii_response` after inference.
    """
    pii_request = PiiDetectionRequest(
        documents=[PiiDocument(id="user-prompt", role="user", document=analysis_task)],
        files=[
            PiiFile(
                file_name=f.name,
                file_url=f.url,
                file_type=f.name.rsplit(".", 1)[-1].lower() if "." in f.name else "",
            )
            for f in files
        ],
        entity_thresholds=_build_entity_thresholds_from_policy(policy) or None,
    )
    pii_result = await client.semantic_proxy.detect_pii_async(pii_request)
    logger.info(
        "PII detection completed: %d document entities, %d file entities",
        sum(len(d.pii_entities) for d in pii_result.response),
        sum(len(f.pii_entities) for f in pii_result.files),
    )

    masked_prompt = analysis_task
    for doc in pii_result.response:
        if doc.id == "user-prompt":
            if doc.masked_document != analysis_task:
                logger.info(
                    "User prompt masked (%d entities replaced)",
                    len(doc.pii_entities),
                )
            masked_prompt = doc.masked_document
            break

    redacted_by_name = {f.file_name: f.file_url for f in pii_result.files}
    if redacted_by_name:
        masked_files = [
            _rename_for_masking(f, redacted_by_name.get(f.name, f.url)) for f in files
        ]
        logger.info("Renamed %d file(s) with pii_masked_ prefix", len(masked_files))
    else:
        masked_files = files

    return masked_prompt, masked_files, pii_result


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

        client = UiPath()
        policy: dict[str, Any] | None = None
        try:
            policy = await client.automation_ops.get_deployed_policy_async()
        except Exception:
            logger.exception("Failed to fetch deployed policy")

        pii_result: PiiDetectionResponse | None = None
        if is_pii_policy_enabled(policy):
            try:
                analysis_task, files, pii_result = await _apply_pii_masking(
                    client, policy, analysis_task, files
                )
            except Exception as e:
                logger.error("PII detection raised: %r", e, exc_info=True)

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

        if pii_result is not None:
            try:
                rehydrated = rehydrate_from_pii_response(analysis_result, pii_result)
                if rehydrated != analysis_result:
                    logger.info("Rehydrated LLM response with PII entities")
                analysis_result = rehydrated
            except Exception:
                logger.exception("Failed to rehydrate LLM response")

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
