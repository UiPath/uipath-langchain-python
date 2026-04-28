import json
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
from langchain_core.runnables.config import RunnableConfig, var_child_runnable_config
from langchain_core.tools import StructuredTool
from opentelemetry import trace as otel_trace
from uipath.agent.models.agent import (
    AgentInternalToolResourceConfig,
)
from uipath.core.tracing.span_utils import UiPathSpanUtils
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.runtime.errors import UiPathErrorCategory
from uipath.tracing import (
    AttachmentDirection,
    AttachmentProvider,
    SpanAttachment,
)

from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)
from uipath_langchain.agent.multimodal import (
    FileInfo,
    build_file_content_blocks_for,
)
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.tools.internal_tools.pii_masker import (
    PiiMasker,
    _masked_name_for,
)
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

# Langchain config metadata key carrying the JSON-serialized SpanAttachment list
# that should render on the llmCall span. The LLMOps callback in uipath-agents
# reads this and stamps it on the llmCall span as the ``attachments`` attribute.
LLM_CALL_ATTACHMENTS_METADATA_KEY = "uipath_llm_call_attachments"


def _original_attachment_id(file: FileInfo) -> str:
    """Return the id to use for the original file in trace attachments.

    Prefers the orchestrator attachment UUID when present; falls back to a
    UUID derived from the file URL for files that did not come from
    orchestrator (defensive, should not happen in production paths).
    """
    if file.attachment_id:
        return file.attachment_id
    return str(uuid.uuid5(uuid.NAMESPACE_URL, file.url))


def _masked_attachment_id(masked_url: str) -> str:
    """Derive a stable GUID from the masked URL for trace attachments.

    The LLMOps traces endpoint validates ``Attachment.Id`` as ``System.Guid``.
    Masked files aren't orchestrator-tracked, so we synthesize a deterministic
    UUID from the redacted blob URL to satisfy the schema while keeping the id
    stable across re-runs.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, masked_url))


def _set_span_attachments(
    span: otel_trace.Span, attachments: list[SpanAttachment]
) -> None:
    """Write a :class:`SpanAttachment` list as a JSON string on the given OTel span."""
    if not attachments or span is None or not span.is_recording():
        return
    try:
        span.set_attribute(
            "attachments",
            json.dumps([att.model_dump(by_alias=True) for att in attachments]),
        )
    except Exception:
        logger.exception("Failed to emit trace attachments")


def _llm_call_attachments_payload(files: list[FileInfo]) -> str | None:
    """Build the JSON attachments payload for the llmCall span.

    Each entry represents the file version actually sent to the model: the
    masked copy when PII masking ran (keyed by the orchestrator UUID from the
    re-upload when available, uuid5 fallback otherwise), else the original
    orchestrator attachment. Direction is ``IN`` because the file is an input
    to the LLM.
    """
    if not files:
        return None
    attachments: list[SpanAttachment] = []
    for file in files:
        if file.masked_attachment_url:
            att_id = file.masked_attachment_id or _masked_attachment_id(
                file.masked_attachment_url
            )
            name = _masked_name_for(file.name)
        else:
            att_id = _original_attachment_id(file)
            name = file.name
        attachments.append(
            SpanAttachment(
                id=att_id,
                file_name=name,
                mime_type=file.mime_type,
                provider=AttachmentProvider.ORCHESTRATOR,
                direction=AttachmentDirection.IN,
            )
        )
    return json.dumps([att.model_dump(by_alias=True) for att in attachments])


def _config_with_llm_call_attachments(
    config: RunnableConfig | None, files: list[FileInfo]
) -> RunnableConfig | None:
    """Return a runnable config carrying the llmCall attachments payload.

    The LLMOps callback in ``uipath-agents`` reads the payload from
    ``metadata[LLM_CALL_ATTACHMENTS_METADATA_KEY]`` and stamps it as the
    ``attachments`` attribute on the llmCall span — so the file actually sent
    to the model (masked copy when PII masking ran, original otherwise)
    renders as a downloadable attachment on the LLM-call boundary in the
    trace UI, mirroring how the PII Masking span renders its files.
    """
    payload = _llm_call_attachments_payload(files)
    if not payload:
        return config
    new_config = cast(RunnableConfig, dict(config) if config else {})
    metadata = dict(new_config.get("metadata") or {})
    metadata[LLM_CALL_ATTACHMENTS_METADATA_KEY] = payload
    new_config["metadata"] = metadata
    return new_config


def _emit_pii_masking_attachments(span: otel_trace.Span, files: list[FileInfo]) -> None:
    """Emit originals (IN) and masked copies (OUT) on the given PII Masking span.

    Originals are keyed by the orchestrator attachment UUID; masked copies are
    keyed by the real orchestrator UUID from the re-upload when available, or
    a uuid5 derived from the redacted URL as a fallback.
    """
    if not files:
        return
    attachments: list[SpanAttachment] = []
    input_files: list[dict[str, Any]] = []
    output_files: list[dict[str, Any]] = []

    for file in files:
        original_id = _original_attachment_id(file)
        attachments.append(
            SpanAttachment(
                id=original_id,
                file_name=file.name,
                mime_type=file.mime_type,
                provider=AttachmentProvider.ORCHESTRATOR,
                direction=AttachmentDirection.IN,
            )
        )
        input_files.append(
            {"id": original_id, "fileName": file.name, "mimeType": file.mime_type}
        )

        if file.masked_attachment_url:
            # Prefer the real orchestrator UUID from the re-upload so the UI
            # can download the file; fall back to the synthesized uuid5.
            masked_id = file.masked_attachment_id or _masked_attachment_id(
                file.masked_attachment_url
            )
            masked_name = _masked_name_for(file.name)
            attachments.append(
                SpanAttachment(
                    id=masked_id,
                    file_name=masked_name,
                    mime_type=file.mime_type,
                    provider=AttachmentProvider.ORCHESTRATOR,
                    direction=AttachmentDirection.OUT,
                )
            )
            output_files.append(
                {"id": masked_id, "fileName": masked_name, "mimeType": file.mime_type}
            )

    _set_span_attachments(span, attachments)

    if span is not None and span.is_recording():
        try:
            input_payload = json.dumps({"files": input_files})
            output_payload = json.dumps({"files": output_files})
            span.set_attribute("input", input_payload)
            span.set_attribute("input.value", input_payload)
            span.set_attribute("input.mime_type", "application/json")
            span.set_attribute("output", output_payload)
            span.set_attribute("output.value", output_payload)
            span.set_attribute("output.mime_type", "application/json")
        except Exception:
            logger.exception("Failed to set PII Masking input/output attributes")


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
            # Reconcile OTel current span with the LangChain/LangGraph external
            # span provider so the new span is parented under the active tool
            # call span and shares its trace id.
            parent_ctx = UiPathSpanUtils.get_parent_context()
            tracer = otel_trace.get_tracer(__name__)
            with tracer.start_as_current_span(
                "PII Masking", context=parent_ctx
            ) as pii_span:
                # Required for the LLMOps exporter's span filter to keep this span.
                pii_span.set_attribute("uipath.custom_instrumentation", True)
                pii_span.set_attribute("span_type", "piiMasking")
                pii_span.set_attribute("type", "piiMasking")
                masker = PiiMasker(client, policy)
                try:
                    analysis_task, files = await masker.apply(analysis_task, files)
                    _emit_pii_masking_attachments(pii_span, files)
                except Exception as exc:
                    pii_span.record_exception(exc)
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
        config = _config_with_llm_call_attachments(config, files)
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
            attachment_id=str(attachment_id),
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
        # Prefer the redacted URL + pii_masked_ name for LLM content when PII masking ran.
        llm_file = (
            FileInfo(
                url=file.masked_attachment_url,
                name=_masked_name_for(file.name),
                mime_type=file.mime_type,
            )
            if file.masked_attachment_url
            else file
        )
        blocks = await build_file_content_blocks_for(llm_file)
        file_content_blocks.extend(blocks)
    return append_content_blocks_to_message(
        message, cast(list[ContentBlock], file_content_blocks)
    )
