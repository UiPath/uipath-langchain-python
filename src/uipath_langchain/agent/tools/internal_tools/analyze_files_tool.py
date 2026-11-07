import base64
import hashlib
import json
import logging
import mimetypes
import uuid
from collections.abc import Iterator
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
from uipath.llm_client import UiPathAPIError, UiPathError
from uipath.llm_client.utils.exceptions import as_uipath_error
from uipath.platform import UiPath
from uipath.platform.errors import EnrichedException
from uipath.runtime.errors import UiPathErrorCategory
from uipath.tracing import (
    AttachmentDirection,
    AttachmentProvider,
    SpanAttachment,
)

from uipath_langchain.agent.attachments.job_attachments import (
    raise_for_job_attachment_error,
)
from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)
from uipath_langchain.agent.exceptions.llm import (
    describe_attachment,
    raise_for_file_analysis_http_error,
    raise_for_llm_client_error,
)
from uipath_langchain.agent.multimodal import (
    FileInfo,
    accepts_only_pdf_documents,
    build_file_content_blocks_for,
    file_part_filename,
    is_pdf,
    normalize_mime_type,
)
from uipath_langchain.agent.react.jsonschema_pydantic_converter import (
    create_model,
    create_output_model,
)
from uipath_langchain.agent.tools.internal_tools.pii_masker import (
    PiiMasker,
    masked_name_for,
)
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)
from uipath_langchain.agent.tools.utils import (
    config_without_streaming,
    sanitize_tool_name,
)
from uipath_langchain.chat.attachment_baggage import llm_call_attachment_ids
from uipath_langchain.chat.helpers import (
    append_content_blocks_to_message,
    extract_text_content,
)

logger = logging.getLogger("uipath")

# The Orchestrator attachment an image part's bytes came from, which the part cannot name.
_SENT_ATTACHMENT_ID_EXTRA = "uipath_sent_attachment_id"
_CONTENT_KEY_PREFIX = "sha256:"
# A TIFF is sent as one image part per page; the cap keeps the trace baggage header small.
_MAX_CONTENT_KEYED_PARTS = 10

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
            name = masked_name_for(file.name)
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


def _llm_call_attachment_ids(
    files: list[FileInfo], message: BaseMessage
) -> dict[str, str]:
    """Map each file part of ``message`` to the Orchestrator attachment it was read from.

    A part sent with a ``filename`` is keyed by it. An image part (an image, a TIFF
    page) carries no filename, so it is keyed by ``sha256:`` and the hex digest of its
    bytes, which the guardrails service computes from the inline file it is handed;
    only the first :data:`_MAX_CONTENT_KEYED_PARTS` are listed.

    Covers the same file versions as :func:`_llm_call_attachments_payload`, but only
    ids Orchestrator can serve: a masked copy without a re-upload id, or a file that
    did not come from Orchestrator, is left out. So is a key two different attachments
    are sent with, which would link one of them to the other.
    """
    ids: dict[str, str] = {}
    ambiguous: set[str] = set()

    def add(key: str, att_id: str) -> None:
        if ids.setdefault(key, att_id) != att_id:
            ambiguous.add(key)

    for file in files:
        att_id, name = _sent_attachment(file)
        filename = file_part_filename(name, file.mime_type)
        if att_id and filename is not None:
            add(filename, att_id)
    for key, att_id in _image_part_keys(message):
        add(key, att_id)
    linked = {key: att_id for key, att_id in ids.items() if key not in ambiguous}
    content_keys = [key for key in linked if key.startswith(_CONTENT_KEY_PREFIX)]
    for key in content_keys[_MAX_CONTENT_KEYED_PARTS:]:
        del linked[key]
    return linked


def _sent_attachment(file: FileInfo) -> tuple[str | None, str]:
    """The Orchestrator id and name of the file version the model is sent."""
    if file.masked_attachment_url:
        return file.masked_attachment_id, masked_name_for(file.name)
    return file.attachment_id, file.name


def _image_part_keys(message: BaseMessage) -> Iterator[tuple[str, str]]:
    if not isinstance(message.content, list):
        return
    for part in message.content:
        if not isinstance(part, dict) or part.get("type") != "image":
            continue
        extras = part.get("extras")
        att_id = (
            extras.get(_SENT_ATTACHMENT_ID_EXTRA) if isinstance(extras, dict) else None
        )
        data = part.get("base64")
        if att_id and isinstance(data, str):
            digest = hashlib.sha256(base64.b64decode(data)).hexdigest()
            yield f"{_CONTENT_KEY_PREFIX}{digest}", att_id


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


# Policy scope values that include files in PII detection. The analyze-files
# tool is a files-only flow, so masking runs only when files are in scope.
_PII_FILE_SCOPES = frozenset({"Both", "Files"})


def _is_pii_scope_for_files(policy: dict[str, Any] | None) -> bool:
    """Return True when the policy's ``pii-detection-scope`` covers files."""
    if not policy:
        return False
    return policy.get("data", {}).get("pii-detection-scope") in _PII_FILE_SCOPES


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
            masked_name = masked_name_for(file.name)
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


def _tool_result(analysis: str) -> dict[str, str]:
    """Build the tool result, carrying the answer under both of its keys.

    ``analysis`` is the field the tool's output schema declares, so it is the one tool
    guardrail rules select. ``analysisResult`` is the key the tool returned before, kept
    so anything that reads it keeps working.
    """
    return {"analysis": analysis, "analysisResult": analysis}


def create_analyze_file_tool(
    resource: AgentInternalToolResourceConfig, llm: BaseChatModel
) -> StructuredTool:
    # Import here to avoid circular dependency
    from uipath_langchain.agent.wrappers import get_job_attachment_wrapper

    tool_name = sanitize_tool_name(resource.name)
    input_model = create_model(resource.input_schema)
    output_model = create_output_model(resource.output_schema, resource.name)

    # Explicitly disable streaming - for conversational, no streaming is needed as this
    # internal tool-call does not produce streamed conversation events.
    non_streaming_llm = llm.model_copy(update={"disable_streaming": True})
    pdf_only_documents = accepts_only_pdf_documents(llm)

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
            return _tool_result("No attachments provided to analyze.")

        client: UiPath | None = None
        policy: dict[str, Any] | None = None
        try:
            client = UiPath()
            policy = await client.automation_ops.get_deployed_policy_async()
        except Exception:
            logger.exception("Failed to fetch deployed policy")

        masker: PiiMasker | None = None
        if (
            client is not None
            and PiiMasker.is_policy_enabled(policy)
            and _is_pii_scope_for_files(policy)
        ):
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
            human_message_with_files = await add_files_to_message(
                human_message, files, pdf_only_documents=pdf_only_documents
            )
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
        config = config_without_streaming(config)
        config = _config_with_llm_call_attachments(config, files)
        # Described by the type the file is sent as, not the one it was declared as.
        sent_types = [normalize_mime_type(f.mime_type, f.name) for f in files]
        attachment_names = [
            describe_attachment(f.name, mime_type)
            for f, mime_type in zip(files, sent_types, strict=True)
        ]
        all_pdf = bool(files) and all(is_pdf(mime_type) for mime_type in sent_types)
        try:
            with llm_call_attachment_ids(
                _llm_call_attachment_ids(files, human_message_with_files)
            ):
                result = await non_streaming_llm.ainvoke(messages, config=config)
        except UiPathAPIError as exc:
            raise_for_file_analysis_http_error(exc, attachment_names, all_pdf=all_pdf)
        except UiPathError as exc:
            raise_for_llm_client_error(exc, attachment_names)
            raise
        except Exception as exc:
            # legacy clients surface raw provider SDK exceptions.
            uipath_error = as_uipath_error(exc)
            if isinstance(uipath_error, UiPathAPIError):
                raise_for_file_analysis_http_error(
                    uipath_error, attachment_names, all_pdf=all_pdf
                )
            raise

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

        return _tool_result(analysis_result)

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


def _attachment_field(attachment: Any, name: str) -> Any:
    """Read an attachment field by its canonical name (``ID``/``FullName``/``MimeType``).

    Supports both mapping-shaped attachments (plain dicts with those keys, as
    produced by ``model_dump`` on the generated input model) and attribute-shaped
    objects (the coerced input-model instances the analyze-files tool receives, or
    a ``SimpleNamespace``). This lets the analyze-files tool and the memory
    file-key recall path share one resolver.
    """
    if isinstance(attachment, dict):
        return attachment.get(name)
    return getattr(attachment, name, None)


async def resolve_attachments_to_file_infos(
    attachments: list[Any],
) -> list[FileInfo]:
    """Resolve job attachments to FileInfo objects.

    Args:
        attachments: List of job attachments, each either a mapping with
            ``ID``/``FullName``/``MimeType`` keys or an object exposing those as
            attributes.

    Returns:
        List of FileInfo objects with blob URIs for each attachment
    """
    client: UiPath | None = None
    file_infos: list[FileInfo] = []

    for attachment in attachments:
        # Access using the Pydantic field aliases (ID, FullName, MimeType) that
        # are dynamically created from the JSON schema, or the equivalent dict keys.
        attachment_id_value = _attachment_field(attachment, "ID")
        if attachment_id_value is None:
            continue
        # Construct the platform client only once a resolvable attachment exists —
        # memory recall may call this per field, often with nothing to resolve.
        client = client or UiPath()
        attachment_name = _attachment_field(attachment, "FullName") or "<unknown>"

        try:
            attachment_id = uuid.UUID(attachment_id_value)
        except (ValueError, TypeError) as e:
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.INVALID_ATTACHMENT_ID,
                title="Invalid attachment id",
                detail=(
                    f"Attachment id {attachment_id_value!r} ('{attachment_name}') is not a valid UUID"
                ),
                category=UiPathErrorCategory.SYSTEM,
            ) from e
        try:
            blob_info = await client.attachments.get_blob_file_access_uri_async(
                key=attachment_id
            )
        except EnrichedException as e:
            raise_for_job_attachment_error(
                e,
                title="Failed to resolve job attachment",
                attachment_name=attachment_name,
                attachment_id=attachment_id,
            )
            raise

        input_mime_type = _attachment_field(attachment, "MimeType")
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


async def _resolve_job_attachment_arguments(
    attachments: list[Any],
) -> list[FileInfo]:
    """Backwards-compatible wrapper around :func:`resolve_attachments_to_file_infos`."""
    return await resolve_attachments_to_file_infos(attachments)


async def add_files_to_message(
    message: BaseMessage,
    files: list[FileInfo],
    *,
    pdf_only_documents: bool = False,
) -> BaseMessage:
    """Add file attachments to a message.

    Args:
        message: The message to add files to (any BaseMessage subclass)
        files: List of file attachments to add
        pdf_only_documents: The model accepts base64 documents only as PDF, so
            other files are sent as text documents (see
            :func:`build_file_content_blocks_for`).

    Returns:
        New message of the same type with file content blocks appended. An image
        part names, in its extras, the Orchestrator attachment its bytes came from.
    """
    if not files:
        return message

    file_content_blocks: list[DataContentBlock] = []
    for file in files:
        # Prefer the redacted URL + pii_masked_ name for LLM content when PII masking ran.
        llm_file = (
            FileInfo(
                url=file.masked_attachment_url,
                name=masked_name_for(file.name),
                mime_type=file.mime_type,
            )
            if file.masked_attachment_url
            else file
        )
        blocks = await build_file_content_blocks_for(
            llm_file, pdf_only_documents=pdf_only_documents
        )
        sent_id, _ = _sent_attachment(file)
        if sent_id:
            for block in blocks:
                if block.get("type") == "image":
                    extras = cast(dict[str, Any], block).setdefault("extras", {})
                    extras[_SENT_ATTACHMENT_ID_EXTRA] = sent_id
        file_content_blocks.extend(blocks)
    return append_content_blocks_to_message(
        message, cast(list[ContentBlock], file_content_blocks)
    )
