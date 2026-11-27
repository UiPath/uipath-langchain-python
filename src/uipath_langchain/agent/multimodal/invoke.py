"""LLM invocation with multimodal file attachments."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    DataContentBlock,
    HumanMessage,
)
from langchain_core.messages.content import create_file_block, create_image_block
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain._utils._attachments import (
    ATTACHMENT_ID_EXTRA,
    ATTACHMENT_NAME_EXTRA,
)

from ..exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from .types import MAX_FILE_SIZE_BYTES, FileInfo
from .utils import (
    FileTooLargeError,
    base_mime_type,
    decode_text,
    download_file_base64,
    is_image,
    is_pdf,
    is_text,
    is_tiff,
    is_unlabeled_pdf,
    is_unlabeled_text,
    normalize_mime_type,
    sanitize_filename,
    stream_tiff_to_content_blocks,
)

logger = logging.getLogger("uipath")

_GONE_STATUSES = frozenset({404, 410})


def file_part_filename(name: str, mime_type: str) -> str | None:
    """Return the ``filename`` a file is sent to the model with.

    Mirrors :func:`build_file_content_blocks_for`: images and TIFF pages go out as
    image blocks, which carry no filename, so they return ``None``.
    """
    sent_mime_type = normalize_mime_type(mime_type, name)
    if is_tiff(sent_mime_type) or is_image(sent_mime_type):
        return None
    return sanitize_filename(name)


def _file_error(
    title: str,
    detail: str,
    category: UiPathErrorCategory = UiPathErrorCategory.USER,
    status: int | None = None,
) -> AgentRuntimeError:
    return AgentRuntimeError(
        code=AgentRuntimeErrorCode.FILE_ERROR,
        title=title,
        detail=detail,
        category=category,
        status=status,
    )


def _empty_file_error(file_info: FileInfo) -> AgentRuntimeError:
    return _file_error(
        "Attachment is empty", f"File '{file_info.name}' has no content."
    )


@contextmanager
def _download_errors(file_info: FileInfo) -> Iterator[None]:
    try:
        yield
    except FileTooLargeError as exc:
        raise _file_error(
            "File attachment too large", f"File '{file_info.name}': {exc}"
        ) from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        phrase = (
            f" ({exc.response.reason_phrase})" if exc.response.reason_phrase else ""
        )
        raise _file_error(
            "Attachment could not be downloaded",
            f"Downloading '{file_info.name}' from storage failed with HTTP "
            f"{status}{phrase}.",
            category=(
                UiPathErrorCategory.USER
                if status in _GONE_STATUSES
                else UiPathErrorCategory.SYSTEM
            ),
            status=status,
        ) from exc
    except httpx.RequestError as exc:
        raise _file_error(
            "Attachment could not be downloaded",
            f"Downloading '{file_info.name}' from storage failed: "
            f"{type(exc).__name__}.",
            category=UiPathErrorCategory.SYSTEM,
        ) from exc


async def _tiff_content_blocks(
    file_info: FileInfo, max_size: int
) -> list[DataContentBlock]:
    with _download_errors(file_info):
        try:
            pages = await stream_tiff_to_content_blocks(
                file_info.url, max_size=max_size
            )
        except (httpx.HTTPError, FileTooLargeError):
            raise
        except Exception as exc:
            raise _file_error(
                "Attachment could not be read",
                f"File '{file_info.name}' is not a valid TIFF image.",
            ) from exc
    if not pages:
        raise _empty_file_error(file_info)
    return pages


async def build_file_content_blocks_for(
    file_info: FileInfo,
    *,
    max_size: int = MAX_FILE_SIZE_BYTES,
    pdf_only_documents: bool = False,
) -> list[DataContentBlock]:
    """Build LangChain content blocks for a single file attachment.

    The declared MIME type is first reconciled with the filename extension
    (see :func:`normalize_mime_type`), so a ``.csv`` labeled
    ``application/vnd.ms-excel`` by a Windows uploader is sent as ``text/csv``
    instead of being parsed by the provider as a corrupt binary workbook. A
    file declared ``application/octet-stream`` (or with no type) whose bytes
    are a PDF is sent as ``application/pdf``, whatever its extension; one whose
    type is still unknown after normalization (``client.ovpn``) and whose bytes
    are text is sent as ``text/plain``.
    Images then become image blocks, TIFFs are split into per-page PNG blocks,
    and every other MIME type — PDF, text, office documents, and any
    arbitrary binary — is wrapped in a generic file block. Provider
    compatibility for non-image formats is delegated to the LLM, except with
    ``pdf_only_documents``: then a text file, such as a CSV, is sent as a
    text document, since the model rejects any base64 document but a PDF.
    An image or file block built from a job attachment carries the
    attachment's id and name in ``extras``, so a guardrail can forward a
    reference to it. A text document carries none: Claude clients send it to
    the provider as is, and the provider rejects unknown fields.

    Args:
        file_info: File URL, name, and MIME type.
        max_size: Maximum allowed raw file size in bytes. LLM providers
            enforce payload limits; base64 encoding adds ~30% overhead.
        pdf_only_documents: The model accepts base64 documents only as PDF,
            as Claude does (see :func:`accepts_only_pdf_documents`).

    Returns:
        A list of DataContentBlock instances for the file.

    Raises:
        AgentRuntimeError: A ``FILE_ERROR`` if the file exceeds the size limit
            for LLM payloads, is empty, is a TIFF that cannot be decoded, or
            cannot be downloaded.
    """
    mime_type = normalize_mime_type(file_info.mime_type, file_info.name)
    if mime_type != file_info.mime_type:
        logger.info(
            "Attachment '%s' was declared as %r; sending it as %r based on its "
            "file extension",
            file_info.name,
            file_info.mime_type,
            mime_type,
        )

    if is_tiff(mime_type):
        return await _tiff_content_blocks(file_info, max_size)

    with _download_errors(file_info):
        base64_file = await download_file_base64(file_info.url, max_size=max_size)

    if not base64_file:
        raise _empty_file_error(file_info)

    tags = (
        {
            ATTACHMENT_ID_EXTRA: str(file_info.attachment_id),
            ATTACHMENT_NAME_EXTRA: file_info.name,
        }
        if file_info.attachment_id
        else {}
    )

    # The declared type, not the extension's: a converted PDF can keep its
    # source name (``Budget.docx``), which normalization would trust.
    if is_unlabeled_pdf(file_info.mime_type, base64_file):
        logger.info(
            "Attachment '%s' was declared as %r; sending it as 'application/pdf' "
            "based on its content",
            file_info.name,
            file_info.mime_type,
        )
        mime_type = "application/pdf"
    elif is_unlabeled_text(mime_type, base64_file):
        logger.info(
            "Attachment '%s' was declared as %r; sending it as 'text/plain' "
            "based on its content",
            file_info.name,
            file_info.mime_type,
        )
        mime_type = "text/plain"

    if is_image(mime_type):
        return [
            create_image_block(
                base64=base64_file, mime_type=base_mime_type(mime_type), **tags
            )
        ]

    if is_pdf(mime_type):
        # Providers match the media type exactly: parameters such as
        # ``; name="inv.pdf"`` make Anthropic reject the document.
        mime_type = "application/pdf"
    elif pdf_only_documents and is_text(mime_type):
        text = decode_text(base64_file)
        if text is not None:
            # Anthropic-native: LangChain's ``text-plain`` block is not converted
            # by every Claude client (Vertex passes it through as is).
            document = {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": text},
                "title": file_info.name or "document",
            }
            return [cast(DataContentBlock, document)]

    return [
        create_file_block(
            base64=base64_file,
            mime_type=mime_type or "application/octet-stream",
            filename=file_part_filename(file_info.name, file_info.mime_type),
            **tags,
        )
    ]


async def build_file_content_blocks(files: list[FileInfo]) -> list[DataContentBlock]:
    """Build content blocks from file attachments.

    Files are processed sequentially to avoid loading multiple large files
    into memory simultaneously.

    Args:
        files: List of file information to convert to content blocks

    Returns:
        List of DataContentBlock instances for the files
    """
    if not files:
        return []

    file_content_blocks: list[DataContentBlock] = []
    for file in files:
        blocks = await build_file_content_blocks_for(file)
        file_content_blocks.extend(blocks)
    return file_content_blocks


async def llm_call_with_files(
    messages: list[AnyMessage],
    files: list[FileInfo],
    model: BaseChatModel,
) -> AIMessage:
    """Invoke an LLM with file attachments.

    Downloads files, creates content blocks, and appends them as a HumanMessage.
    If no files are provided, equivalent to model.ainvoke().

    Args:
        messages: The conversation messages to send to the LLM.
        files: List of file attachments to include.
        model: The LLM model to invoke.

    Returns:
        The AIMessage response from the LLM.

    Raises:
        TypeError: If the LLM returns something other than AIMessage.
    """
    if not files:
        response = await model.ainvoke(messages)
        if not isinstance(response, AIMessage):
            raise TypeError(
                f"LLM returned {type(response).__name__} instead of AIMessage"
            )
        return response

    content_blocks: list[Any] = []
    for file_info in files:
        blocks = await build_file_content_blocks_for(file_info)
        content_blocks.extend(blocks)

    file_message = HumanMessage(content_blocks=content_blocks)
    all_messages = list(messages) + [file_message]

    response = await model.ainvoke(all_messages)

    del all_messages, file_message, content_blocks

    if not isinstance(response, AIMessage):
        raise TypeError(f"LLM returned {type(response).__name__} instead of AIMessage")
    return response
