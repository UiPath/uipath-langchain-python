"""LLM invocation with multimodal file attachments."""

import logging
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    DataContentBlock,
    HumanMessage,
)
from langchain_core.messages.content import create_file_block, create_image_block

from .types import MAX_FILE_SIZE_BYTES, FileInfo
from .utils import (
    base_mime_type,
    decode_text,
    download_file_base64,
    is_image,
    is_pdf,
    is_text,
    is_tiff,
    is_unlabeled_pdf,
    normalize_mime_type,
    sanitize_filename,
    stream_tiff_to_content_blocks,
)

logger = logging.getLogger("uipath")


def file_part_filename(name: str, mime_type: str) -> str | None:
    """Return the ``filename`` a file is sent to the model with.

    Mirrors :func:`build_file_content_blocks_for`: images and TIFF pages go out as
    image blocks, which carry no filename, so they return ``None``.
    """
    sent_mime_type = normalize_mime_type(mime_type, name)
    if is_tiff(sent_mime_type) or is_image(sent_mime_type):
        return None
    return sanitize_filename(name)


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
    are a PDF is sent as ``application/pdf``.
    Images then become image blocks, TIFFs are split into per-page PNG blocks,
    and every other MIME type — PDF, text, office documents, and any
    arbitrary binary — is wrapped in a generic file block. Provider
    compatibility for non-image formats is delegated to the LLM, except with
    ``pdf_only_documents``: then a text file, such as a CSV, is sent as a
    text document, since the model rejects any base64 document but a PDF.

    Args:
        file_info: File URL, name, and MIME type.
        max_size: Maximum allowed raw file size in bytes. LLM providers
            enforce payload limits; base64 encoding adds ~30% overhead.
        pdf_only_documents: The model accepts base64 documents only as PDF,
            as Claude does (see :func:`accepts_only_pdf_documents`).

    Returns:
        A list of DataContentBlock instances for the file.

    Raises:
        ValueError: If the file exceeds the size limit for LLM payloads.
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
        try:
            return await stream_tiff_to_content_blocks(file_info.url, max_size=max_size)
        except ValueError as exc:
            raise ValueError(f"File '{file_info.name}': {exc}") from exc

    try:
        base64_file = await download_file_base64(file_info.url, max_size=max_size)
    except ValueError as exc:
        raise ValueError(f"File '{file_info.name}': {exc}") from exc

    if is_unlabeled_pdf(mime_type, base64_file):
        logger.info(
            "Attachment '%s' was declared as %r; sending it as 'application/pdf' "
            "based on its content",
            file_info.name,
            mime_type,
        )
        mime_type = "application/pdf"

    if is_image(mime_type):
        return [
            create_image_block(base64=base64_file, mime_type=base_mime_type(mime_type))
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
