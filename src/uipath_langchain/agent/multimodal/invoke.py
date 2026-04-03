"""LLM invocation with multimodal file attachments."""

import logging
from typing import Any

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
    download_file_base64,
    is_image,
    is_pdf,
    is_tiff,
    sanitize_filename,
    stream_tiff_to_content_blocks,
)

logger = logging.getLogger("uipath")


async def build_file_content_blocks_for(
    file_info: FileInfo,
    *,
    max_size: int = MAX_FILE_SIZE_BYTES,
) -> list[DataContentBlock]:
    """Build LangChain content blocks for a single file attachment.

    Handles all supported MIME types in one place: images, PDFs, and
    TIFFs (multi-page, converted to individual PNG blocks).

    Args:
        file_info: File URL, name, and MIME type.
        max_size: Maximum allowed raw file size in bytes. LLM providers
            enforce payload limits; base64 encoding adds ~30% overhead.

    Returns:
        A list of DataContentBlock instances for the file.

    Raises:
        ValueError: If the MIME type is not supported or the file exceeds
            the size limit for LLM payloads.
    """
    if is_tiff(file_info.mime_type):
        try:
            return await stream_tiff_to_content_blocks(file_info.url, max_size=max_size)
        except ValueError as exc:
            raise ValueError(f"File '{file_info.name}': {exc}") from exc

    try:
        base64_file = await download_file_base64(file_info.url, max_size=max_size)
    except ValueError as exc:
        raise ValueError(f"File '{file_info.name}': {exc}") from exc

    if is_image(file_info.mime_type):
        return [create_image_block(base64=base64_file, mime_type=file_info.mime_type)]
    if is_pdf(file_info.mime_type):
        return [
            create_file_block(
                base64=base64_file,
                mime_type=file_info.mime_type,
                filename=sanitize_filename(file_info.name),
            )
        ]

    raise ValueError(f"Unsupported mime_type={file_info.mime_type}")


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
