"""LLM invocation with multimodal file attachments."""

import asyncio
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    DataContentBlock,
    HumanMessage,
)
from langchain_core.messages.content import (
    create_file_block,
    create_image_block,
    create_plaintext_block,
)

from .types import FileInfo
from .utils import (
    download_file_base64,
    download_file_text,
    is_image,
    is_pdf,
    is_text,
    sanitize_filename,
)

logger = logging.getLogger("uipath")


async def build_file_content_block(
    file_info: FileInfo,
) -> DataContentBlock:
    """Build a LangChain content block for a file attachment.

    Args:
        file_info: File URL, name, and MIME type.

    Returns:
        A DataContentBlock for the file (image, PDF, or text).
        For unsupported MIME types, returns a plaintext block with an
        error description instead of raising.
    """
    if is_text(file_info.mime_type):
        text_content = await download_file_text(file_info.url)
        return create_plaintext_block(
            text=text_content,
            title=file_info.name,
        )

    base64_file = await download_file_base64(file_info.url)

    if is_image(file_info.mime_type):
        return create_image_block(base64=base64_file, mime_type=file_info.mime_type)
    if is_pdf(file_info.mime_type):
        return create_file_block(
            base64=base64_file,
            mime_type=file_info.mime_type,
            filename=sanitize_filename(file_info.name),
        )

    logger.warning(
        "Unsupported MIME type '%s' for file '%s'; "
        "returning descriptive placeholder to LLM.",
        file_info.mime_type,
        file_info.name,
    )
    return create_plaintext_block(
        text=(
            f"[File '{file_info.name}' could not be processed: "
            f"unsupported file type '{file_info.mime_type}'. "
            f"Summarize your response based on available information.]"
        ),
        title=file_info.name,
    )


async def build_file_content_blocks(files: list[FileInfo]) -> list[DataContentBlock]:
    """Build content blocks from file attachments.

    Args:
        files: List of file information to convert to content blocks

    Returns:
        List of DataContentBlock instances for the files
    """
    if not files:
        return []

    file_content_blocks: list[DataContentBlock] = await asyncio.gather(
        *[build_file_content_block(file) for file in files]
    )
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
        content_block = await build_file_content_block(file_info)
        content_blocks.append(content_block)

    file_message = HumanMessage(content_blocks=content_blocks)
    all_messages = list(messages) + [file_message]

    response = await model.ainvoke(all_messages)
    if not isinstance(response, AIMessage):
        raise TypeError(f"LLM returned {type(response).__name__} instead of AIMessage")
    return response
