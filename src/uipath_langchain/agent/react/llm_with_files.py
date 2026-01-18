"""LLM invocation with file attachments support."""

from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from uipath_langchain.llm import get_content_builder


@dataclass
class FileInfo:
    """File information for LLM file attachments."""

    url: str
    name: str
    mime_type: str


async def build_file_content_part(
    file_info: FileInfo,
    model: BaseChatModel,
) -> dict[str, Any]:
    """Build a provider-specific message content part for a file attachment.

    Args:
        file_info: File URL, name, and MIME type.
        model: The LLM model instance (must have llm_provider and api_flavor attributes).

    Returns:
        Provider-specific content part dictionary.
    """
    builder = get_content_builder(model)
    return await builder.build_file_content_part(
        url=file_info.url,
        filename=file_info.name,
        mime_type=file_info.mime_type,
    )


async def llm_call_with_files(
    messages: list[AnyMessage],
    files: list[FileInfo],
    model: BaseChatModel,
) -> AIMessage:
    """Invoke an LLM with file attachments.

    Downloads files, creates provider-specific content parts, and appends them
    as a HumanMessage. If no files are provided, equivalent to model.ainvoke().
    """
    if not files:
        response = await model.ainvoke(messages)
        if not isinstance(response, AIMessage):
            raise TypeError(
                f"LLM returned {type(response).__name__} instead of AIMessage"
            )
        return response

    content_parts: list[str | dict[Any, Any]] = []
    for file_info in files:
        content_part = await build_file_content_part(file_info, model)
        content_parts.append(content_part)

    file_message = HumanMessage(content=content_parts)
    all_messages = list(messages) + [file_message]

    response = await model.ainvoke(all_messages)
    if not isinstance(response, AIMessage):
        raise TypeError(f"LLM returned {type(response).__name__} instead of AIMessage")
    return response
