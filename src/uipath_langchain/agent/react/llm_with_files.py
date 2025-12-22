"""LLM invocation with file attachments support."""

from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from .file_type_handler import build_message_content_part_from_data


@dataclass
class FileInfo:
    """File information for LLM file attachments."""

    url: str
    name: str
    mime_type: str


def _get_model_name(model: BaseChatModel) -> str:
    """Extract model name from a BaseChatModel instance."""
    for attr in ["model_name", "_model_name", "model", "model_id"]:
        value = getattr(model, attr, None)
        if value and isinstance(value, str):
            return value
    raise ValueError(f"Model name not found in model {model}")


async def create_part_for_file(
    file_info: FileInfo,
    model: BaseChatModel,
) -> dict[str, Any]:
    """Create a provider-specific message content part for a file attachment.

    Downloads the file from file_info.url and formats it for the model's provider.
    """
    model_name = _get_model_name(model)
    return await build_message_content_part_from_data(
        url=file_info.url,
        filename=file_info.name,
        mime_type=file_info.mime_type,
        model=model_name,
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
        content_part = await create_part_for_file(file_info, model)
        content_parts.append(content_part)

    file_message = HumanMessage(content=content_parts)
    all_messages = list(messages) + [file_message]

    response = await model.ainvoke(all_messages)
    if not isinstance(response, AIMessage):
        raise TypeError(f"LLM returned {type(response).__name__} instead of AIMessage")
    return response
