"""Page-reading companion tool for a semantic index context."""

from typing import Any

import httpx
from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, Field
from uipath.agent.models.agent import AgentContextResourceConfig
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.platform.common import UiPathConfig
from uipath.platform.context_grounding import ContextGroundingPageRange
from uipath.platform.errors import (
    ContextGroundingIndexNotFoundError,
    EnrichedException,
)
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain._utils import get_execution_folder_path
from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)

from .structured_tool_with_output_type import StructuredToolWithOutputType
from .utils import sanitize_tool_name

READ_PAGES_TOOL_SUFFIX = "_read_pages"


class ReadPagesInput(BaseModel):
    source_document_id: str = Field(
        ...,
        description="The source_document_id of a search result from this context.",
    )
    start_page: int = Field(
        ...,
        description="The first page to read, numbered like page_number in search results.",
    )
    end_page: int = Field(
        ...,
        description="The last page to read. At most 20 pages per call.",
    )


def read_pages_tool_name(resource: AgentContextResourceConfig) -> str:
    """Name of the page tool, kept within the 64-character limit with its suffix."""
    base = sanitize_tool_name(resource.name)
    return base[: 64 - len(READ_PAGES_TOOL_SUFFIX)] + READ_PAGES_TOOL_SUFFIX


def _read_pages_error_message(e: EnrichedException, index_name: str) -> str:
    """The reason a page read was rejected, worded for the LLM.

    ECS puts the actionable part of a validation error (such as a file name passed
    as the document id) under ``errors``, which ``error_info`` does not extract.
    """
    cause = e.__cause__
    if isinstance(cause, httpx.HTTPStatusError):
        try:
            body = cause.response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            errors = body.get("errors")
            if isinstance(errors, dict):
                details = [
                    str(message)
                    for messages in errors.values()
                    for message in (
                        messages if isinstance(messages, list) else [messages]
                    )
                ]
                if details:
                    return " ".join(details)
            if body.get("message"):
                return str(body["message"])
    info = e.error_info
    if info and info.message:
        return info.message
    if e.status_code == 404:
        return f"Reading pages is not available for index '{index_name}'."
    return f"The page read failed with status {e.status_code}."


def create_read_pages_tool(resource: AgentContextResourceConfig) -> StructuredTool:
    """Create a tool that reads a range of pages of a document in the context's index.

    The tool takes a ``source_document_id`` from the context's search results, so it
    is created next to the search tool when the context sets ``readPages``.
    """
    index_name = resource.index_name or ""
    description = (
        f"Read pages of one document found by the '{resource.name}' search. Pass a "
        "search result's source_document_id and a page range, numbered like the "
        "results' page_number (at most 20 pages per call). Use it when the user "
        "names pages, or when a search result is cut off or continues on another "
        "page. In the result, has_previous / has_next tell whether the document has "
        "a page just before / after the range, and missing_pages are past its end."
    )

    @mockable(
        name=read_pages_tool_name(resource),
        description=description,
        input_schema=ReadPagesInput.model_json_schema(),
        output_schema=ContextGroundingPageRange.model_json_schema(),
        example_calls=[],
    )
    async def read_pages_fn(
        source_document_id: str, start_page: int, end_page: int
    ) -> dict[str, Any]:
        try:
            result = await UiPath().context_grounding.read_pages_async(
                name=index_name,
                source_document_id=source_document_id,
                start_page=start_page,
                end_page=end_page,
                folder_path=get_execution_folder_path(),
                include_system_indexes=UiPathConfig.is_studio_project,
            )
        except ContextGroundingIndexNotFoundError as e:
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.CONTEXT_GROUNDING_INDEX_NOT_FOUND,
                title=f"Context grounding index '{index_name}' was not found",
                detail=e.message,
                category=UiPathErrorCategory.DEPLOYMENT,
            ) from e
        except EnrichedException as e:
            # A wrong id or range is the LLM's to fix: return the reason as the
            # tool result instead of failing the run.
            if e.status_code in (400, 404):
                raise ToolException(_read_pages_error_message(e, index_name)) from e
            raise
        return result.model_dump()

    return StructuredToolWithOutputType(
        name=read_pages_tool_name(resource),
        description=description,
        args_schema=ReadPagesInput,
        coroutine=read_pages_fn,
        output_type=ContextGroundingPageRange,
        handle_tool_error=True,
        metadata={
            "tool_type": "context",
            "display_name": f"{resource.name} (read pages)",
            "index_name": index_name,
        },
    )
