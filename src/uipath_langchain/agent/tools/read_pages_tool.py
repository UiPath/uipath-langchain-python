"""Page-reading companion tool for a semantic index context."""

from typing import Any

import httpx
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import BaseTool, StructuredTool, ToolException
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
from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.tool_node import (
    ToolWrapperMixin,
    ToolWrapperReturnType,
)

from .structured_tool_with_output_type import StructuredToolWithOutputType
from .utils import sanitize_tool_name

READ_PAGES_TOOL_SUFFIX = "_read_pages"

# Pages the agent may read for one user message. Without a cap an LLM can keep
# paging through a document (has_next stays true) until it runs out of iterations.
MAX_PAGES_PER_TURN = 20


class StructuredToolWithWrapper(StructuredToolWithOutputType, ToolWrapperMixin):
    pass


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


def _page_count(args: dict[str, Any]) -> int:
    try:
        return max(0, int(args["end_page"]) - int(args["start_page"]) + 1)
    except (KeyError, TypeError, ValueError):
        return 0


def pages_read_this_turn(messages: list[AnyMessage], tool_name: str) -> int:
    """Pages requested by this tool's successful calls since the last user message."""
    statuses: dict[str, str] = {}
    calls: list[ToolCall] = []
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, ToolMessage):
            statuses[message.tool_call_id] = message.status
        elif isinstance(message, AIMessage):
            calls.extend(c for c in message.tool_calls if c["name"] == tool_name)
    return sum(
        _page_count(c["args"])
        for c in calls
        if statuses.get(c["id"] or "") == "success"
    )


def _read_pages_error_message(
    e: EnrichedException, index_name: str, source_document_id: str
) -> str:
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
            if e.status_code == 404 and body.get("message") == "File not found.":
                return (
                    f"No document with id '{source_document_id}' is in this index. Use "
                    "the source_document_id of a search result from this context; if "
                    "no result is from the document you need, search for it first."
                )
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
    tool_name = read_pages_tool_name(resource)
    index_name = resource.index_name or ""
    description = (
        f"Read pages of one document found by the '{resource.name}' search. Pass a "
        "search result's source_document_id and a page range, numbered like the "
        "results' page_number (at most 20 pages per call and "
        f"{MAX_PAGES_PER_TURN} per question). Use it when the user names pages, or "
        "when a search result is cut off or continues on another page. In the "
        "result, has_previous / has_next tell whether the document has a page just "
        "before / after the range, and missing_pages are past its end."
    )

    _pages_already_read = 0

    @mockable(
        name=tool_name,
        description=description,
        input_schema=ReadPagesInput.model_json_schema(),
        output_schema=ContextGroundingPageRange.model_json_schema(),
        example_calls=[],
    )
    async def read_pages_fn(
        source_document_id: str, start_page: int, end_page: int
    ) -> dict[str, Any]:
        remaining = MAX_PAGES_PER_TURN - _pages_already_read
        if end_page - start_page + 1 > remaining:
            raise ToolException(
                f"Only {max(remaining, 0)} more pages can be read for this question "
                f"(at most {MAX_PAGES_PER_TURN}). Answer with the pages already read, "
                "or read a smaller range."
            )
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
                raise ToolException(
                    _read_pages_error_message(e, index_name, source_document_id)
                ) from e
            raise
        return result.model_dump()

    async def read_pages_budget_wrapper(
        tool: BaseTool,
        call: ToolCall,
        state: AgentGraphState,
    ) -> ToolWrapperReturnType:
        nonlocal _pages_already_read
        _pages_already_read = pages_read_this_turn(state.messages, tool.name)
        return await tool.ainvoke(call)

    tool = StructuredToolWithWrapper(
        name=tool_name,
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
    tool.set_tool_wrappers(awrapper=read_pages_budget_wrapper)
    return tool
