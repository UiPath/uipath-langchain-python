"""The read_pages page budget for agents whose loop LangChain runs.

The standard graph checks a page read against the budget in the tool's wrapper,
which only ``UiPathToolNode`` honors. An advanced agent's tools run in LangChain's
``ToolNode``, so this middleware makes the same check around each call, from the
call's own graph state.
"""

from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from uipath_langchain.agent.tools.read_pages_tool import (
    PAGE_BUDGET_METADATA_KEY,
    page_budget_refusal,
)


class PageBudgetMiddleware(AgentMiddleware[AgentState[Any], Any]):
    """Refuse a read_pages call that would pass the pages allowed per question."""

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        tool = request.tool
        metadata = tool.metadata if tool is not None else None
        if not isinstance(metadata, dict) or PAGE_BUDGET_METADATA_KEY not in metadata:
            return await handler(request)
        refusal = page_budget_refusal(
            request.state.get("messages") or [],
            request.tool_call,
            request.tool_call["name"],
        )
        if refusal is not None:
            return refusal
        return await handler(request)
