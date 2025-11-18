"""LLM node implementation for LangGraph."""

from typing import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.tools import BaseTool

from .constants import MAX_SUCCESSIVE_COMPLETIONS
from .types import AgentGraphState
from .utils import count_successive_completions


def create_llm_node(
    model: BaseChatModel,
    tools: Sequence[BaseTool] | None = None,
):
    """Invoke LLM with tools and dynamically control tool_choice based on successive completions.

    When successive completions reach the limit, tool_choice is set to "required" to force
    the LLM to use a tool and prevent infinite reasoning loops.
    """
    bindable_tools = list(tools) if tools else []
    base_llm = model.bind_tools(bindable_tools) if bindable_tools else model

    async def llm_node(state: AgentGraphState):
        messages: list[AnyMessage] = state.messages

        successive_completions = count_successive_completions(messages)
        if successive_completions >= MAX_SUCCESSIVE_COMPLETIONS:
            llm = base_llm.bind(tool_choice="required")
        else:
            llm = base_llm

        response = await llm.ainvoke(messages)
        if not isinstance(response, AIMessage):
            raise TypeError(
                f"LLM returned {type(response).__name__} instead of AIMessage"
            )

        return {"messages": [response]}

    return llm_node
