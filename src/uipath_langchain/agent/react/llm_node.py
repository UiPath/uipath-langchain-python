"""LLM node for ReAct Agent graph."""

from typing import Sequence, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, ToolCall
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from uipath.runtime.errors import UiPathErrorCategory, UiPathErrorCode

from uipath_langchain.agent.tools.static_args import (
    apply_static_argument_properties_to_schema,
)
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)
from uipath_langchain.llm import get_payload_handler

from ..exceptions import AgentTerminationException
from .constants import (
    DEFAULT_MAX_CONSECUTIVE_THINKING_MESSAGES,
    DEFAULT_MAX_LLM_MESSAGES,
)
from .types import FLOW_CONTROL_TOOLS, AgentGraphState
from .utils import count_consecutive_thinking_messages, extract_input_data_from_state


def _filter_control_flow_tool_calls(
    tool_calls: list[ToolCall],
) -> list[ToolCall]:
    """Remove control flow tools when multiple tool calls exist."""
    if len(tool_calls) <= 1:
        return tool_calls

    return [tc for tc in tool_calls if tc.get("name") not in FLOW_CONTROL_TOOLS]


StateT = TypeVar("StateT", bound=AgentGraphState)
InputT = TypeVar("InputT", bound=BaseModel)


def create_llm_node(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    input_schema: type[InputT] | None = None,
    is_conversational: bool = False,
    llm_messages_limit: int = DEFAULT_MAX_LLM_MESSAGES,
    thinking_messages_limit: int = DEFAULT_MAX_CONSECUTIVE_THINKING_MESSAGES,
):
    """Create LLM node with dynamic tool_choice enforcement.

    Controls when to force tool usage based on consecutive thinking steps
    to prevent infinite loops and ensure progress.

    Args:
        model: The chat model to use
        tools: Available tools to bind
        is_conversational: Whether this is a conversational agent
        llm_messages_limit: Maximum number of LLM calls allowed per execution
        thinking_messages_limit: Max consecutive LLM responses without tool calls
            before enforcing tool usage. 0 = force tools every time.
    """
    bindable_tools = list(tools) if tools else []
    payload_handler = get_payload_handler(model)
    tool_choice_required_value = payload_handler.get_required_tool_choice()

    async def llm_node(state: StateT):
        messages: list[AnyMessage] = state.messages
        agent_ai_messages = sum(1 for msg in messages if isinstance(msg, AIMessage))
        if agent_ai_messages >= llm_messages_limit:
            raise AgentTerminationException(
                code=UiPathErrorCode.EXECUTION_ERROR,
                title=f"Maximum iterations of '{llm_messages_limit}' reached.",
                detail="Verify the agent's trajectory or consider increasing the max iterations in the agent's settings.",
                category=UiPathErrorCategory.USER,
            )

        consecutive_thinking_messages = count_consecutive_thinking_messages(messages)

        static_schema_tools = _apply_tool_argument_properties(
            bindable_tools, state, input_schema
        )
        base_llm = model.bind_tools(static_schema_tools)

        if (
            not is_conversational
            and bindable_tools
            and consecutive_thinking_messages >= thinking_messages_limit
        ):
            llm = base_llm.bind(tool_choice=tool_choice_required_value)
        else:
            llm = base_llm

        response = await llm.ainvoke(messages)
        if not isinstance(response, AIMessage):
            raise TypeError(
                f"LLM returned {type(response).__name__} instead of AIMessage"
            )

        # filter out flow control tools when multiple tool calls exist
        if response.tool_calls:
            filtered_tool_calls = _filter_control_flow_tool_calls(response.tool_calls)
            if len(filtered_tool_calls) != len(response.tool_calls):
                # todo: this does not actually work, but fixing tool call modifying is a separate task
                response.tool_calls = filtered_tool_calls

        return {"messages": [response]}

    return llm_node


def _apply_tool_argument_properties(
    tools: list[BaseTool],
    state: StateT,
    input_schema: type[InputT] | None = None,
) -> list[BaseTool]:
    """Apply dynamic schema modifications to tools based on their argument_properties."""

    agent_input = extract_input_data_from_state(state, input_schema or type(state))
    return [
        apply_static_argument_properties_to_schema(tool, agent_input)
        if isinstance(tool, StructuredToolWithArgumentProperties)
        else tool
        for tool in tools
    ]
