"""LLM node for ReAct Agent graph."""

from typing import Literal, Sequence, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    ToolCall,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from uipath.agent.react import RAISE_ERROR_TOOL
from uipath.llm_client import UiPathAPIError, UiPathError
from uipath.llm_client.utils.exceptions import as_uipath_error
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.chat.handlers import get_payload_handler
from uipath_langchain.chat.handlers.anthropic import anthropic_thinking_type

from ..exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from ..exceptions.licensing import raise_for_provider_http_error
from ..exceptions.llm import raise_for_llm_client_error
from ..messages.message_utils import replace_tool_calls
from ..tools.static_args import StaticArgsHandler
from .constants import (
    DEFAULT_MAX_CONSECUTIVE_THINKING_MESSAGES,
    DEFAULT_MAX_LLM_MESSAGES,
)
from .forced_extraction import build_extraction_call
from .types import FLOW_CONTROL_TOOLS, AgentGraphState
from .utils import count_consecutive_thinking_messages


def _filter_control_flow_tool_calls(
    tool_calls: list[ToolCall],
) -> list[ToolCall]:
    """Remove control flow tool calls only when regular tool calls exist alongside them.

    When only control flow tool calls are present and raise_error is among them,
    keep only the first raise_error (takes precedence over end_execution).
    """
    if len(tool_calls) <= 1:
        return tool_calls

    non_control_flow_tool_calls = [
        tc for tc in tool_calls if tc.get("name") not in FLOW_CONTROL_TOOLS
    ]
    if not non_control_flow_tool_calls:
        raise_error_calls = [
            tc for tc in tool_calls if tc.get("name") == RAISE_ERROR_TOOL.name
        ]
        return raise_error_calls[:1] if raise_error_calls else tool_calls

    return non_control_flow_tool_calls


StateT = TypeVar("StateT", bound=AgentGraphState)
InputT = TypeVar("InputT", bound=BaseModel)


def create_llm_node(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    input_schema: type[InputT] | None = None,
    is_conversational: bool = False,
    llm_messages_limit: int = DEFAULT_MAX_LLM_MESSAGES,
    thinking_messages_limit: int = DEFAULT_MAX_CONSECUTIVE_THINKING_MESSAGES,
    tool_choice: Literal["auto", "any"] = "auto",
    parallel_tool_calls: bool = True,
    strict_mode: bool = False,
):
    """Create LLM node with dynamic tool_choice enforcement.

    Controls when to force tool usage based on consecutive thinking steps
    to prevent infinite loops and ensure progress. Stall accounting is
    provider-agnostic, because forcing can be silently downgraded on any
    transport (Bedrock handlers under thinking, langchain_anthropic) or
    ignored by a BYOM deployment: after `thinking_messages_limit` tool-less
    turns tool_choice is forced, one more tool-less turn retries via the
    forced-extraction call (thinking off, which every provider honors), and
    a further stall raises THINKING_LIMIT_EXCEEDED.

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
    static_args_handler = StaticArgsHandler()

    async def llm_node(state: StateT):
        messages: list[AnyMessage] = state.messages
        initial_count = state.inner_state.initial_message_count or 0
        current_turn_messages = messages[initial_count:]
        agent_ai_messages = sum(
            1 for msg in current_turn_messages if isinstance(msg, AIMessage)
        )
        if agent_ai_messages >= llm_messages_limit:
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.TERMINATION_MAX_ITERATIONS,
                title=f"Maximum iterations of '{llm_messages_limit}' reached.",
                detail="Verify the agent's trajectory or consider increasing the max iterations in the agent's settings.",
                category=UiPathErrorCategory.USER,
            )

        static_schema_tools = static_args_handler.initialize(
            bindable_tools, state, input_schema or type(state)
        )

        current_tool_choice: Literal["auto", "any"] = tool_choice
        consecutive_thinking = count_consecutive_thinking_messages(messages)
        uses_anthropic_thinking = anthropic_thinking_type(model) is not None
        effective_limit = thinking_messages_limit if uses_anthropic_thinking else 0
        call_model: BaseChatModel = model
        call_messages: list[AnyMessage] = messages
        handler = payload_handler
        if not is_conversational and bindable_tools:
            if consecutive_thinking > effective_limit + 1:
                raise AgentRuntimeError(
                    code=AgentRuntimeErrorCode.THINKING_LIMIT_EXCEEDED,
                    title="Agent kept responding without calling a tool.",
                    detail="The model produced consecutive responses without tool calls "
                    "even after the forced extraction retry. If you are using a BYOM "
                    "configuration, verify your model deployment respects tool_choice.",
                    category=UiPathErrorCategory.SYSTEM,
                )
            if consecutive_thinking >= effective_limit:
                current_tool_choice = "any"
                if uses_anthropic_thinking and consecutive_thinking > 0:
                    call_model, call_messages = build_extraction_call(model, messages)
                    handler = get_payload_handler(call_model)

        binding_kwargs = handler.get_tool_binding_kwargs(
            tools=static_schema_tools,
            tool_choice=current_tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            strict_mode=strict_mode,
        )
        llm = call_model.bind_tools(static_schema_tools, **binding_kwargs)

        try:
            response = await llm.ainvoke(call_messages)
        except UiPathAPIError as e:
            # New LLM clients surface provider HTTP errors as a normalized UiPathAPIError directly.
            raise_for_provider_http_error(e)
        except UiPathError as e:
            raise_for_llm_client_error(e)
            raise
        except Exception as e:
            # Legacy in-repo clients (use_new_llm_clients=False) raise raw provider SDK exceptions.
            # Normalize via as_uipath_error and apply the same mapping when the error is HTTP-shaped; non-HTTP errors propagate.
            uipath_error = as_uipath_error(e)
            if isinstance(uipath_error, UiPathAPIError):
                raise_for_provider_http_error(uipath_error)
            raise
        if not isinstance(response, AIMessage):
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.LLM_INVALID_RESPONSE,
                title=f"LLM returned {type(response).__name__} invalid response.",
                detail="The language model returned an unexpected response type."
                "If you are using a BYOM configuration, verify your model deployment.",
                category=UiPathErrorCategory.SYSTEM,
            )

        payload_handler.check_stop_reason(response)

        # filter out flow control tools when multiple tool calls exist
        if response.tool_calls:
            filtered_tool_calls = _filter_control_flow_tool_calls(response.tool_calls)
            if len(filtered_tool_calls) != len(response.tool_calls):
                response = replace_tool_calls(response, filtered_tool_calls)

        static_args_handler.apply_to_response(response.tool_calls)
        return {"messages": [response]}

    return llm_node
