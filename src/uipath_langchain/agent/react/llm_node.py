"""LLM node for ReAct Agent graph."""

from abc import ABC, abstractmethod
from typing import Generic, Literal, NamedTuple, Sequence, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    ToolCall,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from uipath.agent.react import END_EXECUTION_TOOL, RAISE_ERROR_TOOL
from uipath.llm_client import UiPathAPIError, UiPathError
from uipath.llm_client.utils.exceptions import as_uipath_error
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.chat.handlers import get_payload_handler
from uipath_langchain.chat.handlers.base import ModelPayloadHandler
from uipath_langchain.chat.thinking import thinking_rejects_forced_tool_choice

from ..exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
    max_iterations_error,
)
from ..exceptions.llm import (
    raise_for_llm_client_error,
    raise_for_provider_http_error,
)
from ..messages.message_utils import replace_tool_calls
from ..tools.static_args import StaticArgsHandler
from .constants import DEFAULT_MAX_LLM_MESSAGES
from .forced_extraction import build_extraction_call
from .no_forced_tool_choice import (
    finish_without_forcing,
    finishing_tool_call,
    output_fields,
)
from .types import FLOW_CONTROL_TOOLS, AgentGraphState
from .utils import count_consecutive_tool_less_turns


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


async def _ainvoke(
    llm: Runnable[Sequence[AnyMessage], BaseMessage], messages: Sequence[AnyMessage]
) -> BaseMessage:
    try:
        return await llm.ainvoke(messages)
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


def _should_skip_forced_tool_choice(model: BaseChatModel) -> bool:
    model_details = getattr(model, "model_details", None) or {}
    return bool(model_details.get("shouldSkipForcedToolChoice", False))


def _no_finishing_tool_call_error() -> AgentRuntimeError:
    return AgentRuntimeError(
        code=AgentRuntimeErrorCode.THINKING_LIMIT_EXCEEDED,
        title="Agent responded without calling a tool.",
        detail="The model answered without calling a tool, and when asked to finish "
        "it didn't call "
        + END_EXECUTION_TOOL.name
        + " or "
        + RAISE_ERROR_TOOL.name
        + ".",
        category=UiPathErrorCategory.SYSTEM,
    )


def _tool_less_turns_error() -> AgentRuntimeError:
    return AgentRuntimeError(
        code=AgentRuntimeErrorCode.THINKING_LIMIT_EXCEEDED,
        title="Agent kept responding without calling a tool.",
        detail="The model produced consecutive responses without tool calls "
        "even after the forced extraction retry. If you are using a BYOM "
        "configuration, verify your model deployment respects tool_choice.",
        category=UiPathErrorCategory.SYSTEM,
    )


StateT = TypeVar("StateT", bound=AgentGraphState)
InputT = TypeVar("InputT", bound=BaseModel)


class LLMCall(NamedTuple):
    """How a turn calls the model."""

    model: BaseChatModel
    messages: list[AnyMessage]
    handler: ModelPayloadHandler
    tool_choice: Literal["auto", "any"]


class LLMNode(ABC, Generic[StateT]):
    """One LLM turn. Subclasses decide the tool_choice."""

    def __init__(
        self,
        model: BaseChatModel,
        tools: Sequence[BaseTool],
        input_schema: type[BaseModel] | None = None,
        llm_messages_limit: int = DEFAULT_MAX_LLM_MESSAGES,
        parallel_tool_calls: bool = True,
        strict_mode: bool = False,
    ):
        self.model = model
        self.tools = list(tools) if tools else []
        self.input_schema = input_schema
        self.llm_messages_limit = llm_messages_limit
        self.parallel_tool_calls = parallel_tool_calls
        self.strict_mode = strict_mode
        self.payload_handler = get_payload_handler(model)
        self.static_args_handler = StaticArgsHandler()

    async def __call__(self, state: StateT) -> dict[str, list[AIMessage]]:
        messages: list[AnyMessage] = state.messages
        initial_count = state.inner_state.initial_message_count or 0
        agent_ai_messages = sum(
            1 for msg in messages[initial_count:] if isinstance(msg, AIMessage)
        )
        if agent_ai_messages >= self.llm_messages_limit:
            raise max_iterations_error(self.llm_messages_limit)

        static_schema_tools = self.static_args_handler.initialize(
            self.tools, state, self.input_schema or type(state)
        )

        call = self.prepare_call(messages)
        binding_kwargs = call.handler.get_tool_binding_kwargs(
            tools=static_schema_tools,
            tool_choice=call.tool_choice,
            parallel_tool_calls=self.parallel_tool_calls,
            strict_mode=self.strict_mode,
        )
        llm = call.model.bind_tools(static_schema_tools, **binding_kwargs)

        response = await _ainvoke(llm, call.messages)
        if not isinstance(response, AIMessage):
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.LLM_INVALID_RESPONSE,
                title=f"LLM returned {type(response).__name__} invalid response.",
                detail="The language model returned an unexpected response type."
                "If you are using a BYOM configuration, verify your model deployment.",
                category=UiPathErrorCategory.SYSTEM,
            )

        self.payload_handler.check_stop_reason(response)

        response = await self.finish(response, messages, static_schema_tools)

        # filter out flow control tools when multiple tool calls exist
        if response.tool_calls:
            filtered_tool_calls = _filter_control_flow_tool_calls(response.tool_calls)
            if len(filtered_tool_calls) != len(response.tool_calls):
                response = replace_tool_calls(response, filtered_tool_calls)

        self.static_args_handler.apply_to_response(response.tool_calls)
        return {"messages": [response]}

    @abstractmethod
    def prepare_call(self, messages: list[AnyMessage]) -> LLMCall:
        """Model, messages, handler and tool_choice for this turn."""

    async def finish(
        self,
        response: AIMessage,
        messages: list[AnyMessage],
        tools: Sequence[BaseTool],
    ) -> AIMessage:
        """Post-process the response. Unchanged by default."""
        return response


class ConversationalLLMNode(LLMNode[StateT]):
    """Never forces a tool call. For conversational agents and agents without tools."""

    def __init__(
        self,
        model: BaseChatModel,
        tools: Sequence[BaseTool],
        input_schema: type[BaseModel] | None = None,
        llm_messages_limit: int = DEFAULT_MAX_LLM_MESSAGES,
        parallel_tool_calls: bool = True,
        strict_mode: bool = False,
        tool_choice: Literal["auto", "any"] = "auto",
    ):
        super().__init__(
            model,
            tools,
            input_schema=input_schema,
            llm_messages_limit=llm_messages_limit,
            parallel_tool_calls=parallel_tool_calls,
            strict_mode=strict_mode,
        )
        self.tool_choice: Literal["auto", "any"] = tool_choice

    def prepare_call(self, messages: list[AnyMessage]) -> LLMCall:
        # conversational need to be able to answer in text so we dont ForceToolChoice
        return LLMCall(
            self.model, messages, self.payload_handler, tool_choice=self.tool_choice
        )


class ForcedToolChoiceLLMNode(LLMNode[StateT]):
    """Forces tool_choice any every turn.

    Thinking models retry one stall with thinking off; a second stall fails the run.
    """

    def prepare_call(self, messages: list[AnyMessage]) -> LLMCall:
        consecutive_tool_less = count_consecutive_tool_less_turns(messages)
        if consecutive_tool_less > 1:
            raise _tool_less_turns_error()
        if thinking_rejects_forced_tool_choice(self.model) and consecutive_tool_less:
            call_model, call_messages = build_extraction_call(self.model, messages)
            return LLMCall(
                call_model,
                call_messages,
                get_payload_handler(call_model),
                tool_choice="any",
            )
        return LLMCall(self.model, messages, self.payload_handler, tool_choice="any")


class AutoToolChoiceLLMNode(LLMNode[StateT]):
    """For models whose discovery details say shouldSkipForcedToolChoice: true.

    Runs on auto. A text answer gets one finish call; any other stall fails the run.
    The finish call doesn't count against llm_messages_limit: it belongs to the same
    turn and ends the run, so it can't loop.
    """

    def __init__(
        self,
        model: BaseChatModel,
        tools: Sequence[BaseTool],
        input_schema: type[BaseModel] | None = None,
        llm_messages_limit: int = DEFAULT_MAX_LLM_MESSAGES,
        parallel_tool_calls: bool = True,
        strict_mode: bool = False,
    ):
        super().__init__(
            model,
            tools,
            input_schema=input_schema,
            llm_messages_limit=llm_messages_limit,
            parallel_tool_calls=parallel_tool_calls,
            strict_mode=strict_mode,
        )
        self.output_fields = output_fields(self.tools)

    def prepare_call(self, messages: list[AnyMessage]) -> LLMCall:
        if count_consecutive_tool_less_turns(messages) > 0:
            raise _no_finishing_tool_call_error()
        return LLMCall(self.model, messages, self.payload_handler, tool_choice="auto")

    async def finish(
        self,
        response: AIMessage,
        messages: list[AnyMessage],
        tools: Sequence[BaseTool],
    ) -> AIMessage:
        if self.output_fields is None or response.tool_calls or not response.text:
            return response
        finish_llm, finish_input = finish_without_forcing(
            self.model,
            self.payload_handler,
            tools,
            messages,
            response,
            parallel_tool_calls=self.parallel_tool_calls,
            strict_mode=self.strict_mode,
        )
        reply = await _ainvoke(finish_llm, finish_input)
        if not isinstance(reply, AIMessage):
            return response
        self.payload_handler.check_stop_reason(reply)
        return finishing_tool_call(reply, self.output_fields) or response


def create_llm_node(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    input_schema: type[InputT] | None = None,
    is_conversational: bool = False,
    llm_messages_limit: int = DEFAULT_MAX_LLM_MESSAGES,
    tool_choice: Literal["auto", "any"] = "auto",
    parallel_tool_calls: bool = True,
    strict_mode: bool = False,
) -> LLMNode[AgentGraphState]:
    """Pick the LLM node for the agent and model.

    Conversational or no tools: ConversationalLLMNode. Discovery says
    shouldSkipForcedToolChoice: true: AutoToolChoiceLLMNode. Otherwise:
    ForcedToolChoiceLLMNode.

    Args:
        model: The chat model to use
        tools: Available tools to bind
        input_schema: Agent input schema
        is_conversational: Whether this is a conversational agent
        llm_messages_limit: Maximum number of LLM turns allowed per execution; the
            AutoToolChoiceLLMNode's finish call is part of its turn
        tool_choice: Tool choice for the ConversationalLLMNode
        parallel_tool_calls: Allow parallel tool calls
        strict_mode: Validate tool call arguments
    """
    if is_conversational or not tools:
        return ConversationalLLMNode(
            model,
            tools,
            input_schema=input_schema,
            llm_messages_limit=llm_messages_limit,
            parallel_tool_calls=parallel_tool_calls,
            strict_mode=strict_mode,
            tool_choice=tool_choice,
        )
    if _should_skip_forced_tool_choice(model):
        return AutoToolChoiceLLMNode(
            model,
            tools,
            input_schema=input_schema,
            llm_messages_limit=llm_messages_limit,
            parallel_tool_calls=parallel_tool_calls,
            strict_mode=strict_mode,
        )
    return ForcedToolChoiceLLMNode(
        model,
        tools,
        input_schema=input_schema,
        llm_messages_limit=llm_messages_limit,
        parallel_tool_calls=parallel_tool_calls,
        strict_mode=strict_mode,
    )
