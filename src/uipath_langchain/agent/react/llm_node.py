"""LLM node for ReAct Agent graph."""

from typing import Any, Literal, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.tools import BaseTool

from .constants import MAX_CONSECUTIVE_THINKING_MESSAGES
from .types import AgentGraphState
from .utils import count_consecutive_thinking_messages

OPENAI_COMPATIBLE_CHAT_MODELS = (
    "UiPathChatOpenAI",
    "AzureChatOpenAI",
    "ChatOpenAI",
    "UiPathChat",
    "UiPathAzureChatOpenAI",
)


def _get_required_tool_choice_by_model(
    model: BaseChatModel,
) -> Literal["required", "any"]:
    """Get the appropriate tool_choice value to enforce tool usage based on model type.

    "required" - OpenAI compatible required tool_choice value
    "any" - Vertex and Bedrock parameter for required tool_choice value
    """
    model_class_name = model.__class__.__name__
    if model_class_name in OPENAI_COMPATIBLE_CHAT_MODELS:
        return "required"
    return "any"


def create_llm_node(
    model: BaseChatModel,
    tools: Sequence[BaseTool] | None = None,
    thinking_messages_limit: int = MAX_CONSECUTIVE_THINKING_MESSAGES,
):
    """Create LLM node with dynamic tool_choice enforcement.

    Controls when to force tool usage based on consecutive thinking steps
    to prevent infinite loops and ensure progress.

    Args:
        model: The chat model to use
        tools: Available tools to bind
        thinking_messages_limit: Max consecutive LLM responses without tool calls
            before enforcing tool usage. 0 = force tools every time.
    """
    bindable_tools = list(tools) if tools else []
    base_llm = model.bind_tools(bindable_tools) if bindable_tools else model
    tool_choice_required_value = _get_required_tool_choice_by_model(model)

    async def llm_node(state: Any):
        # we need to use Any here because LangGraph has weird edge behavior
        # if the type annotation for the state in the edge function is Any/BaseModel/dict/etc aka not a specific model
        # then LangGraph will pass the **same** state that was passed to the previous node
        # meaning if we want the full state in the edge, we need to pass the full state here as well
        # unfortunately, using AgentGraphState in the annotation and relying on extra="allow" does not work
        # so we are doing the validation manually here
        agent_state = AgentGraphState.model_validate(state, from_attributes=True)
        messages: list[AnyMessage] = agent_state.messages

        consecutive_thinking_messages = count_consecutive_thinking_messages(messages)

        if bindable_tools and consecutive_thinking_messages >= thinking_messages_limit:
            llm = base_llm.bind(tool_choice=tool_choice_required_value)
        else:
            llm = base_llm

        response = await llm.ainvoke(messages)
        if not isinstance(response, AIMessage):
            raise TypeError(
                f"LLM returned {type(response).__name__} instead of AIMessage"
            )

        return {"messages": [response]}

    return llm_node
