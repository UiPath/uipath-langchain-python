"""LLM node implementation for LangGraph."""

from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from uipath_langchain._tracing.tracer import (
    get_tracer,
    is_custom_instrumentation_enabled,
)

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

    If UIPATH_CUSTOM_INSTRUMENTATION is enabled, creates manual spans matching C# Agents.
    Otherwise, relies on OpenInference auto-instrumentation.
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

        if is_custom_instrumentation_enabled():
            return await _llm_node_instrumented(llm, messages, model)

        # Original behavior - OpenInference handles tracing
        response = await llm.ainvoke(messages)
        if not isinstance(response, AIMessage):
            raise TypeError(
                f"LLM returned {type(response).__name__} instead of AIMessage"
            )
        return {"messages": [response]}

    return llm_node


async def _llm_node_instrumented(
    llm: Runnable[Any, AIMessage], messages: list[AnyMessage], model: BaseChatModel
) -> dict[str, Any]:
    """LLM node with manual instrumentation.

    Creates nested span hierarchy matching C# Agents:
    - LLM call (outer, type: completion) - wraps the iteration
    - Model run (inner, type: llmCall) - actual API call with model name
    """
    tracer = get_tracer()

    # Extract model name from the model instance
    model_name = getattr(model, "model_name", getattr(model, "model", "unknown"))

    with tracer.start_llm_call():
        with tracer.start_model_run(model_name=model_name):
            response = await llm.ainvoke(messages)

            if not isinstance(response, AIMessage):
                raise TypeError(
                    f"LLM returned {type(response).__name__} instead of AIMessage"
                )

            return {"messages": [response]}
