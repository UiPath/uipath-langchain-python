"""GENERATE_CONVERSATIONAL_OUTPUT node for the Agent graph.

This intermediate node runs after AGENT for conversational agents whose
output schema declares custom fields beyond `uipath__agent_response_messages`.
It performs a focused LLM call with only the `set_conversational_output`
tool bound and `tool_choice="any"` to reliably extract the structured
output for the turn — decoupling conversational quality from schema
compliance.

The LLM call is tagged with `TAG_NOSTREAM` so its tokens / events never
reach the chat-UI message stream. TERMINATE then reads the tool call's
args from `state.messages[-1]`.
"""

from typing import TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import var_child_runnable_config
from pydantic import BaseModel
from uipath.agent.react.conversational_prompts import (
    get_generate_output_prompt,
)
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.chat.handlers import get_payload_handler

from ..exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from ..exceptions.licensing import raise_for_provider_http_error
from ..tools.utils import config_without_streaming
from .tools.tools import create_set_conversational_output_tool
from .types import AgentGraphState

StateT = TypeVar("StateT", bound=AgentGraphState)


def create_conversational_output_node(
    model: BaseChatModel,
    agent_output_schema: type[BaseModel],
):
    """Build the focused structured-output extraction node.

    Args:
        model: The chat model to invoke for the extraction call. Reused from
            the AGENT loop; rebinding is stateless.
        agent_output_schema: The agent's declared output schema. Used to
            construct the `set_conversational_output` tool with the
            LLM-fillable fields (`uipath__agent_response_messages` stripped).
    """
    set_conversational_output_tool = create_set_conversational_output_tool(
        agent_output_schema
    )
    # Disable streaming on this internal LLM call
    non_streaming_model = model.model_copy(update={"disable_streaming": True})
    payload_handler = get_payload_handler(non_streaming_model)
    binding_kwargs = payload_handler.get_tool_binding_kwargs(
        tools=[set_conversational_output_tool],
        tool_choice="any",
        parallel_tool_calls=False,
    )
    llm = non_streaming_model.bind_tools(
        [set_conversational_output_tool], **binding_kwargs
    )
    output_prompt = get_generate_output_prompt()

    async def conversational_output_node(state: StateT):
        # The appended HumanMessage stays local to this LLM call — only the
        # response is returned to state, so the framework instruction never
        # enters the persisted conversation history.
        messages = [*state.messages, HumanMessage(content=output_prompt)]
        config = config_without_streaming(var_child_runnable_config.get(None))

        try:
            response = await llm.ainvoke(messages, config=config)
        except Exception as e:
            raise_for_provider_http_error(e)
            raise

        if not isinstance(response, AIMessage):
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.LLM_INVALID_RESPONSE,
                title=f"Structured-output LLM returned {type(response).__name__} invalid response.",
                detail=(
                    "The language model returned an unexpected response type."
                    "If you are using a BYOM configuration, verify your model deployment.",
                ),
                category=UiPathErrorCategory.SYSTEM,
            )

        return {"messages": [response]}

    return conversational_output_node
