"""GENERATE_CONVERSATIONAL_OUTPUT node for the Agent graph.

This intermediate node runs after AGENT for conversational agents whose
output schema declares custom fields beyond `uipath__agent_response_messages`.
It performs a focused LLM call with only the `set_conversational_output`
tool bound and `tool_choice="any"` to extract the structured output for the turn.
"""

from typing import TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import var_child_runnable_config
from pydantic import BaseModel
from uipath.agent.react import SET_CONVERSATIONAL_OUTPUT_TOOL
from uipath.agent.react.conversational_prompts import (
    get_generate_output_prompt,
)
from uipath.llm_client import UiPathAPIError, UiPathError
from uipath.llm_client.utils.exceptions import as_uipath_error
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.chat.handlers import get_payload_handler

from ..exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from ..exceptions.licensing import raise_for_provider_http_error
from ..exceptions.llm import raise_for_llm_client_error
from ..tools.utils import config_without_streaming
from .tools.tools import create_set_conversational_output_tool
from .types import AgentGraphState

StateT = TypeVar("StateT", bound=AgentGraphState)


def create_conversational_output_node(
    model: BaseChatModel,
    agent_output_schema: type[BaseModel],
):
    """Build the conversational structured-output node.

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
        messages = [*state.messages, HumanMessage(content=output_prompt)]
        config = config_without_streaming(var_child_runnable_config.get(None))

        try:
            response = await llm.ainvoke(messages, config=config)
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
                title=f"Structured-output LLM returned {type(response).__name__} invalid response.",
                detail=(
                    "The language model returned an unexpected response type. "
                    "If you are using a BYOM configuration, verify your model deployment.",
                ),
                category=UiPathErrorCategory.SYSTEM,
            )

        payload_handler.check_stop_reason(response)

        set_output_call = next(
            (
                tc
                for tc in (response.tool_calls or [])
                if tc["name"] == SET_CONVERSATIONAL_OUTPUT_TOOL.name
            ),
            None,
        )
        if set_output_call is None:
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.LLM_INVALID_RESPONSE,
                title="Structured-output LLM did not return set_conversational_output.",
                detail=(
                    "The language model was expected to call the set_conversational_output tool "
                    "to return the structured output for the turn, but no such call was made."
                ),
                category=UiPathErrorCategory.SYSTEM,
            )

        return {"inner_state": {"conversational_output": set_output_call["args"]}}

    return conversational_output_node
