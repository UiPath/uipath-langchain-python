"""Structured output for the advanced main agent on models that reject forcing.

create_agent binds tool_choice="any" on every call that carries a ToolStrategy,
and some models (Opus 5.5, Sonnet 5.5) answer any forced tool choice with a 400.
For those, the call goes out with no response_format and the output tool bound as
a plain tool on "auto", and the structured response is built here the way
create_agent builds it for a ToolStrategy (langchain.agents.factory, the
ToolStrategy branch of the model node's output handling).

Only that path's default behavior is mirrored: JSON-schema dict schemas, which
parse without validation, and handle_errors=True for several output calls.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.factory import STRUCTURED_OUTPUT_ERROR_TEMPLATE
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.agents.structured_output import (
    MultipleStructuredOutputsError,
    OutputToolBinding,
    ToolStrategy,
)
from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from uipath_langchain.agent.react.no_forced_tool_choice import (
    should_skip_forced_tool_choice,
)


def _tool_messages(calls: list[ToolCall], content: str) -> list[ToolMessage]:
    return [
        ToolMessage(content=content, tool_call_id=call["id"], name=call["name"])
        for call in calls
    ]


class ToolChoiceMiddleware(AgentMiddleware[AgentState[Any], Any]):
    """Send a structured-output call unforced when the model rejects forcing.

    Sits inside ``_PayloadHandlerMiddleware``, so that middleware still sees the
    ToolStrategy and retries a turn that ends without the output tool. Needs the
    graph to be built with the ToolStrategy: create_agent's routing and its
    structured_response state come from it, and this only rewrites each call.
    """

    @staticmethod
    def _bindings(
        request: ModelRequest[Any],
    ) -> tuple[ToolStrategy[Any], dict[str, OutputToolBinding[Any]]] | None:
        strategy = request.response_format
        if not isinstance(strategy, ToolStrategy):
            return None
        if not should_skip_forced_tool_choice(request.model):
            return None
        return strategy, {
            spec.name: OutputToolBinding.from_schema_spec(spec)
            for spec in strategy.schema_specs
        }

    @staticmethod
    def _unforced(
        request: ModelRequest[Any], bindings: dict[str, OutputToolBinding[Any]]
    ) -> ModelRequest[Any]:
        output_tools = [convert_to_openai_tool(b.tool) for b in bindings.values()]
        return request.override(
            response_format=None,
            tools=[*output_tools, *request.tools],
            tool_choice="auto",
        )

    @staticmethod
    def _structured(
        response: ModelResponse[Any],
        strategy: ToolStrategy[Any],
        bindings: dict[str, OutputToolBinding[Any]],
    ) -> ModelResponse[Any]:
        output = next(
            (m for m in reversed(response.result) if isinstance(m, AIMessage)), None
        )
        calls = (
            [c for c in output.tool_calls if c["name"] in bindings] if output else []
        )
        if not output or not calls:
            return response
        if len(calls) > 1:
            error = MultipleStructuredOutputsError([c["name"] for c in calls], output)
            content = STRUCTURED_OUTPUT_ERROR_TEMPLATE.format(error=str(error))
            return ModelResponse(
                result=[*response.result, *_tool_messages(calls, content)]
            )
        structured = bindings[calls[0]["name"]].parse(calls[0]["args"])
        content = (
            strategy.tool_message_content
            or f"Returning structured response: {structured}"
        )
        return ModelResponse(
            result=[*response.result, *_tool_messages(calls, content)],
            structured_response=structured,
        )

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        found = self._bindings(request)
        if found is None:
            return handler(request)
        strategy, bindings = found
        response = handler(self._unforced(request, bindings))
        return self._structured(response, strategy, bindings)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        found = self._bindings(request)
        if found is None:
            return await handler(request)
        strategy, bindings = found
        response = await handler(self._unforced(request, bindings))
        return self._structured(response, strategy, bindings)
