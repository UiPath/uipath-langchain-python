import functools
import inspect
from inspect import Parameter
from typing import Annotated, Any, Callable

from langchain_core.tools import BaseTool, InjectedToolCallId
from langchain_core.tools import tool as langchain_tool
from langgraph.types import interrupt
from uipath.core.chat import (
    UiPathConversationToolCallConfirmationValue,
)

_CANCELLED_MESSAGE = "Cancelled by user"


def _request_approval(
    tool_args: dict[str, Any],
    tool: BaseTool,
) -> dict[str, Any] | None:
    """Interrupt the graph to request user approval for a tool call.

    Returns the (possibly edited) tool arguments if approved, or None if rejected.
    """
    tool_call_id: str = tool_args.pop("tool_call_id")

    input_schema: dict[str, Any] = {}
    tool_call_schema = getattr(
        tool, "tool_call_schema", None
    )  # doesn't include InjectedToolCallId (tool id from claude/oai/etc.)
    if tool_call_schema is not None:
        input_schema = tool_call_schema.model_json_schema()

    response = interrupt(
        UiPathConversationToolCallConfirmationValue(
            tool_call_id=tool_call_id,
            tool_name=tool.name,
            input_schema=input_schema,
            input_value=tool_args,
        )
    )

    # The resume payload from CAS has shape:
    #   {"type": "uipath_cas_tool_call_confirmation",
    #    "value": {"approved": bool, "input": <edited args | None>}}
    if not isinstance(response, dict):
        return tool_args

    confirmation = response.get("value", response)
    if not confirmation.get("approved", True):
        return None

    return confirmation.get("input") or tool_args


def hitl_tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    args_schema: type | None = None,
    return_direct: bool = False,
) -> BaseTool | Callable[..., BaseTool]:

    def decorator(fn: Callable[..., Any]) -> BaseTool:
        _created_tool: list[BaseTool] = []

        # wrap the tool/function
        @functools.wraps(fn)
        def wrapper(**tool_args: Any) -> Any:
            approved_args = _request_approval(tool_args, _created_tool[0])
            if approved_args is None:
                return _CANCELLED_MESSAGE
            return fn(**approved_args)

        # rewrite the signature: e.g. (query: str) -> (query: str, *, tool_call_id: str)
        original_sig = inspect.signature(fn)
        params = list[Parameter](original_sig.parameters.values()) + [
            inspect.Parameter(
                "tool_call_id",
                inspect.Parameter.KEYWORD_ONLY,
                annotation=Annotated[str, InjectedToolCallId],
            ),
        ]
        wrapper.__signature__ = original_sig.replace(parameters=params)  # type: ignore[attr-defined]
        wrapper.__annotations__ = {
            **fn.__annotations__,
            "tool_call_id": Annotated[str, InjectedToolCallId],
        }

        # Create the LangChain tool
        if name is not None:
            result: BaseTool = langchain_tool(
                name,
                description=description,
                args_schema=args_schema,
                return_direct=return_direct,
            )(wrapper)
        else:
            result = langchain_tool(
                wrapper,
                description=description,
                args_schema=args_schema,
                return_direct=return_direct,
            )

        _created_tool.append(result)
        return result

    if func is not None:
        return decorator(func)
    return decorator
