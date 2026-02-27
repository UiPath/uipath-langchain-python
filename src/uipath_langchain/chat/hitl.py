import functools
import inspect
from inspect import Parameter
from typing import Annotated, Any, Callable, NamedTuple

from langchain_core.messages.tool import ToolCall, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId
from langchain_core.tools import tool as langchain_tool
from langgraph.types import interrupt
from uipath.core.chat import (
    UiPathConversationToolCallConfirmationValue,
)

CANCELLED_MESSAGE = "Cancelled by user"
ARGS_MODIFIED_MESSAGE = "Tool arguments were modified by the user"
CONVERSATIONAL_APPROVED_TOOL_ARGS = "conversational_approved_tool_args"
REQUIRE_CONVERSATIONAL_CONFIRMATION = "require_conversational_confirmation"


class ConfirmationResult(NamedTuple):
    """Result of a tool confirmation check."""

    cancelled: ToolMessage | None  # ToolMessage if cancelled, None if approved
    args_modified: bool
    approved_args: dict[str, Any] | None = None

    def annotate_result(self, output: dict[str, Any] | Any) -> None:
        """Apply confirmation metadata to a tool result message."""
        msg = None
        if isinstance(output, dict):
            messages = output.get("messages")
            if messages:
                msg = messages[0]
        if msg is None:
            return
        if self.approved_args is not None:
            msg.response_metadata[CONVERSATIONAL_APPROVED_TOOL_ARGS] = (
                self.approved_args
            )
        if self.args_modified:
            msg.content = (
                f'{{"meta": "{ARGS_MODIFIED_MESSAGE}", "result": {msg.content}}}'
            )


def _patch_span_input(approved_args: dict[str, Any]) -> None:
    """Update the current tracer Run so the span records the approved tool args.

    LangChain tracers (including OpenInference) read ``run.inputs`` when the
    span ends, overwriting any earlier ``set_attribute`` call.  We patch the
    Run object directly via the public ``BaseTracer.run_map`` API so that
    every tracer picks up the correct value.
    """
    try:
        from langchain_core.callbacks import BaseCallbackManager
        from langchain_core.runnables.config import var_child_runnable_config
        from langchain_core.tracers.base import BaseTracer

        config = var_child_runnable_config.get()
        if not isinstance(config, dict):
            return

        run_id = None
        managers: list[BaseCallbackManager] = []
        for v in config.values():
            if isinstance(v, BaseCallbackManager):
                managers.append(v)
                if not run_id:
                    run_id = v.parent_run_id

        if not run_id:
            return

        serialized = str(approved_args)
        key = str(run_id)
        for mgr in managers:
            for handler in mgr.handlers:
                if isinstance(handler, BaseTracer):
                    run = handler.run_map.get(key)
                    if run is not None:
                        run.inputs = {"input": serialized}
    except Exception:
        pass


def request_approval(
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

    return (
        confirmation.get("input")
        if confirmation.get("input") is not None
        else tool_args
    )


def check_tool_confirmation(
    call: ToolCall, tool: BaseTool
) -> ConfirmationResult | None:
    if not (tool.metadata and tool.metadata.get(REQUIRE_CONVERSATIONAL_CONFIRMATION)):
        return None

    original_args = call["args"]
    approved_args = request_approval(
        {**original_args, "tool_call_id": call["id"]}, tool
    )
    if approved_args is None:
        cancelled_msg = ToolMessage(
            content=CANCELLED_MESSAGE,
            name=call["name"],
            tool_call_id=call["id"],
        )
        cancelled_msg.response_metadata[CONVERSATIONAL_APPROVED_TOOL_ARGS] = (
            original_args
        )
        return ConfirmationResult(cancelled=cancelled_msg, args_modified=False)
    call["args"] = approved_args
    return ConfirmationResult(
        cancelled=None,
        args_modified=approved_args != original_args,
        approved_args=approved_args,
    )


def requires_approval(
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
            approved_args = request_approval(tool_args, _created_tool[0])
            if approved_args is None:
                return {"meta": CANCELLED_MESSAGE}
            _patch_span_input(approved_args)
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
