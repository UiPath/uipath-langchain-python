"""Force a structured end_execution out of a thinking model that stalled.

Anthropic won't honor a forced tool_choice while thinking is on, so a thinking model can
answer in plain text and never call end_execution. build_extraction_call retries that
turn with thinking off and the tool call forced, which every provider honors.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from uipath.agent.react import END_EXECUTION_TOOL

_REASONING_BLOCK_TYPES = {"reasoning_content", "reasoning", "thinking"}
_END_EXECUTION_NAME = getattr(
    END_EXECUTION_TOOL.name, "value", str(END_EXECUTION_TOOL.name)
)


def _without_thinking(model: BaseChatModel) -> BaseChatModel:
    """Copy of the model with thinking config stripped, so forcing is honored.

    Thinking lives in a different place per transport: native `thinking`, Bedrock Invoke
    `model_kwargs`, Bedrock Converse `additional_model_request_fields`.
    """
    updates: dict[str, object] = {}
    request_fields = getattr(model, "additional_model_request_fields", None)
    if isinstance(request_fields, dict) and (
        "thinking" in request_fields or "output_config" in request_fields
    ):
        updates["additional_model_request_fields"] = {
            k: v
            for k, v in request_fields.items()
            if k not in ("thinking", "output_config")
        }
    model_kwargs = getattr(model, "model_kwargs", None)
    if isinstance(model_kwargs, dict) and "thinking" in model_kwargs:
        updates["model_kwargs"] = {
            k: v for k, v in model_kwargs.items() if k != "thinking"
        }
    if getattr(model, "thinking", None) is not None:
        updates["thinking"] = None
    if not updates:
        return model
    try:
        return model.model_copy(update=updates)
    except Exception:
        return model


def _strip_reasoning_blocks(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Drop reasoning blocks from AI messages (keep text + tool calls).

    They can't be replayed on a thinking-off call — an orphaned thinking block 400s.
    """
    stripped: list[AnyMessage] = []
    for message in messages:
        if isinstance(message, AIMessage) and isinstance(message.content, list):
            kept = [
                block
                for block in message.content
                if not (
                    isinstance(block, dict)
                    and block.get("type") in _REASONING_BLOCK_TYPES
                )
            ]
            if len(kept) != len(message.content):
                # a turn that was only reasoning is now empty — drop it
                if not kept and not message.tool_calls:
                    continue
                message = message.model_copy(update={"content": kept})
        stripped.append(message)
    return stripped


def _with_extraction_nudge(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Append (or merge into) a trailing user turn telling the model to call end_execution.

    Has to end on a user turn: native/Vertex rejects a forced call that ends on the
    stalled assistant turn (a prefill). Merge instead of appending so roles stay
    alternating even if the stalled turn was dropped as empty.
    """
    nudge = f"Provide the final result now by calling the {_END_EXECUTION_NAME} tool."
    if messages and isinstance(messages[-1], HumanMessage):
        last = messages[-1]
        if isinstance(last.content, str):
            merged: Any = f"{last.content}\n\n{nudge}" if last.content else nudge
        elif isinstance(last.content, list):
            merged = list(last.content) + [{"type": "text", "text": nudge}]
        else:
            merged = nudge
        return list(messages[:-1]) + [HumanMessage(content=merged)]
    return list(messages) + [HumanMessage(content=nudge)]


def build_extraction_call(
    model: BaseChatModel, messages: list[AnyMessage]
) -> tuple[BaseChatModel, list[AnyMessage]]:
    """The (model, messages) for the extraction call: thinking off, reasoning blocks
    dropped, and a nudge to call end_execution — the caller then forces tool_choice."""
    return _without_thinking(model), _with_extraction_nudge(
        _strip_reasoning_blocks(messages)
    )
