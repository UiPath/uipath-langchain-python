"""Force a structured end_execution out of a thinking model that stalled.

Anthropic won't honor a forced tool_choice while thinking is on, so a thinking model can
answer in plain text and never call end_execution. build_extraction_call retries that
turn with thinking off and the tool call forced, which every provider honors.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from uipath.agent.react import END_EXECUTION_TOOL

from uipath_langchain.chat.thinking import is_reasoning_block, strip_thinking

_END_EXECUTION_NAME = getattr(
    END_EXECUTION_TOOL.name, "value", str(END_EXECUTION_TOOL.name)
)


def _strip_reasoning_blocks(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Drop reasoning blocks from AI messages (keep text + tool calls).

    They can't be replayed on a thinking-off call — an orphaned thinking block 400s.
    """
    stripped: list[AnyMessage] = []
    for message in messages:
        if isinstance(message, AIMessage) and isinstance(message.content, list):
            kept = [
                block for block in message.content if not is_reasoning_block(block)
            ]
            if len(kept) != len(message.content):
                # a turn that was only reasoning is now empty — drop it
                if not kept and not message.tool_calls:
                    continue
                message = message.model_copy(update={"content": kept})
        stripped.append(message)
    return stripped


def _with_extraction_nudge(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Append (or merge into) a trailing user turn telling the model to call a tool.

    The wording is tool-neutral: continue the task, or finish with end_execution — so a
    multi-tool agent that stalled mid-task isn't pushed to end early. Has to end on a user
    turn: native/Vertex rejects a forced call that ends on the stalled assistant turn (a
    prefill). Merge instead of appending so roles stay alternating even if the stalled turn
    was dropped as empty.
    """
    nudge = (
        f"Call the tool to continue the task. If you've finished, call "
        f"{_END_EXECUTION_NAME} with the result."
    )
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
    return strip_thinking(model), _with_extraction_nudge(
        _strip_reasoning_blocks(messages)
    )
