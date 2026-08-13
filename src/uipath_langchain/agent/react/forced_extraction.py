"""Force a structured end_execution out of a thinking model that stalled.

Anthropic won't honor a forced tool_choice while thinking is on, so a thinking model can
answer in plain text and never call end_execution. build_extraction_call retries that
turn with thinking off and the tool call forced, which every provider honors.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from uipath_langchain.chat.thinking import is_reasoning_block, strip_thinking


def _strip_reasoning_blocks(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Drop reasoning blocks from AI messages (keep text + tool calls).

    They can't be replayed on a thinking-off call — an orphaned thinking block 400s.
    """
    stripped: list[AnyMessage] = []
    for message in messages:
        if isinstance(message, AIMessage) and isinstance(message.content, list):
            kept = [block for block in message.content if not is_reasoning_block(block)]
            if len(kept) != len(message.content):
                # a turn that was only reasoning is now empty — drop it
                if not kept and not message.tool_calls:
                    continue
                message = message.model_copy(update={"content": kept})
        stripped.append(message)
    return stripped


def _ensure_trailing_user_turn(messages: list[AnyMessage]) -> list[AnyMessage]:
    if messages and not isinstance(messages[-1], AIMessage):
        return messages
    return list(messages) + [HumanMessage(content="Call a tool to continue. Terminal tool calls must contain the final output.")]


def build_extraction_call(
    model: BaseChatModel, messages: list[AnyMessage]
) -> tuple[BaseChatModel, list[AnyMessage]]:
    """The (model, messages) for the extraction call: thinking off, reasoning blocks
    dropped, ending on a user turn — the caller then forces tool_choice."""
    return strip_thinking(model), _ensure_trailing_user_turn(
        _strip_reasoning_blocks(messages)
    )
