import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from uipath.agent.conversation import (
    UiPathConversationContentPartChunkEvent,
    UiPathConversationContentPartEndEvent,
    UiPathConversationContentPartEvent,
    UiPathConversationContentPartStartEvent,
    UiPathConversationEvent,
    UiPathConversationExchangeEvent,
    UiPathConversationMessage,
    UiPathConversationMessageEndEvent,
    UiPathConversationMessageEvent,
    UiPathConversationMessageStartEvent,
    UiPathConversationToolCallEndEvent,
    UiPathConversationToolCallEvent,
    UiPathConversationToolCallStartEvent,
    UiPathInlineValue,
)


def _new_id() -> str:
    return str(uuid.uuid4())


def _wrap_in_conversation_event(
    msg_event: UiPathConversationMessageEvent,
    exchange_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> UiPathConversationEvent:
    """Helper to wrap a message event into a conversation-level event."""
    return UiPathConversationEvent(
        conversation_id=conversation_id or _new_id(),
        exchange=UiPathConversationExchangeEvent(
            exchange_id=exchange_id or _new_id(),
            message=msg_event,
        ),
    )


def _extract_text(content) -> str:
    """Normalize LangGraph message.content to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content or "")


def uipath_to_human_messages(
    uipath_msg: UiPathConversationMessage,
) -> List[HumanMessage]:
    """
    Converts a UiPathConversationMessage into a list of HumanMessages for LangGraph.
    Supports multimodal content parts (text, external content) and preserves metadata.
    """
    human_messages = []

    # Loop over each content part
    if uipath_msg.content_parts:
        for part in uipath_msg.content_parts:
            data = part.data
            content = ""
            metadata: Dict[str, Any] = {
                "message_id": uipath_msg.message_id,
                "content_part_id": part.content_part_id,
                "mime_type": part.mime_type,
                "created_at": uipath_msg.created_at,
                "updated_at": uipath_msg.updated_at,
            }

            if isinstance(data, UiPathInlineValue):
                content = str(data.inline)

            # Append a HumanMessage for this content part
            human_messages.append(HumanMessage(content=content, metadata=metadata))

    # Handle the case where there are no content parts
    else:
        metadata = {
            "message_id": uipath_msg.message_id,
            "role": uipath_msg.role,
            "created_at": uipath_msg.created_at,
            "updated_at": uipath_msg.updated_at,
        }
        human_messages.append(HumanMessage(content="", metadata=metadata))

    return human_messages


def map_message(
    message: BaseMessage,
    exchange_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> Optional[UiPathConversationEvent]:
    """Convert LangGraph BaseMessage (chunk or full) into a UiPathConversationEvent."""
    message_id = getattr(message, "id", None) or _new_id()
    timestamp = datetime.now().isoformat()

    # --- Streaming AIMessageChunk ---
    if isinstance(message, AIMessageChunk):
        msg_event = UiPathConversationMessageEvent(
            message_id=message.id or _new_id(),
        )

        if message.content == []:
            msg_event.start = UiPathConversationMessageStartEvent(
                role="assistant", timestamp=timestamp
            )
            msg_event.content_part = UiPathConversationContentPartEvent(
                content_part_id=f"chunk-{message.id}-{0}",
                start=UiPathConversationContentPartStartEvent(mime_type="text/plain"),
            )

        elif isinstance(message.content, list) and message.content:
            for chunk in message.content:
                if not isinstance(chunk, dict):
                    continue
                idx = chunk.get("index", 0)
                ctype = chunk.get("type")
                id = chunk.get("id", f"chunk-{message.id}-{idx}")

                # Start of a tool call
                if ctype == "tool_use":
                    msg_event.tool_call = UiPathConversationToolCallEvent(
                        tool_call_id=id,
                        start=UiPathConversationToolCallStartEvent(
                            tool_name=chunk.get("name") or "",
                            arguments=UiPathInlineValue(inline=""),
                            timestamp=timestamp,
                        ),
                    )

                # JSON args streaming (content part for tool args)
                elif ctype == "input_json_delta":
                    text = chunk.get("partial_json", "")
                    # first delta: emit content part start + first chunk
                    if text == "":
                        msg_event.content_part = UiPathConversationContentPartEvent(
                            content_part_id=id,
                            start=UiPathConversationContentPartStartEvent(
                                mime_type="application/json"
                            ),
                        )
                    else:
                        msg_event.content_part = UiPathConversationContentPartEvent(
                            content_part_id=id,
                            chunk=UiPathConversationContentPartChunkEvent(
                                data=text,
                                content_part_sequence=idx,
                            ),
                        )

                # Plain text from assistant
                elif ctype == "text":
                    text = chunk.get("text", "")
                    msg_event.content_part = UiPathConversationContentPartEvent(
                        content_part_id=id,
                        chunk=UiPathConversationContentPartChunkEvent(
                            data=text,
                            content_part_sequence=idx,
                        ),
                    )

        stop_reason = message.response_metadata.get("stop_reason")
        if not message.content and stop_reason in ("tool_use", "end_turn"):
            msg_event.end = UiPathConversationMessageEndEvent(timestamp=timestamp)

        if (
            msg_event.start
            or msg_event.content_part
            or msg_event.tool_call
            or msg_event.end
        ):
            return _wrap_in_conversation_event(msg_event, exchange_id, conversation_id)

        return None

    text_content = _extract_text(message.content)

    # --- HumanMessage ---
    if isinstance(message, HumanMessage):
        return _wrap_in_conversation_event(
            UiPathConversationMessageEvent(
                message_id=message_id,
                start=UiPathConversationMessageStartEvent(
                    role="user", timestamp=timestamp
                ),
                content_part=UiPathConversationContentPartEvent(
                    content_part_id=f"cp-{message_id}",
                    start=UiPathConversationContentPartStartEvent(
                        mime_type="text/plain"
                    ),
                    chunk=UiPathConversationContentPartChunkEvent(data=text_content),
                    end=UiPathConversationContentPartEndEvent(),
                ),
                end=UiPathConversationMessageEndEvent(),
            ),
            exchange_id,
            conversation_id,
        )

    # --- AIMessage ---
    if isinstance(message, AIMessage):
        # Extract first tool call if present
        tool_calls = getattr(message, "tool_calls", []) or []
        first_tc = tool_calls[0] if tool_calls else None

        return _wrap_in_conversation_event(
            UiPathConversationMessageEvent(
                message_id=message_id,
                start=UiPathConversationMessageStartEvent(
                    role="assistant", timestamp=timestamp
                ),
                content_part=(
                    UiPathConversationContentPartEvent(
                        content_part_id=f"cp-{message_id}",
                        start=UiPathConversationContentPartStartEvent(
                            mime_type="text/plain"
                        ),
                        chunk=UiPathConversationContentPartChunkEvent(
                            data=text_content
                        ),
                        end=UiPathConversationContentPartEndEvent(),
                    )
                    if text_content
                    else None
                ),
                tool_call=(
                    UiPathConversationToolCallEvent(
                        tool_call_id=first_tc.get("id") or _new_id(),
                        start=UiPathConversationToolCallStartEvent(
                            tool_name=first_tc.get("name"),
                            arguments=UiPathInlineValue(
                                inline=str(first_tc.get("args", ""))
                            ),
                            timestamp=timestamp,
                        ),
                    )
                    if first_tc
                    else None
                ),
                end=UiPathConversationMessageEndEvent(),
            ),
            exchange_id,
            conversation_id,
        )

    # --- ToolMessage ---
    if isinstance(message, ToolMessage):
        return _wrap_in_conversation_event(
            UiPathConversationMessageEvent(
                message_id=message_id,
                tool_call=UiPathConversationToolCallEvent(
                    tool_call_id=message.tool_call_id,
                    start=UiPathConversationToolCallStartEvent(
                        tool_name=message.name or "",
                        arguments=UiPathInlineValue(inline=""),
                        timestamp=timestamp,
                    ),
                    end=UiPathConversationToolCallEndEvent(
                        timestamp=timestamp,
                        result=UiPathInlineValue(inline=message.content),
                    ),
                ),
            ),
            exchange_id,
            conversation_id,
        )

    # --- Fallback ---
    return _wrap_in_conversation_event(
        UiPathConversationMessageEvent(
            message_id=message_id,
            start=UiPathConversationMessageStartEvent(
                role="assistant", timestamp=timestamp
            ),
            content_part=UiPathConversationContentPartEvent(
                content_part_id=f"cp-{message_id}",
                chunk=UiPathConversationContentPartChunkEvent(data=text_content),
            ),
            end=UiPathConversationMessageEndEvent(),
        ),
        exchange_id,
        conversation_id,
    )
