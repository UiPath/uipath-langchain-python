import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, cast

from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    TextContentBlock,
    ToolCallChunk,
    ToolMessage,
)
from uipath.core.chat import (
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

logger = logging.getLogger(__name__)


class UiPathChatMessagesMapper:
    """Stateful mapper that converts LangChain messages to UiPath conversation events.

    Maintains state across multiple message conversions to properly track:
    - The AI message ID associated with each tool call for proper correlation with ToolMessage
    """

    def __init__(self):
        """Initialize the mapper with empty state."""
        self.tool_call_to_ai_message: dict[str, str] = {}
        self.seen_message_ids: set[str] = set()

    def _wrap_in_conversation_event(
        self,
        msg_event: UiPathConversationMessageEvent,
        exchange_id: str | None = None,
        conversation_id: str | None = None,
    ) -> UiPathConversationEvent:
        """Helper to wrap a message event into a conversation-level event."""
        return UiPathConversationEvent(
            conversation_id=conversation_id or str(uuid.uuid4()),
            exchange=UiPathConversationExchangeEvent(
                exchange_id=exchange_id or str(uuid.uuid4()),
                message=msg_event,
            ),
        )

    def _extract_text(self, content: Any) -> str:
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

    def map_messages(self, messages: list[Any]) -> list[Any]:
        """Normalize any 'messages' list into LangChain messages.

        - If already BaseMessage instances: return as-is.
        - If UiPathConversationMessage: convert to HumanMessage.
        """
        if not isinstance(messages, list):
            raise TypeError("messages must be a list")

        if not messages:
            return []

        first = messages[0]

        # Case 1: already LangChain messages
        if isinstance(first, BaseMessage):
            return cast(list[BaseMessage], messages)

        # Case 2: UiPath messages -> convert to HumanMessage
        if isinstance(first, UiPathConversationMessage):
            if not all(isinstance(m, UiPathConversationMessage) for m in messages):
                raise TypeError("Mixed message types not supported")
            return self._map_messages_internal(
                cast(list[UiPathConversationMessage], messages)
            )

        # Fallback: unknown type – just pass through
        return messages

    def _map_messages_internal(
        self, messages: list[UiPathConversationMessage]
    ) -> list[HumanMessage]:
        """
        Converts a UiPathConversationMessage into a list of HumanMessages for LangGraph.
        Supports multimodal content parts (text, external content) and preserves metadata.
        """
        human_messages: list[HumanMessage] = []

        for uipath_msg in messages:
            # Loop over each content part
            if uipath_msg.content_parts:
                for part in uipath_msg.content_parts:
                    data = part.data
                    content = ""
                    metadata: dict[str, Any] = {
                        "message_id": uipath_msg.message_id,
                        "content_part_id": part.content_part_id,
                        "mime_type": part.mime_type,
                        "created_at": uipath_msg.created_at,
                        "updated_at": uipath_msg.updated_at,
                    }

                    if isinstance(data, UiPathInlineValue):
                        content = str(data.inline)

                    # Append a HumanMessage for this content part
                    human_messages.append(
                        HumanMessage(content=content, metadata=metadata)
                    )

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

    def map_event(
        self,
        message: BaseMessage,
        exchange_id: str | None = None,
        conversation_id: str | None = None,
    ) -> UiPathConversationEvent | None:
        """Convert LangGraph BaseMessage (chunk or full) into a UiPathConversationEvent.

        Args:
            message: The LangChain message to convert
            exchange_id: Optional exchange ID for the conversation
            conversation_id: Optional conversation ID

        Returns:
            A UiPathConversationEvent if the message should be emitted, None otherwise
        """
        # Format timestamp as ISO 8601 UTC with milliseconds: 2025-01-04T10:30:00.123Z
        timestamp = (
            datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )

        # --- Streaming AIMessageChunk ---
        if isinstance(message, AIMessageChunk):
            if message.id is None:
                return None

            msg_event = UiPathConversationMessageEvent(
                message_id=message.id,
            )

            # Check if this is the last chunk by examining chunk_position
            if message.chunk_position == "last":
                msg_event.end = UiPathConversationMessageEndEvent(timestamp=timestamp)
                msg_event.content_part = UiPathConversationContentPartEvent(
                    content_part_id=f"chunk-{message.id}-0",
                    end=UiPathConversationContentPartEndEvent(),
                )
                return self._wrap_in_conversation_event(
                    msg_event, exchange_id, conversation_id
                )

            # For every new message_id, start a new message
            if message.id not in self.seen_message_ids:
                self.seen_message_ids.add(message.id)
                msg_event.start = UiPathConversationMessageStartEvent(
                    role="assistant", timestamp=timestamp
                )
                msg_event.content_part = UiPathConversationContentPartEvent(
                    content_part_id=f"chunk-{message.id}-0",
                    start=UiPathConversationContentPartStartEvent(
                        mime_type="text/plain"
                    ),
                )

            elif message.content_blocks:
                for block in message.content_blocks:
                    block_type = block.get("type")

                    if block_type == "text":
                        text_block = cast(TextContentBlock, block)
                        text = text_block["text"]

                        msg_event.content_part = UiPathConversationContentPartEvent(
                            content_part_id=f"chunk-{message.id}-0",
                            chunk=UiPathConversationContentPartChunkEvent(
                                data=text,
                                content_part_sequence=0,
                            ),
                        )

                    elif block_type == "tool_call_chunk":
                        tool_chunk_block = cast(ToolCallChunk, block)

                        tool_call_id = tool_chunk_block.get("id")
                        if tool_call_id:
                            # Track tool_call_id -> ai_message_id mapping
                            self.tool_call_to_ai_message[str(tool_call_id)] = message.id

                        args = tool_chunk_block.get("args") or ""

                        msg_event.content_part = UiPathConversationContentPartEvent(
                            content_part_id=f"chunk-{message.id}-0",
                            chunk=UiPathConversationContentPartChunkEvent(
                                data=args,
                                content_part_sequence=0,
                            ),
                        )
                        # Continue so that multiple tool_call_chunks in the same block list
                        # are handled correctly
                        continue

            # Fallback: raw string content on the chunk (rare when using content_blocks)
            elif isinstance(message.content, str) and message.content:
                msg_event.content_part = UiPathConversationContentPartEvent(
                    content_part_id=f"content-{message.id}",
                    chunk=UiPathConversationContentPartChunkEvent(
                        data=message.content,
                        content_part_sequence=0,
                    ),
                )

            if (
                msg_event.start
                or msg_event.content_part
                or msg_event.tool_call
                or msg_event.end
            ):
                return self._wrap_in_conversation_event(
                    msg_event, exchange_id, conversation_id
                )

            return None

        # --- ToolMessage ---
        if isinstance(message, ToolMessage):
            # Look up the AI message ID using the tool_call_id
            result_message_id = (
                self.tool_call_to_ai_message.get(message.tool_call_id)
                if message.tool_call_id
                else None
            )

            # If no AI message ID was found, we cannot properly associate this tool result
            if not result_message_id:
                logger.warning(
                    f"Tool message {message.tool_call_id} has no associated AI message ID. Skipping."
                )
                return None

            # Clean up the mapping after use
            if message.tool_call_id:
                del self.tool_call_to_ai_message[message.tool_call_id]

            content_value: Any = message.content
            if isinstance(content_value, str):
                try:
                    content_value = json.loads(content_value)
                except (json.JSONDecodeError, TypeError):
                    # Keep as string if not valid JSON
                    pass

            return self._wrap_in_conversation_event(
                UiPathConversationMessageEvent(
                    message_id=result_message_id,
                    tool_call=UiPathConversationToolCallEvent(
                        tool_call_id=message.tool_call_id,
                        start=UiPathConversationToolCallStartEvent(
                            tool_name=message.name,
                            arguments=None,
                            timestamp=timestamp,
                        ),
                        end=UiPathConversationToolCallEndEvent(
                            timestamp=timestamp,
                            result=UiPathInlineValue(inline=content_value),
                        ),
                    ),
                ),
                exchange_id,
                conversation_id,
            )

        # --- Fallback for other BaseMessage types ---
        text_content = self._extract_text(message.content)
        return self._wrap_in_conversation_event(
            UiPathConversationMessageEvent(
                message_id=message.id,
                start=UiPathConversationMessageStartEvent(
                    role="assistant", timestamp=timestamp
                ),
                content_part=UiPathConversationContentPartEvent(
                    content_part_id=f"cp-{message.id}",
                    chunk=UiPathConversationContentPartChunkEvent(data=text_content),
                ),
                end=UiPathConversationMessageEndEvent(),
            ),
            exchange_id,
            conversation_id,
        )


__all__ = ["UiPathChatMessagesMapper"]
