import json
import logging
from datetime import datetime, timezone
from typing import Any, cast
from uuid import uuid4

from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    TextContentBlock,
    ToolCallChunk,
    ToolMessage,
)
from pydantic import ValidationError
from uipath.core.chat import (
    UiPathConversationContentPartChunkEvent,
    UiPathConversationContentPartEndEvent,
    UiPathConversationContentPartEvent,
    UiPathConversationContentPartStartEvent,
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
    """Stateful mapper that converts LangChain messages to UiPath message events.

    Maintains state across multiple message conversions to properly track:
    - The AI message ID associated with each tool call for proper correlation with ToolMessage
    """

    def __init__(self):
        """Initialize the mapper with empty state."""
        self.tool_call_to_ai_message: dict[str, str] = {}
        self.seen_message_ids: set[str] = set()
        self.pending_message_tool_call_count: dict[str, int] = {}

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

        # Case3: List[dict] -> parse to List[UiPathConversationMessage]
        if isinstance(first, dict):
            try:
                parsed_messages = [
                    UiPathConversationMessage.model_validate(message)
                    for message in messages
                ]
                return self._map_messages_internal(parsed_messages)
            except ValidationError:
                pass

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
    ) -> list[UiPathConversationMessageEvent] | None:
        """Convert LangGraph BaseMessage (chunk or full) into a UiPathConversationMessageEvent.

        Args:
            message: The LangChain message to convert

        Returns:
            A UiPathConversationMessageEvent if the message should be emitted, None otherwise.
        """
        # --- Streaming AIMessageChunk ---
        if isinstance(message, AIMessageChunk):
            return self.map_ai_message_chunk_to_events(message)

        # --- ToolMessage ---
        if isinstance(message, ToolMessage):
            return self.map_tool_message_to_events(message)

        # Don't send events for system or user messages. Agent messages are handled above.
        return None

    def get_timestamp(self):
        """Format current time as ISO 8601 UTC with milliseconds: 2025-01-04T10:30:00.123Z"""
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )

    def get_content_part_id(self, message_id: str) -> str:
        return f"chunk-{message_id}-0"

    def map_ai_message_chunk_to_events(
        self, message: AIMessageChunk
    ) -> list[UiPathConversationMessageEvent]:
        if message.id is None:  # Should we throw instead?
            return []

        events: list[UiPathConversationMessageEvent] = []

        # For every new message_id, start a new message
        if message.id not in self.seen_message_ids:
            self.seen_message_ids.add(message.id)
            self.pending_message_tool_call_count[message.id] = 0
            events.append(self.map_to_message_start_event(message.id))

        if message.content_blocks:
            # Generate events for each chunk
            for block in message.content_blocks:
                block_type = block.get("type")
                match block_type:
                    case "text":
                        events.append(
                            self.map_block_to_content_part_chunk_event(
                                message.id, cast(TextContentBlock, block)
                            )
                        )
                    case "tool_call_chunk":
                        events.extend(
                            self.map_block_to_tool_call_start_event(
                                message.id, cast(ToolCallChunk, block)
                            )
                        )

        elif isinstance(message.content, str) and message.content:
            # Fallback: raw string content on the chunk (rare when using content_blocks)
            events.append(
                self.map_content_to_content_part_chunk_event(
                    message.id, message.content
                )
            )

        # Check if this is the last chunk by examining chunk_position, send end message event only if there are no pending tool calls
        if (
            message.chunk_position == "last"
            and self.pending_message_tool_call_count.get(message.id, 0) == 0
        ):
            del self.pending_message_tool_call_count[message.id]
            events.append(self.map_to_message_end_event(message.id))

        return events

    def map_tool_message_to_events(
        self, message: ToolMessage
    ) -> list[UiPathConversationMessageEvent]:
        # Look up the AI message ID using the tool_call_id
        message_id = self.tool_call_to_ai_message.get(message.tool_call_id)
        if message_id is None:
            logger.warning(
                f"Tool message {message.tool_call_id} has no associated AI message ID. Skipping."
            )
            return []

        # Clean up the mapping after use
        del self.tool_call_to_ai_message[message.tool_call_id]

        content_value: Any = message.content
        if isinstance(content_value, str):
            try:
                content_value = json.loads(content_value)
            except (json.JSONDecodeError, TypeError):
                # Keep as string if not valid JSON
                pass

        events = [
            UiPathConversationMessageEvent(
                message_id=message_id,
                tool_call=UiPathConversationToolCallEvent(
                    tool_call_id=message.tool_call_id,
                    end=UiPathConversationToolCallEndEvent(
                        timestamp=self.get_timestamp(),
                        output=UiPathInlineValue(inline=content_value),
                    ),
                ),
            )
        ]

        self.pending_message_tool_call_count[message_id] -= 1
        if self.pending_message_tool_call_count[message_id] == 0:
            events.append(self.map_to_message_end_event(message_id))

        return events

    def map_block_to_tool_call_start_event(
        self, message_id: str, block
    ) -> list[UiPathConversationMessageEvent]:
        tool_call_id = block.get("id")
        if tool_call_id is None:
            return []

        self.tool_call_to_ai_message[tool_call_id] = message_id

        self.pending_message_tool_call_count[message_id] += 1

        tool_name = block.get("name")
        tool_args = block.get("args")

        return [
            UiPathConversationMessageEvent(
                message_id=message_id,
                tool_call=UiPathConversationToolCallEvent(
                    tool_call_id=tool_call_id,
                    start=UiPathConversationToolCallStartEvent(
                        tool_name=tool_name,
                        timestamp=self.get_timestamp(),
                        input=UiPathInlineValue(inline=tool_args),
                    ),
                ),
            )
        ]

    def map_block_to_content_part_chunk_event(
        self, message_id: str, block: TextContentBlock
    ) -> UiPathConversationMessageEvent:
        text = block["text"]
        return UiPathConversationMessageEvent(
            message_id=message_id,
            content_part=UiPathConversationContentPartEvent(
                content_part_id=self.get_content_part_id(message_id),
                chunk=UiPathConversationContentPartChunkEvent(
                    data=text,
                ),
            ),
        )

    def map_content_to_content_part_chunk_event(
        self, message_id: str, content: str
    ) -> UiPathConversationMessageEvent:
        return UiPathConversationMessageEvent(
            message_id=message_id,
            content_part=UiPathConversationContentPartEvent(
                content_part_id=self.get_content_part_id(message_id),
                chunk=UiPathConversationContentPartChunkEvent(
                    data=content,
                ),
            ),
        )

    def map_to_message_start_event(
        self, message_id: str
    ) -> UiPathConversationMessageEvent:
        return UiPathConversationMessageEvent(
            message_id=message_id,
            start=UiPathConversationMessageStartEvent(
                role="assistant", timestamp=self.get_timestamp()
            ),
            content_part=UiPathConversationContentPartEvent(
                content_part_id=self.get_content_part_id(message_id),
                start=UiPathConversationContentPartStartEvent(mime_type="text/plain"),
            ),
        )

    def map_to_message_end_event(
        self, message_id: str
    ) -> UiPathConversationMessageEvent:
        return UiPathConversationMessageEvent(
            message_id=message_id,
            end=UiPathConversationMessageEndEvent(),
            content_part=UiPathConversationContentPartEvent(
                content_part_id=self.get_content_part_id(message_id),
                end=UiPathConversationContentPartEndEvent(),
            ),
        )


__all__ = ["UiPathChatMessagesMapper"]
