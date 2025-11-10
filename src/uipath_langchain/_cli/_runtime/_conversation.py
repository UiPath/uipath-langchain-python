import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from pydantic import TypeAdapter, ValidationError
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

from uipath_langchain.chat.content_blocks import (
    ContentBlock,
    TextContent,
    ToolCallChunkContent,
    ToolCallContent,
)

logger = logging.getLogger(__name__)


def _new_id() -> str:
    return str(uuid.uuid4())


class MessageMapper:
    """Stateful mapper that converts LangChain messages to UiPath conversation events.

    Maintains state across multiple message conversions to properly track:
    - The AI message ID associated with each tool call for proper correlation with ToolMessage
    """

    def __init__(self):
        """Initialize the mapper with empty state."""
        self.tool_call_to_ai_message: Dict[str, str] = {}
        self.seen_message_ids: set[str] = set()

    def _wrap_in_conversation_event(
        self,
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

    def _extract_text(self, content) -> str:
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

    def map_message(
        self,
        message: BaseMessage,
        exchange_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Optional[UiPathConversationEvent]:
        """Convert LangGraph BaseMessage (chunk or full) into a UiPathConversationEvent.

        Args:
            message: The LangChain message to convert
            exchange_id: Optional exchange ID for the conversation
            conversation_id: Optional conversation ID

        Returns:
            A UiPathConversationEvent if the message should be emitted, None otherwise
        """
        # Format timestamp as ISO 8601 UTC with milliseconds: 2025-01-04T10:30:00.123Z
        timestamp = datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')

        # --- Streaming AIMessageChunk ---
        if isinstance(message, AIMessageChunk):
            # Track this AI message ID for associating tool calls
            ai_message_id = message.id

            msg_event = UiPathConversationMessageEvent(
                message_id=ai_message_id,
            )

            # Check if this is the last chunk by examining chunk_position
            chunk = AIMessageChunk(**message.model_dump())
            if hasattr(chunk, "chunk_position") and getattr(chunk, "chunk_position") == "last":
                msg_event.end = UiPathConversationMessageEndEvent(timestamp=timestamp)
                msg_event.content_part = UiPathConversationContentPartEvent(
                    content_part_id=f"chunk-{message.id}-0",
                    end=UiPathConversationContentPartEndEvent()
                )
                return self._wrap_in_conversation_event(msg_event, exchange_id, conversation_id)

            # For every new message_id, start a new message
            if ai_message_id not in self.seen_message_ids:
                self.seen_message_ids.add(ai_message_id)
                msg_event.start = UiPathConversationMessageStartEvent(
                    role="assistant", timestamp=timestamp
                )
                msg_event.content_part = UiPathConversationContentPartEvent(
                    content_part_id=f"chunk-{message.id}-0",
                    start=UiPathConversationContentPartStartEvent(mime_type="text/plain"),
                )

            elif isinstance(message.content, list) and message.content:
                content_adapter = TypeAdapter(ContentBlock)

                for raw_chunk in message.content:
                    if not isinstance(raw_chunk, dict):
                        continue

                    try:
                        # Parse chunk
                        chunk = content_adapter.validate_python(raw_chunk)

                        if isinstance(chunk, TextContent):
                            msg_event.content_part = UiPathConversationContentPartEvent(
                                content_part_id=f"chunk-{message.id}-0",
                                chunk=UiPathConversationContentPartChunkEvent(
                                    data=chunk.text,
                                    content_part_sequence=0,
                                ),
                            )

                        elif isinstance(chunk, ToolCallChunkContent):
                            # Track tool_call_id -> ai_message_id mapping
                            if chunk.id:
                                self.tool_call_to_ai_message[chunk.id] = ai_message_id

                            msg_event.content_part = UiPathConversationContentPartEvent(
                                content_part_id=f"chunk-{message.id}-0",
                                chunk=UiPathConversationContentPartChunkEvent(
                                    data=chunk.args,
                                    content_part_sequence=0,
                                ),
                            )
                            continue

                    except ValidationError as e:
                        # Log and skip unknown/invalid chunk types
                        logger.warning(
                            f"Failed to parse content chunk: {raw_chunk}. Error: {e}"
                        )
                        continue

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
                return self._wrap_in_conversation_event(msg_event, exchange_id, conversation_id)

            return None

        # --- ToolMessage ---
        if isinstance(message, ToolMessage):
            # Look up the AI message ID using the tool_call_id
            result_message_id = self.tool_call_to_ai_message.get(message.tool_call_id) if message.tool_call_id else None

            # If no AI message ID was found, we cannot properly associate this tool result
            if not result_message_id:
                logger.warning(
                    f"Tool message {message.tool_call_id} has no associated AI message ID. Skipping."
                )
                return None

            # Clean up the mapping after use
            if message.tool_call_id:
                del self.tool_call_to_ai_message[message.tool_call_id]

            content_value = message.content
            if isinstance(content_value, str):
                try:
                    content_value = json.loads(content_value)
                except (json.JSONDecodeError, TypeError):
                    pass  # Keep as string if not valid JSON

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
                            result=content_value,
                        ),
                    ),
                ),
                exchange_id,
                conversation_id,
            )

        text_content = self._extract_text(message.content)
        # --- Fallback ---
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
