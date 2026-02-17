import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, cast

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ContentBlock,
    HumanMessage,
    TextContentBlock,
    ToolCall,
    ToolMessage,
)
from langchain_core.messages.content import create_text_block
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
    UiPathExternalValue,
    UiPathInlineValue,
)
from uipath.runtime import UiPathRuntimeStorageProtocol

logger = logging.getLogger(__name__)

STORAGE_NAMESPACE_EVENT_MAPPER = "chat-event-mapper"
STORAGE_KEY_TOOL_CALL_ID_TO_MESSAGE_ID_MAP = "tool_call_map"


class UiPathChatMessagesMapper:
    """Stateful mapper that converts LangChain messages to UiPath message events.

    Maintains state across multiple message conversions to properly track:
    - The AI message ID associated with each tool call for proper correlation with ToolMessage
    """

    def __init__(self, runtime_id: str, storage: UiPathRuntimeStorageProtocol | None):
        """Initialize the mapper with empty state."""
        self.runtime_id = runtime_id
        self.storage = storage
        self.current_message: AIMessageChunk
        self.seen_message_ids: set[str] = set()
        self._storage_lock = asyncio.Lock()

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
        - If UiPathConversationMessage: convert to LangChain message
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

    # def _convert_uipath_message(
    #     self, uipath_msg: UiPathConversationMessage
    # ) -> BaseMessage:
    #     """Convert a single UiPathConversationMessage into a LangChain message.

    #     Supports multimodal content parts (text, external content) and preserves metadata.
    #     Creates appropriate message type based on role (AIMessage for assistant, HumanMessage for user).

    #     All content parts are combined into one LangChain message so the LLM sees text and attachments together.
    #     Format: "text content\n<uip:attachments>[{...},{...}]</uip:attachments>"
    #     """

    #     # Store text content and attachments separately, and join in the end
    #     text_content: str = ""
    #     attachments: list[dict[str, Any]] = []

    #     metadata: dict[str, Any] = {
    #         "message_id": uipath_msg.message_id,
    #         "created_at": uipath_msg.created_at,
    #         "updated_at": uipath_msg.updated_at,
    #     }

    #     if uipath_msg.content_parts:
    #         for part in uipath_msg.content_parts:
    #             data = part.data

    #             if isinstance(data, UiPathInlineValue):
    #                 if text_content:
    #                     text_content += " "
    #                 text_content += str(data.inline)

    #             elif isinstance(data, UiPathExternalValue):
    #                 attachment_id = self.parse_attachment_id_from_content_part_uri(
    #                     data.uri
    #                 )
    #                 full_name = getattr(part, "name", None)
    #                 if attachment_id and full_name:
    #                     attachments.append(
    #                         {
    #                             "id": attachment_id,
    #                             "full_name": full_name,
    #                             "mime_type": part.mime_type,
    #                         }
    #                     )

    #         if attachments:
    #             metadata["attachments"] = attachments

    #     content_parts: list[str] = []
    #     if text_content:
    #         content_parts.append(text_content)

    #     if attachments:
    #         content_parts.append(
    #             f"<uip:attachments>{json.dumps(attachments)}</uip:attachments>"
    #         )
    #     combined_content = "\n".join(content_parts)

    #     if uipath_msg.role == "assistant":
    #         return AIMessage(content=combined_content, metadata=metadata)
    #     else:
    #         return HumanMessage(content=combined_content, metadata=metadata)
    def _map_messages_internal(
        self, messages: list[UiPathConversationMessage]
    ) -> list[BaseMessage]:
        """
        Converts UiPathConversationMessage list to LangChain messages (UserMessage/AIMessage/ToolMessage list).
        - All content parts are combined into content_blocks
        - Tool calls are converted to LangChain ToolCall format, with results stored as ToolMessage
        - Metadata includes message_id, role, timestamps
        """
        converted_messages: list[BaseMessage] = []

        for uipath_message in messages:
            content_blocks: list[ContentBlock] = []

            # Convert content_parts to content_blocks
            # TODO: Convert file-attachment content-parts to content_blocks as well
            if uipath_message.content_parts:
                for uipath_content_part in uipath_message.content_parts:
                    data = uipath_content_part.data
                    if uipath_content_part.mime_type.startswith("text/") and isinstance(
                        data, UiPathInlineValue
                    ):
                        text = str(data.inline)
                        if text:
                            content_blocks.append(
                                create_text_block(
                                    text, id=uipath_content_part.content_part_id
                                )
                            )

            # Metadata for the user/assistant message
            metadata = {
                "message_id": uipath_message.message_id,
                "created_at": uipath_message.created_at,
                "updated_at": uipath_message.updated_at,
            }

            role = uipath_message.role
            if role == "user":
                converted_messages.append(
                    HumanMessage(
                        id=uipath_message.message_id,
                        content_blocks=content_blocks,
                        additional_kwargs=metadata,
                    )
                )

            elif role == "assistant":
                # Convert tool calls to LangChain format
                tool_calls: list[ToolCall] = []
                tool_messages: list[ToolMessage] = []
                if uipath_message.tool_calls:
                    for uipath_tool_call in uipath_message.tool_calls:
                        tool_call = ToolCall(
                            name=uipath_tool_call.name.replace(" ", "_"),
                            args=uipath_tool_call.input or {},
                            id=uipath_tool_call.tool_call_id,
                        )
                        tool_calls.append(tool_call)

                        tool_call_output = (
                            uipath_tool_call.result.output
                            if uipath_tool_call.result
                            else None
                        )
                        tool_call_status = (
                            "success"
                            if uipath_tool_call.result
                            and not uipath_tool_call.result.is_error
                            else "error"
                        )

                        # Serialize output to string if needed
                        if tool_call_output is None:
                            content = ""
                        elif isinstance(tool_call_output, str):
                            content = tool_call_output
                        else:
                            content = json.dumps(tool_call_output)

                        tool_messages.append(
                            ToolMessage(
                                content=content,
                                status=tool_call_status,
                                tool_call_id=uipath_tool_call.tool_call_id,
                            )
                        )

                # Ideally we pass in content_blocks here rather than string content, but when doing so, OpenAI errors unless a msg_ prefix is used for content-block IDs.
                # When needed, we can switch to content_blocks but need to work out a common ID strategy across models for the content-block IDs.
                converted_messages.append(
                    AIMessage(
                        id=uipath_message.message_id,
                        # content_blocks=content_blocks,
                        content=self._extract_text(content_blocks)
                        if content_blocks
                        else "",
                        tool_calls=tool_calls,
                        additional_kwargs=metadata,
                    )
                )
                converted_messages.extend(tool_messages)

        return converted_messages

    async def map_event(
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
            return await self.map_ai_message_chunk_to_events(message)

        # --- ToolMessage ---
        if isinstance(message, ToolMessage):
            return await self.map_tool_message_to_events(message)

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

    def parse_attachment_id_from_content_part_uri(self, uri: str) -> str | None:
        """Parse attachment ID from a URI.

        Extracts the UUID from URIs like:
        "urn:uipath:cas:file:orchestrator:a940a416-b97b-4146-3089-08de5f4d0a87"

        Args:
            uri: The URI to parse

        Returns:
            The attachment ID if found, None otherwise
        """
        if not uri:
            return None

        # The UUID is the last segment after the final colon
        parts = uri.rsplit(":", 1)
        if len(parts) != 2:
            return None

        potential_uuid = parts[1]
        if not potential_uuid:
            return None

        # Validate it's a proper UUID and normalize to lowercase
        try:
            return str(uuid.UUID(potential_uuid))
        except (ValueError, AttributeError):
            return None

    async def map_ai_message_chunk_to_events(
        self, message: AIMessageChunk
    ) -> list[UiPathConversationMessageEvent]:
        if message.id is None:  # Should we throw instead?
            return []

        events: list[UiPathConversationMessageEvent] = []

        # For every new message_id, start a new message
        if message.id not in self.seen_message_ids:
            self.current_message = message
            self.seen_message_ids.add(message.id)
            events.append(self.map_to_message_start_event(message.id))

        if message.content_blocks:
            # Generate events for each chunk
            for block in message.content_blocks:
                block_type = block.get("type")
                match block_type:
                    case "text":
                        events.append(
                            self.map_chunk_to_content_part_chunk_event(
                                message.id, cast(TextContentBlock, block)
                            )
                        )
                    case "tool_call_chunk":
                        # Accumulate the message chunk
                        self.current_message = self.current_message + message

        elif isinstance(message.content, str) and message.content:
            # Fallback: raw string content on the chunk (rare when using content_blocks)
            events.append(
                self.map_content_to_content_part_chunk_event(
                    message.id, message.content
                )
            )

        # Check if this is the last chunk by examining chunk_position, send end message event only if there are no pending tool calls
        if message.chunk_position == "last":
            if (
                self.current_message.tool_calls is not None
                and len(self.current_message.tool_calls) > 0
            ):
                events.extend(
                    await self.map_current_message_to_start_tool_call_events()
                )
            else:
                events.append(self.map_to_message_end_event(message.id))

        return events

    async def map_current_message_to_start_tool_call_events(self):
        events: list[UiPathConversationMessageEvent] = []
        if (
            self.current_message
            and self.current_message.id is not None
            and self.current_message.tool_calls
        ):
            async with self._storage_lock:
                if self.storage is not None:
                    tool_call_id_to_message_id_map: dict[
                        str, str
                    ] = await self.storage.get_value(
                        self.runtime_id,
                        STORAGE_NAMESPACE_EVENT_MAPPER,
                        STORAGE_KEY_TOOL_CALL_ID_TO_MESSAGE_ID_MAP,
                    )

                    if tool_call_id_to_message_id_map is None:
                        tool_call_id_to_message_id_map = {}
                else:
                    tool_call_id_to_message_id_map = {}

                for tool_call in self.current_message.tool_calls:
                    tool_call_id = tool_call["id"]
                    if tool_call_id is not None:
                        tool_call_id_to_message_id_map[tool_call_id] = (
                            self.current_message.id
                        )
                        events.append(
                            self.map_tool_call_to_tool_call_start_event(
                                self.current_message.id, tool_call
                            )
                        )

                if self.storage is not None:
                    await self.storage.set_value(
                        self.runtime_id,
                        STORAGE_NAMESPACE_EVENT_MAPPER,
                        STORAGE_KEY_TOOL_CALL_ID_TO_MESSAGE_ID_MAP,
                        tool_call_id_to_message_id_map,
                    )

        return events

    async def map_tool_message_to_events(
        self, message: ToolMessage
    ) -> list[UiPathConversationMessageEvent]:
        # Look up the AI message ID using the tool_call_id
        message_id, is_last_tool_call = await self.get_message_id_for_tool_call(
            message.tool_call_id
        )
        if message_id is None:
            logger.warning(
                f"Tool message {message.tool_call_id} has no associated AI message ID. Skipping."
            )
            return []

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
                        output=content_value,
                    ),
                ),
            )
        ]

        if is_last_tool_call:
            events.append(self.map_to_message_end_event(message_id))

        return events

    async def get_message_id_for_tool_call(
        self, tool_call_id: str
    ) -> tuple[str | None, bool]:
        if self.storage is None:
            logger.error(
                f"attempt to lookup tool call id {tool_call_id} when no storage provided"
            )
            return None, False

        async with self._storage_lock:
            tool_call_id_to_message_id_map: dict[
                str, str
            ] = await self.storage.get_value(
                self.runtime_id,
                STORAGE_NAMESPACE_EVENT_MAPPER,
                STORAGE_KEY_TOOL_CALL_ID_TO_MESSAGE_ID_MAP,
            )

            if tool_call_id_to_message_id_map is None:
                logger.error(
                    f"attempt to lookup tool call id {tool_call_id} when no map present in storage"
                )
                return None, False

            message_id = tool_call_id_to_message_id_map.get(tool_call_id)
            if message_id is None:
                logger.error(
                    f"tool call to message map does not contain tool call id {tool_call_id}"
                )
                return None, False

            del tool_call_id_to_message_id_map[tool_call_id]

            await self.storage.set_value(
                self.runtime_id,
                STORAGE_NAMESPACE_EVENT_MAPPER,
                STORAGE_KEY_TOOL_CALL_ID_TO_MESSAGE_ID_MAP,
                tool_call_id_to_message_id_map,
            )

            is_last = message_id not in tool_call_id_to_message_id_map.values()

        return message_id, is_last

    def map_tool_call_to_tool_call_start_event(
        self, message_id: str, tool_call: ToolCall
    ) -> UiPathConversationMessageEvent:
        return UiPathConversationMessageEvent(
            message_id=message_id,
            tool_call=UiPathConversationToolCallEvent(
                tool_call_id=tool_call["id"],
                start=UiPathConversationToolCallStartEvent(
                    tool_name=tool_call["name"],
                    timestamp=self.get_timestamp(),
                    input=tool_call["args"],
                ),
            ),
        )

    def map_chunk_to_content_part_chunk_event(
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
