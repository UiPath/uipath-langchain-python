"""Attachment resolution for conversational agents."""

import asyncio
import json
import re
import uuid
from typing import Any, Type, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AnyMessage,
    ContentBlock,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables.config import var_child_runnable_config
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from uipath.platform import UiPath
from uipath.platform.attachments import Attachment

from uipath_langchain.agent.multimodal import FileInfo, build_file_content_block
from uipath_langchain.chat.helpers import (
    append_content_blocks_to_message,
    extract_text_content,
)

_ATTACHMENTS_PATTERN = re.compile(
    r"<uip:attachments>(.*?)</uip:attachments>", re.DOTALL
)

_READ_ATTACHMENTS_SYSTEM_MESSAGE = (
    "Process the provided files to complete the given task. "
    "Analyze the files contents thoroughly to deliver an accurate response "
    "based on the extracted information."
)


async def resolve_attachments(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Resolve file attachments embedded in conversational messages.

    UiPath's runtime injects attachment metadata as a ``<uip:attachments>``
    text block into HumanMessages. This function resolves those references to
    actual file content blocks so the LLM can read the files.

    Messages that contain no ``<uip:attachments>`` tag are returned unchanged,
    so it is safe to call this on the full message history on every turn.

    To persist resolved content to the checkpointer (avoiding re-downloads on
    subsequent turns), include the resolved messages in the node's return value
    alongside the LLM response. LangGraph's ``add_messages`` reducer replaces
    messages with matching IDs rather than appending them::

        async def call_model(state: MessagesState):
            resolved = await resolve_attachments(state["messages"])
            response = await llm.ainvoke([SystemMessage(...)] + resolved)
            return {"messages": resolved + [response]}

    Args:
        messages: List of LangChain messages from the agent state.

    Returns:
        New list where HumanMessages with attachments have their
        ``<uip:attachments>`` text replaced with DataContentBlock items
        containing the actual file content.
    """
    return list(await asyncio.gather(*[_resolve_message(m) for m in messages]))


async def _resolve_message(message: AnyMessage) -> AnyMessage:
    if not isinstance(message, HumanMessage):
        return message

    content = message.content
    if not isinstance(content, list):
        return message

    clean_blocks: list[Any] = []
    file_infos: list[FileInfo] = []

    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            clean_blocks.append(block)
            continue

        text = block.get("text", "")
        match = _ATTACHMENTS_PATTERN.search(text)
        if not match:
            clean_blocks.append(block)
            continue

        # Parse attachment metadata
        attachments: list[dict[str, Any]] = json.loads(match.group(1))
        file_infos.extend(await _resolve_file_infos(attachments))

        # Preserve any text outside the tag
        remaining = _ATTACHMENTS_PATTERN.sub("", text).strip()
        if remaining:
            clean_blocks.append({"type": "text", "text": remaining})

    if not file_infos:
        return message

    file_blocks = list(
        await asyncio.gather(*[build_file_content_block(fi) for fi in file_infos])
    )

    return HumanMessage(
        id=message.id,
        content=clean_blocks + file_blocks,
        additional_kwargs=message.additional_kwargs,
    )


async def _resolve_file_infos(
    attachments: list[dict[str, Any]],
) -> list[FileInfo]:
    client = UiPath()
    file_infos: list[FileInfo] = []

    for att in attachments:
        att_id = att.get("id")
        if not att_id:
            continue

        blob_info = await client.attachments.get_blob_file_access_uri_async(
            key=uuid.UUID(att_id)
        )
        file_infos.append(
            FileInfo(
                url=blob_info.uri,
                name=blob_info.name,
                mime_type=att.get("mime_type", ""),
            )
        )

    return file_infos


class ReadAttachmentsInput(BaseModel):
    """Input schema for the ReadAttachmentsTool."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str = Field(description="What you want to know about or do with the files.")
    attachments: list[Attachment] = Field(
        description="The attachment objects from inside the <uip:attachments> tag."
    )


class ReadAttachmentsTool(BaseTool):
    """Tool that reads and interprets file attachments using the provided LLM.

    The tool downloads each attachment, passes the file content to a non-streaming
    copy of the provided LLM for interpretation, and returns the result as text.
    This keeps multimodal content out of the agent's message state — the original
    ``<uip:attachments>`` metadata in HumanMessages is never modified.

    Example::

        from langchain_openai import ChatOpenAI
        from uipath_langchain.chat import ReadAttachmentsTool

        llm = ChatOpenAI(model="gpt-4.1")
        tool = ReadAttachmentsTool(llm=llm)
    """

    name: str = "read_attachments"
    description: str = (
        "Read and interpret the content of file attachments provided by the user. "
        "Call this when you see a <uip:attachments> tag in a user message, passing "
        "the attachment objects from inside the tag and a query describing what you "
        "want to know about or do with the files."
    )
    args_schema: Type[BaseModel] = ReadAttachmentsInput

    llm: BaseChatModel

    _non_streaming_llm: BaseChatModel = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._non_streaming_llm = self.llm.model_copy(
            update={"disable_streaming": True}
        )

    def _run(self, query: str, attachments: list[Attachment]) -> str:
        raise NotImplementedError("Use async version via arun()")

    async def _arun(self, query: str, attachments: list[Attachment]) -> str:
        file_infos = await _resolve_file_infos(
            [a.model_dump(mode="json") for a in attachments]
        )
        if not file_infos:
            return "No attachments provided to analyze."

        file_blocks = list(
            await asyncio.gather(*[build_file_content_block(fi) for fi in file_infos])
        )

        human_message_with_files = append_content_blocks_to_message(
            HumanMessage(content=query), cast(list[ContentBlock], file_blocks)
        )

        messages = [
            SystemMessage(content=_READ_ATTACHMENTS_SYSTEM_MESSAGE),
            human_message_with_files,
        ]

        config = var_child_runnable_config.get(None)
        result = await self._non_streaming_llm.ainvoke(messages, config=config)
        return extract_text_content(result)
