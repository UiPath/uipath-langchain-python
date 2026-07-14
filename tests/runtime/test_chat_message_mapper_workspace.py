"""Assistant-message workspace file content-parts must be hidden from the LLM."""

from langchain_core.messages import AIMessage, HumanMessage
from uipath.core.chat import (
    UiPathConversationContentPart,
    UiPathConversationMessage,
    UiPathExternalValue,
    UiPathInlineValue,
)

from uipath_langchain.runtime.messages import UiPathChatMessagesMapper

CAS_URI = "urn:uipath:cas:file:orchestrator:a940a416-b97b-4146-3089-08de5f4d0a87"


def _file_part(
    part_id: str,
    name: str,
    *,
    metadata: dict[str, str] | None = None,
) -> UiPathConversationContentPart:
    return UiPathConversationContentPart(
        content_part_id=part_id,
        mime_type="text/markdown",
        data=UiPathExternalValue(uri=CAS_URI),
        name=name,
        metadata=metadata,
        citations=[],
    )


def test_assistant_file_parts_are_skipped() -> None:
    mapper = UiPathChatMessagesMapper("test-runtime", None)
    message = UiPathConversationMessage(
        message_id="a1",
        role="assistant",
        content_parts=[
            UiPathConversationContentPart(
                content_part_id="p1",
                mime_type="text/plain",
                data=UiPathInlineValue(inline="done, see the plan"),
                citations=[],
            ),
            _file_part("ws-a1-0", "plan/todo.md", metadata={"fileKind": "workspace"}),
        ],
        tool_calls=[],
    )

    result = mapper.map_messages([message])

    assert len(result) == 1
    ai = result[0]
    assert isinstance(ai, AIMessage)
    assert "<uip:attachments>" not in ai.content
    assert "attachments" not in ai.additional_kwargs
    assert "done, see the plan" in ai.content


def test_user_file_parts_still_produce_attachments() -> None:
    mapper = UiPathChatMessagesMapper("test-runtime", None)
    message = UiPathConversationMessage(
        message_id="u1",
        role="user",
        content_parts=[_file_part("p1", "report.pdf")],
        tool_calls=[],
    )

    result = mapper.map_messages([message])

    assert len(result) == 1
    user = result[0]
    assert isinstance(user, HumanMessage)
    assert user.additional_kwargs["attachments"][0]["full_name"] == "report.pdf"


def test_ordinary_assistant_file_parts_still_produce_attachments() -> None:
    mapper = UiPathChatMessagesMapper("test-runtime", None)
    message = UiPathConversationMessage(
        message_id="a1",
        role="assistant",
        content_parts=[_file_part("p1", "report.pdf")],
        tool_calls=[],
    )

    result = mapper.map_messages([message])

    assert len(result) == 1
    assistant = result[0]
    assert isinstance(assistant, AIMessage)
    assert assistant.additional_kwargs["attachments"][0]["full_name"] == "report.pdf"


def test_workspace_part_id_is_hidden_without_metadata() -> None:
    mapper = UiPathChatMessagesMapper("test-runtime", None)
    message = UiPathConversationMessage(
        message_id="a1",
        role="assistant",
        content_parts=[_file_part("ws-a1-0", "plan.md")],
        tool_calls=[],
    )

    result = mapper.map_messages([message])

    assert len(result) == 1
    assistant = result[0]
    assert isinstance(assistant, AIMessage)
    assert "attachments" not in assistant.additional_kwargs
