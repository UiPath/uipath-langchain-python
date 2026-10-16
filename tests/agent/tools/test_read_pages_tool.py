"""Tests for read_pages_tool.py module."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from uipath.agent.models.agent import (
    AgentContextQuerySetting,
    AgentContextResourceConfig,
    AgentContextRetrievalMode,
    AgentContextSettings,
)
from uipath.platform.context_grounding import (
    ContextGroundingPage,
    ContextGroundingPageRange,
)
from uipath.platform.errors import (
    ContextGroundingIndexNotFoundError,
    EnrichedException,
)

from uipath_langchain.agent.exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.read_pages_tool import (
    MAX_PAGES_PER_TURN,
    ReadPagesInput,
    create_read_pages_tool,
    pages_read_this_turn,
    read_pages_tool_name,
)

DOC_ID = "1d01be8d-280b-45e7-b532-d1f8f8f36ead"
MODULE = "uipath_langchain.agent.tools.read_pages_tool"


def _resource(name: str = "Annual report") -> AgentContextResourceConfig:
    return AgentContextResourceConfig(
        name=name,
        description="Annual report index",
        resource_type="context",
        context_type="index",
        index_name="report-index",
        folder_path="/test/folder",
        settings=AgentContextSettings(
            result_count=3,
            retrieval_mode=AgentContextRetrievalMode.SEMANTIC,
            query=AgentContextQuerySetting(variant="dynamic"),
            read_pages=True,
        ),
        is_enabled=True,
    )


def _enriched(status_code: int, body: dict[str, Any] | None) -> EnrichedException:
    """An EnrichedException raised from its HTTP error, as the SDK raises it."""
    request = httpx.Request(
        "POST", "https://cloud.uipath.com/org/tenant/ecs_/v2/indexes/i/ReadPages"
    )
    response = httpx.Response(
        status_code=status_code,
        content=json.dumps(body).encode() if body is not None else b"",
        request=request,
    )
    http_error = httpx.HTTPStatusError("error", request=request, response=response)
    try:
        raise EnrichedException(http_error) from http_error
    except EnrichedException as e:
        return e


@pytest.fixture
def sdk():
    mock_sdk = MagicMock()
    mock_sdk.context_grounding.read_pages_async = AsyncMock(
        return_value=ContextGroundingPageRange(
            source_document_id=DOC_ID,
            source="report.pdf",
            pages=[ContextGroundingPage(page_number=7, content="Page seven")],
            missing_pages=[],
            has_previous=True,
            has_next=True,
        )
    )
    with (
        patch(f"{MODULE}.UiPath", return_value=mock_sdk),
        patch(f"{MODULE}.UiPathConfig", MagicMock(is_studio_project=False)),
    ):
        yield mock_sdk


class TestReadPagesToolDefinition:
    def test_name_is_the_context_name_with_a_suffix(self):
        tool = create_read_pages_tool(_resource("Annual report"))

        assert tool.name == "Annual_report_read_pages"

    def test_long_context_name_keeps_the_suffix_within_64_characters(self):
        name = read_pages_tool_name(_resource("x" * 80))

        assert len(name) == 64
        assert name.endswith("_read_pages")

    def test_arguments_are_document_id_and_page_range(self):
        tool = create_read_pages_tool(_resource())

        assert tool.args_schema is ReadPagesInput
        schema = ReadPagesInput.model_json_schema()
        assert set(schema["properties"]) == {
            "source_document_id",
            "start_page",
            "end_page",
        }
        assert set(schema["required"]) == {
            "source_document_id",
            "start_page",
            "end_page",
        }

    def test_description_points_to_the_search_results(self):
        tool = create_read_pages_tool(_resource("Annual report"))

        assert "'Annual report' search" in tool.description
        assert "source_document_id" in tool.description

    def test_metadata_marks_a_context_tool(self):
        tool = create_read_pages_tool(_resource())

        assert tool.metadata == {
            "tool_type": "context",
            "display_name": "Annual report (read pages)",
            "index_name": "report-index",
        }


class TestReadPagesToolCall:
    @pytest.mark.asyncio
    async def test_reads_the_range_from_the_context_index(self, sdk, monkeypatch):
        monkeypatch.setenv("UIPATH_FOLDER_PATH", "Shared/Reports")
        tool = create_read_pages_tool(_resource())

        assert tool.coroutine is not None
        result = await tool.coroutine(
            source_document_id=DOC_ID, start_page=7, end_page=7
        )

        sdk.context_grounding.read_pages_async.assert_awaited_once_with(
            name="report-index",
            source_document_id=DOC_ID,
            start_page=7,
            end_page=7,
            folder_path="Shared/Reports",
            include_system_indexes=False,
        )
        assert result["source_document_id"] == DOC_ID
        assert result["pages"] == [{"page_number": 7, "content": "Page seven"}]
        assert result["has_previous"] is True
        assert result["has_next"] is True

    @pytest.mark.asyncio
    async def test_invalid_document_id_is_returned_to_the_llm(self, sdk):
        message = (
            "SourceDocumentId 'report.pdf' is not a document id. Use the document "
            "id (a GUID) from a search result's source_document_id; if you only "
            "know the document's name or title, search for it first."
        )
        sdk.context_grounding.read_pages_async.side_effect = _enriched(
            400,
            {
                "message": "One or more validation errors occurred.",
                "errors": {"SourceDocumentId": [message]},
            },
        )
        tool = create_read_pages_tool(_resource())

        output = await tool.ainvoke(
            {
                "type": "tool_call",
                "id": "call-1",
                "name": tool.name,
                "args": {
                    "source_document_id": "report.pdf",
                    "start_page": 1,
                    "end_page": 3,
                },
            }
        )

        assert isinstance(output, ToolMessage)
        assert output.status == "error"
        assert output.content == message

    @pytest.mark.asyncio
    async def test_unknown_document_is_returned_to_the_llm(self, sdk):
        sdk.context_grounding.read_pages_async.side_effect = _enriched(
            404, {"message": "File not found."}
        )
        tool = create_read_pages_tool(_resource())

        output = await tool.ainvoke(
            {"source_document_id": DOC_ID, "start_page": 1, "end_page": 3}
        )

        assert output == (
            f"No document with id '{DOC_ID}' is in this index. Use the "
            "source_document_id of a search result from this context; if no result "
            "is from the document you need, search for it first."
        )

    @pytest.mark.asyncio
    async def test_endpoint_unavailable_is_returned_to_the_llm(self, sdk):
        sdk.context_grounding.read_pages_async.side_effect = _enriched(404, None)
        tool = create_read_pages_tool(_resource())

        output = await tool.ainvoke(
            {"source_document_id": DOC_ID, "start_page": 1, "end_page": 3}
        )

        assert output == "Reading pages is not available for index 'report-index'."

    @pytest.mark.asyncio
    async def test_missing_index_fails_the_run(self, sdk):
        sdk.context_grounding.read_pages_async.side_effect = (
            ContextGroundingIndexNotFoundError(index_name="report-index")
        )
        tool = create_read_pages_tool(_resource())

        assert tool.coroutine is not None
        with pytest.raises(AgentRuntimeError) as exc:
            await tool.coroutine(source_document_id=DOC_ID, start_page=1, end_page=3)

        assert exc.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.CONTEXT_GROUNDING_INDEX_NOT_FOUND
        )

    @pytest.mark.asyncio
    async def test_server_errors_are_raised(self, sdk):
        sdk.context_grounding.read_pages_async.side_effect = _enriched(
            500, {"message": "boom"}
        )
        tool = create_read_pages_tool(_resource())

        assert tool.coroutine is not None
        with pytest.raises(EnrichedException):
            await tool.coroutine(source_document_id=DOC_ID, start_page=1, end_page=3)


def _read_call(
    call_id: str, start: int, end: int, name: str = "Annual_report_read_pages"
):
    return {
        "name": name,
        "args": {"source_document_id": DOC_ID, "start_page": start, "end_page": end},
        "id": call_id,
        "type": "tool_call",
    }


def _turn(*reads: tuple[str, int, int, str]) -> list[Any]:
    """A user message followed by page reads, each (call id, start, end, status)."""
    messages: list[Any] = [HumanMessage(content="question")]
    for call_id, start, end, status in reads:
        messages.append(
            AIMessage(content="", tool_calls=[_read_call(call_id, start, end)])
        )
        messages.append(
            ToolMessage(content="pages", tool_call_id=call_id, status=status)
        )
    return messages


class TestPageBudget:
    def test_counts_successful_reads_since_the_last_user_message(self):
        earlier_turn = _turn(("old", 1, 20, "success"))
        messages = earlier_turn + _turn(
            ("a", 1, 3, "success"), ("b", 10, 19, "error"), ("c", 40, 44, "success")
        )

        assert pages_read_this_turn(messages, "Annual_report_read_pages") == 8

    def test_ignores_other_tools(self):
        messages = _turn(("a", 1, 3, "success"))
        messages[1] = AIMessage(
            content="", tool_calls=[_read_call("a", 1, 3, name="other_tool")]
        )

        assert pages_read_this_turn(messages, "Annual_report_read_pages") == 0

    @pytest.mark.asyncio
    async def test_a_read_past_the_budget_is_refused(self, sdk):
        tool = create_read_pages_tool(_resource())
        state = AgentGraphState(
            messages=_turn(("a", 1, MAX_PAGES_PER_TURN - 2, "success"))
        )

        output = await tool.awrapper(  # type: ignore[attr-defined]
            tool, _read_call("next", 30, 32), state
        )

        assert isinstance(output, ToolMessage)
        assert output.status == "error"
        assert "Only 2 more pages can be read" in str(output.content)
        sdk.context_grounding.read_pages_async.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_read_within_the_budget_is_made(self, sdk):
        tool = create_read_pages_tool(_resource())
        state = AgentGraphState(
            messages=_turn(("a", 1, MAX_PAGES_PER_TURN - 2, "success"))
        )

        output = await tool.awrapper(  # type: ignore[attr-defined]
            tool, _read_call("next", 30, 31), state
        )

        assert isinstance(output, ToolMessage)
        assert output.status == "success"
        sdk.context_grounding.read_pages_async.assert_awaited_once()
