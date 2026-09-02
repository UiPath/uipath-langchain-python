"""Tests for the output-file verification node and its graph wiring."""

from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from uipath.agent.react import END_EXECUTION_TOOL, RAISE_ERROR_TOOL
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.attachments.output_files import get_output_file_fields
from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)
from uipath_langchain.agent.react.agent import create_agent
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.react.output_files_node import (
    create_output_files_node,
    output_files_verified,
)
from uipath_langchain.agent.react.types import AgentGraphNode, AgentGraphState
from uipath_langchain.agent.tools.internal_tools.output_file_tool import (
    OUTPUT_FILE_TOOL_NAME,
    create_output_file_tool,
)
from uipath_langchain.agent.tools.internal_tools.schema_utils import (
    JOB_ATTACHMENT_DEFINITION,
)

ATTACHMENT_ID = "11111111-1111-1111-1111-111111111111"
OTHER_ATTACHMENT_ID = "22222222-2222-2222-2222-222222222222"
JOB_KEY = "33333333-3333-3333-3333-333333333333"


def output_schema(required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "report": {
                "$ref": "#/definitions/job-attachment",
                "description": "The generated report",
            },
        },
        "required": required if required is not None else ["summary", "report"],
        "definitions": {"job-attachment": JOB_ATTACHMENT_DEFINITION},
    }


def ticket(attachment_id: str = ATTACHMENT_ID) -> dict[str, str]:
    return {
        "ID": attachment_id,
        "FullName": "report.md",
        "MimeType": "text/markdown",
    }


def state_ending_with(args: dict[str, Any], *, tool_name: str | None = None) -> Any:
    """State whose latest AI message calls a flow-control tool with ``args``."""
    return AgentGraphState(
        messages=[
            HumanMessage(content="go"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool_name or END_EXECUTION_TOOL.name,
                        "args": args,
                        "id": "call-1",
                    }
                ],
            ),
        ]
    )


@pytest.fixture
def fields():
    return get_output_file_fields(create_model(output_schema()))


@pytest.fixture
def linked_job(monkeypatch):
    """A current job whose only linked attachment is ATTACHMENT_ID."""
    monkeypatch.setenv("UIPATH_JOB_KEY", JOB_KEY)
    monkeypatch.delenv("UIPATH_FOLDER_KEY", raising=False)

    class FakeJobs:
        async def list_attachments_async(self, **kwargs: Any) -> list[str]:
            return [ATTACHMENT_ID]

    class FakeUiPath:
        jobs = FakeJobs()

    monkeypatch.setattr(
        "uipath_langchain.agent.attachments.output_files.UiPath",
        lambda *args, **kwargs: FakeUiPath(),
    )


class TestOutputFilesNode:
    async def test_valid_output_clears_the_problem(self, fields, linked_job):
        node = create_output_files_node(fields, max_retries=2)

        update = await node(state_ending_with({"summary": "s", "report": ticket()}))

        assert update["inner_state"]["output_file_problem"] is None
        assert "messages" not in update

    async def test_missing_required_file_returns_a_corrective_tool_message(
        self, fields, linked_job
    ):
        node = create_output_files_node(fields, max_retries=2)

        update = await node(state_ending_with({"summary": "s"}))

        message = update["messages"][0]
        assert isinstance(message, ToolMessage)
        assert message.tool_call_id == "call-1"
        assert message.status == "error"
        assert OUTPUT_FILE_TOOL_NAME in message.content
        assert "'report'" in message.content
        assert update["inner_state"]["output_file_retries"] == 1

    async def test_unlinked_attachment_returns_a_corrective_tool_message(
        self, fields, linked_job
    ):
        node = create_output_files_node(fields, max_retries=2)

        update = await node(
            state_ending_with({"summary": "s", "report": ticket(OTHER_ATTACHMENT_ID)})
        )

        assert OTHER_ATTACHMENT_ID in update["messages"][0].content
        assert update["inner_state"]["output_file_retries"] == 1

    async def test_retries_are_capped_then_the_run_faults(self, fields, linked_job):
        node = create_output_files_node(fields, max_retries=2)
        state = state_ending_with({"summary": "s"})
        state.inner_state.output_file_retries = 2

        with pytest.raises(AgentRuntimeError) as exc_info:
            await node(state)

        assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.OUTPUT_VALIDATION_ERROR
        )
        assert exc_info.value.error_info.category == UiPathErrorCategory.USER

    async def test_optional_file_field_left_empty_passes(self, linked_job):
        fields = get_output_file_fields(
            create_model(output_schema(required=["summary"]))
        )
        node = create_output_files_node(fields, max_retries=2)

        update = await node(state_ending_with({"summary": "s"}))

        assert update["inner_state"]["output_file_problem"] is None

    async def test_non_end_execution_call_is_left_alone(self, fields, linked_job):
        node = create_output_files_node(fields, max_retries=2)

        update = await node(
            state_ending_with({"message": "boom"}, tool_name=RAISE_ERROR_TOOL.name)
        )

        assert update["inner_state"]["output_file_problem"] is None
        assert "messages" not in update


class TestOutputFilesVerified:
    def test_cleared_problem_is_verified(self):
        assert output_files_verified(AgentGraphState()) is True

    def test_recorded_problem_is_not_verified(self):
        state = AgentGraphState()
        state.inner_state.output_file_problem = "missing"

        assert output_files_verified(state) is False


class TestGraphWiring:
    def build(self, schema: dict[str, Any]):
        return create_agent(
            model=GenericFakeChatModel(messages=iter([])),
            tools=[create_output_file_tool()],
            messages=[SystemMessage(content="sys"), HumanMessage(content="go")],
            output_schema=create_model(schema),
        ).compile()

    def test_file_output_adds_the_verification_node(self):
        graph = self.build(output_schema())

        assert AgentGraphNode.VERIFY_OUTPUT_FILES in graph.get_graph().nodes

    def test_no_file_output_leaves_the_graph_unchanged(self):
        graph = self.build(
            {"type": "object", "properties": {"summary": {"type": "string"}}}
        )

        assert AgentGraphNode.VERIFY_OUTPUT_FILES not in graph.get_graph().nodes

    def test_verification_can_reach_both_terminate_and_agent(self):
        edges = self.build(output_schema()).get_graph().edges
        targets = {
            edge.target
            for edge in edges
            if edge.source == AgentGraphNode.VERIFY_OUTPUT_FILES
        }

        assert AgentGraphNode.TERMINATE in targets
        assert AgentGraphNode.AGENT in targets
