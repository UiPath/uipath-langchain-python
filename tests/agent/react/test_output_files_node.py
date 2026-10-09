"""Tests for the output-file verification node and its graph wiring."""

from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from uipath.agent.models.agent import AgentInternalToolResourceConfig
from uipath.agent.react import END_EXECUTION_TOOL, RAISE_ERROR_TOOL
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.attachments.output_files import get_output_file_fields
from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)
from uipath_langchain.agent.react.agent import create_agent
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.react.output_files_node import create_output_files_node
from uipath_langchain.agent.react.types import (
    AgentGraphConfig,
    AgentGraphNode,
    AgentGraphState,
)
from uipath_langchain.agent.tools.internal_tools.create_file_tool import (
    create_file_tool,
)
from uipath_langchain.agent.tools.internal_tools.schema_utils import (
    JOB_ATTACHMENT_DEFINITION,
)

from ..attachments.fake_orchestrator import patch_orchestrator

ATTACHMENT_ID = "11111111-1111-1111-1111-111111111111"
OTHER_ATTACHMENT_ID = "22222222-2222-2222-2222-222222222222"


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
                id="ai-1",
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
    """A current job where only ATTACHMENT_ID exists, already linked to it."""
    return patch_orchestrator(
        monkeypatch, existing={ATTACHMENT_ID: "report.md"}, linked=[ATTACHMENT_ID]
    )


class TestOutputFilesNode:
    async def test_valid_output_proceeds_to_termination(self, fields, linked_job):
        node = create_output_files_node(fields, max_retries=2)

        command = await node(state_ending_with({"summary": "s", "report": ticket()}))

        assert command.goto == AgentGraphNode.TERMINATE
        (message,) = command.update["messages"]
        assert message.tool_calls[0]["args"]["report"]["ID"] == ATTACHMENT_ID

    async def test_a_verified_reference_lands_in_the_registry(self, fields, linked_job):
        """Agent-output guardrails resolve files through the registry, so a file a
        tool never registered is still inspected once Orchestrator vouched for it."""
        node = create_output_files_node(fields, max_retries=2)

        command = await node(state_ending_with({"summary": "s", "report": ticket()}))

        registered = command.update["inner_state"]["job_attachments"]
        assert list(registered) == [ATTACHMENT_ID]
        assert registered[ATTACHMENT_ID].full_name == "report.md"

    async def test_nothing_is_registered_without_a_job_key(self, fields, monkeypatch):
        monkeypatch.delenv("UIPATH_JOB_KEY", raising=False)
        node = create_output_files_node(fields, max_retries=2)

        command = await node(state_ending_with({"summary": "s", "report": ticket()}))

        assert command.goto == AgentGraphNode.TERMINATE
        assert command.update["inner_state"]["job_attachments"] == {}

    async def test_an_edited_reference_reaches_termination_rebuilt(
        self, fields, linked_job
    ):
        node = create_output_files_node(fields, max_retries=2)
        edited = {**ticket(), "FullName": "/report.md"}

        command = await node(state_ending_with({"summary": "s", "report": edited}))

        (message,) = command.update["messages"]
        assert message.id == "ai-1"
        assert message.tool_calls[0]["args"]["report"]["FullName"] == "report.md"

    async def test_missing_required_file_returns_a_corrective_tool_message(
        self, fields, linked_job
    ):
        node = create_output_files_node(fields, max_retries=2)

        command = await node(state_ending_with({"summary": "s"}))

        assert command.goto == AgentGraphNode.AGENT
        message = command.update["messages"][0]
        assert isinstance(message, ToolMessage)
        assert message.tool_call_id == "call-1"
        assert message.status == "error"
        assert "'report'" in message.content
        assert command.update["inner_state"]["output_file_retries"] == 1

    async def test_unknown_attachment_returns_a_corrective_tool_message(
        self, fields, linked_job
    ):
        node = create_output_files_node(fields, max_retries=2)

        command = await node(
            state_ending_with({"summary": "s", "report": ticket(OTHER_ATTACHMENT_ID)})
        )

        assert command.goto == AgentGraphNode.AGENT
        assert OTHER_ATTACHMENT_ID in command.update["messages"][0].content
        assert command.update["inner_state"]["output_file_retries"] == 1

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

        command = await node(state_ending_with({"summary": "s"}))

        assert command.goto == AgentGraphNode.TERMINATE

    async def test_reaching_the_node_without_end_execution_is_loud(
        self, fields, linked_job
    ):
        """The router never sends anything else here. Passing the output through
        would skip verification without saying so, so this raises instead."""
        node = create_output_files_node(fields, max_retries=2)

        with pytest.raises(AgentRuntimeError) as exc_info:
            await node(
                state_ending_with({"message": "boom"}, tool_name=RAISE_ERROR_TOOL.name)
            )

        assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.ROUTING_ERROR
        )
        assert exc_info.value.error_info.category == UiPathErrorCategory.SYSTEM


def make_tool():
    return create_file_tool(
        AgentInternalToolResourceConfig.model_validate(
            {
                "$resourceType": "tool",
                "type": "Internal",
                "name": "Create File",
                "description": "Make a file.",
                "properties": {"toolType": "create-file"},
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "fileName": {"type": "string"},
                        "content": {"type": "string"},
                        "filePath": {"type": "string"},
                    },
                    "required": ["fileName"],
                },
            }
        )
    )


class TestGraphWiring:
    def build(self, schema: dict[str, Any], *, enabled: bool = True, tools=None):
        return create_agent(
            model=GenericFakeChatModel(messages=iter([])),
            tools=tools if tools is not None else [make_tool()],
            messages=[SystemMessage(content="sys"), HumanMessage(content="go")],
            output_schema=create_model(schema),
            config=AgentGraphConfig(output_files_enabled=enabled),
        ).compile()

    def test_file_output_adds_the_verification_node(self):
        graph = self.build(output_schema())

        assert AgentGraphNode.VERIFY_OUTPUT_FILES in graph.get_graph().nodes

    def test_no_file_output_leaves_the_graph_unchanged(self):
        graph = self.build(
            {"type": "object", "properties": {"summary": {"type": "string"}}}
        )

        assert AgentGraphNode.VERIFY_OUTPUT_FILES not in graph.get_graph().nodes

    def test_disabled_flag_leaves_the_graph_unchanged(self):
        graph = self.build(output_schema(), enabled=False)

        assert AgentGraphNode.VERIFY_OUTPUT_FILES not in graph.get_graph().nodes

    def test_a_file_producing_tool_other_than_ours_is_still_verified(self):
        """Any tool can return a real ticket, so the gate cannot key off ours."""
        graph = self.build(output_schema(), tools=[])

        assert AgentGraphNode.VERIFY_OUTPUT_FILES in graph.get_graph().nodes

    def test_verification_can_reach_both_terminate_and_agent(self):
        edges = self.build(output_schema()).get_graph().edges
        targets = {
            edge.target
            for edge in edges
            if edge.source == AgentGraphNode.VERIFY_OUTPUT_FILES
        }

        assert AgentGraphNode.TERMINATE in targets
        assert AgentGraphNode.AGENT in targets
