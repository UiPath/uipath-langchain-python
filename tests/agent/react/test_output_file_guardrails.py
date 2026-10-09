"""A whole agent run that creates a file and returns it, judged by a files-only guardrail
after each LLM call and at agent output."""

import uuid
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.content import create_tool_call
from uipath.core.guardrails import (
    GuardrailValidationResult,
    GuardrailValidationResultType,
)
from uipath.platform.guardrails import BuiltInValidatorGuardrail, GuardrailAttachment

from uipath_langchain.agent.guardrails.actions import BlockAction
from uipath_langchain.agent.react.agent import create_agent
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.react.types import AgentGraphConfig
from uipath_langchain.agent.tools.internal_tools.create_file_tool import (
    create_file_tool_name,
)
from uipath_langchain.agent.tools.internal_tools.schema_utils import (
    JOB_ATTACHMENT_DEFINITION,
)

from ..attachments.fake_orchestrator import patch_orchestrator
from .test_output_files_node import make_tool

_INPUT_ID = "0b6f3a2d-5e4c-4b1a-8f9e-1d2c3b4a5f60"
_OUTPUT_ID = "5d1e2f3a-4b5c-4d6e-8f70-819a2b3c4d5e"
_CODE = "import pygame\npygame.init()\n"


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "definitions": {"job-attachment": JOB_ATTACHMENT_DEFINITION},
    }


_INPUT_SCHEMA = _schema({"cv": {"$ref": "#/definitions/job-attachment"}}, ["cv"])
_OUTPUT_SCHEMA = _schema(
    {"summary": {"type": "string"}, "code": {"$ref": "#/definitions/job-attachment"}},
    ["summary", "code"],
)


def _protected_code_in_files() -> BuiltInValidatorGuardrail:
    return BuiltInValidatorGuardrail.model_validate(
        {
            "$guardrailType": "builtInValidator",
            "id": "ip-code-files",
            "name": "Protected code in files",
            "description": "Blocks protected code in the files the agent produces.",
            "enabledForEvals": True,
            "selector": {"scopes": ["Llm", "Agent"]},
            "validatorType": "intellectual_property",
            "validatorParameters": [
                {"$parameterType": "enum-list", "id": "ipEntities", "value": ["Code"]},
                {"$parameterType": "enum", "id": "appliesTo", "value": "Files"},
            ],
        }
    )


class _RecordingGuardrails:
    def __init__(self) -> None:
        self.attachments: list[list[GuardrailAttachment] | None] = []

    def evaluate_guardrail(
        self, text, guardrail, *, attachments=None, termination_mode=None
    ):
        self.attachments.append(attachments)
        return GuardrailValidationResult(
            result=GuardrailValidationResultType.PASSED, reason="clean"
        )


class _StubChatAnthropic(ChatAnthropic):
    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)


def _scripted_model(file_tool: str) -> Any:
    model: Any = _StubChatAnthropic.model_construct(model="claude-opus-5-5")
    model.model_details = {}
    model.bind_tools = Mock(return_value=model)
    model.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(
                content="",
                tool_calls=[
                    create_tool_call(
                        name=file_tool,
                        args={"fileName": "code_sample.py", "content": _CODE},
                        id="c1",
                    )
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    create_tool_call(
                        name="end_execution",
                        args={
                            "summary": "Saved the code sample.",
                            "code": {
                                "ID": _OUTPUT_ID,
                                "FullName": "code_sample.py",
                                "MimeType": "text/x-python",
                            },
                        },
                        id="c2",
                    )
                ],
            ),
        ]
    )
    return model


@pytest.fixture
def recorded(monkeypatch) -> _RecordingGuardrails:
    patch_orchestrator(
        monkeypatch,
        existing={_INPUT_ID: "cv.pdf", _OUTPUT_ID: "code_sample.py"},
        linked=[_INPUT_ID, _OUTPUT_ID],
    )

    class FakeJobs:
        async def create_attachment_async(self, **_: Any) -> uuid.UUID:
            return uuid.UUID(_OUTPUT_ID)

    class FakeUiPath:
        jobs = FakeJobs()

    monkeypatch.setattr(
        "uipath_langchain.agent.tools.internal_tools.create_file_tool.UiPath",
        lambda *args, **kwargs: FakeUiPath(),
    )

    guardrails = _RecordingGuardrails()

    class GuardrailsUiPath:
        def __init__(self) -> None:
            self.guardrails = guardrails

    monkeypatch.setattr(
        "uipath_langchain.agent.guardrails.guardrail_nodes.UiPath", GuardrailsUiPath
    )
    return guardrails


@pytest.mark.asyncio
async def test_each_check_receives_only_the_files_its_payload_references(
    recorded: _RecordingGuardrails,
) -> None:
    tool = make_tool()
    file_tool = create_file_tool_name([tool])
    assert file_tool is not None
    graph: Any = create_agent(
        _scripted_model(file_tool),
        [tool],
        [SystemMessage(content="sys"), HumanMessage(content="Save the code.")],
        input_schema=create_model(_INPUT_SCHEMA),
        output_schema=create_model(_OUTPUT_SCHEMA),
        guardrails=[(_protected_code_in_files(), BlockAction("protected code"))],
        config=AgentGraphConfig(output_files_enabled=True),
    ).compile()

    result = await graph.ainvoke(
        {
            "cv": {
                "ID": _INPUT_ID,
                "FullName": "cv.pdf",
                "MimeType": "application/pdf",
            }
        }
    )

    assert result["code"]["ID"] == _OUTPUT_ID
    sent = [
        None if refs is None else [ref.file_name for ref in refs]
        for refs in recorded.attachments
    ]
    assert sent == [
        # LLM output, turn 1: create-file carries the code inline, no file yet.
        [],
        # LLM output, turn 2: end_execution references the created file.
        ["code_sample.py"],
        # Agent output: the created file, not the uploaded input.
        ["code_sample.py"],
    ]
