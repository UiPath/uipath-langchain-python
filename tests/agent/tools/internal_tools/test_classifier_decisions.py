"""Tests for the classifier tool with the OpenAI Decisions provider."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import tool_call
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from uipath.agent.models.agent import AgentInternalToolResourceConfig
from uipath.llm_client import UiPathAPIError
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
    AgentStartupError,
)
from uipath_langchain.agent.multimodal import FileInfo
from uipath_langchain.agent.tools.internal_tools.classifier.decisions import (
    build_decisions_input,
)
from uipath_langchain.agent.tools.internal_tools.internal_tool_factory import (
    create_internal_tool,
)
from uipath_langchain.agent.tools.static_args import StaticArgsHandler

PACKAGE = "uipath_langchain.agent.tools.internal_tools.classifier"

pytestmark = pytest.mark.usefixtures("_passthrough_mockable")

DEPARTMENT: dict[str, Any] = {
    "name": "department",
    "type": "choice",
    "instructions": "Which department should handle this complaint?",
    "choices": [
        {"value": "billing", "description": "Payments, invoices, and refunds."},
        {"value": "technical"},
    ],
}
FRUSTRATION: dict[str, Any] = {
    "name": "frustration",
    "type": "score",
    "instructions": "How frustrated the customer appears",
    "levels": [
        {"label": "Calm", "description": "No frustration"},
        {"label": "Annoyed"},
        {"label": "Furious"},
    ],
}
IS_URGENT: dict[str, Any] = {
    "name": "is_urgent",
    "type": "predicate",
    "instructions": "The customer needs an answer today.",
}
QUESTIONS = [DEPARTMENT, FRUSTRATION, IS_URGENT]

DECISIONS_RESPONSE: dict[str, Any] = {
    "answers": [
        {
            "type": "choice",
            "name": "department",
            "choice": "billing",
            "confidence": 0.93,
            "probabilities": [
                {"value": "billing", "probability": 0.93},
                {"value": "technical", "probability": 0.07},
            ],
        },
        {
            "type": "score",
            "name": "frustration",
            "score": 1.4,
            "confidence": 0.61,
            "probabilities": [
                {"value": 0, "label": "Calm", "probability": 0.1},
                {"value": 1, "label": "Annoyed", "probability": 0.4},
                {"value": 2, "label": "Furious", "probability": 0.5},
            ],
        },
        {"type": "predicate", "name": "is_urgent", "probability": 0.82},
    ]
}

QUESTIONS_LIST_SCHEMA: dict[str, Any] = {
    "type": "array",
    "minItems": 1,
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "type": {"type": "string", "enum": ["choice", "score", "predicate"]},
            "instructions": {"type": "string"},
            "choices": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["value"],
                },
            },
            "levels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["label"],
                },
            },
        },
        "required": ["name", "type", "instructions"],
    },
}

JOB_ATTACHMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ID": {"type": "string"},
        "FullName": {"type": "string"},
        "MimeType": {"type": "string"},
    },
    "required": ["ID"],
    "x-uipath-resource-kind": "JobAttachment",
}

IMAGES_SCHEMA: dict[str, Any] = {
    "type": "array",
    "description": "Images to classify with the input.",
    "items": {"$ref": "#/definitions/job-attachment"},
}


def _static(value: Any) -> dict[str, Any]:
    return {"variant": "static", "value": value, "isSensitive": False}


def _input_schema(
    input_schema: dict[str, Any] | None = None,
    questions: dict[str, Any] | None = None,
    images: bool = True,
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "input": input_schema or {"type": "string", "description": "The ticket"},
        "questions": questions or QUESTIONS_LIST_SCHEMA,
    }
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": ["input", "questions"],
    }
    if images:
        properties["images"] = IMAGES_SCHEMA
        schema["definitions"] = {"job-attachment": JOB_ATTACHMENT_SCHEMA}
    return schema


def _resource(
    argument_properties: dict[str, Any] | None = None,
    input_schema: dict[str, Any] | None = None,
) -> AgentInternalToolResourceConfig:
    return AgentInternalToolResourceConfig.model_validate(
        {
            "$resourceType": "tool",
            "name": "Classify Ticket",
            "description": "Classify a support ticket",
            "type": "Internal",
            "inputSchema": input_schema or _input_schema(),
            "outputSchema": {"type": "object", "properties": {}},
            "properties": {
                "toolType": "classifier",
                "settings": {"provider": "openai", "model": "gpt-6-luna"},
            },
            "argumentProperties": (
                {"$['questions']": _static(QUESTIONS)}
                if argument_properties is None
                else argument_properties
            ),
        }
    )


@pytest.fixture
def _passthrough_mockable():
    with patch(f"{PACKAGE}.tool.mockable", lambda **kwargs: lambda f: f):
        yield


@pytest.fixture
def decisions_client() -> Any:
    client = MagicMock()
    client.acreate = AsyncMock(return_value=DECISIONS_RESPONSE)
    with patch(
        f"{PACKAGE}.decisions.UiPathDecisionsClient", return_value=client
    ) as client_cls:
        client.cls = client_cls
        yield client


class _AgentInput(BaseModel):
    ticket: str = "I was charged twice!"


def _args(tool: BaseTool, llm_args: dict[str, Any]) -> dict[str, Any]:
    """The arguments as the tool node passes them: LLM args plus configured ones."""
    handler = StaticArgsHandler()
    handler.initialize([tool], _AgentInput(), _AgentInput)
    call = tool_call(name=tool.name, args=llm_args, id="call-1")
    handler.apply_to_response([call])
    return call["args"]


async def _tool_message(tool: BaseTool, args: dict[str, Any]) -> ToolMessage:
    result = await tool.ainvoke(
        tool_call(name=tool.name, args=_args(tool, args), id="call-1")
    )
    assert isinstance(result, ToolMessage)
    return result


def _decisions_error(status: int, body: Any) -> UiPathAPIError:
    response = httpx.Response(
        status,
        json=body,
        request=httpx.Request("POST", "https://api.openai.com/v1/decisions"),
    )
    return UiPathAPIError.from_response(response)


# --- Request -------------------------------------------------------------------


async def test_static_questions_reach_decisions_in_its_shapes(
    decisions_client: Any,
) -> None:
    tool = create_internal_tool(_resource(), AsyncMock())

    await tool.ainvoke(_args(tool, {"input": "I was charged twice!"}))

    decisions_client.cls.assert_called_once_with(model_name="gpt-6-luna", timeout=60.0)
    sent_input, sent_questions = decisions_client.acreate.await_args.args
    assert sent_input == "I was charged twice!"
    assert sent_questions == [
        {
            "type": "choice",
            "name": "department",
            "instructions": "Which department should handle this complaint?",
            "choices": [
                {"value": "billing", "description": "Payments, invoices, and refunds."},
                {"value": "technical"},
            ],
        },
        {
            "type": "score",
            "name": "frustration",
            "instructions": "How frustrated the customer appears",
            "levels": [
                {"label": "Calm", "description": "No frustration"},
                {"label": "Annoyed"},
                {"label": "Furious"},
            ],
        },
        {
            "type": "predicate",
            "name": "is_urgent",
            "instructions": "The customer needs an answer today.",
        },
    ]


async def test_object_input_is_sent_as_json_text(decisions_client: Any) -> None:
    tool = create_internal_tool(
        _resource(input_schema=_input_schema({"type": "object"}, images=False)),
        AsyncMock(),
    )

    await tool.ainvoke(
        _args(tool, {"input": {"subject": "Refund", "body": "Charged twice"}})
    )

    sent_input, _ = decisions_client.acreate.await_args.args
    assert sent_input == '{"subject": "Refund", "body": "Charged twice"}'


async def test_answers_are_keyed_by_name_in_decisions_shapes(
    decisions_client: Any,
) -> None:
    tool = create_internal_tool(_resource(), AsyncMock())

    result = await tool.ainvoke(_args(tool, {"input": "I was charged twice!"}))

    assert result == {
        "department": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.93,
            "probabilities": [
                {"value": "billing", "probability": 0.93},
                {"value": "technical", "probability": 0.07},
            ],
        },
        "frustration": {
            "type": "score",
            "score": 1.4,
            "confidence": 0.61,
            "probabilities": DECISIONS_RESPONSE["answers"][1]["probabilities"],
        },
        "is_urgent": {"type": "predicate", "probability": 0.82},
    }


async def test_builder_questions_use_decisions_fields(decisions_client: Any) -> None:
    builder_schema = {
        "type": "object",
        "required": ["department", "is_urgent"],
        "properties": {
            "department": {
                "type": "object",
                "required": ["type", "instructions", "choices"],
                "properties": {
                    "type": {"type": "string", "enum": ["choice"]},
                    "instructions": {"type": "string"},
                    "choices": QUESTIONS_LIST_SCHEMA["items"]["properties"]["choices"],
                },
            },
            "is_urgent": {
                "type": "object",
                "required": ["type", "instructions"],
                "properties": {
                    "type": {"type": "string", "enum": ["predicate"]},
                    "instructions": {"type": "string"},
                },
            },
        },
    }
    tool = create_internal_tool(
        _resource(
            {
                "$['questions']": {"variant": "objectBuilder"},
                "$['questions']['department']": {"variant": "objectBuilder"},
                "$['questions']['department']['instructions']": _static(
                    DEPARTMENT["instructions"]
                ),
                "$['questions']['department']['choices']": _static(
                    DEPARTMENT["choices"]
                ),
                "$['questions']['is_urgent']": {"variant": "objectBuilder"},
                "$['questions']['is_urgent']['instructions']": _static(
                    IS_URGENT["instructions"]
                ),
            },
            input_schema=_input_schema(questions=builder_schema),
        ),
        AsyncMock(),
    )
    decisions_client.acreate.return_value = {
        "answers": [DECISIONS_RESPONSE["answers"][0], DECISIONS_RESPONSE["answers"][2]]
    }

    result = await tool.ainvoke(
        {
            "input": "I was charged twice!",
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": DEPARTMENT["instructions"],
                    "choices": DEPARTMENT["choices"],
                },
                "is_urgent": {
                    "type": "predicate",
                    "instructions": IS_URGENT["instructions"],
                },
            },
        }
    )

    _, sent_questions = decisions_client.acreate.await_args.args
    assert [question["name"] for question in sent_questions] == [
        "department",
        "is_urgent",
    ]
    assert set(result) == {"department", "is_urgent"}


# --- Images --------------------------------------------------------------------


def test_build_input_with_images_is_one_user_message() -> None:
    assert build_decisions_input(
        "Is this a receipt?", ["data:image/png;base64,AA"]
    ) == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Is this a receipt?"},
                {"type": "input_image", "image_url": "data:image/png;base64,AA"},
            ],
        }
    ]
    assert build_decisions_input("", ["data:image/png;base64,AA"]) == [
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,AA"}
            ],
        }
    ]


ATTACHMENT = {
    "ID": "6f1d2c3b-0000-4000-8000-000000000001",
    "FullName": "receipt.png",
    "MimeType": "image/png",
}


async def test_images_reach_decisions_as_data_urls(decisions_client: Any) -> None:
    tool = create_internal_tool(_resource(), AsyncMock())
    file = FileInfo(
        url="https://blob/receipt.png", name="receipt.png", mime_type="image/png"
    )

    with (
        patch(
            f"{PACKAGE}.tool.resolve_attachments_to_file_infos",
            AsyncMock(return_value=[file]),
        ) as resolve,
        patch(f"{PACKAGE}.tool.download_file_base64", AsyncMock(return_value="QUJD")),
    ):
        await tool.ainvoke(_args(tool, {"input": "", "images": [ATTACHMENT]}))

    assert resolve.await_args is not None
    [attachments] = resolve.await_args.args
    assert attachments[0]["ID"] == ATTACHMENT["ID"]
    sent_input, _ = decisions_client.acreate.await_args.args
    assert sent_input == [
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,QUJD"}
            ],
        }
    ]


async def test_a_file_that_is_not_an_image_is_a_tool_error(
    decisions_client: Any,
) -> None:
    tool = create_internal_tool(_resource(), AsyncMock())
    file = FileInfo(url="https://blob/a.pdf", name="a.pdf", mime_type="application/pdf")

    with patch(
        f"{PACKAGE}.tool.resolve_attachments_to_file_infos",
        AsyncMock(return_value=[file]),
    ):
        message = await _tool_message(
            tool, {"input": "text", "images": [{**ATTACHMENT, "FullName": "a.pdf"}]}
        )

    assert message.status == "error"
    assert "'a.pdf'" in message.content
    assert "is not an image" in message.content
    decisions_client.acreate.assert_not_awaited()


async def test_empty_input_without_images_is_a_tool_error(
    decisions_client: Any,
) -> None:
    tool = create_internal_tool(_resource(), AsyncMock())

    message = await _tool_message(tool, {"input": "  "})

    assert message.status == "error"
    assert "Argument 'input' must be a non-empty string" in message.content


# --- Errors --------------------------------------------------------------------


async def test_refusal_is_a_tool_error(decisions_client: Any) -> None:
    decisions_client.acreate.return_value = {
        "answers": [
            DECISIONS_RESPONSE["answers"][0],
            {"type": "refusal", "name": "frustration"},
            DECISIONS_RESPONSE["answers"][2],
        ]
    }
    tool = create_internal_tool(_resource(), AsyncMock())

    message = await _tool_message(tool, {"input": "text"})

    assert message.status == "error"
    assert "refused to answer the question(s) 'frustration'" in message.content


@pytest.mark.parametrize(
    ("answers", "detail"),
    [
        pytest.param({"answers": {}}, "Expected a list of answers", id="not-a-list"),
        pytest.param(
            {"answers": DECISIONS_RESPONSE["answers"][:2]},
            "Question 'is_urgent': no answer returned",
            id="missing",
        ),
        pytest.param(
            {
                "answers": [
                    {**DECISIONS_RESPONSE["answers"][0], "probabilities": {"a": 1}},
                    *DECISIONS_RESPONSE["answers"][1:],
                ]
            },
            "malformed answer (probabilities=",
            id="probabilities-not-a-list",
        ),
        pytest.param(
            {
                "answers": [
                    *DECISIONS_RESPONSE["answers"][:2],
                    {"type": "predicate", "name": "is_urgent", "probability": "high"},
                ]
            },
            "malformed answer (probability='high')",
            id="probability-not-a-number",
        ),
    ],
)
async def test_malformed_response_is_an_invalid_response_error(
    decisions_client: Any, answers: dict[str, Any], detail: str
) -> None:
    decisions_client.acreate.return_value = answers
    tool = create_internal_tool(_resource(), AsyncMock())

    with pytest.raises(AgentRuntimeError) as exc_info:
        await tool.ainvoke(_args(tool, {"input": "text"}))

    error = exc_info.value.error_info
    assert error.code == AgentRuntimeError.full_code(
        AgentRuntimeErrorCode.LLM_INVALID_RESPONSE
    )
    assert error.title == "Invalid OpenAI Decisions response"
    assert detail in error.detail


async def test_unknown_model_is_a_user_runtime_error(decisions_client: Any) -> None:
    decisions_client.acreate.side_effect = _decisions_error(
        404,
        {"error": {"code": "model_not_found", "message": "The model does not exist"}},
    )
    tool = create_internal_tool(_resource(), AsyncMock())

    with pytest.raises(AgentRuntimeError) as exc_info:
        await tool.ainvoke(_args(tool, {"input": "text"}))

    error = exc_info.value.error_info
    assert error.category == UiPathErrorCategory.USER
    assert error.title == "Unknown OpenAI Decisions model"
    assert "gpt-6-luna" in error.detail


async def test_input_rejected_for_llm_content_is_a_tool_error(
    decisions_client: Any,
) -> None:
    decisions_client.acreate.side_effect = _decisions_error(
        400, {"error": {"message": "Input is too long."}}
    )
    tool = create_internal_tool(_resource(), AsyncMock())

    message = await _tool_message(tool, {"input": "text"})

    assert message.status == "error"
    assert "OpenAI Decisions rejected the input (HTTP 400): Input is too long." in (
        message.content
    )


@pytest.mark.parametrize(
    ("argument_properties", "input_schema"),
    [
        pytest.param(
            {
                "$['questions']": _static(
                    [
                        {
                            "name": "department",
                            "type": "choice",
                            "instructions": "Which team?",
                            "options": [{"name": "a"}, {"name": "b"}],
                        }
                    ]
                )
            },
            None,
            id="jev-options",
        ),
        pytest.param(
            {
                "$['questions']": _static(
                    [{"name": "urgent", "type": "noul", "instructions": "Urgent?"}]
                )
            },
            None,
            id="jev-type",
        ),
        pytest.param(
            None,
            {
                "type": "object",
                "properties": {
                    "state": {"type": "string"},
                    "questions": QUESTIONS_LIST_SCHEMA,
                },
                "required": ["state", "questions"],
            },
            id="jev-state-argument",
        ),
        pytest.param(
            None,
            {
                **_input_schema(images=False),
                "properties": {
                    **_input_schema(images=False)["properties"],
                    "images": {"type": "string"},
                },
            },
            id="images-not-a-list",
        ),
    ],
)
def test_jev_shapes_fail_at_startup(
    argument_properties: dict[str, Any] | None, input_schema: dict[str, Any] | None
) -> None:
    with pytest.raises(AgentStartupError):
        create_internal_tool(_resource(argument_properties, input_schema), AsyncMock())
