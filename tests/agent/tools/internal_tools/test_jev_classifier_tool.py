"""Tests for jev_classifier_tool.py module."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from uipath.agent.models.agent import (
    AgentInternalJevClassifierSettings,
    AgentInternalJevClassifierToolProperties,
    AgentInternalToolResourceConfig,
)
from uipath.llm_client import UiPathAPIError

from uipath_langchain.agent.exceptions import AgentRuntimeError
from uipath_langchain.agent.tools.internal_tools.internal_tool_factory import (
    create_internal_tool,
)
from uipath_langchain.agent.tools.internal_tools.jev_classifier_tool import (
    build_jev_output_schema,
    build_jev_questions,
)

MODULE = "uipath_langchain.agent.tools.internal_tools.jev_classifier_tool"

pytestmark = pytest.mark.usefixtures("_passthrough_mockable")

QUESTIONS: list[dict[str, Any]] = [
    {
        "name": "department",
        "type": "choice",
        "instructions": "Which team should handle this",
        "options": [
            {"name": "billing", "description": "Payment issues"},
            {"name": "technical", "description": None},
        ],
    },
    {
        "name": "frustration",
        "type": "score",
        "instructions": "How frustrated the customer appears",
        "levels": ["Calm", "Frustrated", "Very angry"],
    },
    {"name": "is_urgent", "type": "noul", "instructions": "Is this urgent?"},
]

JEV_RESPONSE: dict[str, Any] = {
    "model": "jev-1.13.0",
    "answers": {
        "department": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.81,
            "probabilities": {"billing": 0.88, "technical": 0.12},
        },
        "frustration": {
            "type": "score",
            "score": 1.6,
            "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
            "probabilities": {"0": 0.0, "1": 0.4, "2": 0.6},
            "confidence": 0.7,
        },
        "is_urgent": {"type": "noul", "noul": 0.3},
    },
    "usage": {"input_tokens": 20, "output_tokens": 0},
}


@pytest.fixture
def _passthrough_mockable():
    with patch(f"{MODULE}.mockable", lambda **kwargs: lambda f: f):
        yield


@pytest.fixture
def resource() -> AgentInternalToolResourceConfig:
    return AgentInternalToolResourceConfig.model_validate(
        {
            "$resourceType": "tool",
            "name": "Classify Ticket",
            "description": "Classify a support ticket",
            "type": "Internal",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Ticket body"}
                },
                "required": ["text"],
            },
            "outputSchema": {"type": "object", "properties": {}},
            "properties": {
                "toolType": "jev-classifier",
                "settings": {"model": "jev-latest", "questions": QUESTIONS},
            },
        }
    )


def _settings(
    resource: AgentInternalToolResourceConfig,
) -> AgentInternalJevClassifierSettings:
    assert isinstance(resource.properties, AgentInternalJevClassifierToolProperties)
    return resource.properties.settings


@pytest.fixture
def jev_client() -> Any:
    client = MagicMock()
    client.asystem_one = AsyncMock(return_value=JEV_RESPONSE)
    with patch(f"{MODULE}.UiPathJevClient", return_value=client) as client_cls:
        client.cls = client_cls
        yield client


def test_factory_creates_tool_with_fixed_input_and_derived_output(
    resource: AgentInternalToolResourceConfig,
) -> None:
    tool = create_internal_tool(resource, AsyncMock())

    assert tool.name == "Classify_Ticket"
    assert tool.description == "Classify a support ticket"
    input_schema = tool.args_schema.model_json_schema()  # type: ignore[union-attr]
    assert list(input_schema["properties"]) == ["text"]
    assert input_schema["properties"]["text"]["description"] == "Ticket body"
    assert tool.metadata is not None
    output_schema = tool.metadata["output_schema"].model_json_schema()
    assert set(output_schema["properties"]) == {
        "department",
        "frustration",
        "is_urgent",
    }


def test_questions_payload_matches_typesafe_contract(
    resource: AgentInternalToolResourceConfig,
) -> None:
    assert build_jev_questions(_settings(resource)) == {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this",
            "criteria": {"billing": "Payment issues", "technical": None},
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated the customer appears",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
        "is_urgent": {"type": "noul", "instructions": "Is this urgent?"},
    }


def test_output_schema_lists_choices_and_levels(
    resource: AgentInternalToolResourceConfig,
) -> None:
    schema = build_jev_output_schema(_settings(resource))

    department = schema["properties"]["department"]
    assert department["properties"]["choice"]["enum"] == ["billing", "technical"]
    assert department["description"] == "Which team should handle this"
    frustration = schema["properties"]["frustration"]
    assert frustration["properties"]["level"]["enum"] == [
        "Calm",
        "Frustrated",
        "Very angry",
    ]
    assert schema["properties"]["is_urgent"]["required"] == ["noul", "answer"]
    assert schema["required"] == ["department", "frustration", "is_urgent"]


async def test_invoke_calls_jev_and_maps_answers(
    resource: AgentInternalToolResourceConfig, jev_client: Any
) -> None:
    tool = create_internal_tool(resource, AsyncMock())

    result = await tool.ainvoke({"text": "I was charged twice!"})

    jev_client.cls.assert_called_once_with(model_name="jev-latest")
    jev_client.asystem_one.assert_awaited_once()
    state, questions = jev_client.asystem_one.await_args.args
    assert state == "I was charged twice!"
    assert set(questions) == {"department", "frustration", "is_urgent"}
    assert result == {
        "department": {
            "choice": "billing",
            "confidence": 0.81,
            "probabilities": {"billing": 0.88, "technical": 0.12},
        },
        "frustration": {
            "score": 1.6,
            "level": "Very angry",
            "confidence": 0.7,
            "probabilities": {"0": 0.0, "1": 0.4, "2": 0.6},
        },
        "is_urgent": {"noul": 0.3, "answer": False},
    }


async def test_client_is_created_once(
    resource: AgentInternalToolResourceConfig, jev_client: Any
) -> None:
    tool = create_internal_tool(resource, AsyncMock())

    await tool.ainvoke({"text": "one"})
    await tool.ainvoke({"text": "two"})

    jev_client.cls.assert_called_once()
    assert jev_client.asystem_one.await_count == 2


async def test_empty_text_is_rejected(
    resource: AgentInternalToolResourceConfig, jev_client: Any
) -> None:
    tool = create_internal_tool(resource, AsyncMock())

    with pytest.raises(AgentRuntimeError):
        await tool.coroutine(text="  ")  # type: ignore[misc]
    jev_client.asystem_one.assert_not_awaited()


async def test_missing_answer_raises_runtime_error(
    resource: AgentInternalToolResourceConfig, jev_client: Any
) -> None:
    jev_client.asystem_one.return_value = {"answers": {}}
    tool = create_internal_tool(resource, AsyncMock())

    with pytest.raises(AgentRuntimeError, match="department"):
        await tool.ainvoke({"text": "text"})


async def test_http_error_is_mapped_to_runtime_error(
    resource: AgentInternalToolResourceConfig, jev_client: Any
) -> None:
    response = httpx.Response(
        429, request=httpx.Request("POST", "https://api.typesafe.ai/v1/systemone")
    )
    jev_client.asystem_one.side_effect = UiPathAPIError.from_response(response)
    tool = create_internal_tool(resource, AsyncMock())

    with pytest.raises(AgentRuntimeError):
        await tool.ainvoke({"text": "text"})


async def test_client_configuration_failure_is_runtime_error(
    resource: AgentInternalToolResourceConfig,
) -> None:
    with patch(f"{MODULE}.UiPathJevClient", side_effect=ValueError("no settings")):
        tool = create_internal_tool(resource, AsyncMock())

        with pytest.raises(AgentRuntimeError, match="configure access to Jev"):
            await tool.ainvoke({"text": "text"})
