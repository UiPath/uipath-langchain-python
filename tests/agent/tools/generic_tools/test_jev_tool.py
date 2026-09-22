"""Tests for the Jev generic tool."""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel
from pytest_httpx import HTTPXMock
from uipath.agent.models.agent import (
    AgentGenericToolProperties,
    AgentGenericToolResourceConfig,
)

from uipath_langchain.agent.exceptions import AgentRuntimeError, AgentStartupError
from uipath_langchain.agent.tools.generic_tools.jev import (
    JevClient,
    JevClientConfig,
    JevToolSettings,
    create_jev_tool,
)
from uipath_langchain.agent.tools.generic_tools.jev.jev_client import (
    build_gateway_url,
)

SYSTEM_ONE_URL = "https://api.typesafe.ai/v1/systemone"

SETTINGS: dict[str, Any] = {
    "model": "jev-1.13.0",
    "questions": [
        {
            "name": "department",
            "type": "choice",
            "instructions": "Which team should handle this",
            "options": [
                {"name": "billing", "description": "Payment issues"},
                {"name": "technical"},
            ],
        },
        {
            "name": "frustration",
            "type": "score",
            "instructions": "How frustrated the customer appears",
            "levels": ["Calm", "Frustrated", "Angry"],
        },
        {
            "name": "is_urgent",
            "type": "noul",
            "instructions": "The message conveys urgency",
            "trueDescription": "Needs action now",
        },
    ],
}

API_RESPONSE: dict[str, Any] = {
    "model": "jev-1.13.0",
    "answers": {
        "department": {
            "type": "choice",
            "choice": "technical",
            "confidence": 0.78,
            "probabilities": {"technical": 0.85, "billing": 0.15},
        },
        "frustration": {
            "type": "score",
            "score": 1.0,
            "confidence": 1.0,
            "legend": {"0": "Calm", "1": "Frustrated", "2": "Angry"},
            "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0},
        },
        "is_urgent": {"type": "noul", "noul": 0.97},
    },
    "usage": {"input_tokens": 392, "output_tokens": 65},
}

EXPECTED_OUTPUT: dict[str, Any] = {
    "department": {
        "choice": "technical",
        "confidence": 0.78,
        "probabilities": {"technical": 0.85, "billing": 0.15},
    },
    "frustration": {
        "score": 1.0,
        "confidence": 1.0,
        "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0},
    },
    "is_urgent": {"noul": 0.97},
}

pytestmark = pytest.mark.usefixtures("_passthrough_mockable", "_enable_jev")


@pytest.fixture
def _passthrough_mockable() -> Any:
    with patch(
        "uipath_langchain.agent.tools.generic_tools.jev.jev_tool.mockable",
        lambda **kwargs: lambda f: f,
    ):
        yield


@pytest.fixture
def _enable_jev(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UIPATH_FEATURE_EnableJevTool", "true")
    monkeypatch.delenv("UIPATH_FEATURE_EnableJevViaLlmGateway", raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)


def _resource(settings: dict[str, Any] | None = None) -> AgentGenericToolResourceConfig:
    return AgentGenericToolResourceConfig(
        name="Classify ticket",
        description="Classify a support ticket",
        input_schema={},
        properties=AgentGenericToolProperties(
            sub_type="jev", settings=SETTINGS if settings is None else settings
        ),
    )


class TestJevSettings:
    def test_api_questions(self) -> None:
        settings = JevToolSettings.model_validate(SETTINGS)
        assert settings.api_questions() == {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this",
                "criteria": {"billing": "Payment issues", "technical": None},
            },
            "frustration": {
                "type": "score",
                "instructions": "How frustrated the customer appears",
                "criteria": ["Calm", "Frustrated", "Angry"],
            },
            "is_urgent": {
                "type": "noul",
                "instructions": "The message conveys urgency",
                "criteria": {"true": "Needs action now"},
            },
        }

    def test_noul_without_descriptions_has_no_criteria(self) -> None:
        settings = JevToolSettings.model_validate(
            {"questions": [{"name": "q", "type": "noul", "instructions": "x"}]}
        )
        assert settings.model == "jev-latest"
        assert settings.api_questions() == {"q": {"type": "noul", "instructions": "x"}}

    def test_output_schema(self) -> None:
        schema = JevToolSettings.model_validate(SETTINGS).output_schema()
        assert schema["required"] == ["department", "frustration", "is_urgent"]
        department = schema["properties"]["department"]
        assert department["properties"]["choice"]["enum"] == ["billing", "technical"]
        assert department["description"] == "Which team should handle this"
        assert schema["properties"]["frustration"]["required"] == [
            "score",
            "confidence",
            "probabilities",
        ]
        assert schema["properties"]["is_urgent"]["required"] == ["noul"]

    @pytest.mark.parametrize(
        "settings",
        [
            {"questions": []},
            {"questions": [{"name": "q", "type": "unknown", "instructions": "x"}]},
            {
                "questions": [
                    {"name": "q", "type": "noul", "instructions": "x"},
                    {"name": "q", "type": "noul", "instructions": "y"},
                ]
            },
            {
                "questions": [
                    {
                        "name": "q",
                        "type": "choice",
                        "instructions": "x",
                        "options": [{"name": "only_one"}],
                    }
                ]
            },
            {
                "questions": [
                    {
                        "name": "q",
                        "type": "choice",
                        "instructions": "x",
                        "options": [{"name": "a"}, {"name": "a"}],
                    }
                ]
            },
            {
                "questions": [
                    {"name": "q", "type": "score", "instructions": "x", "levels": ["a"]}
                ]
            },
            {"questions": [{"name": "1bad", "type": "noul", "instructions": "x"}]},
        ],
    )
    def test_invalid_settings_fail_startup(self, settings: dict[str, Any]) -> None:
        with pytest.raises(AgentStartupError):
            create_jev_tool(_resource(settings), MagicMock())


class TestJevTool:
    def test_feature_flag_disabled_fails_startup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("UIPATH_FEATURE_EnableJevTool", "false")
        with pytest.raises(AgentStartupError) as exc_info:
            create_jev_tool(_resource(), MagicMock())
        assert "EnableJevTool" in str(exc_info.value.error_info.detail)

    def test_tool_schema(self) -> None:
        tool = create_jev_tool(_resource(), MagicMock())
        assert tool.name == "Classify_ticket"
        assert tool.description == "Classify a support ticket"
        args_schema = tool.args_schema
        assert isinstance(args_schema, type) and issubclass(args_schema, BaseModel)
        assert list(args_schema.model_json_schema()["properties"]) == ["state"]

    async def test_classifies_with_api_key(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=SYSTEM_ONE_URL, method="POST", json=API_RESPONSE)
        tool = create_jev_tool(_resource(), MagicMock())

        result = await tool.ainvoke({"state": "My Stripe integration keeps failing"})

        assert result == EXPECTED_OUTPUT
        request = httpx_mock.get_requests()[0]
        assert request.headers["Authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["state"] == "My Stripe integration keeps failing"
        assert body["model"] == "jev-1.13.0"
        assert set(body["questions"]) == {"department", "frustration", "is_urgent"}

    async def test_honours_base_url_override(
        self, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TYPESAFE_BASE_URL", "https://jev.example.com/")
        httpx_mock.add_response(
            url="https://jev.example.com/v1/systemone", json=API_RESPONSE
        )
        tool = create_jev_tool(_resource(), MagicMock())
        assert await tool.ainvoke({"state": "hi"}) == EXPECTED_OUTPUT

    async def test_missing_api_key_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TYPESAFE_API_KEY")
        tool = create_jev_tool(_resource(), MagicMock())
        with pytest.raises(AgentRuntimeError) as exc_info:
            await tool.ainvoke({"state": "hi"})
        assert "TYPESAFE_API_KEY" in str(exc_info.value.error_info.detail)

    @pytest.mark.parametrize("status", [401, 422, 500])
    async def test_http_errors_raise(self, httpx_mock: HTTPXMock, status: int) -> None:
        httpx_mock.add_response(url=SYSTEM_ONE_URL, status_code=status, text="nope")
        tool = create_jev_tool(_resource(), MagicMock())
        with pytest.raises(AgentRuntimeError) as exc_info:
            await tool.ainvoke({"state": "hi"})
        assert f"HTTP {status}" in str(exc_info.value.error_info.detail)

    async def test_retries_rate_limited_requests(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=SYSTEM_ONE_URL, status_code=429, headers={"retry-after": "0"}
        )
        httpx_mock.add_response(url=SYSTEM_ONE_URL, json=API_RESPONSE)
        tool = create_jev_tool(_resource(), MagicMock())
        assert await tool.ainvoke({"state": "hi"}) == EXPECTED_OUTPUT
        assert len(httpx_mock.get_requests()) == 2

    async def test_missing_answer_raises(self, httpx_mock: HTTPXMock) -> None:
        response = {**API_RESPONSE, "answers": {"department": {"choice": "billing"}}}
        httpx_mock.add_response(url=SYSTEM_ONE_URL, json=response)
        tool = create_jev_tool(_resource(), MagicMock())
        with pytest.raises(AgentRuntimeError):
            await tool.ainvoke({"state": "hi"})


class TestJevGateway:
    def _settings(self, url: str) -> MagicMock:
        settings = MagicMock()
        settings.build_base_url.return_value = url
        settings.build_auth_headers.return_value = {"X-UiPath-Internal-TenantId": "t"}
        settings.build_auth_pipeline.return_value = None
        return settings

    @pytest.mark.parametrize(
        ("completions_url", "expected"),
        [
            (
                "https://cloud/org/tenant/agenthub_/llm/raw/vendor/typesafe/model/jev-latest/completions",
                "https://cloud/org/tenant/agenthub_/llm/raw/vendor/typesafe/model/jev-latest/systemone",
            ),
            (
                "https://gw/api/raw/vendor/typesafe/model/jev-latest/completions?x=1",
                "https://gw/api/raw/vendor/typesafe/model/jev-latest/systemone?x=1",
            ),
        ],
    )
    def test_build_gateway_url(self, completions_url: str, expected: str) -> None:
        settings = self._settings(completions_url)
        assert build_gateway_url(settings, "jev-latest") == expected
        api_config = settings.build_base_url.call_args.kwargs["api_config"]
        assert api_config.vendor_type == "typesafe"

    async def test_routes_via_gateway(self, httpx_mock: HTTPXMock) -> None:
        url = "https://cloud/org/tenant/agenthub_/llm/raw/vendor/typesafe/model/jev-1.13.0/systemone"
        httpx_mock.add_response(url=url, json=API_RESPONSE)
        settings = self._settings(url.replace("/systemone", "/completions"))
        client = JevClient(
            JevClientConfig(use_llm_gateway=True, api_key="ignored"),
            gateway_settings=settings,
        )
        tool = create_jev_tool(_resource(), MagicMock(), client=client)

        assert await tool.ainvoke({"state": "hi"}) == EXPECTED_OUTPUT
        request = httpx_mock.get_requests()[0]
        assert request.headers["X-UiPath-Internal-TenantId"] == "t"
        assert "Authorization" not in request.headers

    def test_gateway_flag_is_read_from_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("UIPATH_FEATURE_EnableJevViaLlmGateway", "true")
        assert JevClientConfig.from_environment().use_llm_gateway is True
