"""Tests for ExceptionMapper behavior."""

import httpx
import pytest
from uipath.platform.errors import EnrichedException
from uipath.runtime.errors import UiPathErrorCategory
from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
    AgentStartupError,
    AgentStartupErrorCode,
)

from uipath_agents._errors.exception_mapper import ExceptionMapper

# --- Fixtures ---


@pytest.fixture()
def http_status_error() -> httpx.HTTPStatusError:
    response = httpx.Response(
        status_code=429, request=httpx.Request("GET", "https://api.example.com")
    )
    return httpx.HTTPStatusError(
        "rate limited", request=response.request, response=response
    )


@pytest.fixture()
def enriched_exception(http_status_error: httpx.HTTPStatusError) -> EnrichedException:
    return EnrichedException(http_status_error)


@pytest.fixture()
def existing_runtime_error() -> AgentRuntimeError:
    return AgentRuntimeError(
        code=AgentRuntimeErrorCode.UNEXPECTED_ERROR,
        title="already mapped",
        detail="should pass through",
        category=UiPathErrorCategory.SYSTEM,
    )


@pytest.fixture()
def existing_startup_error() -> AgentStartupError:
    return AgentStartupError(
        code=AgentStartupErrorCode.UNEXPECTED_ERROR,
        title="already mapped",
        detail="should pass through",
        category=UiPathErrorCategory.SYSTEM,
    )


# --- map_runtime ---


class TestMapRuntime:
    def test_passthrough_existing_runtime_error(
        self, existing_runtime_error: AgentRuntimeError
    ) -> None:
        result = ExceptionMapper.map_runtime(existing_runtime_error)

        assert result is existing_runtime_error

    def test_generic_exception_returns_agent_runtime_error(self) -> None:
        result = ExceptionMapper.map_runtime(ValueError("bad value"))

        assert isinstance(result, AgentRuntimeError)
        assert result.error_info.category == UiPathErrorCategory.UNKNOWN

    def test_generic_exception_includes_original_message(self) -> None:
        result = ExceptionMapper.map_runtime(ValueError("bad value"))

        assert "bad value" in result.error_info.detail

    def test_http_status_error_returns_agent_runtime_error(
        self, http_status_error: httpx.HTTPStatusError
    ) -> None:
        result = ExceptionMapper.map_runtime(http_status_error)

        assert isinstance(result, AgentRuntimeError)
        assert result.error_info.status == 429

    def test_enriched_exception_returns_agent_runtime_error(
        self, enriched_exception: EnrichedException
    ) -> None:
        result = ExceptionMapper.map_runtime(enriched_exception)

        assert isinstance(result, AgentRuntimeError)
        assert result.error_info.status == 429


# --- map_config ---


class TestMapConfig:
    def test_passthrough_existing_startup_error(
        self, existing_startup_error: AgentStartupError
    ) -> None:
        result = ExceptionMapper.map_config(existing_startup_error)

        assert result is existing_startup_error

    def test_generic_exception_returns_agent_startup_error(self) -> None:
        result = ExceptionMapper.map_config(KeyError("missing key"))

        assert isinstance(result, AgentStartupError)
        assert result.error_info.category == UiPathErrorCategory.UNKNOWN

    def test_generic_exception_includes_original_message(self) -> None:
        result = ExceptionMapper.map_config(RuntimeError("config failed"))

        assert "config failed" in result.error_info.detail

    def test_http_status_error_returns_agent_startup_error(
        self, http_status_error: httpx.HTTPStatusError
    ) -> None:
        result = ExceptionMapper.map_config(http_status_error)

        assert isinstance(result, AgentStartupError)
        assert result.error_info.status == 429

    def test_enriched_exception_returns_agent_startup_error(
        self, enriched_exception: EnrichedException
    ) -> None:
        result = ExceptionMapper.map_config(enriched_exception)

        assert isinstance(result, AgentStartupError)
        assert result.error_info.status == 429
