"""Tests for execution type environment variable handling."""

import pytest

from uipath_agents._observability.llmops.spans.span_attributes import (
    ENV_UIPATH_IS_DEBUG,
    ENV_UIPATH_PROCESS_VERSION,
    ExecutionType,
    get_agent_version,
    get_execution_type,
)


@pytest.fixture(autouse=True)
def clear_env_caches():
    """Clear cached environment variable functions before each test."""
    get_execution_type.cache_clear()
    get_agent_version.cache_clear()
    yield
    get_execution_type.cache_clear()
    get_agent_version.cache_clear()


class TestExecutionType:
    def test_debug_when_env_true_lowercase(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_UIPATH_IS_DEBUG, "true")
        assert get_execution_type() == ExecutionType.DEBUG

    def test_debug_when_env_True_csharp_style(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Recognize 'True' (capitalized) as debug mode."""
        monkeypatch.setenv(ENV_UIPATH_IS_DEBUG, "True")
        assert get_execution_type() == ExecutionType.DEBUG

    def test_debug_when_env_TRUE_uppercase(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_UIPATH_IS_DEBUG, "TRUE")
        assert get_execution_type() == ExecutionType.DEBUG

    def test_runtime_when_env_False_csharp_style(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Recognize 'False' (capitalized) as runtime mode."""
        monkeypatch.setenv(ENV_UIPATH_IS_DEBUG, "False")
        assert get_execution_type() == ExecutionType.RUNTIME

    def test_runtime_when_env_false_lowercase(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_UIPATH_IS_DEBUG, "false")
        assert get_execution_type() == ExecutionType.RUNTIME

    def test_runtime_when_env_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default to RUNTIME when env not set (per Orchestrator behavior)."""
        monkeypatch.delenv(ENV_UIPATH_IS_DEBUG, raising=False)
        assert get_execution_type() == ExecutionType.RUNTIME

    def test_runtime_when_env_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default to RUNTIME when env is empty."""
        monkeypatch.setenv(ENV_UIPATH_IS_DEBUG, "")
        assert get_execution_type() == ExecutionType.RUNTIME


class TestAgentVersion:
    def test_version_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_UIPATH_PROCESS_VERSION, "1.2.3")
        assert get_agent_version() == "1.2.3"

    def test_none_when_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_UIPATH_PROCESS_VERSION, raising=False)
        assert get_agent_version() is None

    def test_none_when_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_UIPATH_PROCESS_VERSION, "")
        assert get_agent_version() is None
