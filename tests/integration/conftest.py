"""Pytest configuration for integration tests.

Re-exports fixtures from e2e tests and provides AgentTraceTest base class.
"""

from pathlib import Path
from typing import ClassVar

import pytest

from tests.e2e.conftest import (
    auth_env,
    authenticated_session,
    base_url,
    client_id,
    client_secret,
    organization_id,
    project_id,
    run_uipath_command,
    tenant_id,
)

from .trace_assertions import (
    assert_no_extra_spans,
    assert_span_attributes,
    assert_span_hierarchy,
    get_golden_span_names,
    print_trace_summary,
)
from .trace_assertions.trace_assert import TRACE_OUTPUT_PATH

REPO_ROOT = Path(__file__).parent.parent.parent

__all__ = [
    "AgentTraceTest",
    "auth_env",
    "authenticated_session",
    "base_url",
    "client_id",
    "client_secret",
    "organization_id",
    "project_id",
    "tenant_id",
]


class AgentTraceTest:
    """Base for per-agent trace integration tests.

    Subclasses set GOLDEN, CONFIG, AGENT_DIR, and optionally AGENT_INPUT.
    """

    GOLDEN: ClassVar[Path]
    CONFIG: ClassVar[Path]
    AGENT_DIR: ClassVar[str]
    AGENT_INPUT: ClassVar[str] = "{}"

    @pytest.fixture(autouse=True, scope="class")
    def run_agent(self, authenticated_session: dict[str, str]) -> None:
        """Run agent once, shared by all tests in the class."""
        agent_path = REPO_ROOT / "examples" / self.AGENT_DIR

        env = authenticated_session.copy()
        env["LLMOPS_TRACE_FILE"] = TRACE_OUTPUT_PATH

        result = run_uipath_command(
            command=["run", "agent.json", self.AGENT_INPUT],
            cwd=agent_path,
            env=env,
            timeout=120,
        )

        assert result.returncode == 0, f"Agent run failed: {result.stderr}"

        print_trace_summary(
            golden_path=self.GOLDEN,
            actual_path=TRACE_OUTPUT_PATH,
            config_path=self.CONFIG,
        )

    def pytest_generate_tests(self, metafunc: pytest.Metafunc) -> None:
        """Parametrize span_name from each subclass's golden/config."""
        if "span_name" in metafunc.fixturenames:
            spans = get_golden_span_names(self.GOLDEN, self.CONFIG)
            metafunc.parametrize("span_name", spans)

    @pytest.mark.e2e
    def test_no_extra_spans(self) -> None:
        """Validate no unexpected spans in actual trace."""
        assert_no_extra_spans(
            golden_path=self.GOLDEN,
            actual_path=TRACE_OUTPUT_PATH,
            config_path=self.CONFIG,
        )

    @pytest.mark.e2e
    def test_span_hierarchy(self, span_name: str) -> None:
        """Validate span exists with correct parent."""
        assert_span_hierarchy(
            span_name=span_name,
            golden_path=self.GOLDEN,
            actual_path=TRACE_OUTPUT_PATH,
            config_path=self.CONFIG,
        )

    @pytest.mark.e2e
    def test_span_attributes(self, span_name: str) -> None:
        """Validate span attributes match golden."""
        assert_span_attributes(
            span_name=span_name,
            golden_path=self.GOLDEN,
            actual_path=TRACE_OUTPUT_PATH,
            config_path=self.CONFIG,
        )
