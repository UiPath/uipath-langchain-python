"""Trace validation tests for basic agent."""

from pathlib import Path

from tests.integration.conftest import AgentTraceTest

EXPECTED = Path(__file__).parent / "expected_traces"


class TestBasicAgent(AgentTraceTest):
    GOLDEN = EXPECTED / "golden.json"
    CONFIG = EXPECTED / "config.json"
    AGENT_DIR = "basic"
