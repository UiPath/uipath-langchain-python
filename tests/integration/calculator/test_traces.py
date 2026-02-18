"""Trace validation tests for calculator agent."""

from pathlib import Path

from tests.integration.conftest import AgentTraceTest

EXPECTED = Path(__file__).parent / "expected_traces"


class TestCalculatorAgent(AgentTraceTest):
    GOLDEN = EXPECTED / "golden.json"
    CONFIG = EXPECTED / "config.json"
    AGENT_DIR = "calculator"
    AGENT_INPUT = '{"a": 10, "operator": "*", "b": 5}'
