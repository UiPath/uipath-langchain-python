"""
E2E tests for `uipath run` command on sample agents.

Tests that each example agent can be executed successfully using
the uipath run command with proper authentication.
"""

import json

import pytest

from .conftest import EXAMPLES_DIR, run_uipath_command


class TestUiPathRun:
    """Tests for the uipath run command."""

    @pytest.mark.e2e
    def test_run_calculator_with_input(self, authenticated_session: dict[str, str]):
        """Test running calculator agent with JSON input."""
        example_dir = EXAMPLES_DIR / "calculator"

        # Test input for calculator: 5 + 3 = 8
        input_data = json.dumps({"a": 5, "b": 3, "operator": "+"})

        result = run_uipath_command(
            command=["run", "agent.json", input_data],
            cwd=example_dir,
            env=authenticated_session,
            timeout=120,
        )

        assert result.returncode == 0, f"Run failed: {result.stderr}"
        assert "error" not in result.stderr.lower(), f"Error in output: {result.stderr}"

        # Check output contains result
        output = result.stdout
        assert output, "No output from run command"

    @pytest.mark.e2e
    def test_run_calculator_multiplication(self, authenticated_session: dict[str, str]):
        """Test calculator with multiplication."""
        example_dir = EXAMPLES_DIR / "calculator"

        input_data = json.dumps({"a": 6, "b": 7, "operator": "*"})

        result = run_uipath_command(
            command=["run", "agent.json", input_data],
            cwd=example_dir,
            env=authenticated_session,
            timeout=120,
        )

        assert result.returncode == 0, f"Run failed: {result.stderr}"

    @pytest.mark.e2e
    def test_run_basic_agent(self, authenticated_session: dict[str, str]):
        """Test running basic agent without input."""
        example_dir = EXAMPLES_DIR / "basic"

        if not (example_dir / "agent.json").exists():
            pytest.skip("Basic agent not configured")

        result = run_uipath_command(
            command=["run", "agent.json"],
            cwd=example_dir,
            env=authenticated_session,
            timeout=120,
        )

        # Basic agent has empty input schema, should succeed without input
        assert result.returncode == 0, f"Run failed: {result.stderr}"

    @pytest.mark.e2e
    def test_run_simulated_rpa_tool(self, authenticated_session: dict[str, str]):
        """Test running adder agent with simulated RPA tool that adds two numbers."""
        example_dir = EXAMPLES_DIR / "adderwithrpatool" / "Agent"

        input_data = json.dumps({"number1": 5, "number2": 17})

        # Remove UIPATH_PROJECT_ID so debug skips the Studio API call
        # (ResourceOverwritesContext) which the CI service account can't access
        env = {
            k: v for k, v in authenticated_session.items() if k != "UIPATH_PROJECT_ID"
        }

        result = run_uipath_command(
            command=["debug", "agent.json", input_data],
            cwd=example_dir,
            env=env,
            timeout=120,
            stdin_input="c\n",
        )

        assert result.returncode == 0, f"Debug failed: {result.stderr}"

        output = result.stdout
        assert output, "No output from debug command"
        assert "22" in output, f"Expected sum of 22 in output, got: {output}"

    @pytest.mark.e2e
    def test_run_simulated_hitl_escalation(self, authenticated_session: dict[str, str]):
        """Test running HITL agent with simulated escalation tool.

        The HITL_1 agent sends a color approval request via an escalation tool.
        With simulation.json (instructions: "always approve"), the escalation
        tool is mocked by an LLM instead of calling Action Center.
        """
        example_dir = EXAMPLES_DIR / "HITL_1" / "Agent"

        input_data = json.dumps({"color": "Red"})

        # Remove UIPATH_PROJECT_ID so debug skips the Studio API call
        # (ResourceOverwritesContext) which the CI service account can't access
        env = {
            k: v for k, v in authenticated_session.items() if k != "UIPATH_PROJECT_ID"
        }

        result = run_uipath_command(
            command=["debug", "agent.json", input_data],
            cwd=example_dir,
            env=env,
            timeout=120,
            stdin_input="c\n",
        )

        assert result.returncode == 0, f"Debug failed: {result.stderr}"

        output = result.stdout
        assert output, "No output from debug command"

        output_lower = output.lower()
        assert "red" in output_lower, f"Expected color 'red' in output, got: {output}"
        assert "confirm" in output_lower or "approv" in output_lower, (
            f"Expected confirmation/approval language in output, got: {output}"
        )

    @pytest.mark.e2e
    def test_run_with_invalid_input(self, authenticated_session: dict[str, str]):
        """Test that invalid input returns non-zero exit code (negative test)."""
        example_dir = EXAMPLES_DIR / "calculator"

        # Missing required fields (b and operator)
        input_data = json.dumps({"a": 5})

        result = run_uipath_command(
            command=["run", "agent.json", input_data],
            cwd=example_dir,
            env=authenticated_session,
            timeout=120,
        )

        # This is a NEGATIVE test: we expect the command to fail
        # because required fields are missing
        assert result.returncode != 0, "Expected non-zero exit code for invalid input"
        assert (
            "validation" in result.stderr.lower() or "missing" in result.stderr.lower()
        ), f"Expected validation error message, got: {result.stderr}"

    @pytest.mark.e2e
    def test_run_nonexistent_agent(self, authenticated_session: dict[str, str]):
        """Test that non-existent agent returns non-zero exit code (negative test)."""
        example_dir = EXAMPLES_DIR / "calculator"

        result = run_uipath_command(
            command=["run", "nonexistent.json"],
            cwd=example_dir,
            env=authenticated_session,
            timeout=30,
        )

        # This is a NEGATIVE test: we expect the command to fail
        # because the agent file doesn't exist
        assert result.returncode != 0, (
            "Expected non-zero exit code for non-existent agent"
        )
        assert "not found" in result.stderr.lower(), (
            f"Expected 'not found' error message, got: {result.stderr}"
        )


class TestRunAllExamples:
    """Parametrized tests that run on all examples."""

    @pytest.mark.e2e
    def test_agent_json_exists(self, example_name: str):
        """Verify each example has an agent.json file."""
        example_dir = EXAMPLES_DIR / example_name
        agent_file = example_dir / "agent.json"

        assert agent_file.exists(), f"agent.json missing in {example_name}"

    @pytest.mark.e2e
    def test_agent_json_valid(self, example_name: str):
        """Verify each agent.json is valid JSON."""
        example_dir = EXAMPLES_DIR / example_name
        agent_file = example_dir / "agent.json"

        with open(agent_file) as f:
            data = json.load(f)

        # Check required fields
        assert "name" in data, f"Missing 'name' in {example_name}/agent.json"
        assert "settings" in data, f"Missing 'settings' in {example_name}/agent.json"
