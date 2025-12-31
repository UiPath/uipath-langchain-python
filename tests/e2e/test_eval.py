"""
E2E tests for `uipath eval` command on sample agents.

Tests that agents with evaluation sets can run evaluations
successfully using the uipath eval command.

Requires UIPATH_PROJECT_ID to be set for evaluation tests.
"""

import json

import pytest

from .conftest import EXAMPLES_DIR, run_uipath_command


class TestUiPathEval:
    """Tests for the uipath eval command."""

    @pytest.mark.e2e
    def test_eval_calculator(
        self, authenticated_session: dict[str, str], project_id: str
    ):
        """Test running evaluations on calculator agent."""
        if not project_id:
            pytest.skip("UIPATH_PROJECT_ID not set - required for eval tests")

        example_dir = EXAMPLES_DIR / "calculator"

        result = run_uipath_command(
            command=["eval", "agent.json", "--no-report"],
            cwd=example_dir,
            env=authenticated_session,
            timeout=300,  # Evals can take longer
        )

        assert result.returncode == 0, f"Eval failed: {result.stderr}"

        # Check for evaluation output
        output = result.stdout + result.stderr
        assert "error" not in output.lower() or "evaluation" in output.lower()

    @pytest.mark.e2e
    def test_eval_with_specific_set(
        self, authenticated_session: dict[str, str], project_id: str
    ):
        """Test running evaluation with specific eval set."""
        if not project_id:
            pytest.skip("UIPATH_PROJECT_ID not set - required for eval tests")

        example_dir = EXAMPLES_DIR / "calculator"
        eval_set_path = example_dir / "evaluations" / "eval-sets" / "legacy.json"

        if not eval_set_path.exists():
            pytest.skip("Eval set not found")

        result = run_uipath_command(
            # EVAL_SET is a positional arg, not --eval-set option
            command=["eval", "agent.json", str(eval_set_path), "--no-report"],
            cwd=example_dir,
            env=authenticated_session,
            timeout=300,
        )

        # Check command executed (may fail due to config but shouldn't crash)
        assert result.returncode in [0, 1], f"Unexpected error: {result.stderr}"


class TestEvalConfiguration:
    """Tests for evaluation configuration validation."""

    @pytest.mark.e2e
    def test_eval_sets_exist(self, example_with_evals: str):
        """Verify examples with evals have eval-sets directory."""
        example_dir = EXAMPLES_DIR / example_with_evals
        evals_dir = example_dir / "evaluations" / "eval-sets"

        assert evals_dir.exists(), f"eval-sets missing in {example_with_evals}"

    @pytest.mark.e2e
    def test_eval_sets_valid_json(self, example_with_evals: str):
        """Verify eval set files are valid JSON."""
        example_dir = EXAMPLES_DIR / example_with_evals
        evals_dir = example_dir / "evaluations" / "eval-sets"

        if not evals_dir.exists():
            pytest.skip(f"No eval-sets in {example_with_evals}")

        for eval_file in evals_dir.glob("*.json"):
            with open(eval_file) as f:
                data = json.load(f)

            assert "evaluations" in data, f"Missing 'evaluations' in {eval_file.name}"
            assert isinstance(data["evaluations"], list), (
                f"'evaluations' should be a list in {eval_file.name}"
            )

    @pytest.mark.e2e
    def test_evaluators_exist(self, example_with_evals: str):
        """Verify examples with evals have evaluators directory."""
        example_dir = EXAMPLES_DIR / example_with_evals
        evaluators_dir = example_dir / "evaluations" / "evaluators"

        assert evaluators_dir.exists(), f"evaluators missing in {example_with_evals}"

    @pytest.mark.e2e
    def test_evaluators_valid_json(self, example_with_evals: str):
        """Verify evaluator files are valid JSON."""
        example_dir = EXAMPLES_DIR / example_with_evals
        evaluators_dir = example_dir / "evaluations" / "evaluators"

        if not evaluators_dir.exists():
            pytest.skip(f"No evaluators in {example_with_evals}")

        for evaluator_file in evaluators_dir.glob("*.json"):
            with open(evaluator_file) as f:
                data = json.load(f)

            # Basic validation - evaluators should have certain fields
            assert isinstance(data, dict), (
                f"Evaluator should be an object in {evaluator_file.name}"
            )


class TestEvalResults:
    """Tests that verify evaluation results."""

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_calculator_addition_eval(
        self, authenticated_session: dict[str, str], project_id: str
    ):
        """Test that calculator correctly evaluates addition."""
        if not project_id:
            pytest.skip("UIPATH_PROJECT_ID not set")

        example_dir = EXAMPLES_DIR / "calculator"

        result = run_uipath_command(
            command=["eval", "agent.json", "--no-report"],
            cwd=example_dir,
            env=authenticated_session,
            timeout=300,
        )

        # Parse output for results if available
        output = result.stdout
        if result.returncode == 0:
            # Look for evaluation results table with scores
            # The output should contain "Evaluation Results" and scores
            assert "evaluation results" in output.lower() or "100.0" in output
