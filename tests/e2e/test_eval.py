"""
E2E tests for `uipath eval` command on sample agents.

Tests that agents with evaluation sets can run evaluations
successfully using the uipath eval command.

Requires UIPATH_PROJECT_ID to be set for evaluation tests.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

from .conftest import EXAMPLES_DIR, run_uipath_command

# --- GitHub Actions Helpers ---


def emit_github_annotation(
    level: str, message: str, title: Optional[str] = None
) -> None:
    """Emit a GitHub Actions annotation if running in CI."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        prefix = {"notice": "ℹ", "warning": "⚠", "error": "✗"}.get(level, "")
        print(f"{prefix} {title or ''}: {message}")
        return

    if title:
        print(f"::{level} title={title}::{message}")
    else:
        print(f"::{level}::{message}")


def write_github_summary(content: str) -> None:
    """Write markdown content to GitHub Actions job summary."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(content + "\n")
    else:
        print(content)


def _get_eval_results_filepath(filename: str = "eval-summary.md") -> str:
    output_path = os.environ.get("GITHUB_WORKSPACE", ".")
    return os.path.join(output_path, filename)


def write_eval_results_file(content: str, filename: str = "eval-summary.md") -> None:
    """Write evaluation results to a file for PR comment posting."""
    filepath = _get_eval_results_filepath(filename)
    with open(filepath, "a") as f:
        f.write(content + "\n")


@pytest.fixture(scope="module", autouse=True)
def clear_eval_results_at_start():
    """Clear eval results file at the start of the test module."""
    filepath = _get_eval_results_filepath()
    if os.path.exists(filepath):
        os.remove(filepath)
    yield


# --- Eval Result Parsing ---


@dataclass
class EvalResult:
    """Result of a single evaluation test case."""

    name: str
    passed: bool
    scores: dict[str, float]
    error: Optional[str] = None


def parse_eval_output(output: str) -> list[EvalResult]:
    """Parse uipath eval output to extract individual test results.

    Parses output like:
        ▌ Test Addition
          LegacyExactMatchEvaluator        100.0
        ✗ Test Subtraction
          Error: ...

    The output shows each test twice: first as "○ Name - Running..." (pending),
    then as "▌ Name" (passed) or "✗ Name" (failed). We skip pending entries
    and only keep final results.
    """
    results: dict[str, EvalResult] = {}
    current_test: Optional[str] = None
    current_passed: bool = True
    current_scores: dict[str, float] = {}
    current_error: Optional[str] = None
    is_pending: bool = False

    # Markers: ○ = pending/running, ▌/✓ = passed, ✗ = failed
    test_pattern = re.compile(r"^\s*([▌✓✗○])\s+(.+?)(?:\s*-\s*Running\.\.\.)?$")
    score_pattern = re.compile(r"^\s+(\w+(?:Evaluator)?)\s+([\d.]+)\s*$")

    for line in output.split("\n"):
        test_match = test_pattern.match(line)
        if test_match:
            # Save previous test (if not pending)
            if current_test and not is_pending:
                results[current_test] = EvalResult(
                    name=current_test,
                    passed=current_passed,
                    scores=current_scores.copy(),
                    error=current_error,
                )

            marker = test_match.group(1)
            current_test = test_match.group(2).strip()
            current_passed = marker != "✗"
            is_pending = marker == "○"
            current_scores = {}
            current_error = None
            continue

        score_match = score_pattern.match(line)
        if score_match and current_test:
            current_scores[score_match.group(1)] = float(score_match.group(2))
            continue

        if current_test and "Error:" in line:
            current_error = line.strip()

    # Save last test
    if current_test and not is_pending:
        results[current_test] = EvalResult(
            name=current_test,
            passed=current_passed,
            scores=current_scores.copy(),
            error=current_error,
        )

    return list(results.values())


# --- Common Test Helpers ---


def run_eval_and_parse(
    example_dir: Path,
    authenticated_session: dict[str, str],
    eval_set_path: Optional[Path] = None,
) -> tuple[int, list[EvalResult], str]:
    """Run eval command and parse results.

    Returns:
        Tuple of (return_code, eval_results, raw_output)
    """
    command = ["eval", "agent.json"]
    if eval_set_path:
        command.append(str(eval_set_path))
    command.append("--no-report")

    result = run_uipath_command(
        command=command,
        cwd=example_dir,
        env=authenticated_session,
        timeout=300,
    )

    output = result.stdout + result.stderr
    eval_results = parse_eval_output(output)

    return result.returncode, eval_results, output


def build_results_summary(
    title: str, eval_results: list[EvalResult]
) -> tuple[str, list[EvalResult]]:
    """Build markdown summary and return failures.

    Returns:
        Tuple of (markdown_summary, failed_results)
    """
    lines = [
        f"## {title}",
        "",
        "| Test Case | Status | Details |",
        "|-----------|--------|---------|",
    ]

    for r in eval_results:
        if r.passed:
            scores_str = ", ".join(f"{k}: {v}%" for k, v in r.scores.items())
            lines.append(f"| {r.name} | ✅ Pass | {scores_str or 'N/A'} |")
            emit_github_annotation(
                "notice", scores_str or "PASSED", f"PASSED: {r.name}"
            )
        else:
            error_msg = r.error or "Failed"
            lines.append(f"| {r.name} | ❌ Fail | {error_msg} |")
            emit_github_annotation("error", error_msg, f"FAILED: {r.name}")

    passed = sum(1 for r in eval_results if r.passed)
    failed = sum(1 for r in eval_results if not r.passed)
    lines.extend(
        [
            "",
            f"**Total:** {len(eval_results)} | ✅ {passed} passed | ❌ {failed} failed",
            "",
        ]
    )

    failures = [r for r in eval_results if not r.passed]
    return "\n".join(lines), failures


def assert_eval_success(
    returncode: int,
    eval_results: list[EvalResult],
    output: str,
) -> None:
    """Assert that eval command succeeded and check for failures."""
    assert returncode == 0, f"Eval command failed:\n{output[:1000]}"
    assert eval_results, f"No evaluation results parsed from output:\n{output[:500]}"

    failures = [r for r in eval_results if not r.passed]
    if failures:
        details = "\n".join(f"  - {r.name}: {r.error or 'Failed'}" for r in failures)
        pytest.fail(f"Some evaluations failed:\n{details}")


# --- Test Classes ---


class TestEvalConfiguration:
    """Tests for evaluation configuration validation (no auth required)."""

    @pytest.mark.e2e
    def test_eval_sets_exist(self, example_with_evals: str):
        """Verify examples with evals have eval-sets directory."""
        evals_dir = EXAMPLES_DIR / example_with_evals / "evaluations" / "eval-sets"
        assert evals_dir.exists(), f"eval-sets missing in {example_with_evals}"

    @pytest.mark.e2e
    def test_eval_sets_valid_json(self, example_with_evals: str):
        """Verify eval set files are valid JSON with required fields."""
        evals_dir = EXAMPLES_DIR / example_with_evals / "evaluations" / "eval-sets"
        if not evals_dir.exists():
            pytest.skip(f"No eval-sets in {example_with_evals}")

        for eval_file in evals_dir.glob("*.json"):
            with open(eval_file) as f:
                data = json.load(f)
            assert "evaluations" in data, f"Missing 'evaluations' in {eval_file.name}"
            assert isinstance(data["evaluations"], list)

    @pytest.mark.e2e
    def test_evaluators_exist(self, example_with_evals: str):
        """Verify examples with evals have evaluators directory."""
        evaluators_dir = (
            EXAMPLES_DIR / example_with_evals / "evaluations" / "evaluators"
        )
        assert evaluators_dir.exists(), f"evaluators missing in {example_with_evals}"

    @pytest.mark.e2e
    def test_evaluators_valid_json(self, example_with_evals: str):
        """Verify evaluator files are valid JSON."""
        evaluators_dir = (
            EXAMPLES_DIR / example_with_evals / "evaluations" / "evaluators"
        )
        if not evaluators_dir.exists():
            pytest.skip(f"No evaluators in {example_with_evals}")

        for evaluator_file in evaluators_dir.glob("*.json"):
            with open(evaluator_file) as f:
                data = json.load(f)
            assert isinstance(data, dict)


class TestEvalExecution:
    """Tests that run evaluations and report results."""

    @pytest.mark.e2e
    def test_calculator_evaluations(
        self, authenticated_session: dict[str, str], project_id: str
    ):
        """Run calculator evaluations."""
        if not project_id:
            pytest.skip("UIPATH_PROJECT_ID not set")

        returncode, eval_results, output = run_eval_and_parse(
            EXAMPLES_DIR / "calculator", authenticated_session
        )

        summary, _ = build_results_summary(
            "📊 Calculator Agent Evaluation Results", eval_results
        )
        write_github_summary(summary)
        write_eval_results_file(summary)

        assert_eval_success(returncode, eval_results, output)

    @pytest.mark.e2e
    def test_calculator_same_as_agent_evaluations(
        self, authenticated_session: dict[str, str], project_id: str
    ):
        """Run calculator_same_as_agent evaluations.

        This test validates that:
        1. The same-as-agent model configuration works
        2. Sequential evaluations run correctly (each eval has isolated state)
        """
        if not project_id:
            pytest.skip("UIPATH_PROJECT_ID not set")

        example_dir = EXAMPLES_DIR / "calculator_same_as_agent"
        if not example_dir.exists():
            pytest.skip("calculator_same_as_agent example not found")

        returncode, eval_results, output = run_eval_and_parse(
            example_dir, authenticated_session
        )

        summary, _ = build_results_summary(
            "🧮 Calculator (Same as Agent) Evaluation Results", eval_results
        )
        write_github_summary(summary)
        write_eval_results_file(summary)

        assert_eval_success(returncode, eval_results, output)

    @pytest.mark.e2e
    def test_calculator_same_as_agent_model_settings_override(
        self, authenticated_session: dict[str, str], project_id: str
    ):
        """Test model settings override functionality with calculator_same_as_agent.

        This test validates that:
        1. --model-settings-id "default" works (uses same-as-agent, no override)
        2. --model-settings-id with specific ID works (overrides agent model)
        3. Evaluators with "same-as-agent" adapt to the overridden agent model
        """
        if not project_id:
            pytest.skip("UIPATH_PROJECT_ID not set")

        example_dir = EXAMPLES_DIR / "calculator_same_as_agent"
        if not example_dir.exists():
            pytest.skip("calculator_same_as_agent example not found")

        # Test 1: Model settings ID "default" (same-as-agent)
        command_default = [
            "eval",
            "agent.json",
            "--model-settings-id",
            "default",
            "--no-report",
        ]
        result_default = run_uipath_command(
            command=command_default,
            cwd=example_dir,
            env=authenticated_session,
            timeout=300,
        )

        output_default = result_default.stdout + result_default.stderr
        eval_results_default = parse_eval_output(output_default)

        # Verify default works and uses original agent model
        assert result_default.returncode == 0, (
            f"Eval with --model-settings-id default failed:\n{output_default[:1000]}"
        )
        assert eval_results_default, (
            f"No results with default settings:\n{output_default[:500]}"
        )

        # Test 2: Model settings ID with override (gpt-5-2025-08-07)
        command_override = [
            "eval",
            "agent.json",
            "--model-settings-id",
            "604d96fd-5e89-484b-b750-ccf3f516e2e1",
            "--no-report",
        ]
        result_override = run_uipath_command(
            command=command_override,
            cwd=example_dir,
            env=authenticated_session,
            timeout=300,
        )

        output_override = result_override.stdout + result_override.stderr
        eval_results_override = parse_eval_output(output_override)

        # Verify override works
        assert result_override.returncode == 0, (
            f"Eval with model override failed:\n{output_override[:1000]}"
        )
        assert eval_results_override, (
            f"No results with override:\n{output_override[:500]}"
        )

        # Verify logs show the override is happening
        assert "Applying model settings override" in output_override, (
            "Expected log message about model settings override not found"
        )
        assert "gpt-5-2025-08-07" in output_override, (
            "Expected overridden model name not found in logs"
        )

        # Verify evaluators resolve to the overridden model
        assert (
            "Resolving 'same-as-agent' to agent model: gpt-5-2025-08-07"
            in output_override
        ), "Evaluators should resolve 'same-as-agent' to the overridden agent model"

        # Generate summaries for both runs
        summary_default, _ = build_results_summary(
            "🧮 Calculator Model Settings - Default (same-as-agent)",
            eval_results_default,
        )
        summary_override, _ = build_results_summary(
            "🧮 Calculator Model Settings - Override (gpt-5-2025-08-07)",
            eval_results_override,
        )

        write_github_summary(summary_default)
        write_github_summary(summary_override)
        write_eval_results_file(summary_default)
        write_eval_results_file(summary_override)

        # Assert both runs succeeded
        assert_eval_success(
            result_default.returncode, eval_results_default, output_default
        )
        assert_eval_success(
            result_override.returncode, eval_results_override, output_override
        )

    @pytest.mark.e2e
    def test_calculator_with_specific_eval_set(
        self, authenticated_session: dict[str, str], project_id: str
    ):
        """Test running evaluation with specific eval set."""
        if not project_id:
            pytest.skip("UIPATH_PROJECT_ID not set")

        example_dir = EXAMPLES_DIR / "calculator"
        eval_set_path = example_dir / "evaluations" / "eval-sets" / "legacy.json"

        if not eval_set_path.exists():
            pytest.skip("legacy.json eval set not found")

        returncode, eval_results, output = run_eval_and_parse(
            example_dir, authenticated_session, eval_set_path
        )

        assert_eval_success(returncode, eval_results, output)

    @pytest.mark.e2e
    def test_calculator_trace_hierarchy(
        self, authenticated_session: dict[str, str], project_id: str, tmp_path: Path
    ):
        """Test evaluation trace has correct span hierarchy.

        Verifies that the trace structure follows the expected hierarchy:
        - Evaluation Set Run (root)
          - Evaluation
            - root
              - Agent run
          - Evaluator
            - Evaluation output
        """
        if not project_id:
            pytest.skip("UIPATH_PROJECT_ID not set")

        example_dir = EXAMPLES_DIR / "calculator"
        trace_file = tmp_path / "trace.jsonl"

        # Run eval with trace output
        command = ["eval", "agent.json", "--trace-file", str(trace_file), "--no-report"]
        result = run_uipath_command(
            command=command,
            cwd=example_dir,
            env=authenticated_session,
            timeout=300,
        )

        assert result.returncode == 0, f"Eval command failed:\n{result.stdout}"
        assert trace_file.exists(), "Trace file was not created"

        # Parse trace spans
        spans = []
        with open(trace_file) as f:
            for line in f:
                if line.strip():
                    spans.append(json.loads(line))

        assert spans, "No spans found in trace file"

        # Verify expected hierarchy: Evaluation Set Run -> Evaluation -> root -> Agent run
        eval_set_run = next(
            (s for s in spans if s.get("name") == "Evaluation Set Run"), None
        )
        assert eval_set_run, "No 'Evaluation Set Run' span found"
        eval_set_run_id = eval_set_run.get("context", {}).get("span_id")

        # Find Evaluation child
        evaluation = next(
            (
                s
                for s in spans
                if s.get("name") == "Evaluation"
                and s.get("parent_id") == eval_set_run_id
            ),
            None,
        )
        assert evaluation, "No 'Evaluation' span found as child of Evaluation Set Run"
        evaluation_id = evaluation.get("context", {}).get("span_id")

        # Find root child
        root = next(
            (
                s
                for s in spans
                if s.get("name") == "root" and s.get("parent_id") == evaluation_id
            ),
            None,
        )
        assert root, "No 'root' span found as child of Evaluation"
        root_id = root.get("context", {}).get("span_id")

        # Find Agent run child
        agent_run = next(
            (
                s
                for s in spans
                if "Agent run" in s.get("name", "") and s.get("parent_id") == root_id
            ),
            None,
        )
        assert agent_run, "No 'Agent run' span found as child of root"

        # Verify evaluator structure: Evaluation -> Evaluator -> Evaluation output
        evaluator = next(
            (
                s
                for s in spans
                if "Evaluator:" in s.get("name", "")
                and s.get("parent_id") == evaluation_id
            ),
            None,
        )
        assert evaluator, "No 'Evaluator' span found as child of Evaluation"
        evaluator_id = evaluator.get("context", {}).get("span_id")

        # Find Evaluation output child
        eval_output = next(
            (
                s
                for s in spans
                if s.get("name") == "Evaluation output"
                and s.get("parent_id") == evaluator_id
            ),
            None,
        )
        assert eval_output, "No 'Evaluation output' span found as child of Evaluator"
