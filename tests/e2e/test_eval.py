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
from typing import Optional

import pytest

from .conftest import EXAMPLES_DIR, run_uipath_command


def emit_github_annotation(
    level: str, message: str, title: Optional[str] = None
) -> None:
    """Emit a GitHub Actions annotation if running in CI.

    Args:
        level: "notice", "warning", or "error"
        message: The annotation message
        title: Optional title for the annotation
    """
    if os.environ.get("GITHUB_ACTIONS") != "true":
        # Not in GitHub Actions, just print
        prefix = {"notice": "ℹ", "warning": "⚠", "error": "✗"}.get(level, "")
        print(f"{prefix} {title or ''}: {message}")
        return

    # Format: ::notice title=<title>::<message>
    # Note: No space between level and properties
    if title:
        print(f"::{level} title={title}::{message}")
    else:
        print(f"::{level}::{message}")


def write_github_summary(content: str) -> None:
    """Write markdown content to GitHub Actions job summary.

    This creates a visible summary in the workflow run that appears
    prominently in the Actions UI.

    Args:
        content: Markdown content to append to the summary
    """
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(content + "\n")
    else:
        # Not in GitHub Actions, just print
        print(content)


def write_eval_results_file(content: str, filename: str = "eval-summary.md") -> None:
    """Write evaluation results to a file for PR comment posting.

    Args:
        content: Markdown content to write
        filename: Output filename (default: eval-summary.md)
    """
    # Write to workspace root or current directory
    output_path = os.environ.get("GITHUB_WORKSPACE", ".")
    filepath = os.path.join(output_path, filename)

    # Append to file (multiple tests may write to it)
    with open(filepath, "a") as f:
        f.write(content + "\n")


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
          LegacyJsonSimilarityEvaluator    100.0
        ✗ Test Subtraction
          Error: ...

    Returns list of EvalResult with test name, pass/fail status, and scores.
    Deduplicates results by test name, keeping the entry with scores.
    """
    results_dict: dict[str, EvalResult] = {}
    current_test: Optional[str] = None
    current_passed: bool = True
    current_scores: dict[str, float] = {}
    current_error: Optional[str] = None

    lines = output.split("\n")

    def save_current_test() -> None:
        """Save current test, keeping entry with more scores."""
        nonlocal current_test, current_passed, current_scores, current_error
        if current_test:
            new_result = EvalResult(
                name=current_test,
                passed=current_passed,
                scores=current_scores.copy(),
                error=current_error,
            )
            # Keep the entry with more scores (or error info)
            existing = results_dict.get(current_test)
            if not existing or len(new_result.scores) > len(existing.scores):
                results_dict[current_test] = new_result

    for _i, line in enumerate(lines):
        # Check for test start markers
        # ▌ or ✓ = passing, ✗ = failing
        test_match = re.match(r"^\s*[▌✓✗○]\s+(.+?)(?:\s*-\s*Running\.\.\.)?$", line)
        if test_match:
            # Save previous test if exists
            save_current_test()

            current_test = test_match.group(1).strip()
            current_passed = "✗" not in line
            current_scores = {}
            current_error = None
            continue

        # Check for evaluator scores (indented lines with evaluator name and score)
        score_match = re.match(r"^\s+(\w+(?:Evaluator)?)\s+([\d.]+)\s*$", line)
        if score_match and current_test:
            evaluator_name = score_match.group(1)
            score = float(score_match.group(2))
            current_scores[evaluator_name] = score
            continue

        # Check for error messages
        if current_test and "Error:" in line:
            current_error = line.strip()
            continue

    # Don't forget the last test
    save_current_test()

    return list(results_dict.values())


def parse_results_table(output: str) -> dict[str, dict[str, float]]:
    """Parse the final results table from eval output.

    Parses output like:
        ┃  Evaluation     ┃  LegacyExactMatchEvaluator  ┃
        │  Test Addition  │                      100.0  │
        │  Average        │                       25.0  │

    Returns dict mapping test name to evaluator scores.
    """
    results: dict[str, dict[str, float]] = {}

    # Find the table header to get evaluator names
    header_match = re.search(r"┃\s*Evaluation\s*┃(.+?)┃", output)
    if not header_match:
        return results

    # Parse evaluator names from header
    header_content = header_match.group(1)
    evaluator_names = [
        name.strip() for name in re.split(r"┃", header_content) if name.strip()
    ]

    # Find data rows (lines starting with │)
    for line in output.split("\n"):
        if line.strip().startswith("│") and "─" not in line:
            # Parse the row
            parts = [p.strip() for p in re.split(r"│", line) if p.strip()]
            if len(parts) >= 2:
                test_name = parts[0]
                if test_name.lower() == "average":
                    continue  # Skip average row
                scores = {}
                for i, score_str in enumerate(parts[1:]):
                    if i < len(evaluator_names):
                        try:
                            scores[evaluator_names[i]] = float(score_str)
                        except ValueError:
                            pass
                if scores:
                    results[test_name] = scores

    return results


class TestUiPathEval:
    """Tests for the uipath eval command."""

    @pytest.mark.e2e
    def test_eval_calculator(
        self, authenticated_session: dict[str, str], project_id: str
    ):
        """Test running evaluations on calculator agent with individual result reporting."""
        if not project_id:
            pytest.skip("UIPATH_PROJECT_ID not set - required for eval tests")

        example_dir = EXAMPLES_DIR / "calculator"

        # Note: --no-report is required when using external applications (client credentials)
        # because reporting requires user context (personal workspace) which is not available
        # with machine-to-machine authentication
        result = run_uipath_command(
            command=["eval", "agent.json", "--no-report"],
            cwd=example_dir,
            env=authenticated_session,
            timeout=300,  # Evals can take longer
        )

        assert result.returncode == 0, f"Eval failed: {result.stderr}"

        # Parse and report individual evaluation results
        output = result.stdout + result.stderr
        eval_results = parse_eval_output(output)
        table_results = parse_results_table(output)

        # Build GitHub Summary markdown
        summary_lines = [
            "## 🧮 Calculator Agent Evaluation (test_eval_calculator)",
            "",
            "| Test Case | Status | Scores |",
            "|-----------|--------|--------|",
        ]

        # Report each evaluation result as a GitHub Actions annotation
        failed_evals = []
        for eval_result in eval_results:
            if not eval_result.passed:
                failed_evals.append(
                    f"{eval_result.name}: FAILED - {eval_result.error or 'Unknown error'}"
                )
                emit_github_annotation(
                    "error",
                    eval_result.error or "Unknown error",
                    title=f"Eval FAILED: {eval_result.name}",
                )
                error_msg = eval_result.error or "Unknown error"
                summary_lines.append(f"| {eval_result.name} | ❌ Fail | {error_msg} |")
            else:
                # Log scores for passing tests
                score_info = ", ".join(
                    f"{k}={v}%" for k, v in eval_result.scores.items()
                )
                emit_github_annotation(
                    "notice",
                    score_info or "PASSED",
                    title=f"Eval PASSED: {eval_result.name}",
                )
                scores_str = ", ".join(
                    f"{k}: {v}%" for k, v in eval_result.scores.items()
                )
                summary_lines.append(
                    f"| {eval_result.name} | ✅ Pass | {scores_str or 'N/A'} |"
                )

        # Also check table results for scores
        for test_name, scores in table_results.items():
            for evaluator, score in scores.items():
                if score < 100.0:
                    emit_github_annotation(
                        "warning",
                        f"{evaluator}: {score}%",
                        title=f"Eval Score: {test_name}",
                    )

        # Add summary statistics
        passed_count = sum(1 for r in eval_results if r.passed)
        failed_count = sum(1 for r in eval_results if not r.passed)
        summary_lines.extend(
            [
                "",
                f"**Total:** {len(eval_results)} evaluations | "
                f"✅ {passed_count} passed | ❌ {failed_count} failed",
                "",
            ]
        )

        # Write GitHub summary only (PR comment handled by detailed test)
        write_github_summary("\n".join(summary_lines))

        if failed_evals:
            pytest.fail("Some evaluations failed:\n" + "\n".join(failed_evals))

        # Verify we got some results
        assert eval_results or table_results, (
            f"No evaluation results parsed from output: {output[:500]}"
        )

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

        assert result.returncode == 0, f"Eval failed: {result.stderr}"

        # Parse and verify results
        output = result.stdout + result.stderr
        eval_results = parse_eval_output(output)
        table_results = parse_results_table(output)

        # Report individual results as GitHub Actions annotations
        for eval_result in eval_results:
            if eval_result.passed:
                score_info = ", ".join(
                    f"{k}={v}%" for k, v in eval_result.scores.items()
                )
                emit_github_annotation(
                    "notice",
                    score_info or "PASSED",
                    title=f"Eval PASSED: {eval_result.name}",
                )
            else:
                emit_github_annotation(
                    "error",
                    eval_result.error or "Failed",
                    title=f"Eval FAILED: {eval_result.name}",
                )

        # Verify we got results
        assert (
            eval_results or table_results or "evaluation results" in output.lower()
        ), f"Expected evaluation results in output, got: {output[:500]}"


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


class TestEvalResultsDetailed:
    """Tests that run evaluations and report detailed results per test case."""

    # TODO: Add test_calculator_same_as_agent_evaluations after PR #62 is merged

    @pytest.mark.e2e
    def test_calculator_evaluations_detailed(
        self, authenticated_session: dict[str, str], project_id: str
    ):
        """Run calculator evaluations and report each test case individually."""
        if not project_id:
            pytest.skip("UIPATH_PROJECT_ID not set")

        example_dir = EXAMPLES_DIR / "calculator"

        result = run_uipath_command(
            command=["eval", "agent.json", "--no-report"],
            cwd=example_dir,
            env=authenticated_session,
            timeout=300,
        )

        # Parse output even if command fails
        output = result.stdout + result.stderr
        eval_results = parse_eval_output(output)
        table_results = parse_results_table(output)

        # Build detailed report and emit GitHub Actions annotations
        report_lines = ["Evaluation Results:"]

        # Build GitHub Summary markdown
        summary_lines = [
            "## 📊 Calculator Agent Evaluation Results",
            "",
            "| Test Case | Status | Scores |",
            "|-----------|--------|--------|",
        ]

        # Report from parsed inline results
        for eval_result in eval_results:
            status = "PASS" if eval_result.passed else "FAIL"
            report_lines.append(f"\n  [{status}] {eval_result.name}")

            if eval_result.passed:
                score_info = ", ".join(
                    f"{k}={v}%" for k, v in eval_result.scores.items()
                )
                emit_github_annotation(
                    "notice",
                    score_info or "PASSED",
                    title=f"Eval PASSED: {eval_result.name}",
                )
                for evaluator, score in eval_result.scores.items():
                    report_lines.append(f"    - {evaluator}: {score}%")
                # Add to summary table
                scores_str = ", ".join(
                    f"{k}: {v}%" for k, v in eval_result.scores.items()
                )
                summary_lines.append(
                    f"| {eval_result.name} | ✅ Pass | {scores_str or 'N/A'} |"
                )
            else:
                emit_github_annotation(
                    "error",
                    eval_result.error or "Failed",
                    title=f"Eval FAILED: {eval_result.name}",
                )
                if eval_result.error:
                    report_lines.append(f"    - Error: {eval_result.error}")
                # Add to summary table
                error_msg = eval_result.error or "Failed"
                summary_lines.append(f"| {eval_result.name} | ❌ Fail | {error_msg} |")

        # Report from table results
        if table_results:
            report_lines.append("\n  Summary Table:")
            for test_name, scores in table_results.items():
                score_str = ", ".join(f"{k}: {v}%" for k, v in scores.items())
                report_lines.append(f"    {test_name}: {score_str}")

        # Add summary statistics
        passed_count = sum(1 for r in eval_results if r.passed)
        failed_count = sum(1 for r in eval_results if not r.passed)
        summary_lines.extend(
            [
                "",
                f"**Total:** {len(eval_results)} evaluations | "
                f"✅ {passed_count} passed | ❌ {failed_count} failed",
                "",
            ]
        )

        # Write GitHub summary and results file for PR comment
        summary_content = "\n".join(summary_lines)
        write_github_summary(summary_content)
        write_eval_results_file(summary_content)

        print("\n".join(report_lines))

        # Assert command succeeded
        assert result.returncode == 0, (
            f"Eval command failed:\n{result.stderr}\n\n"
            f"Parsed results:\n" + "\n".join(report_lines)
        )

        # Assert we got results
        assert eval_results or table_results, "No evaluation results found in output"

        # Check for failures
        failures = [r for r in eval_results if not r.passed]
        if failures:
            failure_details = "\n".join(
                f"  - {r.name}: {r.error or 'Failed'}" for r in failures
            )
            pytest.fail(f"Some evaluations failed:\n{failure_details}")
