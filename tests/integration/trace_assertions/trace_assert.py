"""Core trace assertion utilities for LLMOPS format."""

import json
import os
from pathlib import Path
from typing import Any

from .matchers import FormatValidator, ValueMatcher

TRACE_OUTPUT_PATH = os.getenv("TRACE_OUTPUT_PATH", "/tmp/trace.json")


def load_trace(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load trace from LLMOPS JSONL file.

    Args:
        path: Path to trace file. Defaults to TRACE_OUTPUT_PATH.

    Returns:
        List of span dicts with parsed attributes.
    """
    path = Path(path) if path else Path(TRACE_OUTPUT_PATH)
    spans: list[dict[str, Any]] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                span = json.loads(line)
                # Parse JSON string attributes
                if isinstance(span.get("Attributes"), str):
                    span["Attributes"] = json.loads(span["Attributes"])
                spans.append(span)

    return spans


def load_expected(path: str | Path) -> dict[str, Any]:
    """Load expected trace definition from JSON file.

    Args:
        path: Path to expected trace JSON.

    Returns:
        Expected trace definition dict.
    """
    with open(path) as f:
        return json.load(f)


class TraceAsserter:
    """Asserts trace spans match expected definitions."""

    def __init__(self, spans: list[dict[str, Any]]) -> None:
        """Initialize with loaded spans.

        Args:
            spans: List of span dicts from load_trace().
        """
        self.spans = spans
        self._by_id: dict[str, dict[str, Any]] = {s["Id"]: s for s in spans}
        self._by_name: dict[str, list[dict[str, Any]]] = {}
        for s in spans:
            self._by_name.setdefault(s["Name"], []).append(s)

    def find_span(
        self, name: str, parent_name: str | None = None
    ) -> dict[str, Any] | None:
        """Find span by name and optional parent.

        Args:
            name: Span name to find.
            parent_name: Expected parent span name, or None for root.

        Returns:
            Matching span or None.
        """
        candidates = self._by_name.get(name, [])
        for span in candidates:
            parent_id = span.get("ParentId")
            if parent_name is None:
                if parent_id is None:
                    return span
            else:
                if parent_id and parent_id in self._by_id:
                    parent = self._by_id[parent_id]
                    if parent["Name"] == parent_name:
                        return span
        return None

    def assert_span(
        self,
        span: dict[str, Any],
        expected: dict[str, Any],
    ) -> list[str]:
        """Assert span matches expected definition.

        Args:
            span: Actual span dict.
            expected: Expected span definition with:
                - attributes: exact/wildcard matches
                - attributes_format: format validations
                - optional_attributes: validate if present
                - forbidden_attributes: must not exist

        Returns:
            List of error messages (empty if all pass).
        """
        errors: list[str] = []
        attrs = span.get("Attributes", {})

        # Check required attributes (exact/wildcard)
        for key, exp_val in expected.get("attributes", {}).items():
            if key not in attrs:
                errors.append(f"Missing attribute: {key}")
            elif not ValueMatcher.matches(attrs[key], exp_val):
                errors.append(f"Attribute {key}: expected {exp_val}, got {attrs[key]}")

        # Check format validations
        for key, fmt in expected.get("attributes_format", {}).items():
            if key not in attrs:
                errors.append(f"Missing attribute for format check: {key}")
            elif not FormatValidator.validate(attrs[key], fmt):
                errors.append(
                    f"Attribute {key}: invalid {fmt} format, got {attrs[key]}"
                )

        # Check optional attributes (validate if present)
        for key, exp_val in expected.get("optional_attributes", {}).items():
            if key in attrs and not ValueMatcher.matches(attrs[key], exp_val):
                errors.append(
                    f"Optional attribute {key}: expected {exp_val}, got {attrs[key]}"
                )

        # Check forbidden attributes
        for key in expected.get("forbidden_attributes", []):
            if key in attrs:
                errors.append(f"Forbidden attribute present: {key}")

        return errors

    def assert_all(self, expected_def: dict[str, Any]) -> None:
        """Assert all required spans match expected definition.

        Args:
            expected_def: Full expected trace definition with required_spans.

        Raises:
            AssertionError: If any span missing or validation fails.
        """
        all_errors: list[str] = []

        for req_span in expected_def.get("required_spans", []):
            name = req_span["name"]
            parent = req_span.get("parent")

            span = self.find_span(name, parent)
            if span is None:
                parent_desc = f" (parent: {parent})" if parent else " (root)"
                all_errors.append(f"Missing span: {name}{parent_desc}")
                continue

            errors = self.assert_span(span, req_span)
            for err in errors:
                all_errors.append(f"[{name}] {err}")

        if all_errors:
            raise AssertionError(
                "Trace assertion failed:\n" + "\n".join(f"  - {e}" for e in all_errors)
            )


def assert_trace(
    expected_path: str | Path,
    trace_path: str | Path | None = None,
) -> None:
    """Convenience function to assert trace against expected definition.

    Args:
        expected_path: Path to expected trace JSON.
        trace_path: Path to actual trace. Defaults to TRACE_OUTPUT_PATH.

    Raises:
        AssertionError: If validation fails.
    """
    spans = load_trace(trace_path)
    expected = load_expected(expected_path)
    asserter = TraceAsserter(spans)
    asserter.assert_all(expected)
