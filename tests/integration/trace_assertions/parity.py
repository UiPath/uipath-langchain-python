"""Parity-based trace assertions — golden is truth, config overrides exceptions."""

import fnmatch
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from .matchers import FormatValidator


@dataclass
class MatchOverrides:
    """Override rules for fields or attributes. Supports glob wildcards."""

    format: dict[str, str] = field(default_factory=dict)
    ignore: list[str] = field(default_factory=list)


@dataclass
class ParityConfig:
    """Parity config: everything in golden is checked by default.

    Override specific fields/attributes with format checks or ignores.
    Wildcards supported (e.g. ``*Id``, ``*Time*``).
    """

    description: str = ""
    ignore_spans: list[str] = field(default_factory=list)
    fields: MatchOverrides = field(default_factory=MatchOverrides)
    attributes: MatchOverrides = field(default_factory=MatchOverrides)


def load_trace_json(path: Path) -> list[dict[str, Any]]:
    """Load trace from JSON array or JSONL file."""
    with open(path) as f:
        content = f.read().strip()

    if content.startswith("["):
        spans = json.loads(content)
    else:
        spans = [json.loads(line) for line in content.splitlines() if line.strip()]

    for span in spans:
        if isinstance(span.get("Attributes"), str):
            span["Attributes"] = json.loads(span["Attributes"])

    return spans


def load_config(path: Path | None) -> ParityConfig:
    """Load config from JSON file or return defaults."""
    if path is None:
        return ParityConfig()

    with open(path) as f:
        data = json.load(f)

    fields_data = data.get("fields", {})
    attrs_data = data.get("attributes", {})

    return ParityConfig(
        description=data.get("description", ""),
        ignore_spans=data.get("ignore_spans", []),
        fields=MatchOverrides(
            format=fields_data.get("format", {}),
            ignore=fields_data.get("ignore", []),
        ),
        attributes=MatchOverrides(
            format=attrs_data.get("format", {}),
            ignore=attrs_data.get("ignore", []),
        ),
    )


# --- Matching helpers ---


def _matches_any(name: str, patterns: list[str]) -> bool:
    """Check if name matches any glob pattern."""
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def _find_format(name: str, format_rules: dict[str, str]) -> str | None:
    """Find the format rule for a name. Exact match wins over wildcards."""
    if name in format_rules:
        return format_rules[name]
    for pattern, fmt in format_rules.items():
        if fnmatch.fnmatch(name, pattern):
            return fmt
    return None


def _check_overrides(
    golden_dict: dict[str, Any],
    actual_dict: dict[str, Any],
    overrides: MatchOverrides,
    prefix: str = "",
) -> list[str]:
    """Check golden keys against actual using override rules.

    For each key in golden:
      1. Matches ignore pattern → skip
      2. Matches format pattern → validate format in actual
      3. Otherwise → exact match
    """
    errors: list[str] = []

    for key, golden_val in golden_dict.items():
        # 1. Ignore
        if _matches_any(key, overrides.ignore):
            continue

        # 2. Format check (null golden values fall through to exact match)
        fmt = _find_format(key, overrides.format)
        if fmt is not None and golden_val is not None:
            actual_val = actual_dict.get(key)
            if actual_val is None:
                errors.append(f"{prefix}{key}: missing in actual (format: {fmt})")
            elif not FormatValidator.validate(actual_val, fmt):
                errors.append(f"{prefix}{key}: invalid {fmt} format: {actual_val!r}")
            continue

        # 3. Exact match
        actual_val = actual_dict.get(key)
        if key not in actual_dict:
            errors.append(f"{prefix}{key}: missing in actual, expected {golden_val!r}")
        elif golden_val != actual_val:
            errors.append(
                f"{prefix}{key}:\n"
                f"      expected: {golden_val!r}\n"
                f"      actual:   {actual_val!r}"
            )

    return errors


# --- ASCII Tree Rendering ---


def render_span_tree(spans: list[dict[str, Any]]) -> str:
    """Render spans as an ASCII tree showing parent-child relationships."""
    children: dict[str | None, list[dict[str, Any]]] = {}
    for s in spans:
        children.setdefault(s.get("ParentId"), []).append(s)

    def _render(parent_id: str | None, prefix: str) -> list[str]:
        lines: list[str] = []
        kids = children.get(parent_id, [])
        for i, child in enumerate(kids):
            is_last = i == len(kids) - 1
            connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
            lines.append(f"{prefix}{connector}{child['Name']}")
            extension = "    " if is_last else "\u2502   "
            lines.extend(_render(child["Id"], prefix + extension))
        return lines

    roots = children.get(None, [])
    result: list[str] = []
    for root in roots:
        result.append(root["Name"])
        result.extend(_render(root["Id"], ""))
    return "\n".join(result)


def _tree_diff_message(
    golden: list[dict[str, Any]], actual: list[dict[str, Any]]
) -> str:
    """Build a combined tree-diff error message."""
    return (
        "\n\nExpected tree:\n"
        + _indent(render_span_tree(golden))
        + "\n\nActual tree:\n"
        + _indent(render_span_tree(actual))
    )


def _indent(text: str, spaces: int = 2) -> str:
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())


def print_trace_summary(
    golden_path: Path | str,
    actual_path: Path | str,
    config_path: Path | str | None = None,
) -> None:
    """Print a visual summary of golden vs actual trace trees and config to stderr."""
    golden = load_trace_json(Path(golden_path))
    actual = load_trace_json(Path(actual_path))
    config = load_config(Path(config_path) if config_path else None)

    golden_filtered = [s for s in golden if s["Name"] not in config.ignore_spans]
    actual_filtered = [s for s in actual if s["Name"] not in config.ignore_spans]

    w = sys.stderr.write

    w("\n")
    w("=" * 60 + "\n")
    desc = config.description or "Trace Parity"
    w(f"  {desc}\n")
    w("=" * 60 + "\n")

    # Fields
    w("\n  Field checks:\n")
    if config.fields.format:
        fmts = [f"{p}={v}" for p, v in config.fields.format.items()]
        w(f"    Format: {', '.join(fmts)}\n")
    if config.fields.ignore:
        w(f"    Ignore: {', '.join(config.fields.ignore)}\n")
    w("    Exact:  everything else in golden\n")

    # Attributes
    w("\n  Attribute checks:\n")
    if config.attributes.format:
        fmts = [f"{p}={v}" for p, v in config.attributes.format.items()]
        w(f"    Format: {', '.join(fmts)}\n")
    if config.attributes.ignore:
        ign = config.attributes.ignore
        w(f"    Ignore: {', '.join(ign[:6])}")
        if len(ign) > 6:
            w(f" ... (+{len(ign) - 6} more)")
        w("\n")
    w("    Exact:  everything else in golden\n")

    w(f"\n  Expected tree ({len(golden_filtered)} spans):\n")
    for line in render_span_tree(golden_filtered).splitlines():
        w(f"    {line}\n")

    w(f"\n  Actual tree ({len(actual_filtered)} spans):\n")
    for line in render_span_tree(actual_filtered).splitlines():
        w(f"    {line}\n")

    match = "MATCH" if len(golden_filtered) == len(actual_filtered) else "MISMATCH"
    w(
        f"\n  Span count: {len(golden_filtered)} expected, {len(actual_filtered)} actual [{match}]\n"
    )
    w("=" * 60 + "\n\n")


# --- Per-Span Assertions ---


def get_golden_span_names(
    golden_path: Path | str,
    config_path: Path | str | None = None,
) -> list[str]:
    """Return span names from golden file (filtered by config) for parametrization."""
    golden = load_trace_json(Path(golden_path))
    config = load_config(Path(config_path) if config_path else None)
    return [s["Name"] for s in golden if s["Name"] not in config.ignore_spans]


def assert_span_hierarchy(
    span_name: str,
    golden_path: Path | str,
    actual_path: Path | str,
    config_path: Path | str | None = None,
) -> None:
    """Assert a single span exists in actual with the correct parent.

    Raises:
        AssertionError: With ASCII tree diff on failure.
    """
    golden = load_trace_json(Path(golden_path))
    actual = load_trace_json(Path(actual_path))
    config = load_config(Path(config_path) if config_path else None)

    golden_filtered = [s for s in golden if s["Name"] not in config.ignore_spans]
    actual_filtered = [s for s in actual if s["Name"] not in config.ignore_spans]

    golden_by_id = {s["Id"]: s for s in golden}
    actual_by_id = {s["Id"]: s for s in actual}

    golden_span = next((s for s in golden_filtered if s["Name"] == span_name), None)
    if golden_span is None:
        raise AssertionError(f"Span {span_name!r} not found in golden file")

    expected_parent = _get_parent_name(golden_span, golden_by_id)
    actual_span = next((s for s in actual_filtered if s["Name"] == span_name), None)

    if actual_span is None:
        parent_str = f" (parent: {expected_parent})" if expected_parent else " (root)"
        raise AssertionError(
            f"Missing span: {span_name!r}{parent_str}"
            + _tree_diff_message(golden_filtered, actual_filtered)
        )

    actual_parent = _get_parent_name(actual_span, actual_by_id)
    if actual_parent != expected_parent:
        raise AssertionError(
            f"Span {span_name!r}: expected parent {expected_parent!r}, "
            f"got {actual_parent!r}"
            + _tree_diff_message(golden_filtered, actual_filtered)
        )


def assert_no_extra_spans(
    golden_path: Path | str,
    actual_path: Path | str,
    config_path: Path | str | None = None,
) -> None:
    """Assert actual trace has no spans beyond those in golden.

    Raises:
        AssertionError: With ASCII tree diff listing extra spans.
    """
    golden = load_trace_json(Path(golden_path))
    actual = load_trace_json(Path(actual_path))
    config = load_config(Path(config_path) if config_path else None)

    golden_filtered = [s for s in golden if s["Name"] not in config.ignore_spans]
    actual_filtered = [s for s in actual if s["Name"] not in config.ignore_spans]

    golden_hier = _build_hierarchy(golden_filtered, golden)
    actual_hier = _build_hierarchy(actual_filtered, actual)

    extras = actual_hier - golden_hier
    if extras:
        lines = [
            f"  - {name!r} (parent: {p})" if p else f"  - {name!r} (root)"
            for name, p in extras
        ]
        raise AssertionError(
            "Extra spans in actual trace:\n"
            + "\n".join(lines)
            + _tree_diff_message(golden_filtered, actual_filtered)
        )


def assert_span_attributes(
    span_name: str,
    golden_path: Path | str,
    actual_path: Path | str,
    config_path: Path | str | None = None,
) -> None:
    """Assert a single span's fields and attributes match golden.

    Default: exact match. Config can override to format check or ignore.

    Raises:
        AssertionError: With readable diff on failure.
    """
    golden = load_trace_json(Path(golden_path))
    actual = load_trace_json(Path(actual_path))
    config = load_config(Path(config_path) if config_path else None)

    golden_by_name = {
        s["Name"]: s for s in golden if s["Name"] not in config.ignore_spans
    }
    actual_by_name = {
        s["Name"]: s for s in actual if s["Name"] not in config.ignore_spans
    }

    if span_name not in golden_by_name:
        raise AssertionError(f"Span {span_name!r} not found in golden file")
    if span_name not in actual_by_name:
        pytest.skip(
            f"Span {span_name!r} missing from actual (checked by hierarchy test)"
        )

    golden_span = golden_by_name[span_name]
    actual_span = actual_by_name[span_name]

    errors: list[str] = []

    # Check top-level fields (exclude Attributes — checked separately)
    golden_fields = {k: v for k, v in golden_span.items() if k != "Attributes"}
    actual_fields = {k: v for k, v in actual_span.items() if k != "Attributes"}
    errors.extend(_check_overrides(golden_fields, actual_fields, config.fields))

    # Check nested Attributes
    golden_attrs = golden_span.get("Attributes", {}) or {}
    actual_attrs = actual_span.get("Attributes", {}) or {}
    errors.extend(
        _check_overrides(golden_attrs, actual_attrs, config.attributes, "Attributes.")
    )

    if errors:
        raise AssertionError(
            "Attribute mismatches:\n" + "\n".join(f"  - {e}" for e in errors)
        )


# --- Hierarchy helpers ---


def _get_parent_name(
    span: dict[str, Any], spans_by_id: dict[str, dict[str, Any]]
) -> str | None:
    """Get parent span name from ParentId."""
    parent_id = span.get("ParentId")
    if parent_id is None:
        return None
    parent = spans_by_id.get(parent_id)
    return parent["Name"] if parent else f"<unknown:{parent_id}>"


def _build_hierarchy(
    spans: list[dict[str, Any]], all_spans: list[dict[str, Any]]
) -> set[tuple[str, str | None]]:
    """Build set of (span_name, parent_name) tuples."""
    by_id = {s["Id"]: s for s in all_spans}
    return {(s["Name"], _get_parent_name(s, by_id)) for s in spans}


# --- Bulk Assertions ---


def assert_hierarchy(
    golden_path: Path | str,
    actual_path: Path | str,
    config_path: Path | str | None = None,
) -> None:
    """Assert actual trace has same span hierarchy as golden.

    Raises:
        AssertionError: If hierarchy doesn't match, includes ASCII tree diff.
    """
    golden = load_trace_json(Path(golden_path))
    actual = load_trace_json(Path(actual_path))
    config = load_config(Path(config_path) if config_path else None)

    golden_filtered = [s for s in golden if s["Name"] not in config.ignore_spans]
    actual_filtered = [s for s in actual if s["Name"] not in config.ignore_spans]

    errors: list[str] = []

    golden_hier = _build_hierarchy(golden_filtered, golden)
    actual_hier = _build_hierarchy(actual_filtered, actual)

    for name, parent in golden_hier - actual_hier:
        parent_str = f" (parent: {parent})" if parent else " (root)"
        errors.append(f"Missing span: {name!r}{parent_str}")

    for name, parent in actual_hier - golden_hier:
        parent_str = f" (parent: {parent})" if parent else " (root)"
        errors.append(f"Extra span: {name!r}{parent_str}")

    if errors:
        raise AssertionError(
            "Hierarchy assertion failed:\n"
            + "\n".join(f"  - {e}" for e in errors)
            + _tree_diff_message(golden_filtered, actual_filtered)
        )


def assert_attributes(
    golden_path: Path | str,
    actual_path: Path | str,
    config_path: Path | str | None = None,
) -> None:
    """Assert actual trace fields and attributes match golden per config rules.

    Raises:
        AssertionError: If any span has mismatches.
    """
    golden = load_trace_json(Path(golden_path))
    actual = load_trace_json(Path(actual_path))
    config = load_config(Path(config_path) if config_path else None)

    golden_filtered = [s for s in golden if s["Name"] not in config.ignore_spans]
    actual_filtered = [s for s in actual if s["Name"] not in config.ignore_spans]

    all_errors: list[str] = []
    golden_by_name = {s["Name"]: s for s in golden_filtered}
    actual_by_name = {s["Name"]: s for s in actual_filtered}

    for name, golden_span in golden_by_name.items():
        if name not in actual_by_name:
            continue

        actual_span = actual_by_name[name]

        golden_fields = {k: v for k, v in golden_span.items() if k != "Attributes"}
        actual_fields = {k: v for k, v in actual_span.items() if k != "Attributes"}
        for err in _check_overrides(golden_fields, actual_fields, config.fields):
            all_errors.append(f"[{name}] {err}")

        golden_attrs = golden_span.get("Attributes", {}) or {}
        actual_attrs = actual_span.get("Attributes", {}) or {}
        for err in _check_overrides(
            golden_attrs, actual_attrs, config.attributes, "Attributes."
        ):
            all_errors.append(f"[{name}] {err}")

    if all_errors:
        raise AssertionError(
            "Attributes assertion failed:\n" + "\n".join(f"  - {e}" for e in all_errors)
        )


def assert_parity(
    golden_path: Path | str,
    actual_path: Path | str,
    config_path: Path | str | None = None,
) -> None:
    """Assert both hierarchy and attributes match golden."""
    assert_hierarchy(golden_path, actual_path, config_path)
    assert_attributes(golden_path, actual_path, config_path)
