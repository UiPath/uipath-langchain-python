"""Tests for span utility classes."""

from typing import Generator
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from uipath_agents._observability.llmops.span_hierarchy import SpanHierarchyManager
from uipath_agents._observability.llmops.spans.span_keys import SpanKeys


class TestSpanKeys:
    """Tests for SpanKeys utility class."""

    def test_model_key_differs_from_run_id(self) -> None:
        run_id = UUID("12345678-1234-5678-1234-567812345678")
        model_key = SpanKeys.model(run_id)
        assert model_key != run_id

    def test_model_key_is_deterministic(self) -> None:
        run_id = UUID("12345678-1234-5678-1234-567812345678")
        key1 = SpanKeys.model(run_id)
        key2 = SpanKeys.model(run_id)
        assert key1 == key2

    def test_model_key_uses_xor_1(self) -> None:
        run_id = UUID("12345678-1234-5678-1234-567812345678")
        expected = UUID(int=run_id.int ^ 1)
        assert SpanKeys.model(run_id) == expected

    def test_tool_child_key_differs_from_run_id(self) -> None:
        run_id = UUID("12345678-1234-5678-1234-567812345678")
        child_key = SpanKeys.tool_child(run_id)
        assert child_key != run_id

    def test_tool_child_key_differs_from_model_key(self) -> None:
        run_id = UUID("12345678-1234-5678-1234-567812345678")
        model_key = SpanKeys.model(run_id)
        child_key = SpanKeys.tool_child(run_id)
        assert model_key != child_key

    def test_tool_child_key_is_deterministic(self) -> None:
        run_id = UUID("12345678-1234-5678-1234-567812345678")
        key1 = SpanKeys.tool_child(run_id)
        key2 = SpanKeys.tool_child(run_id)
        assert key1 == key2

    def test_tool_child_key_uses_xor_2(self) -> None:
        run_id = UUID("12345678-1234-5678-1234-567812345678")
        expected = UUID(int=run_id.int ^ 2)
        assert SpanKeys.tool_child(run_id) == expected

    def test_all_three_keys_are_unique(self) -> None:
        run_id = UUID("12345678-1234-5678-1234-567812345678")
        model_key = SpanKeys.model(run_id)
        child_key = SpanKeys.tool_child(run_id)
        assert len({run_id, model_key, child_key}) == 3


class TestSpanHierarchyManager:
    """Tests for SpanHierarchyManager class."""

    @pytest.fixture(autouse=True)
    def cleanup(self) -> Generator[None, None, None]:
        """Clean up all stacks before and after each test."""
        SpanHierarchyManager.clear_all()
        yield
        SpanHierarchyManager.clear_all()

    def test_initialize_creates_stack_with_root_span(self) -> None:
        run_id = UUID("12345678-1234-5678-1234-567812345678")
        root_span = MagicMock()

        SpanHierarchyManager.initialize(run_id, root_span)

        assert SpanHierarchyManager.has_stack(run_id)
        assert SpanHierarchyManager.current(run_id) == root_span

    def test_push_adds_span_to_stack(self) -> None:
        run_id = UUID("12345678-1234-5678-1234-567812345678")
        root_span = MagicMock()
        child_span = MagicMock()

        SpanHierarchyManager.initialize(run_id, root_span)
        SpanHierarchyManager.push(run_id, child_span)

        assert SpanHierarchyManager.current(run_id) == child_span

    def test_push_returns_true_on_success(self) -> None:
        run_id = UUID("12345678-1234-5678-1234-567812345678")
        span = MagicMock()

        result = SpanHierarchyManager.push(run_id, span)

        assert result is True

    def test_push_returns_false_for_none_span(self) -> None:
        run_id = UUID("12345678-1234-5678-1234-567812345678")

        result = SpanHierarchyManager.push(run_id, None)  # type: ignore

        assert result is False

    def test_pop_returns_lifo_order(self) -> None:
        run_id = UUID("12345678-1234-5678-1234-567812345678")
        span1 = MagicMock(name="span1")
        span2 = MagicMock(name="span2")

        SpanHierarchyManager.initialize(run_id, span1)
        SpanHierarchyManager.push(run_id, span2)

        assert SpanHierarchyManager.pop(run_id) == span2
        assert SpanHierarchyManager.pop(run_id) == span1

    def test_pop_empty_stack_returns_none(self) -> None:
        run_id = UUID("12345678-1234-5678-1234-567812345678")
        span = MagicMock()

        SpanHierarchyManager.initialize(run_id, span)
        SpanHierarchyManager.pop(run_id)  # Pop the root

        result = SpanHierarchyManager.pop(run_id)
        assert result is None

    def test_pop_nonexistent_stack_returns_none(self) -> None:
        run_id = UUID("12345678-1234-5678-1234-567812345678")

        result = SpanHierarchyManager.pop(run_id)

        assert result is None

    def test_current_returns_top_without_popping(self) -> None:
        run_id = UUID("12345678-1234-5678-1234-567812345678")
        span = MagicMock()

        SpanHierarchyManager.initialize(run_id, span)

        assert SpanHierarchyManager.current(run_id) == span
        assert SpanHierarchyManager.current(run_id) == span  # Still there

    def test_current_empty_stack_returns_none(self) -> None:
        run_id = UUID("12345678-1234-5678-1234-567812345678")

        result = SpanHierarchyManager.current(run_id)

        assert result is None

    def test_ancestors_returns_all_spans(self) -> None:
        run_id = UUID("12345678-1234-5678-1234-567812345678")
        span1 = MagicMock(name="span1")
        span2 = MagicMock(name="span2")
        span3 = MagicMock(name="span3")

        SpanHierarchyManager.initialize(run_id, span1)
        SpanHierarchyManager.push(run_id, span2)
        SpanHierarchyManager.push(run_id, span3)

        ancestors = SpanHierarchyManager.ancestors(run_id)

        assert ancestors == [span1, span2, span3]

    def test_ancestors_returns_copy(self) -> None:
        run_id = UUID("12345678-1234-5678-1234-567812345678")
        span = MagicMock()

        SpanHierarchyManager.initialize(run_id, span)
        ancestors = SpanHierarchyManager.ancestors(run_id)

        ancestors.append(MagicMock())  # Modify returned list
        assert len(SpanHierarchyManager.ancestors(run_id)) == 1  # Original unchanged

    def test_cleanup_removes_stack(self) -> None:
        run_id = UUID("12345678-1234-5678-1234-567812345678")
        span = MagicMock()

        SpanHierarchyManager.initialize(run_id, span)
        SpanHierarchyManager.cleanup(run_id)

        assert not SpanHierarchyManager.has_stack(run_id)

    def test_cleanup_nonexistent_stack_is_safe(self) -> None:
        run_id = UUID("12345678-1234-5678-1234-567812345678")

        SpanHierarchyManager.cleanup(run_id)  # Should not raise

    def test_clear_all_removes_all_stacks(self) -> None:
        run_id1 = UUID("12345678-1234-5678-1234-567812345678")
        run_id2 = UUID("87654321-4321-8765-4321-876543218765")

        SpanHierarchyManager.initialize(run_id1, MagicMock())
        SpanHierarchyManager.initialize(run_id2, MagicMock())

        SpanHierarchyManager.clear_all()

        assert not SpanHierarchyManager.has_stack(run_id1)
        assert not SpanHierarchyManager.has_stack(run_id2)

    def test_parallel_runs_isolated(self) -> None:
        run_id1 = UUID("12345678-1234-5678-1234-567812345678")
        run_id2 = UUID("87654321-4321-8765-4321-876543218765")
        span1 = MagicMock(name="span1")
        span2 = MagicMock(name="span2")

        SpanHierarchyManager.initialize(run_id1, span1)
        SpanHierarchyManager.initialize(run_id2, span2)

        assert SpanHierarchyManager.current(run_id1) == span1
        assert SpanHierarchyManager.current(run_id2) == span2

        SpanHierarchyManager.cleanup(run_id1)
        assert not SpanHierarchyManager.has_stack(run_id1)
        assert SpanHierarchyManager.has_stack(run_id2)
