"""Span hierarchy management for parent-child relationships.

Encapsulates global state for parallel agent execution support.
Each run_id has its own hierarchy to track span parent-child relationships.
"""

import logging
from typing import ClassVar, Dict, List, Optional
from uuid import UUID

from opentelemetry.trace import Span

logger = logging.getLogger(__name__)


class SpanHierarchyManager:
    """Manages span parent-child relationships via per-run stacks.

    Encapsulates global state for parallel agent execution support.
    Each run_id has its own stack to track span hierarchy.

    Usage:
        # Initialize for a new run
        SpanHierarchyManager.initialize(run_id, root_span)

        # Push/pop spans
        SpanHierarchyManager.push(run_id, child_span)
        parent = SpanHierarchyManager.current(run_id)
        child = SpanHierarchyManager.pop(run_id)

        # Cleanup when done
        SpanHierarchyManager.cleanup(run_id)
    """

    _stacks: ClassVar[Dict[UUID, List[Span]]] = {}

    @classmethod
    def initialize(cls, run_id: UUID, root_span: Span) -> None:
        """Initialize stack for a new run with root span.

        Args:
            run_id: Unique identifier for this execution run
            root_span: The root agent span for this run
        """
        cls._stacks[run_id] = [root_span]

    @classmethod
    def push(cls, run_id: UUID, span: Span) -> bool:
        """Push span onto run's stack.

        Args:
            run_id: The run ID to push the span to
            span: The span to push

        Returns:
            True if push succeeded, False if span is None
        """
        if span is None:
            logger.error("Attempted to push None span for run_id %s", run_id)
            return False
        if run_id not in cls._stacks:
            cls._stacks[run_id] = []
        cls._stacks[run_id].append(span)
        return True

    @classmethod
    def pop(cls, run_id: UUID) -> Optional[Span]:
        """Pop span from run's stack.

        Args:
            run_id: The run ID to pop from

        Returns:
            The popped span, or None if stack is empty/missing
        """
        if run_id not in cls._stacks:
            logger.warning("No span stack for run_id %s", run_id)
            return None
        if not cls._stacks[run_id]:
            logger.warning(
                "Empty span stack for run_id %s - possible push/pop mismatch", run_id
            )
            return None
        return cls._stacks[run_id].pop()

    @classmethod
    def current(cls, run_id: UUID) -> Optional[Span]:
        """Get current span without popping.

        Args:
            run_id: The run ID to get current span for

        Returns:
            The current (top) span, or None if stack is empty/missing
        """
        if run_id in cls._stacks and cls._stacks[run_id]:
            return cls._stacks[run_id][-1]
        return None

    @classmethod
    def ancestors(cls, run_id: UUID) -> List[Span]:
        """Get all ancestor spans for a run.

        Args:
            run_id: The run ID to get ancestors for

        Returns:
            Copy of the span stack (root first, current last)
        """
        return list(cls._stacks.get(run_id, []))

    @classmethod
    def cleanup(cls, run_id: UUID) -> None:
        """Remove run's stack to prevent memory leak.

        Args:
            run_id: The run ID to clean up
        """
        if run_id in cls._stacks:
            del cls._stacks[run_id]

    @classmethod
    def clear_all(cls) -> None:
        """Clear all stacks (for testing)."""
        cls._stacks.clear()

    @classmethod
    def has_stack(cls, run_id: UUID) -> bool:
        """Check if a stack exists for the given run_id.

        Args:
            run_id: The run ID to check

        Returns:
            True if a stack exists, False otherwise
        """
        return run_id in cls._stacks
