"""Span key derivation utilities for nested span storage.

When storing multiple related spans per run_id (e.g., LLM + model, tool + child),
we need unique keys. XOR-based key derivation creates deterministic, collision-free
keys that can be computed at both storage and retrieval time.
"""

from uuid import UUID


class SpanKeys:
    """Derives unique span keys from run_id using XOR operations.

    Enables storing multiple related spans per run_id in a single dictionary:
    - LLM call span at run_id
    - Model span at run_id ^ 1
    - Tool child span at run_id ^ 2

    Usage:
        spans[run_id] = llm_span
        spans[SpanKeys.model(run_id)] = model_span
        spans[SpanKeys.tool_child(run_id)] = child_span
    """

    # XOR constants for different span types
    MODEL_XOR = 1
    TOOL_CHILD_XOR = 2

    @staticmethod
    def model(run_id: UUID) -> UUID:
        """Derive key for model span from LLM run_id.

        We create two spans per LLM call: outer LLM span + inner model span.
        XOR with 1 creates a distinct but deterministic key.

        Args:
            run_id: The LLM call's run_id

        Returns:
            Unique key for storing the model span
        """
        return UUID(int=run_id.int ^ SpanKeys.MODEL_XOR)

    @staticmethod
    def tool_child(run_id: UUID) -> UUID:
        """Derive key for interruptible tool child span from tool run_id.

        Tool calls may have child spans (escalation, process, agent).
        XOR with 2 creates a distinct key from both run_id and model keys.

        Args:
            run_id: The tool call's run_id

        Returns:
            Unique key for storing the tool child span
        """
        return UUID(int=run_id.int ^ SpanKeys.TOOL_CHILD_XOR)
