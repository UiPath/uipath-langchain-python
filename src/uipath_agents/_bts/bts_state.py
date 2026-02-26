"""Shared state for BTS runtime and callback."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

from uipath.platform.automation_tracker import (
    AutomationTrackerService,
    OperationStatus,
    TransactionStatus,
)


@dataclass
class ToolOperationState:
    """Tracks an in-flight BTS tool operation."""

    operation_id: str
    transaction_id: str
    parent_operation_id: str
    name: str
    tool_type: str
    fingerprint: str
    status: OperationStatus = OperationStatus.UNKNOWN
    result: Optional[str] = None
    attributes: dict[str, str] = field(default_factory=dict)
    start_operation_fired: bool = False
    # Runtime-only ref to tool's _bts_context dict (not serialized)
    bts_context_ref: Optional[dict[str, Any]] = field(default=None, repr=False)


@dataclass
class BtsState:
    """Shared state between BtsRuntime and BtsCallback."""

    # --- Transaction ---
    transaction_id: Optional[str] = None
    transaction_created: bool = False
    transaction_status: TransactionStatus = TransactionStatus.UNKNOWN
    transaction_name: Optional[str] = None
    transaction_fingerprint: Optional[str] = None
    transaction_result: Optional[str] = None

    # --- Agent operation ---
    agent_operation_id: Optional[str] = None
    agent_operation_status: OperationStatus = OperationStatus.UNKNOWN
    agent_operation_fingerprint: Optional[str] = None
    agent_operation_result: Optional[str] = None

    # --- Tool operations (keyed by LangChain run_id) ---
    tool_operations: dict[UUID, ToolOperationState] = field(default_factory=dict)

    # --- Ended tool operations saved for suspend/resume dedup ---
    ended_tool_operations: list[ToolOperationState] = field(default_factory=list)

    # --- Resumed tool operations keyed by (tool_type, name) for dedup ---
    resumed_tool_operations: dict[tuple[str, str], ToolOperationState] = field(
        default_factory=dict
    )

    # --- Background task tracking ---
    pending_tasks: set[asyncio.Task[None]] = field(default_factory=set)

    # --- Service reference ---
    tracker_service: Optional[AutomationTrackerService] = None
