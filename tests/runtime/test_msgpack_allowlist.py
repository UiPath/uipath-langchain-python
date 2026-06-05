"""Tests for registering UiPath HITL types in LangGraph's msgpack safe-type set.

The HITL interrupt payloads (CreateEscalation, TaskRecipient, ...) are stored in
the LangGraph checkpoint. Without registration, the permissive serializer logs a
"Deserializing unregistered type ..." warning and threatens to block them under
strict msgpack. ``register_uipath_msgpack_safe_types`` adds them to the safe set.
"""

import logging

import pytest
from langgraph.checkpoint.serde import _msgpack as lg_msgpack
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from uipath.platform.action_center.tasks import TaskRecipient, TaskRecipientType
from uipath.platform.common import CreateEscalation

from uipath_langchain.runtime._msgpack_allowlist import (
    register_uipath_msgpack_safe_types,
)


def test_registers_hitl_types_as_safe() -> None:
    register_uipath_msgpack_safe_types()
    safe = lg_msgpack.SAFE_MSGPACK_TYPES
    assert ("uipath.platform.common.interrupt_models", "CreateEscalation") in safe
    assert ("uipath.platform.common.interrupt_models", "CreateTask") in safe
    assert ("uipath.platform.action_center.tasks", "TaskRecipient") in safe
    assert ("uipath.platform.action_center.tasks", "TaskRecipientType") in safe


def test_is_idempotent() -> None:
    register_uipath_msgpack_safe_types()
    first = lg_msgpack.SAFE_MSGPACK_TYPES
    register_uipath_msgpack_safe_types()
    # No-op when everything is already registered (same frozenset object reused).
    assert lg_msgpack.SAFE_MSGPACK_TYPES is first


def test_round_trip_without_unregistered_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    register_uipath_msgpack_safe_types()
    serializer = JsonPlusSerializer()
    payload = CreateEscalation(
        app_name="EscApp",
        title="review",
        recipient=TaskRecipient(type=TaskRecipientType.EMAIL, value="x@y.com"),
    )

    with caplog.at_level(logging.WARNING, logger="langgraph.checkpoint.serde.jsonplus"):
        decoded = serializer.loads_typed(serializer.dumps_typed(payload))

    assert isinstance(decoded, CreateEscalation)
    assert decoded.recipient is not None
    assert decoded.recipient.type == TaskRecipientType.EMAIL
    assert decoded.recipient.value == "x@y.com"

    uipath_warnings = [
        r.getMessage()
        for r in caplog.records
        if "unregistered type" in r.getMessage() and "uipath.platform" in r.getMessage()
    ]
    assert uipath_warnings == [], uipath_warnings
