"""Register UiPath HITL interrupt/recipient types as safe LangGraph msgpack types.

The UiPath HITL primitives (``interrupt(CreateTask / CreateEscalation / ...)``)
store their payloads in the LangGraph checkpoint. LangGraph's default
(permissive) msgpack serializer logs a ``Deserializing unregistered type ...``
warning for every type that isn't in its built-in ``SAFE_MSGPACK_TYPES``
allowlist, and warns that such types "will be blocked in a future version".

LangGraph's allowlist is binary — permissive (warn on everything unknown) or
strict (block everything not explicitly listed). Flipping the whole runtime to
strict would risk blocking unrelated agent-state types (custom Pydantic models,
etc.), so instead we extend the *safe* set with the UiPath interrupt / recipient
/ task types. HITL resume keeps working and stays quiet, while every other type
retains the default permissive behavior.

This is a best-effort compatibility shim: it tolerates LangGraph internals
changing (degrades to a no-op) and the UiPath types being unavailable.
"""

import logging

logger = logging.getLogger(__name__)


def _collect_safe_type_keys() -> set[tuple[str, str]]:
    """Collect ``(module, qualname)`` keys for the UiPath HITL payload types."""
    keys: set[tuple[str, str]] = set()

    def _add(*classes: type) -> None:
        for cls in classes:
            keys.add((cls.__module__, cls.__name__))

    try:
        from uipath.platform.common import interrupt_models as im

        _add(
            im.CreateTask,
            im.CreateEscalation,
            im.WaitTask,
            im.WaitEscalation,
            im.InvokeProcess,
            im.InvokeProcessRaw,
            im.WaitJob,
            im.WaitJobRaw,
        )
    except Exception:  # pragma: no cover - defensive
        logger.debug(
            "UiPath interrupt_models unavailable for msgpack registration",
            exc_info=True,
        )

    try:
        from uipath.platform.action_center.tasks import (
            Task,
            TaskRecipient,
            TaskRecipientType,
        )

        _add(Task, TaskRecipient, TaskRecipientType)
    except Exception:  # pragma: no cover - defensive
        logger.debug("UiPath task types unavailable for msgpack registration")

    try:
        from uipath.platform.orchestrator.job import Job

        _add(Job)
    except Exception:  # pragma: no cover - defensive
        logger.debug("UiPath Job type unavailable for msgpack registration")

    return keys


def register_uipath_msgpack_safe_types() -> None:
    """Extend LangGraph's msgpack safe-type allowlist with UiPath HITL types.

    Idempotent and best-effort — safe to call more than once, and a no-op if
    LangGraph's internals are not in the expected shape.
    """
    try:
        from langgraph.checkpoint.serde import _msgpack as lg_msgpack
    except Exception:  # pragma: no cover - LangGraph layout changed
        logger.debug("LangGraph msgpack module not found; skipping registration")
        return

    extra = _collect_safe_type_keys()
    if not extra:
        return

    try:
        current = set(lg_msgpack.SAFE_MSGPACK_TYPES)
        if extra <= current:
            return
        # jsonplus reads ``_msgpack.SAFE_MSGPACK_TYPES`` dynamically per check,
        # so rebinding the module attribute is honored by all serializers.
        lg_msgpack.SAFE_MSGPACK_TYPES = frozenset(current | extra)
    except Exception:  # pragma: no cover - defensive
        logger.debug("Could not extend SAFE_MSGPACK_TYPES", exc_info=True)
