"""Project a run's attachments into guardrail attachment references.

Two sources feed the same reference shape:

* Agent- and LLM-scope guardrails judge the conversation so far, so they read the whole
  job-attachment registry (:func:`resolve_guardrail_attachments`).
* Tool-scope guardrails judge one tool call, so they read only the attachments that call
  mentions: the ``{"ID": ...}`` objects in its arguments before the tool runs, or in its
  result afterwards (:func:`resolve_referenced_attachments`). A tool call that names no
  file forwards nothing, even when the run holds files elsewhere; otherwise every tool
  call would ship every file to the backend.

Any built-in guardrail forwards references unless it is scoped to prompts. The runtime
forwards id, file name and mime type only; the backend's feature flag decides whether
they are used at all, the backend decides which validators and file types it can
inspect, and it resolves the id through Orchestrator.

Nothing in this module raises: the guardrail node re-raises any exception, which would
end the run over a single malformed attachment.
"""

import logging
import uuid
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from pydantic import BaseModel
from uipath.platform.attachments import Attachment
from uipath.platform.guardrails import BuiltInValidatorGuardrail, GuardrailAttachment

logger = logging.getLogger(__name__)

#: Limits enforced by the validate API.
_MAX_ATTACHMENTS = 5
_MAX_FILE_NAME_LENGTH = 260
#: ``appliesTo`` guardrail parameter; only ``Prompts`` excludes files (default is ``Both``).
_APPLIES_TO_PARAMETER = "appliesto"
_PROMPTS_ONLY = "prompts"
#: Wire key of a job attachment reference as the model and the tools exchange it.
_ID_KEY = "ID"
#: Bounds for scanning a tool payload, which can be arbitrarily large or deep.
_MAX_SCAN_DEPTH = 32
_MAX_SCAN_NODES = 10_000


def _scope_includes_files(guardrail: BuiltInValidatorGuardrail) -> bool:
    try:
        for parameter in guardrail.validator_parameters:
            if parameter.id.lower() != _APPLIES_TO_PARAMETER:
                continue
            if isinstance(parameter.value, str):
                return parameter.value.strip().lower() != _PROMPTS_ONLY
    except Exception:
        logger.debug(
            "Could not read the guardrail scope; assuming files apply.", exc_info=True
        )
    return True


async def resolve_guardrail_attachments(
    job_attachments: dict[str, Attachment],
    guardrail: BuiltInValidatorGuardrail,
) -> list[GuardrailAttachment]:
    """Return up to five references for every attachment the run knows about.

    For Agent- and LLM-scope guardrails, which evaluate the conversation as a whole.
    Empty when the guardrail is scoped to prompts or the run has no attachments. Never
    raises.
    """
    if not job_attachments:
        return []
    if not _scope_includes_files(guardrail):
        logger.debug(
            "Guardrail '%s' is scoped to prompts; skipping attachment resolution.",
            guardrail.name,
        )
        return []
    return _collect(job_attachments.values())


def resolve_referenced_attachments(
    data: Any,
    job_attachments: dict[str, Attachment] | None,
    guardrail: BuiltInValidatorGuardrail,
) -> list[GuardrailAttachment]:
    """Return up to five references for the attachments ``data`` mentions.

    For Tool-scope guardrails: ``data`` is the tool call's arguments (before the tool
    runs) or its parsed result (after). A mention is a mapping with an ``ID`` that parses
    as a UUID, the shape the model emits and the tool wrapper expands. Only ids the run's
    registry holds are forwarded, with the registry's name and type: the registry is the
    set of files this run legitimately has (agent input plus files its tools returned),
    so a mention the run never held, or a non-attachment resource id, is skipped rather
    than sent to the backend for lookup. Empty when nothing is mentioned or the guardrail
    is scoped to prompts. Never raises.
    """
    if data is None:
        return []
    if not _scope_includes_files(guardrail):
        logger.debug(
            "Guardrail '%s' is scoped to prompts; skipping attachment resolution.",
            guardrail.name,
        )
        return []
    try:
        mentions = list(_iter_attachment_mentions(data))
    except Exception:
        logger.warning(
            "Could not scan the tool payload for attachments; guardrail '%s' evaluates "
            "without files.",
            guardrail.name,
            exc_info=True,
        )
        return []
    registry = job_attachments or {}
    resolved = (_resolve_mention(mention, registry) for mention in mentions)
    return _collect(attachment for attachment in resolved if attachment is not None)


def _iter_attachment_mentions(data: Any) -> Iterator[Any]:
    """Yield every attachment-shaped value in ``data``, depth first, within bounds."""
    stack: list[tuple[Any, int]] = [(data, 0)]
    visited = 0
    while stack:
        value, depth = stack.pop()
        visited += 1
        if visited > _MAX_SCAN_NODES:
            logger.debug(
                "Stopped scanning the tool payload for attachments after %d values.",
                _MAX_SCAN_NODES,
            )
            return
        if depth > _MAX_SCAN_DEPTH:
            continue
        if isinstance(value, Attachment):
            yield value
        elif isinstance(value, BaseModel):
            stack.append((value.model_dump(by_alias=True), depth + 1))
        elif isinstance(value, Mapping):
            if _is_mention(value):
                yield value
            else:
                stack.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            stack.extend((item, depth + 1) for item in value)


def _is_mention(value: Mapping[Any, Any]) -> bool:
    raw_id = value.get(_ID_KEY)
    if raw_id is None or isinstance(raw_id, bool):
        return False
    try:
        uuid.UUID(str(raw_id))
    except ValueError:
        return False
    return True


def _resolve_mention(mention: Any, registry: dict[str, Attachment]) -> Any | None:
    """Look one mention up in the run's registry; None when the run never held it."""
    try:
        raw_id = mention.id if isinstance(mention, Attachment) else mention[_ID_KEY]
        attachment_id = str(uuid.UUID(str(raw_id)))
        known = registry.get(attachment_id)
        if known is not None:
            return known
        logger.debug(
            "Skipping attachment reference '%s': this run does not hold it.",
            attachment_id,
        )
    except Exception:
        logger.debug(
            "Skipping a malformed attachment reference in the tool payload.",
            exc_info=True,
        )
    return None


def _collect(attachments: Iterable[Any]) -> list[GuardrailAttachment]:
    """Build references, dropping unusable and duplicate ones, capped to the API limit."""
    references: list[GuardrailAttachment] = []
    seen: set[str] = set()
    for attachment in attachments:
        reference = _to_reference(attachment)
        if reference is None or reference.id in seen:
            continue
        seen.add(reference.id)
        references.append(reference)
        if len(references) == _MAX_ATTACHMENTS:
            break
    return references


def _to_reference(attachment: Any) -> GuardrailAttachment | None:
    """Build one reference, or None when the attachment cannot be forwarded."""
    try:
        attachment_id = str(uuid.UUID(str(getattr(attachment, "id", None))))
        file_name = str(getattr(attachment, "full_name", "") or "")
        mime_type = str(getattr(attachment, "mime_type", "") or "")
        if not file_name or not mime_type:
            # The validate API rejects the whole request over an empty name or type.
            logger.debug(
                "Skipping attachment '%s' for guardrail inspection: missing name or type.",
                file_name or attachment_id,
            )
            return None
        return GuardrailAttachment(
            id=attachment_id,
            file_name=file_name[:_MAX_FILE_NAME_LENGTH],
            mime_type=mime_type,
        )
    except Exception:
        logger.warning(
            "Skipping attachment '%s' for guardrail inspection: invalid reference.",
            getattr(attachment, "full_name", "?"),
            exc_info=True,
        )
        return None
