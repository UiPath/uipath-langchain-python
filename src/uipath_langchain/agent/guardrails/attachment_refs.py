"""Project a run's attachments into guardrail attachment references.

Two sources feed the same reference shape:

* Pre-execution guardrails at Agent and LLM scope judge the conversation so far, so they
  read the whole job-attachment registry (:func:`resolve_guardrail_attachments`).
* Every other guardrail judges one payload, so it reads only the attachments that payload
  mentions (:func:`resolve_referenced_attachments`): the ``{"ID": ...}`` objects in a tool
  call's arguments or result, in the tool-call arguments an LLM produced, or in the
  agent's result. A payload that names no file forwards none, even when the run holds
  files elsewhere; otherwise every check would ship every file to the backend, and an
  input file would be judged again as if the agent had produced it.

Both return one of three answers, which the validate API reads differently:

* ``None``: say nothing about files. The guardrail is scoped to text only, or the
  payload could not be scanned; the backend evaluates the text as before.
* ``[]``: files apply and there are none to inspect. A files-only guardrail passes
  without judging the text.
* a list: the files to inspect.

The runtime forwards id, file name and mime type only; the backend's feature flag
decides whether they are used at all, the backend decides which validators and file
types it can inspect, and it resolves the id through Orchestrator. The scan and the
reference rules live in ``uipath.platform.guardrails.attachments``, shared with the
coded-agent guardrails.

Nothing in this module raises: the guardrail node re-raises any exception, which would
end the run over a single malformed attachment.
"""

import logging
from typing import Any

from uipath.platform.attachments import Attachment
from uipath.platform.guardrails import (
    BuiltInValidatorGuardrail,
    GuardrailAttachment,
    detection_scope_includes_files,
    referenced_attachments,
    to_references,
)
from uipath.platform.guardrails.attachments import MAX_ATTACHMENTS

logger = logging.getLogger(__name__)

#: Limit enforced by the validate API.
_MAX_ATTACHMENTS = MAX_ATTACHMENTS


def _is_text_only(guardrail: BuiltInValidatorGuardrail) -> bool:
    if detection_scope_includes_files(guardrail):
        return False
    logger.debug(
        "Guardrail '%s' is scoped to text only; skipping attachment resolution.",
        guardrail.name,
    )
    return True


async def resolve_guardrail_attachments(
    job_attachments: dict[str, Attachment],
    guardrail: BuiltInValidatorGuardrail,
) -> list[GuardrailAttachment] | None:
    """Return up to five references for every attachment the run knows about.

    For pre-execution guardrails at Agent and LLM scope, which evaluate the conversation
    as a whole. ``None`` when the guardrail is scoped to text only, empty when the run
    has no attachments. Never raises.
    """
    if _is_text_only(guardrail):
        return None
    if not job_attachments:
        return []
    return to_references(job_attachments.values())


def resolve_referenced_attachments(
    data: Any,
    job_attachments: dict[str, Attachment] | None,
    guardrail: BuiltInValidatorGuardrail,
) -> list[GuardrailAttachment] | None:
    """Return up to five references for the attachments ``data`` mentions, in order.

    ``data`` is the payload one guardrail judges: a tool call's arguments or parsed
    result, an LLM's tool-call arguments, or the agent's result. A mention is a mapping
    with an ``ID`` that parses as a UUID, the shape the model emits and the tool wrapper
    expands. Only ids the run's registry holds are forwarded, with the registry's name
    and type: the registry is the set of files this run legitimately has (agent input
    plus files its tools returned or its output verified), so a mention the run never
    held, or a non-attachment resource id, is skipped rather than sent to the backend
    for lookup. ``None`` when the guardrail is scoped to text only or ``data`` cannot be
    scanned; empty when nothing is mentioned. Never raises.
    """
    if _is_text_only(guardrail):
        return None
    if data is None:
        return []
    return referenced_attachments(data, job_attachments or {})
