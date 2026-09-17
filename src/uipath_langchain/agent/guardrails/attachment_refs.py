"""Project a run's job-attachment registry into guardrail attachment references.

Any built-in guardrail forwards the run's attachments unless it is scoped to prompts. The
runtime forwards id, file name and mime type only; the backend's feature flag decides
whether they are used at all, and the backend decides which validators and file types it
can inspect and resolves the id through Orchestrator.

Nothing in this module raises: the guardrail node re-raises any exception, which would end
the run over a single malformed attachment.
"""

import logging
import uuid
from typing import Any

from uipath.platform.attachments import Attachment
from uipath.platform.guardrails import BuiltInValidatorGuardrail, GuardrailAttachment

logger = logging.getLogger(__name__)

#: Limits enforced by the validate API.
_MAX_ATTACHMENTS = 5
_MAX_FILE_NAME_LENGTH = 260
#: ``appliesTo`` guardrail parameter; only ``Prompts`` excludes files (default is ``Both``).
_APPLIES_TO_PARAMETER = "appliesto"
_PROMPTS_ONLY = "prompts"


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
    """Return up to five attachment references for the guardrail, or an empty list.

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

    references: list[GuardrailAttachment] = []
    for attachment in job_attachments.values():
        reference = _to_reference(attachment)
        if reference is not None:
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
