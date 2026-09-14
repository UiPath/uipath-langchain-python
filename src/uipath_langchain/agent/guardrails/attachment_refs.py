"""Project a run's job-attachment registry into guardrail attachment references.

Only the ``llm_as_judge`` validator consumes attachments today, only behind a feature flag,
and only when the guardrail's author scoped it to files, so all three gates live here:
resolving a SAS URL costs an Orchestrator round-trip and there is no point spending it for a
validator -- or a guardrail configuration -- that will ignore the result.

Nothing in this module raises. The low-code guardrail node re-raises any exception it sees
(see ``guardrail_nodes._create_guardrail_node``), which terminates the agent run — so a
transient storage failure must never escape. An unreadable file degrades to "the guardrail
evaluates the text payload alone", which is the safer failure direction for a guardrail.
"""

import asyncio
import logging
import uuid
from typing import Any

from uipath.core.feature_flags import FeatureFlags
from uipath.platform import UiPath
from uipath.platform.attachments import Attachment
from uipath.platform.guardrails import BuiltInValidatorGuardrail, GuardrailAttachment

logger = logging.getLogger(__name__)

#: Kill switch. Names are case-sensitive and the env var is an exact concatenation:
#: ``UIPATH_FEATURE_GuardrailAttachmentsEnabled``.
GUARDRAIL_ATTACHMENTS_FEATURE_FLAG = "GuardrailAttachmentsEnabled"

#: The only validator that can read file contents today. The backend enforces this too.
_LLM_AS_JUDGE = "llm_as_judge"

#: Matches the ceiling the validate API enforces; resolving more would be wasted work.
_MAX_ATTACHMENTS = 5

#: The validate API rejects longer file names with a 400.
_MAX_FILE_NAME_LENGTH = 260

#: Optional guardrail parameter scoping the evaluation to ``Prompts``, ``Files`` or ``Both``.
#: Parameter ids are matched case-insensitively, as the backend does.
_APPLIES_TO_PARAMETER = "appliesto"

#: The one value that puts files out of scope. Anything else -- including an absent parameter,
#: which is every guardrail configured before it existed -- keeps them in, matching the
#: backend's default of "Both". Narrowing on an unrecognized value would silently stop
#: scanning files for a guardrail whose author never asked for that.
_PROMPTS_ONLY = "prompts"

#: What the backend can inspect. The runtime only forwards references — the backend decides
#: how to read each type, so this set exists to avoid spending an Orchestrator round-trip on a
#: file that would be skipped anyway.
SUPPORTED_MIME_TYPES = frozenset(
    {
        # Already text: the backend decodes these and inlines them into the judged payload.
        "text/plain",
        "text/csv",
        "application/csv",
        "text/markdown",
        "text/tab-separated-values",
        "application/json",
        "text/xml",
        "application/xml",
        # Binary: the backend sends these to a vision-capable judge model as content parts.
        "application/pdf",
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
    }
)


def _is_enabled(guardrail: BuiltInValidatorGuardrail) -> bool:
    """Both gates: the validator must be the judge and the feature flag must be on."""
    if getattr(guardrail, "validator_type", None) != _LLM_AS_JUDGE:
        return False
    return FeatureFlags.is_flag_enabled(
        GUARDRAIL_ATTACHMENTS_FEATURE_FLAG, default=False
    )


def _scope_includes_files(guardrail: BuiltInValidatorGuardrail) -> bool:
    """Whether the guardrail's ``appliesTo`` parameter puts the run's files in scope.

    The backend gates on this too, but it can only do so after the URLs are resolved. Reading
    it here is what actually saves the Orchestrator round-trip per file on a prompts-only
    guardrail -- the reason this projection is gated at all.
    """
    try:
        for parameter in getattr(guardrail, "validator_parameters", None) or []:
            if str(getattr(parameter, "id", "")).lower() != _APPLIES_TO_PARAMETER:
                continue
            value = getattr(parameter, "value", None)
            if isinstance(value, str):
                return value.strip().lower() != _PROMPTS_ONLY
    except Exception:
        # This module promises never to raise: the caller re-raises, which ends the run. A
        # parameter list that is not shaped as expected falls back to the backend's default.
        logger.debug(
            "Could not read the guardrail scope; assuming files apply.", exc_info=True
        )
    return True


async def resolve_guardrail_attachments(
    job_attachments: dict[str, Attachment],
    guardrail: BuiltInValidatorGuardrail,
) -> list[GuardrailAttachment]:
    """Resolve the run's job attachments into references the guardrails API can read.

    Args:
        job_attachments: The per-run registry from ``state.inner_state.job_attachments``.
        guardrail: The guardrail about to be evaluated; gates on its validator type.

    Returns:
        Resolved attachment references, or an empty list when the feature is off, the
        validator cannot use them, the guardrail is scoped to prompts only, none are of a
        supported type, or resolution failed. Never raises.
    """
    if not job_attachments or not _is_enabled(guardrail):
        return []
    if not _scope_includes_files(guardrail):
        logger.debug(
            "Guardrail '%s' is scoped to prompts; skipping attachment resolution.",
            getattr(guardrail, "name", "?"),
        )
        return []

    candidates = [
        attachment
        for attachment in job_attachments.values()
        if attachment.id is not None
        and (attachment.mime_type or "").lower() in SUPPORTED_MIME_TYPES
    ][:_MAX_ATTACHMENTS]
    if not candidates:
        return []

    client = UiPath()
    resolved = await asyncio.gather(
        *(_resolve_one(client, attachment) for attachment in candidates)
    )
    return [reference for reference in resolved if reference is not None]


async def _resolve_one(
    client: Any, attachment: Attachment
) -> GuardrailAttachment | None:
    """Resolve one attachment to a SAS URL, or return None if that is not possible."""
    try:
        blob_info = await client.attachments.get_blob_file_access_uri_async(
            key=uuid.UUID(str(attachment.id))
        )
        # The validate API caps file names at 260 characters; a longer name would be a 400
        # for the whole request, and this is a display label, not an identifier.
        file_name = (attachment.full_name or blob_info.name)[:_MAX_FILE_NAME_LENGTH]
        return GuardrailAttachment(
            id=str(attachment.id),
            file_name=file_name,
            mime_type=attachment.mime_type,
            url=blob_info.uri,
        )
    except Exception:
        # Deliberately broad. The low-code guardrail node re-raises everything, so letting
        # a transient Orchestrator failure escape here would kill a production run over an
        # unscanned file. Log the file name, never the URL — the resolved URL carries a SAS
        # signature.
        logger.warning(
            "Could not resolve attachment '%s' for guardrail inspection; "
            "the guardrail will evaluate without it.",
            attachment.full_name,
            exc_info=True,
        )
        return None
