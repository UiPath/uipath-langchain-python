"""Project a run's job-attachment registry into guardrail attachment references.

Only the ``llm_as_judge`` validator consumes attachments today, and only behind a feature
flag, so both gates live here: resolving a SAS URL costs an Orchestrator round-trip and
there is no point spending it for a validator that will ignore the result.

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

#: Phase 1 — types that are already text, inlined into the judge payload by the backend.
#: Phase 2 adds pdf and the image types, which the backend sends to the model as content parts.
SUPPORTED_MIME_TYPES = frozenset(
    {
        "text/plain",
        "text/csv",
        "application/csv",
        "text/markdown",
        "text/tab-separated-values",
        "application/json",
        "text/xml",
        "application/xml",
    }
)


def _is_enabled(guardrail: BuiltInValidatorGuardrail) -> bool:
    """Both gates: the validator must be the judge and the feature flag must be on."""
    if getattr(guardrail, "validator_type", None) != _LLM_AS_JUDGE:
        return False
    return FeatureFlags.is_flag_enabled(
        GUARDRAIL_ATTACHMENTS_FEATURE_FLAG, default=False
    )


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
        validator cannot use them, none are of a supported type, or resolution failed.
        Never raises.
    """
    if not job_attachments or not _is_enabled(guardrail):
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
        return GuardrailAttachment(
            id=str(attachment.id),
            file_name=attachment.full_name or blob_info.name,
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
