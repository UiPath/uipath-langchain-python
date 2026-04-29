"""PII masking for the analyze-files tool.

Encapsulates the policy evaluation, PII detection request, and rehydration of
masked LLM output behind a single :class:`PiiMasker` class.
"""

import base64
import logging
from typing import Any

from uipath.core.feature_flags import FeatureFlags
from uipath.platform import UiPath
from uipath.platform.semantic_proxy import (
    PiiDetectionRequest,
    PiiDetectionResponse,
    PiiDocument,
    PiiEntityThreshold,
    PiiFile,
    rehydrate_from_pii_response,
)

from uipath_langchain.agent.multimodal import FileInfo, download_file_base64

logger = logging.getLogger("uipath")

_FEATURE_FLAG = "FilePiiMaskingEnabled"


def masked_name_for(name: str) -> str:
    """Apply the ``pii_masked_`` filename prefix for re-uploaded masked files."""
    if "." in name:
        base, ext = name.rsplit(".", 1)
        return f"pii_masked_{base}.{ext}"
    return f"pii_masked_{name}"


class PiiMasker:
    """Runs PII detection against prompts/files and rehydrates masked LLM output.

    Two gates (both must allow) control whether masking runs:

    1. Opt-in kill-switch — the ``FilePiiMaskingEnabled`` feature flag
       (defaults to ``False``; enable via ``FeatureFlags.configure_flags``
       or the ``UIPATH_FEATURE_FilePiiMaskingEnabled`` env var).
    2. Platform policy — ``data.container.pii-in-flight-agents`` from the
       AutomationOps deployed-policy response.
    """

    def __init__(self, client: UiPath, policy: dict[str, Any] | None) -> None:
        self._client = client
        self._policy = policy
        self._result: PiiDetectionResponse | None = None

    @staticmethod
    def is_policy_enabled(policy: dict[str, Any] | None) -> bool:
        """Return True when both the feature flag and platform policy allow masking."""
        flag_enabled = FeatureFlags.is_flag_enabled(_FEATURE_FLAG, default=False)
        logger.info("PII masking feature flag %s=%s", _FEATURE_FLAG, flag_enabled)
        if not flag_enabled:
            return False
        if not policy:
            return False
        container = policy.get("data", {}).get("container", {})
        return bool(container.get("pii-in-flight-agents", False))

    async def apply(
        self, analysis_task: str, files: list[FileInfo]
    ) -> tuple[str, list[FileInfo]]:
        """Run PII detection and return the masked prompt and annotated files.

        Files returned keep their original ``url``/``name``/``attachment_id`` and
        additionally carry ``masked_attachment_url`` (redacted blob URL from the
        PII service) and ``masked_attachment_id`` (orchestrator UUID from the
        re-upload, when the upload succeeds). This lets observability callers
        reference both versions while the LLM path substitutes the masked URL at
        the message boundary.

        The underlying detection response is retained so the LLM output can be
        rehydrated later via :meth:`rehydrate`.
        """
        request = PiiDetectionRequest(
            documents=[
                PiiDocument(id="user-prompt", role="user", document=analysis_task)
            ],
            files=[
                PiiFile(
                    file_name=f.name,
                    file_url=f.url,
                    file_type=f.name.rsplit(".", 1)[-1].lower()
                    if "." in f.name
                    else "",
                )
                for f in files
            ],
            entity_thresholds=self._entity_thresholds_from_policy() or None,
        )
        self._result = await self._client.semantic_proxy.detect_pii_async(request)
        logger.info(
            "PII detection completed: %d document entities, %d file entities",
            sum(len(d.pii_entities) for d in self._result.response),
            sum(len(f.pii_entities) for f in self._result.files),
        )

        masked_prompt = analysis_task
        for doc in self._result.response:
            if doc.id == "user-prompt":
                if doc.masked_document != analysis_task:
                    logger.info(
                        "User prompt masked (%d entities replaced)",
                        len(doc.pii_entities),
                    )
                masked_prompt = doc.masked_document
                break

        redacted_by_name = {f.file_name: f.file_url for f in self._result.files}
        if redacted_by_name:
            masked_files = [
                self._with_masked_url(f, redacted_by_name.get(f.name, f.url))
                for f in files
            ]
            logger.info(
                "Populated masked_attachment_url on %d file(s)", len(masked_files)
            )

            # Re-upload redacted bytes to orchestrator so LLMOps traces can
            # resolve a download URL for the masked attachment.
            for idx, masked_file in enumerate(masked_files):
                uploaded_id = await self._upload_masked_to_orchestrator(masked_file)
                if uploaded_id:
                    masked_files[idx] = FileInfo(
                        url=masked_file.url,
                        name=masked_file.name,
                        mime_type=masked_file.mime_type,
                        masked_attachment_url=masked_file.masked_attachment_url,
                        attachment_id=masked_file.attachment_id,
                        masked_attachment_id=uploaded_id,
                    )
        else:
            masked_files = files

        return masked_prompt, masked_files

    def rehydrate(self, text: str) -> str:
        """Replace masked placeholders in ``text`` with the original PII values.

        Returns ``text`` unchanged if :meth:`apply` hasn't been called.
        """
        if self._result is None:
            return text
        rehydrated = rehydrate_from_pii_response(text, self._result)
        if rehydrated != text:
            logger.info("Rehydrated LLM response with PII entities")
        return rehydrated

    def _entity_thresholds_from_policy(self) -> list[PiiEntityThreshold]:
        """Extract enabled entity thresholds from the policy's ``pii-entity-table``."""
        if not self._policy:
            return []
        table = self._policy.get("data", {}).get("pii-entity-table", [])
        thresholds: list[PiiEntityThreshold] = []
        for entry in table:
            if not entry.get("pii-entity-is-enabled", False):
                continue
            category = entry.get("pii-entity-category")
            confidence = entry.get("pii-entity-confidence-threshold")
            if category is None or confidence is None:
                continue
            thresholds.append(
                PiiEntityThreshold(
                    category=category,
                    confidence_threshold=confidence,
                )
            )
        return thresholds

    @staticmethod
    def _with_masked_url(file: FileInfo, redacted_url: str) -> FileInfo:
        """Return a new ``FileInfo`` carrying the redacted URL on ``masked_attachment_url``.

        The original ``url``, ``name``, and ``attachment_id`` are preserved so
        observability callers (LLMOps traces) can reference both versions.
        """
        return FileInfo(
            url=file.url,
            name=file.name,
            mime_type=file.mime_type,
            masked_attachment_url=redacted_url,
            attachment_id=file.attachment_id,
        )

    async def _upload_masked_to_orchestrator(self, file: FileInfo) -> str | None:
        """Re-upload the redacted blob to orchestrator so LLMOps can download it.

        The PII service returns a blob URL that LLMOps has no way to resolve, so
        clicking the masked attachment in the trace viewer fails. Fetching the
        bytes and uploading them via ``client.jobs.create_attachment_async``
        gives us a real orchestrator UUID that the UI knows how to download,
        and links the attachment to the current job so it shows up under the
        job's attachments (job_key falls back to the running job's instance_key).

        Returns the uploaded attachment id, or ``None`` on failure (callers fall
        back to a synthesized uuid5 — the trace still shows the file, just not
        downloadable).
        """
        if not file.masked_attachment_url:
            return None
        try:
            content = base64.b64decode(
                await download_file_base64(file.masked_attachment_url)
            )
            masked_name = masked_name_for(file.name)
            attachment_key = await self._client.jobs.create_attachment_async(
                name=masked_name,
                content=content,
                category="pii masked",
            )
            logger.info(
                "Uploaded masked attachment '%s' as id=%s",
                masked_name,
                attachment_key,
            )
            return str(attachment_key)
        except Exception:
            logger.exception("Failed to upload masked file to orchestrator")
            return None
