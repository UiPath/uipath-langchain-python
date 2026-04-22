"""PII masking for the analyze-files tool.

Encapsulates the policy evaluation, PII detection request, and rehydration of
masked LLM output behind a single :class:`PiiMasker` class.
"""

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

from uipath_langchain.agent.multimodal import FileInfo

logger = logging.getLogger("uipath")

_FEATURE_FLAG = "FilePiiMaskingEnabled"


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
        """Run PII detection and return the masked prompt and redacted files.

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
                self._rename_for_masking(f, redacted_by_name.get(f.name, f.url))
                for f in files
            ]
            logger.info("Renamed %d file(s) with pii_masked_ prefix", len(masked_files))
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
    def _rename_for_masking(file: FileInfo, redacted_url: str) -> FileInfo:
        """Return a FileInfo pointing at ``redacted_url`` with a ``pii_masked_`` prefix."""
        if "." in file.name:
            base, ext = file.name.rsplit(".", 1)
            new_name = f"pii_masked_{base}.{ext}"
        else:
            new_name = f"pii_masked_{file.name}"
        return FileInfo(url=redacted_url, name=new_name, mime_type=file.mime_type)
