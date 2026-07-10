"""Fetch ontology component files (OWL, R2RML) from Data Fabric for the ontology tool.

A Data Fabric ontology exposes typed files — OWL (the semantic schema) and R2RML
(the ontology→table/column mapping) — via
``EntitiesService.get_ontology_file_async``. This module provides:

* :func:`fetch_ontology_file` — a thin fetch of one file's raw content, which
  **raises** on failure or oversize. Callers decide whether a given file is
  critical (R2RML — it is also the entity allow-list) or optional (OWL — the
  prompt can degrade without it).
* :func:`fence_ontology_block` — wraps content in a labelled block for prompt
  injection, tagging the file type and media type (e.g. Turtle vs OWL Functional
  Notation) so the LLM knows how to read it.

Ontology names/folders are pinned from configuration, never supplied by the LLM.
"""

import logging

from uipath.platform.entities import EntitiesService

logger = logging.getLogger(__name__)

# Hard cap per file. Above this, an ontology file would consume too much of the
# prompt/token budget once injected and degrade answer quality, so it is rejected
# outright (not truncated) rather than silently bloating the prompt. Real OWL/R2RML
# files for a typical ontology are well under this (tens of KB).
_MAX_ONTOLOGY_BYTES = 256_000


async def fetch_ontology_file(
    entities_service: EntitiesService,
    name: str,
    file_type: str,
    folder_key: str | None,
) -> tuple[str, str]:
    """Fetch one ontology file's ``(content, media_type)``.

    Args:
        entities_service: Authenticated SDK service used for the REST call.
        name: Ontology name (pinned from configuration).
        file_type: The ontology file to fetch — ``"owl"`` or ``"r2rml"``.
        folder_key: Key of the folder the ontology lives in.

    Returns:
        ``(content, media_type)`` — the raw file body and its media type.

    Raises:
        Exception: Propagates the underlying fetch error. Also raises
            ``ValueError`` if the body exceeds :data:`_MAX_ONTOLOGY_BYTES`.
            Callers apply the critical/optional policy (raise vs degrade).
    """
    data = await entities_service.get_ontology_file_async(name, file_type, folder_key)
    content = data.get("content") or ""
    media_type = data.get("mediaType") or ""
    size = len(content.encode("utf-8"))
    if size > _MAX_ONTOLOGY_BYTES:
        logger.warning(
            "Ontology '%s' %s is %d bytes, over the %d-byte limit; refusing to "
            "inject it — it would consume too much of the prompt/token budget and "
            "degrade answer quality.",
            name,
            file_type,
            size,
            _MAX_ONTOLOGY_BYTES,
        )
        raise ValueError(
            f"Ontology '{name}' {file_type} is {size} bytes, over the "
            f"{_MAX_ONTOLOGY_BYTES}-byte limit; not injected into the prompt "
            "(would consume too much of the token budget)."
        )
    return content, media_type


def fence_ontology_block(
    name: str, file_type: str, content: str, media_type: str = ""
) -> str:
    """Wrap ontology file content in a labelled block for prompt injection."""
    label = file_type.upper()
    if media_type:
        label = f"{label}, {media_type}"
    return (
        f"--- {label}: {name} ---\n{content}\n--- END {file_type.upper()}: {name} ---"
    )
