"""Fetches ontology OWL schemas from Data Fabric for prompt injection.

A Data Fabric context may attach one or more ontologies (mirroring the entity
set). This module fetches each configured ontology's OWL via the SDK
(``EntitiesService.get_ontology_file_async``) and returns them concatenated,
ready to embed in the inner SQL agent's system prompt.

Fetching is deterministic — done once when the sub-graph is built — rather than
an LLM-decided tool call, so the model always has the ontology in context.
Ontology names/folders are pinned from configuration, never supplied by the LLM.
"""

import asyncio
import logging

from uipath.platform.entities import EntitiesService

logger = logging.getLogger(__name__)

# Defensive cap per ontology so a malformed/oversized OWL can't blow up the
# prompt/token budget.
_MAX_OWL_BYTES = 1_000_000


def _notation_label(media_type: str) -> str:
    """Best-effort label for the OWL serialization (Turtle or OFN)."""
    mt = (media_type or "").lower()
    if "turtle" in mt or mt.endswith("ttl"):
        return "Turtle"
    if "functional" in mt or "ofn" in mt:
        return "OWL Functional Notation"
    return "Turtle or OWL Functional Notation"


async def _fetch_one(
    entities_service: EntitiesService, name: str, folder_key: str | None
) -> str:
    try:
        data = await entities_service.get_ontology_file_async(name, "owl", folder_key)
        owl = data.get("content") or ""
        media_type = data.get("mediaType") or ""
        if len(owl.encode("utf-8")) > _MAX_OWL_BYTES:
            raise ValueError(f"Ontology '{name}' OWL exceeds the size limit.")
    except Exception as e:
        logger.warning("Ontology fetch failed for %r: %s", name, e)
        return (
            f"Ontology '{name}' is unavailable ({type(e).__name__}). "
            "Proceed using the entity schemas in the system prompt."
        )
    notation = _notation_label(media_type)
    return f"--- ONTOLOGY: {name} ({notation}) ---\n{owl}\n--- END ONTOLOGY: {name} ---"


async def fetch_ontology_text(
    entities_service: EntitiesService,
    ontologies: list[tuple[str, str | None]],
) -> str:
    """Fetch and concatenate the OWL of every configured ontology.

    Args:
        entities_service: Authenticated SDK service used for the REST call.
        ontologies: ``(name, folder_key)`` pairs to fetch (pinned from config).

    Returns:
        The concatenated ontology text ready for prompt injection, or ``""`` when
        no ontologies are configured. Individual fetch failures degrade to a
        short "unavailable, use entity schemas" note rather than raising, so a
        missing ontology never fails the run.
    """
    if not ontologies:
        return ""
    # Fetch concurrently — each fetch is independent; gather preserves order so
    # the concatenation is deterministic.
    blocks = await asyncio.gather(
        *(_fetch_one(entities_service, name, folder) for name, folder in ontologies)
    )
    return "\n\n".join(blocks)
