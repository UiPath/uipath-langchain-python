"""Client for fetching ontology files from UiPath Data Fabric (QueryEngine).

The QueryEngine ontology REST API is hosted under the same ``datafabric_``
service as Data Fabric entities, so we reuse the SDK's authenticated
``EntitiesService`` — its ``request_async`` already injects auth, tenant/account
scoping, and retries — instead of building a second auth path. The only
caller-influenced value is ``ontology_name``, which is validated against the
QueryEngine name contract before it becomes part of the request URL.

The ``owl`` file's content may be serialized as Turtle (.ttl) or as OWL
Functional Notation (.ofn) — both are valid OWL 2 QL serializations and both
are plain text. To stay agnostic to the stored serialization we request the
JSON wrapper (``Accept: application/json``), which always returns ``content``
plus its ``mediaType`` regardless of notation. Requesting a specific text type
(e.g. ``text/turtle``) would 406 when the stored file is the other notation.

Naming follows the REST API: the resource is identified by ``ontologyName``
(``OntologyController`` route ``/{ontologyName}/files/{fileType}``).
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# QueryEngine ontology name contract (OntologyCreateRequestValidator):
# lowercase, must start with a letter, max 64 chars.
_ONTOLOGY_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,63}$")

# Defensive cap so a malformed or oversized file can never blow up the prompt
# or token budget. Real OWL 2 QL files are a few KB; QueryEngine caps at 10 MB.
_MAX_OWL_BYTES = 1_000_000

_FOLDER_KEY_HEADER = "X-UiPath-FolderKey"


def _validate_ontology_name(ontology_name: str) -> str:
    """Validate the ontology name against the QueryEngine name contract.

    The name becomes a path segment in the request URL, so only the documented
    charset is permitted. This blocks path-segment injection and traversal via
    crafted name values.

    Args:
        ontology_name: The ontology name to validate.

    Returns:
        The validated name (unchanged).

    Raises:
        ValueError: If the name does not match ``^[a-z][a-z0-9-]{0,63}$``.
    """
    if not isinstance(ontology_name, str) or not _ONTOLOGY_NAME_RE.match(
        ontology_name
    ):
        raise ValueError(
            f"Invalid ontology name {ontology_name!r}. "
            "Must match ^[a-z][a-z0-9-]{0,63}$."
        )
    return ontology_name


async def fetch_ontology_owl(
    entities_service: Any,
    ontology_name: str,
    folder_key: str | None = None,
) -> tuple[str, str]:
    """Fetch the OWL file for an ontology from Data Fabric.

    Args:
        entities_service: An authenticated SDK ``EntitiesService``. Reused for
            its ``request_async`` (auth headers, base-URL scoping, retries).
        ontology_name: Ontology name. Validated against the QE name contract.
        folder_key: Optional UiPath folder key for folder-scoped resolution.

    Returns:
        A ``(content, media_type)`` tuple. ``content`` is the OWL text in
        whatever serialization is stored — Turtle or OWL Functional Notation;
        ``media_type`` is the stored media type (e.g. ``text/turtle``), usable
        to label the notation.

    Raises:
        ValueError: If the name is invalid or the content exceeds the size cap.
            Transport/HTTP errors propagate from the SDK as raised exceptions
            (the caller decides how to degrade).
    """
    safe_name = _validate_ontology_name(ontology_name)
    # Same datafabric_ service the entities calls target; matches the
    # QueryEngine ontology route GET /ontologies/{ontologyName}/files/{fileType}.
    endpoint = f"datafabric_/api/ontologies/{safe_name}/files/owl"

    # JSON wrapper: notation-agnostic (works for Turtle or OFN) and returns the
    # stored mediaType. A text/* Accept would 406 on a serialization mismatch.
    headers = {"Accept": "application/json"}
    if folder_key:
        headers[_FOLDER_KEY_HEADER] = folder_key

    response = await entities_service.request_async(
        "GET", endpoint, scoped="tenant", headers=headers
    )

    data = response.json()
    content = data.get("content") or ""
    media_type = data.get("mediaType") or ""

    if len(content.encode("utf-8")) > _MAX_OWL_BYTES:
        raise ValueError(
            f"Ontology OWL for {safe_name!r} exceeds the "
            f"{_MAX_OWL_BYTES} byte limit."
        )
    logger.debug(
        "Fetched ontology OWL for %r (%d chars, mediaType=%s)",
        safe_name,
        len(content),
        media_type,
    )
    return content, media_type
