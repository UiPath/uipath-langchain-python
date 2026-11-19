"""Typed seam to the Ontology service client the SDK exposes as ``UiPath().ontology``.

The client itself lives in ``uipath-platform`` (``OntologyService``). This module
only names the slice of it the ontology tool uses, as ``Protocol`` types, so the
tool is typed and testable without importing the concrete classes — the pinned
``uipath-platform`` may predate the client. At runtime the client is looked up by
attribute; a missing attribute is reported as a structured runtime error instead
of an ``AttributeError`` deep in the first query.

Ontology ids come from the agent definition (the picker saves the catalog key),
never from the model, and are still validated before they reach a URL.
"""

import re
from typing import Any, Mapping, Protocol, Sequence

from uipath.runtime.errors import UiPathErrorCategory

from ....exceptions import AgentRuntimeError, AgentRuntimeErrorCode

#: The only ontology state the tool queries. Anything else fails the call.
DEPLOYED_STATE = "DEPLOYED"

# Catalog keys are GUIDs; the service also accepts identifiers. Either way the id
# becomes a URL path segment inside the SDK, so reject anything that could escape
# it (slashes, dot sequences, whitespace, query characters).
_SAFE_ONTOLOGY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


class OntologyDetailsLike(Protocol):
    """``GET /ontology/{id}``: the catalog view the picker also reads."""

    id: str
    name: str
    display_name: str | None
    description: str | None
    folder_id: str | None
    state: str | None


class OntologyArtifactContentLike(Protocol):
    """One downloaded artifact, checksum-verified by the client."""

    file_name: str
    content: bytes
    media_type: str
    etag: str | None


class SqlColumnLike(Protocol):
    """One column of a SQL result: its name and the runtime's type name."""

    name: str
    type: str | None


class SqlResultLike(Protocol):
    """``POST /ontology/{id}/sql`` as the client returns it.

    ``rows`` are dicts keyed by column name; ``column_names`` flattens ``columns``.
    ``row_count`` is the number of rows in this answer, not of all matches.
    """

    columns: Sequence[SqlColumnLike]
    rows: Sequence[Mapping[str, Any]]
    row_count: int
    generated_sql: str | None

    @property
    def column_names(self) -> Sequence[str]: ...


class OntologyServiceClient(Protocol):
    """The ``OntologyService`` methods this tool calls."""

    async def get_async(self, ontology_id: str) -> OntologyDetailsLike: ...

    async def get_artifact_by_type_async(
        self, ontology_id: str, artifact_type: str
    ) -> OntologyArtifactContentLike: ...

    async def get_graph_async(self, ontology_id: str) -> dict[str, Any]: ...

    async def execute_sql_async(
        self,
        ontology_id: str,
        statement: str,
        params: dict[str, Any] | None = None,
    ) -> SqlResultLike: ...


def validate_ontology_id(ontology_id: str | None) -> str:
    """Return ``ontology_id`` if it is a safe path segment, else raise."""
    if not ontology_id or not _SAFE_ONTOLOGY_ID.match(ontology_id):
        raise AgentRuntimeError(
            code=AgentRuntimeErrorCode.ONTOLOGY_INVALID_REFERENCE,
            title="The ontology context has no usable ontology id",
            detail=(
                "The Ontology service addresses an ontology by the id the picker "
                "saved (letters, digits, '_' and '-'). Remove the ontology from "
                "this agent's context and add it again."
            ),
            category=UiPathErrorCategory.DEPLOYMENT,
        )
    return ontology_id


def resolve_ontology_client(sdk: Any) -> OntologyServiceClient:
    """Return ``sdk.ontology`` or explain why the service path cannot run."""
    client = getattr(sdk, "ontology", None)
    if client is None:
        raise AgentRuntimeError(
            code=AgentRuntimeErrorCode.ONTOLOGY_SERVICE_UNAVAILABLE,
            title="The Ontology service client is not available",
            detail=(
                "The installed uipath-platform exposes no Ontology service client "
                "(UiPath().ontology). Upgrade uipath-platform to a version that "
                "includes it."
            ),
            category=UiPathErrorCategory.DEPLOYMENT,
        )
    return client


def ensure_deployed(details: OntologyDetailsLike, ontology_id: str) -> None:
    """Fail the call when the ontology is not ``DEPLOYED``; no retry makes sense."""
    state = (details.state or "").upper()
    if state == DEPLOYED_STATE:
        return
    label = ontology_label(details, ontology_id)
    raise AgentRuntimeError(
        code=AgentRuntimeErrorCode.ONTOLOGY_NOT_DEPLOYED,
        title=f"Ontology '{label}' is not deployed",
        detail=(
            f"Ontology '{label}' is in state '{details.state or 'unknown'}'. "
            "Only a deployed ontology can be queried; deploy it again or remove "
            "it from this agent."
        ),
        category=UiPathErrorCategory.DEPLOYMENT,
    )


def ontology_label(details: OntologyDetailsLike, fallback: str) -> str:
    """The name people know the ontology by, for messages and logs."""
    return details.display_name or details.name or fallback
