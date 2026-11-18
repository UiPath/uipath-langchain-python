"""Tests for the Ontology-service grounding sources and the source flag.

Covers ``datafabric_tool/ontology/ontology_grounding.py`` and the client seam in
``ontology_service_client.py``: the flag's values and default, the files source
(checksum-verified download, size cap, no-schema failure), the graph rendering,
and the deployed-state guard.
"""

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from uipath.core.feature_flags import FeatureFlags

from uipath_langchain.agent.exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from uipath_langchain.agent.tools.datafabric_tool.ontology.ontology_grounding import (
    ONTOLOGY_SOURCE_FF,
    FilesGroundingSource,
    GraphGroundingSource,
    OntologySource,
    read_ontology_source,
    render_graph,
    select_grounding_source,
)
from uipath_langchain.agent.tools.datafabric_tool.ontology.ontology_service_client import (
    ensure_deployed,
    resolve_ontology_client,
    validate_ontology_id,
)

# --- fakes ------------------------------------------------------------------


@dataclass
class _Artifact:
    file_name: str
    type: str = "schema"
    media_type: str = "text/owl-functional"
    size_bytes: int = 10
    checksum: str = "abc"


@dataclass
class _Content:
    content: bytes
    media_type: str = "text/owl-functional"
    etag: str | None = '"abc"'


@dataclass
class _FakeClient:
    """Stands in for ``UiPath().ontology``; records what the tool asked for."""

    artifacts: list[_Artifact] = field(default_factory=list)
    contents: dict[str, bytes] = field(default_factory=dict)
    graph: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, Any]] = field(default_factory=list)

    async def list_artifacts_async(self, ontology_id, artifact_type=None):
        self.calls.append(("list", (ontology_id, artifact_type)))
        return [a for a in self.artifacts if artifact_type in (None, a.type)]

    async def get_artifact_async(self, ontology_id, file_name, expected=None):
        self.calls.append(("get", (ontology_id, file_name, expected)))
        return _Content(content=self.contents[file_name])

    async def get_graph_async(self, ontology_id):
        self.calls.append(("graph", ontology_id))
        return self.graph


_DETAILS = SimpleNamespace(
    id="onto-1",
    name="nxorders",
    display_name="Orders",
    description="Sales orders",
    folder_id="f-1",
    state="DEPLOYED",
)


# --- flag -------------------------------------------------------------------


def test_source_defaults_to_files(monkeypatch):
    monkeypatch.setattr(FeatureFlags, "get_flag", lambda name, default=None: default)
    assert read_ontology_source() is OntologySource.FILES


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("files", OntologySource.FILES),
        ("Graph", OntologySource.GRAPH),
        (" graph ", OntologySource.GRAPH),
    ],
)
def test_source_reads_each_value(monkeypatch, raw, expected):
    monkeypatch.setattr(FeatureFlags, "get_flag", lambda name, default=None: raw)
    assert read_ontology_source() is expected


def test_unknown_source_falls_back_to_files(monkeypatch, caplog):
    monkeypatch.setattr(FeatureFlags, "get_flag", lambda name, default=None: "cloud")
    with caplog.at_level(logging.WARNING):
        assert read_ontology_source() is OntologySource.FILES
    assert ONTOLOGY_SOURCE_FF in caplog.text


def test_select_grounding_source_per_value():
    assert isinstance(
        select_grounding_source(OntologySource.FILES), FilesGroundingSource
    )
    assert isinstance(
        select_grounding_source(OntologySource.GRAPH), GraphGroundingSource
    )


# --- files source -----------------------------------------------------------


async def test_files_source_downloads_schema_files_with_their_manifest_entries():
    schema = _Artifact("nxorders.ofn", size_bytes=20, checksum="c1")
    mapping = _Artifact("nxorders.yarrrml.yml", type="mapping")
    client = _FakeClient(
        artifacts=[schema, mapping],
        contents={"nxorders.ofn": b"Ontology(<urn:x>)"},
    )

    grounding = await FilesGroundingSource().ground(client, "onto-1", _DETAILS)

    assert grounding.kind == "files"
    assert "Ontology(<urn:x>)" in grounding.text
    assert "BEGIN SCHEMA nxorders.ofn (text/owl-functional)" in grounding.text
    assert "not instructions" in grounding.text
    assert grounding.summary == "nxorders.ofn (20 B)"
    # only schema files are listed, and each download names its manifest entry
    # so the client can verify size and checksum
    assert client.calls[0] == ("list", ("onto-1", "schema"))
    assert client.calls[1] == ("get", ("onto-1", "nxorders.ofn", schema))


async def test_files_source_skips_oversize_files_and_fails_when_none_remain(caplog):
    big = _Artifact("huge.ofn", size_bytes=999)
    client = _FakeClient(artifacts=[big], contents={"huge.ofn": b"x" * 999})

    with caplog.at_level(logging.WARNING):
        with pytest.raises(AgentRuntimeError) as err:
            await FilesGroundingSource(max_bytes=100).ground(client, "onto-1", _DETAILS)

    assert err.value.error_info.code == AgentRuntimeError.full_code(
        AgentRuntimeErrorCode.ONTOLOGY_GROUNDING_UNAVAILABLE
    )
    assert "over the 100-byte cap" in caplog.text
    # oversize files are never downloaded
    assert all(kind != "get" for kind, _ in client.calls)


async def test_files_source_skips_a_download_that_outgrows_the_cap():
    small = _Artifact("a.ofn", size_bytes=5)
    ok = _Artifact("b.ofn", size_bytes=5)
    client = _FakeClient(
        artifacts=[small, ok], contents={"a.ofn": b"x" * 50, "b.ofn": b"fine"}
    )

    grounding = await FilesGroundingSource(max_bytes=10).ground(
        client, "onto-1", _DETAILS
    )

    assert "fine" in grounding.text
    assert "x" * 50 not in grounding.text


# --- graph source -----------------------------------------------------------


_GRAPH = {
    "namespace": "https://ontology.uipath.com/nxorders#",
    "objects": [
        {
            "name": "SalesOrder",
            "label": "Sales Order",
            "domain": "Sales",
            "objectType": "entity",
            "key": "id",
            "description": "An order placed by a customer.\nMay span lines.",
            "synonyms": ["order", "SO"],
            "propertyCount": 7,
        },
        {
            "name": "Customer",
            "domain": "Sales",
            "objectType": "entity",
            "synonyms": [],
            "properties": [
                {"name": "id", "range": "xsd:string", "required": True},
                {
                    "name": "name",
                    "range": "xsd:string",
                    "label": "Customer name",
                    "synonyms": ["account"],
                    "description": "Legal name",
                },
            ],
        },
    ],
    "relationships": [
        {
            "name": "PLACED_BY",
            "reverseName": "PLACED",
            "from": "SalesOrder",
            "to": "Customer",
            "cardinality": "EXACTLY_ONE",
            "join": {"fromProperty": "customer_id", "toProperty": "id"},
        }
    ],
}


def test_render_graph_lists_objects_properties_and_joins():
    text = render_graph(_GRAPH)

    assert "Namespace: https://ontology.uipath.com/nxorders#" in text
    assert "### Domain: Sales" in text
    assert "- SalesOrder (Sales Order) [entity; key: id]" in text
    assert "An order placed by a customer. May span lines." in text
    assert "Also called: order, SO" in text
    assert "Properties: 7 (names not listed in this view)" in text
    assert "- Customer [entity]" in text
    assert "- id: xsd:string, required" in text
    assert "- name: xsd:string (Customer name); also account — Legal name" in text
    assert (
        "- SalesOrder --PLACED_BY--> Customer [exactly one]; "
        "join SalesOrder.customer_id = Customer.id; reverse: PLACED"
    ) in text


async def test_graph_source_fences_the_rendering_and_summarises_counts():
    client = _FakeClient(graph=_GRAPH)

    grounding = await GraphGroundingSource().ground(client, "onto-1", _DETAILS)

    assert grounding.kind == "graph"
    assert grounding.text.startswith("--- BEGIN ONTOLOGY GRAPH Orders")
    assert grounding.summary == "2 objects, 1 relationships"
    assert client.calls == [("graph", "onto-1")]


async def test_graph_source_cuts_at_the_cap(caplog):
    client = _FakeClient(graph=_GRAPH)
    with caplog.at_level(logging.WARNING):
        grounding = await GraphGroundingSource(max_bytes=120).ground(
            client, "onto-1", _DETAILS
        )
    assert "rendering cut at the size cap" in grounding.text
    assert "was cut" in caplog.text


# --- client seam ------------------------------------------------------------


def test_validate_ontology_id_accepts_keys_and_rejects_path_tricks():
    assert validate_ontology_id("c0d9f850-ba58-443e-8c54-0bc4bd68f202")
    assert validate_ontology_id("nxorders_v2")
    for bad in (None, "", "../other", "a/b", "a b", "x?y=1"):
        with pytest.raises(AgentRuntimeError) as err:
            validate_ontology_id(bad)
        assert err.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.ONTOLOGY_INVALID_REFERENCE
        )


def test_resolve_client_requires_the_sdk_attribute():
    sdk = SimpleNamespace(ontology=object())
    assert resolve_ontology_client(sdk) is sdk.ontology
    with pytest.raises(AgentRuntimeError) as err:
        resolve_ontology_client(SimpleNamespace())
    assert err.value.error_info.code == AgentRuntimeError.full_code(
        AgentRuntimeErrorCode.ONTOLOGY_SERVICE_UNAVAILABLE
    )


def test_ensure_deployed_passes_deployed_and_names_other_states():
    ensure_deployed(_DETAILS, "onto-1")
    broken = SimpleNamespace(**{**_DETAILS.__dict__, "state": "BROKEN"})
    with pytest.raises(AgentRuntimeError) as err:
        ensure_deployed(broken, "onto-1")
    assert err.value.error_info.code == AgentRuntimeError.full_code(
        AgentRuntimeErrorCode.ONTOLOGY_NOT_DEPLOYED
    )
    assert "Orders" in err.value.error_info.title
    assert "BROKEN" in err.value.error_info.detail
