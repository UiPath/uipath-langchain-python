"""Tests for the YARRRML mapping parser (datafabric_tool/ontology/ontology_yarrrml.py)."""

import pytest

from uipath_langchain.agent.tools.datafabric_tool.ontology.ontology_yarrrml import (
    MappedEntity,
    YarrrmlParseError,
    parse_yarrrml_entities,
)

_MAPPING = """
prefixes:
  onto: https://ontology.uipath.com/hls/clinical#
mappings:
  Claim:
    sources:
      - access: datafabric
        table: claim
        entityId: e-claim
    s: onto:Claim/$(id)
  Patient:
    sources:
      - access: datafabric
        table: patient
        entity-id: e-patient
        folderId: other-folder
  ClaimAgain:
    sources:
      access: datafabric
      table: claim
      entityId: e-claim
  NoEntity:
    sources:
      - access: datafabric
        table: lookup
  Ignored:
    sources:
      - "data.csv~csv"
      - {table: no_access}
"""


def test_parses_entities_with_ids_and_folders():
    assert parse_yarrrml_entities(_MAPPING, "onto-folder") == [
        MappedEntity("claim", "e-claim", "onto-folder"),
        MappedEntity("patient", "e-patient", "other-folder"),
        MappedEntity("lookup", None, "onto-folder"),
    ]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("mappings: [unclosed", "not valid YAML"),
        ("prefixes: {}", "No `mappings`"),
        ("- just a list", "No `mappings`"),
        ("mappings:\n  A:\n    sources: ['x.csv~csv']", "binds no Data Fabric entity"),
    ],
)
def test_rejects_unusable_mappings(text, message):
    with pytest.raises(YarrrmlParseError, match=message):
        parse_yarrrml_entities(text, None)
