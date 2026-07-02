"""Tests for the R2RML parser (datafabric_tool/ontology_r2rml.py)."""

import pytest

from uipath_langchain.agent.tools.datafabric_tool.ontology_r2rml import (
    R2RMLParseError,
    parse_r2rml_entities,
)


def _block(subject: str, table: str, folder: str, folder_first: bool = False) -> str:
    table_line = f'  rr:logicalTable [ rr:tableName "{table}" ] ;'
    folder_line = f'  uipath:folderPath "{folder}" ;'
    body = (
        f"{folder_line}\n{table_line}"
        if folder_first
        else f"{table_line}\n{folder_line}"
    )
    return (
        f"{subject} a rr:TriplesMap ;\n{body}\n  rr:subjectMap [ rr:class onto:X ] .\n"
    )


def test_single_triplesmap():
    doc = _block("map:TM_a", "alpha", "Shared/Fin")
    assert parse_r2rml_entities(doc) == [("alpha", "Shared/Fin")]


def test_multiple_triplesmaps_preserve_order():
    doc = _block("map:TM_a", "alpha", "F/a") + _block("map:TM_b", "beta", "F/b")
    assert parse_r2rml_entities(doc) == [("alpha", "F/a"), ("beta", "F/b")]


def test_pairing_is_per_block_not_global_zip():
    # folderPath appears before tableName in one block, after in the other.
    # A naive global-zip would mis-pair; block parsing must keep them correct.
    doc = _block("map:TM_a", "alpha", "F/a", folder_first=True) + _block(
        "map:TM_b", "beta", "F/b", folder_first=False
    )
    assert parse_r2rml_entities(doc) == [("alpha", "F/a"), ("beta", "F/b")]


def test_duplicate_pairs_are_deduped():
    doc = _block("map:TM_a", "alpha", "F/a") + _block("map:TM_a2", "alpha", "F/a")
    assert parse_r2rml_entities(doc) == [("alpha", "F/a")]


def test_same_name_different_folder_kept_distinct():
    doc = _block("map:TM_a", "shared", "F/a") + _block("map:TM_b", "shared", "F/b")
    assert parse_r2rml_entities(doc) == [("shared", "F/a"), ("shared", "F/b")]


def test_missing_folder_path_raises():
    doc = 'map:TM_a a rr:TriplesMap ;\n  rr:logicalTable [ rr:tableName "alpha" ] .\n'
    with pytest.raises(R2RMLParseError):
        parse_r2rml_entities(doc)


def test_missing_table_name_raises():
    doc = 'map:TM_a a rr:TriplesMap ;\n  uipath:folderPath "F/a" .\n'
    with pytest.raises(R2RMLParseError):
        parse_r2rml_entities(doc)


def test_no_triplesmap_raises():
    with pytest.raises(R2RMLParseError):
        parse_r2rml_entities("@prefix rr: <http://www.w3.org/ns/r2rml#> .\n")


def test_realistic_three_entity_mapping():
    doc = (
        "@prefix rr: <http://www.w3.org/ns/r2rml#> .\n"
        "@prefix uipath: <http://uipath.com/ns/datafabric#> .\n"
        + _block("map:TM_frpm", "frpm", "Shared/CaliforniaSchools")
        + _block("map:TM_satscores", "satscores", "Shared/CaliforniaSchools")
        + _block("map:TM_schools", "schools", "Shared/CaliforniaSchools")
    )
    assert parse_r2rml_entities(doc) == [
        ("frpm", "Shared/CaliforniaSchools"),
        ("satscores", "Shared/CaliforniaSchools"),
        ("schools", "Shared/CaliforniaSchools"),
    ]
