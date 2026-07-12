"""Tests for the rdflib R2RML parser (datafabric_tool/ontology/ontology_r2rml.py)."""

import pytest

from uipath_langchain.agent.tools.datafabric_tool.ontology.ontology_r2rml import (
    R2RMLParseError,
    parse_r2rml_entities,
)

# Every document must declare the prefixes it uses — rdflib is a real Turtle
# parser (unlike the previous regex scanner), so undeclared prefixes are errors.
_PREFIXES = (
    "@prefix rr:     <http://www.w3.org/ns/r2rml#> .\n"
    "@prefix uipath: <http://uipath.com/ns/datafabric#> .\n"
    "@prefix map:    <http://ex.org/mapping#> .\n"
    "@prefix onto:   <http://ex.org/ontology#> .\n"
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


def _doc(*blocks: str) -> str:
    return _PREFIXES + "\n".join(blocks)


def test_single_triplesmap():
    doc = _doc(_block("map:TM_a", "alpha", "Shared/Fin"))
    assert parse_r2rml_entities(doc) == [("alpha", "Shared/Fin")]


def test_multiple_triplesmaps_sorted_deterministically():
    # rdflib is graph-based (no document order); the parser returns a sorted,
    # de-duplicated allow-list so output is deterministic across runs.
    doc = _doc(_block("map:TM_b", "beta", "F/b"), _block("map:TM_a", "alpha", "F/a"))
    assert parse_r2rml_entities(doc) == [("alpha", "F/a"), ("beta", "F/b")]


def test_pairing_is_per_triplesmap_not_global():
    # folderPath before tableName in one map, after in the other. Pairing is per
    # TriplesMap subject, so table always pairs with the folder from its own map.
    doc = _doc(
        _block("map:TM_a", "alpha", "F/a", folder_first=True),
        _block("map:TM_b", "beta", "F/b", folder_first=False),
    )
    assert parse_r2rml_entities(doc) == [("alpha", "F/a"), ("beta", "F/b")]


def test_duplicate_pairs_are_deduped():
    doc = _doc(_block("map:TM_a", "alpha", "F/a"), _block("map:TM_a2", "alpha", "F/a"))
    assert parse_r2rml_entities(doc) == [("alpha", "F/a")]


def test_same_name_different_folder_kept_distinct():
    doc = _doc(_block("map:TM_a", "shared", "F/a"), _block("map:TM_b", "shared", "F/b"))
    assert parse_r2rml_entities(doc) == [("shared", "F/a"), ("shared", "F/b")]


def test_missing_folder_path_raises():
    doc = _doc(
        'map:TM_a a rr:TriplesMap ;\n  rr:logicalTable [ rr:tableName "alpha" ] .\n'
    )
    with pytest.raises(R2RMLParseError):
        parse_r2rml_entities(doc)


def test_missing_table_name_raises():
    doc = _doc('map:TM_a a rr:TriplesMap ;\n  uipath:folderPath "F/a" .\n')
    with pytest.raises(R2RMLParseError):
        parse_r2rml_entities(doc)


def test_no_triplesmap_raises():
    with pytest.raises(R2RMLParseError, match="No `rr:TriplesMap`"):
        parse_r2rml_entities("@prefix rr: <http://www.w3.org/ns/r2rml#> .\n")


def test_invalid_turtle_raises():
    with pytest.raises(R2RMLParseError, match="not valid Turtle"):
        parse_r2rml_entities("this is not turtle @@@ <")


def test_compact_one_line_triplesmap_parses():
    # rdflib handles compact single-line syntax the regex parser would have
    # struggled with — a table paired with its folder on one physical line.
    doc = _doc(
        'map:TM_a a rr:TriplesMap ; rr:logicalTable [ rr:tableName "alpha" ] ; '
        'uipath:folderPath "F/a" .'
    )
    assert parse_r2rml_entities(doc) == [("alpha", "F/a")]


def test_realistic_three_entity_mapping():
    doc = _doc(
        _block("map:TM_frpm", "frpm", "Shared/CaliforniaSchools"),
        _block("map:TM_satscores", "satscores", "Shared/CaliforniaSchools"),
        _block("map:TM_schools", "schools", "Shared/CaliforniaSchools"),
    )
    assert parse_r2rml_entities(doc) == [
        ("frpm", "Shared/CaliforniaSchools"),
        ("satscores", "Shared/CaliforniaSchools"),
        ("schools", "Shared/CaliforniaSchools"),
    ]
