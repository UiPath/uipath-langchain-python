"""Parse a Data Fabric R2RML mapping into ``(entity_name, folder_path)`` pairs.

Dependency-free (no ``rdflib``). The mapping follows the UiPath authoring
template: each entity is one *contiguous* ``rr:TriplesMap`` block that declares
exactly one ``rr:tableName "<EntityName>"`` and exactly one
``uipath:folderPath "<Folder/Path>"``. The document is split into TriplesMap
blocks and the two literals are read *per block*, so a table is always paired
with the folder from its own block (never mis-zipped across maps).

The extracted ``(entity_name, folder_path)`` set is the closed allow-list of
entities the ontology tool is permitted to resolve and query.
"""

import re

# Boundary of a TriplesMap declaration: ``<subject> a rr:TriplesMap``.
_TRIPLES_MAP_RE = re.compile(r"\ba\s+rr:TriplesMap\b")
_TABLE_NAME_RE = re.compile(r'rr:tableName\s+"([^"]+)"')
_FOLDER_PATH_RE = re.compile(r'uipath:folderPath\s+"([^"]+)"')


class R2RMLParseError(ValueError):
    """Raised when the R2RML mapping does not satisfy the authoring contract."""


def parse_r2rml_entities(r2rml_text: str) -> list[tuple[str, str]]:
    """Extract the ``(entity_name, folder_path)`` allow-list from an R2RML mapping.

    Splits the mapping into ``rr:TriplesMap`` blocks and reads the single
    ``rr:tableName`` and ``uipath:folderPath`` literal from each, preserving
    document order and de-duplicating identical pairs.

    Args:
        r2rml_text: The R2RML mapping document (Turtle).

    Returns:
        Ordered, de-duplicated ``(entity_name, folder_path)`` pairs.

    Raises:
        R2RMLParseError: No ``rr:TriplesMap`` was found, or a block does not
            declare exactly one ``rr:tableName`` and one ``uipath:folderPath``.
    """
    starts = [m.start() for m in _TRIPLES_MAP_RE.finditer(r2rml_text)]
    if not starts:
        raise R2RMLParseError("No `rr:TriplesMap` found in the R2RML mapping.")

    bounds = starts + [len(r2rml_text)]
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for i in range(len(starts)):
        block = r2rml_text[bounds[i] : bounds[i + 1]]
        names = _TABLE_NAME_RE.findall(block)
        folders = _FOLDER_PATH_RE.findall(block)
        if len(names) != 1 or len(folders) != 1:
            raise R2RMLParseError(
                "Each TriplesMap must declare exactly one rr:tableName and one "
                f"uipath:folderPath; found tableName={names} folderPath={folders}."
            )
        pair = (names[0], folders[0])
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)

    return pairs
