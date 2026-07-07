"""Parse a Data Fabric R2RML mapping into ``(entity_name, folder_path)`` pairs.

Uses ``rdflib`` to parse the mapping as an RDF graph rather than scanning the
text, so authoring variations (predicate order, whitespace, one-line vs
multi-line blocks, ``@base``/prefix differences) are handled by a real Turtle
parser instead of regexes.

The mapping follows the UiPath authoring contract: each entity is one
``rr:TriplesMap`` that declares exactly one ``rr:tableName`` (via
``rr:logicalTable``) naming the Data Fabric entity, and exactly one
``uipath:folderPath`` naming the folder that entity lives in. Because each pair
is read from a single TriplesMap subject, a table is always paired with the
folder from its own map (never mis-zipped across maps).

The extracted ``(entity_name, folder_path)`` set is the closed allow-list of
entities the ontology tool is permitted to resolve and query.
"""

from rdflib import Graph, Namespace
from rdflib.namespace import RDF

# R2RML vocabulary and the UiPath extension carrying the folder path.
RR = Namespace("http://www.w3.org/ns/r2rml#")
UIPATH = Namespace("http://uipath.com/ns/datafabric#")


class R2RMLParseError(ValueError):
    """Raised when the R2RML mapping does not satisfy the authoring contract."""


def parse_r2rml_entities(r2rml_text: str) -> list[tuple[str, str]]:
    """Extract the ``(entity_name, folder_path)`` allow-list from an R2RML mapping.

    Parses the mapping as Turtle and, for each ``rr:TriplesMap``, reads the
    single ``rr:tableName`` (under ``rr:logicalTable``) and the single
    ``uipath:folderPath`` literal. Pairs are de-duplicated and returned in a
    deterministic (sorted) order.

    Args:
        r2rml_text: The R2RML mapping document (Turtle).

    Returns:
        Sorted, de-duplicated ``(entity_name, folder_path)`` pairs.

    Raises:
        R2RMLParseError: The document is not valid Turtle, contains no
            ``rr:TriplesMap``, or a TriplesMap does not declare exactly one
            ``rr:tableName`` and one ``uipath:folderPath``.
    """
    graph = Graph()
    try:
        graph.parse(data=r2rml_text, format="turtle")
    except Exception as e:  # rdflib raises assorted parser errors
        raise R2RMLParseError(f"R2RML mapping is not valid Turtle: {e}") from e

    triples_maps = list(graph.subjects(RDF.type, RR.TriplesMap))
    if not triples_maps:
        raise R2RMLParseError("No `rr:TriplesMap` found in the R2RML mapping.")

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for triples_map in triples_maps:
        table_names = [
            str(name)
            for logical_table in graph.objects(triples_map, RR.logicalTable)
            for name in graph.objects(logical_table, RR.tableName)
        ]
        folder_paths = [
            str(folder) for folder in graph.objects(triples_map, UIPATH.folderPath)
        ]
        if len(table_names) != 1 or len(folder_paths) != 1:
            raise R2RMLParseError(
                "Each TriplesMap must declare exactly one rr:tableName (via "
                "rr:logicalTable) and one uipath:folderPath; found "
                f"tableName={table_names} folderPath={folder_paths}."
            )
        pair = (table_names[0], folder_paths[0])
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)

    return sorted(pairs)
