"""``$ref`` inspection shared by both schema-to-model backends.

These helpers work on the JSON Schema document alone, so they are identical
whichever library builds the models.
"""

from typing import Any

# Marker left on any OUTPUT-schema node whose $ref target could not be resolved.
# Both backends discard $defs names and non-standard (x-*) keys but preserve the
# standard `title`/`description` annotations on a property, so the marker lives as
# annotations rather than a named type. Downstream can detect an unresolved field
# via ``title == UNRESOLVED_TYPE_TITLE``. See create_output_model.
UNRESOLVED_TYPE_TITLE = "UiPathUnresolvedType"


def ref_resolves(ref: str, root: dict[str, Any]) -> bool:
    """Whether a local JSON-pointer ``$ref`` (``#/...``) resolves within `root`.

    External/URL refs and the bare ``#`` (whole-document) ref return False:
    neither backend can resolve them, so they are treated as dangling.
    """
    if not ref.startswith("#/"):
        return False
    node: Any = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")  # JSON-pointer unescape
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return False
    return True


def resolve_pointer(schema: dict[str, Any], ref: str) -> Any:
    """Resolve a local JSON pointer against `schema`, or None."""
    if not ref.startswith("#/"):
        return None
    node: Any = schema
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def neutralize_dangling_refs(
    schema: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Return a copy of `schema` with every unresolvable ``$ref`` replaced.

    A ``$ref`` is dangling when its target is not present under ``$defs``/
    ``definitions`` (e.g. a .NET ``Nullable<decimal>`` serialized without its
    definition). Each dangling ref node is replaced *in place* by a permissive,
    self-documenting placeholder (accepts any value; the original ref is kept in
    its ``description``), so valid sibling fields and valid ``$ref``s -- including
    those nested in arrays, objects, or ``$defs`` -- are preserved. This keeps the
    output schema usable by best-effort features instead of discarding it whole.

    Returns:
        A tuple of (sanitized schema copy, list of the dangling ref strings found).
    """
    dropped: list[str] = []

    def visit(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and not ref_resolves(ref, schema):
                dropped.append(ref)
                return {
                    "title": UNRESOLVED_TYPE_TITLE,
                    "description": (
                        f"Unresolved $ref '{ref}'; original type could not be "
                        "resolved at startup, so this field accepts any value."
                    ),
                }
            return {key: visit(value) for key, value in node.items()}
        if isinstance(node, list):
            return [visit(item) for item in node]
        return node

    return visit(schema), dropped
