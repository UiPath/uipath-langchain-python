"""JSONPath utilities."""

from jsonpath_ng import JSONPath, parse
from jsonpath_ng.jsonpath import Child, Fields, Root


def parse_jsonpath_segments(json_path: str) -> list[str]:
    """Parse JSON path $['a']['b'] into list of segments ['a', 'b']."""
    jsonpath_expr = parse(json_path)
    return _extract_segments(jsonpath_expr)


def _extract_segments(expr: JSONPath) -> list[str]:
    """Recursively extract segments from a JSONPath expression tree.

    Args:
        expr: The JSONPath expression node (Root, Child, Fields, etc.)
    """
    parts: list[str] = []

    if isinstance(expr, Root):
        pass
    elif isinstance(expr, Fields):
        # Leaf node, add it to the path
        parts.extend(expr.fields)
    elif isinstance(expr, Child):
        # Child node, walk left then right to maintain order
        parts.extend(_extract_segments(expr.left))
        parts.extend(_extract_segments(expr.right))

    return parts
