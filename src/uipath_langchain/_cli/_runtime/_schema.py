from dataclasses import dataclass
from typing import Any, Dict

from langchain_core.runnables.graph import Graph
from langgraph.graph.state import CompiledStateGraph
from uipath.runtime.schema import (
    UiPathRuntimeEdge,
    UiPathRuntimeGraph,
    UiPathRuntimeNode,
)


@dataclass
class SchemaDetails:
    schema: dict[str, Any]
    has_input_circular_dependency: bool
    has_output_circular_dependency: bool


def _convert_graph_to_uipath(graph: Graph) -> UiPathRuntimeGraph:
    """Helper to convert a LangGraph Graph object to UiPathRuntimeGraph.

    Args:
        graph: A LangGraph Graph object (from get_graph() call)

    Returns:
        UiPathRuntimeGraph with nodes and edges
    """
    nodes: list[UiPathRuntimeNode] = []
    for _, node in graph.nodes.items():
        nodes.append(
            UiPathRuntimeNode(
                id=node.id,
                name=node.name or node.id,
                type=node.__class__.__name__,
                subgraph=None,
            )
        )

    edges: list[UiPathRuntimeEdge] = []
    for edge in graph.edges:
        edges.append(
            UiPathRuntimeEdge(
                source=edge.source,
                target=edge.target,
                label=getattr(edge, "data", None) or getattr(edge, "label", None),
            )
        )

    return UiPathRuntimeGraph(nodes=nodes, edges=edges)


def langgraph_to_uipath_graph(
    compiled_graph: CompiledStateGraph[Any, Any, Any, Any], xray: int = 1
) -> UiPathRuntimeGraph:
    """Convert a compiled LangGraph to UiPathRuntimeGraph structure.

    Args:
        compiled_graph: A compiled LangGraph (Pregel instance)
        xray: Depth of subgraph expansion (0 = no subgraphs, 1 = one level, etc.)

    Returns:
        UiPathRuntimeGraph with hierarchical subgraph structure
    """
    graph: Graph = compiled_graph.get_graph(xray=xray)

    subgraphs_dict: dict[str, Graph] = {}
    if xray:
        for name, subgraph_pregel in compiled_graph.get_subgraphs():
            next_xray: int | bool = (
                xray - 1 if isinstance(xray, int) and xray > 1 else False
            )
            subgraph_graph: Graph = subgraph_pregel.get_graph(xray=next_xray)
            subgraphs_dict[name] = subgraph_graph

    nodes: list[UiPathRuntimeNode] = []
    for node_id, node in graph.nodes.items():
        subgraph: UiPathRuntimeGraph | None = None
        if node_id in subgraphs_dict:
            subgraph = _convert_graph_to_uipath(subgraphs_dict[node_id])

        nodes.append(
            UiPathRuntimeNode(
                id=node.id,
                name=node.name or node.id,
                type=node.__class__.__name__,
                subgraph=subgraph,
            )
        )

    edges: list[UiPathRuntimeEdge] = []
    for edge in graph.edges:
        edges.append(
            UiPathRuntimeEdge(
                source=edge.source,
                target=edge.target,
                label=getattr(edge, "data", None) or getattr(edge, "label", None),
            )
        )

    return UiPathRuntimeGraph(nodes=nodes, edges=edges)


def generate_schema_from_graph(
    graph: CompiledStateGraph[Any, Any, Any],
) -> SchemaDetails:
    """Extract input/output schema from a LangGraph graph"""
    input_circular_dependency = False
    output_circular_dependency = False
    schema = {
        "input": {"type": "object", "properties": {}, "required": []},
        "output": {"type": "object", "properties": {}, "required": []},
    }

    if hasattr(graph, "input_schema"):
        if hasattr(graph.input_schema, "model_json_schema"):
            input_schema = graph.input_schema.model_json_schema()
            unpacked_ref_def_properties, input_circular_dependency = _resolve_refs(
                input_schema
            )

            # Process the schema to handle nullable types
            processed_properties = process_nullable_types(
                unpacked_ref_def_properties.get("properties", {})
            )

            schema["input"]["properties"] = processed_properties
            schema["input"]["required"] = unpacked_ref_def_properties.get(
                "required", []
            )

    if hasattr(graph, "output_schema"):
        if hasattr(graph.output_schema, "model_json_schema"):
            output_schema = graph.output_schema.model_json_schema()
            unpacked_ref_def_properties, output_circular_dependency = _resolve_refs(
                output_schema
            )

            # Process the schema to handle nullable types
            processed_properties = process_nullable_types(
                unpacked_ref_def_properties.get("properties", {})
            )

            schema["output"]["properties"] = processed_properties
            schema["output"]["required"] = unpacked_ref_def_properties.get(
                "required", []
            )

    return SchemaDetails(schema, input_circular_dependency, output_circular_dependency)


def _resolve_refs(schema, root=None, visited=None):
    """Recursively resolves $ref references in a JSON schema, handling circular references.

    Returns:
        tuple: (resolved_schema, has_circular_dependency)
    """
    if root is None:
        root = schema

    if visited is None:
        visited = set()

    has_circular = False

    if isinstance(schema, dict):
        if "$ref" in schema:
            ref_path = schema["$ref"]

            if ref_path in visited:
                # Circular dependency detected
                return {
                    "type": "object",
                    "description": f"Circular reference to {ref_path}",
                }, True

            visited.add(ref_path)

            # Resolve the reference
            ref_parts = ref_path.lstrip("#/").split("/")
            ref_schema = root
            for part in ref_parts:
                ref_schema = ref_schema.get(part, {})

            result, circular = _resolve_refs(ref_schema, root, visited)
            has_circular = has_circular or circular

            # Remove from visited after resolution (allows the same ref in different branches)
            visited.discard(ref_path)

            return result, has_circular

        resolved_dict = {}
        for k, v in schema.items():
            resolved_value, circular = _resolve_refs(v, root, visited)
            resolved_dict[k] = resolved_value
            has_circular = has_circular or circular
        return resolved_dict, has_circular

    elif isinstance(schema, list):
        resolved_list = []
        for item in schema:
            resolved_item, circular = _resolve_refs(item, root, visited)
            resolved_list.append(resolved_item)
            has_circular = has_circular or circular
        return resolved_list, has_circular

    return schema, False


def process_nullable_types(
    schema: Dict[str, Any] | list[Any] | Any,
) -> Dict[str, Any] | list[Any]:
    """Process the schema to handle nullable types by removing anyOf with null and keeping the base type."""
    if isinstance(schema, dict):
        if "anyOf" in schema and len(schema["anyOf"]) == 2:
            types = [t.get("type") for t in schema["anyOf"]]
            if "null" in types:
                non_null_type = next(
                    t for t in schema["anyOf"] if t.get("type") != "null"
                )
                return non_null_type

        return {k: process_nullable_types(v) for k, v in schema.items()}
    elif isinstance(schema, list):
        return [process_nullable_types(item) for item in schema]
    return schema
