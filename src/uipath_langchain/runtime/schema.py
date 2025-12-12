from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.graph import Graph, Node
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.pregel._read import PregelNode
from langgraph.pregel._write import ChannelWrite, ChannelWriteEntry
from uipath.runtime.schema import (
    UiPathRuntimeEdge,
    UiPathRuntimeGraph,
    UiPathRuntimeNode,
)

try:
    from langgraph._internal._runnable import RunnableCallable
except ImportError:
    RunnableCallable = None  # type: ignore

T = TypeVar("T")


@dataclass
class SchemaDetails:
    schema: dict[str, Any]
    has_input_circular_dependency: bool
    has_output_circular_dependency: bool


def _unwrap_runnable_callable(
    runnable: Runnable[Any, Any],
    target_type: type[T],
    _seen: set[int] | None = None,
) -> T | None:
    """Try to find an instance of target_type (e.g., BaseChatModel)
    inside a Runnable.

    Handles:
    - Direct model runnables
    - LangGraph RunnableCallable
    - LangChain function runnables (RunnableLambda, etc.)
    - RunnableBinding / RunnableSequence with nested steps
    """
    if isinstance(runnable, target_type):
        return runnable

    if _seen is None:
        _seen = set()
    obj_id = id(runnable)
    if obj_id in _seen:
        return None
    _seen.add(obj_id)

    func: Callable[..., Any] | None = None

    # 1) LangGraph internal RunnableCallable
    if RunnableCallable is not None and isinstance(runnable, RunnableCallable):
        func = getattr(runnable, "func", None)

    # 2) Generic LangChain function-wrapping runnables
    if func is None:
        for attr_name in ("func", "_func", "afunc", "_afunc"):
            maybe = getattr(runnable, attr_name, None)
            if callable(maybe):
                func = maybe
                break

    # 3) Look into the function closure for a model
    if func is not None:
        closure = getattr(func, "__closure__", None) or ()
        for cell in closure:
            content = getattr(cell, "cell_contents", None)
            if isinstance(content, target_type):
                return content
            if isinstance(content, Runnable):
                found = _unwrap_runnable_callable(content, target_type, _seen)
                if found is not None:
                    return found

    # 4) Deep-scan attributes, including nested runnables / containers
    def _scan_value(value: Any) -> T | None:
        if isinstance(value, target_type):
            return value
        if isinstance(value, Runnable):
            return _unwrap_runnable_callable(value, target_type, _seen)
        if isinstance(value, dict):
            for v in value.values():
                found = _scan_value(v)
                if found is not None:
                    return found
        # Handle lists, tuples, sets, etc. but avoid strings/bytes
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            for item in value:
                found = _scan_value(item)
                if found is not None:
                    return found
        return None

    try:
        attrs = vars(runnable)
    except TypeError:
        attrs = {}

    for value in attrs.values():
        found = _scan_value(value)
        if found is not None:
            return found

    return None


def _get_node_type(node: Node) -> str:
    """Determine the type of a LangGraph node using strongly-typed isinstance checks.

    Args:
        node: A Node object from the graph

    Returns:
        String representing the node type
    """
    if node.id in ("__start__", "__end__"):
        return node.id

    if node.data is None:
        return "node"

    if not isinstance(node.data, Runnable):
        return "node"

    tool_node = _unwrap_runnable_callable(node.data, ToolNode)
    if tool_node is not None:
        return "tool"

    chat_model = _unwrap_runnable_callable(node.data, BaseChatModel)  # type: ignore[type-abstract]
    if chat_model is not None:
        return "model"

    language_model = _unwrap_runnable_callable(node.data, BaseLanguageModel)  # type: ignore[type-abstract]
    if language_model is not None:
        return "model"

    return "node"


def _get_node_metadata(node: Node) -> dict[str, Any]:
    """Extract metadata from a node in a type-safe manner.

    Args:
        node: A Node object from the graph

    Returns:
        Dictionary containing node metadata
    """
    if node.data is None:
        return {}

    # Early return if data is not a Runnable
    if not isinstance(node.data, Runnable):
        return {}

    metadata: dict[str, Any] = {}

    tool_node = _unwrap_runnable_callable(node.data, ToolNode)
    if tool_node is not None:
        if hasattr(tool_node, "_tools_by_name"):
            tools_by_name = tool_node._tools_by_name
            metadata["tool_names"] = list(tools_by_name.keys())
            metadata["tool_count"] = len(tools_by_name)
        return metadata

    chat_model = _unwrap_runnable_callable(node.data, BaseChatModel)  # type: ignore[type-abstract]
    if chat_model is not None:
        if hasattr(chat_model, "model") and isinstance(chat_model.model, str):
            metadata["model_name"] = chat_model.model
        elif hasattr(chat_model, "model_name") and chat_model.model_name:
            metadata["model_name"] = chat_model.model_name

        if hasattr(chat_model, "temperature") and chat_model.temperature is not None:
            metadata["temperature"] = chat_model.temperature

        if hasattr(chat_model, "max_tokens") and chat_model.max_tokens is not None:
            metadata["max_tokens"] = chat_model.max_tokens
        elif (
            hasattr(chat_model, "max_completion_tokens")
            and chat_model.max_completion_tokens is not None
        ):
            metadata["max_tokens"] = chat_model.max_completion_tokens

    return metadata


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
                type=_get_node_type(node),
                metadata=_get_node_metadata(node),
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


def get_graph_schema(
    compiled_graph: CompiledStateGraph[Any, Any, Any, Any], xray: int = 1
) -> UiPathRuntimeGraph:
    """Convert a compiled LangGraph to UiPathRuntimeGraph structure.

    Args:
        compiled_graph: A compiled LangGraph (Pregel instance)
        xray: Depth of subgraph expansion (0 = no subgraphs, 1 = one level, etc.)

    Returns:
        UiPathRuntimeGraph with hierarchical subgraph structure
    """
    graph: Graph = compiled_graph.get_graph(xray=0)  # Keep parent at xray=0

    subgraphs_dict: dict[str, UiPathRuntimeGraph] = {}
    if xray:
        for name, subgraph_pregel in compiled_graph.get_subgraphs():
            next_xray: int = xray - 1 if isinstance(xray, int) and xray > 0 else 0
            subgraph_graph: Graph = subgraph_pregel.get_graph(xray=next_xray)
            subgraphs_dict[name] = _convert_graph_to_uipath(subgraph_graph)

    nodes: list[UiPathRuntimeNode] = []
    for node_id, node in graph.nodes.items():
        subgraph: UiPathRuntimeGraph | None = subgraphs_dict.get(node_id)
        nodes.append(
            UiPathRuntimeNode(
                id=node.id,
                name=node.name or node.id,
                type=_get_node_type(node),
                metadata=_get_node_metadata(node),
                subgraph=subgraph,
            )
        )

    # Use a set to track unique edges (source, target)
    seen_edges: set[tuple[str, str]] = set()
    edges: list[UiPathRuntimeEdge] = []

    # First, add edges from graph.edges (static edges)
    for edge in graph.edges:
        edge_tuple = (edge.source, edge.target)
        if edge_tuple not in seen_edges:
            seen_edges.add(edge_tuple)
            edges.append(
                UiPathRuntimeEdge(
                    source=edge.source,
                    target=edge.target,
                    label=getattr(edge, "data", None) or getattr(edge, "label", None),
                )
            )

    # Build a map of channel -> target node
    channel_to_target: dict[str, str] = {}
    node_spec: PregelNode
    for node_name, node_spec in compiled_graph.nodes.items():
        for trigger in node_spec.triggers:
            if isinstance(trigger, str) and trigger.startswith("branch:to:"):
                channel_to_target[trigger] = node_name

    # Extract edges by looking at what each node writes to (dynamic edges from Command)
    for source_name, node_spec in compiled_graph.nodes.items():
        writer: Runnable[Any, Any]
        for writer in node_spec.writers:
            if isinstance(writer, ChannelWrite):
                write_entry: Any
                for write_entry in writer.writes:
                    if isinstance(write_entry, ChannelWriteEntry) and isinstance(
                        write_entry.channel, str
                    ):
                        target = channel_to_target.get(write_entry.channel)
                        if target:
                            edge_tuple = (source_name, target)
                            if edge_tuple not in seen_edges:
                                seen_edges.add(edge_tuple)
                                edges.append(
                                    UiPathRuntimeEdge(
                                        source=source_name, target=target, label=None
                                    )
                                )

    return UiPathRuntimeGraph(nodes=nodes, edges=edges)


def get_entrypoints_schema(
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
            processed_properties = _process_nullable_types(
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
            processed_properties = _process_nullable_types(
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


def _process_nullable_types(
    schema: dict[str, Any] | list[Any] | Any,
) -> dict[str, Any] | list[Any]:
    """Process the schema to handle nullable types by removing anyOf with null and keeping the base type."""
    if isinstance(schema, dict):
        if "anyOf" in schema and len(schema["anyOf"]) == 2:
            types = [t.get("type") for t in schema["anyOf"]]
            if "null" in types:
                non_null_type = next(
                    t for t in schema["anyOf"] if t.get("type") != "null"
                )
                return non_null_type

        return {k: _process_nullable_types(v) for k, v in schema.items()}
    elif isinstance(schema, list):
        return [_process_nullable_types(item) for item in schema]
    return schema


__all__ = [
    "get_graph_schema",
    "get_entrypoints_schema",
]
