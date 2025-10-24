from typing import Any, Dict

from langgraph.graph.state import CompiledStateGraph


def resolve_refs(schema, root=None):
    """Recursively resolves $ref references in a JSON schema."""
    if root is None:
        root = schema  # Store the root schema to resolve $refs

    if isinstance(schema, dict):
        if "$ref" in schema:
            ref_path = schema["$ref"].lstrip("#/").split("/")
            ref_schema = root
            for part in ref_path:
                ref_schema = ref_schema.get(part, {})
            return resolve_refs(ref_schema, root)

        return {k: resolve_refs(v, root) for k, v in schema.items()}

    elif isinstance(schema, list):
        return [resolve_refs(item, root) for item in schema]

    return schema


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


def generate_schema_from_graph(
    graph: CompiledStateGraph[Any, Any, Any],
) -> Dict[str, Any]:
    """Extract input/output schema from a LangGraph graph"""
    schema = {
        "input": {"type": "object", "properties": {}, "required": []},
        "output": {"type": "object", "properties": {}, "required": []},
    }

    if hasattr(graph, "input_schema"):
        if hasattr(graph.input_schema, "model_json_schema"):
            input_schema = graph.input_schema.model_json_schema()
            unpacked_ref_def_properties = resolve_refs(input_schema)

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
            unpacked_ref_def_properties = resolve_refs(output_schema)

            # Process the schema to handle nullable types
            processed_properties = process_nullable_types(
                unpacked_ref_def_properties.get("properties", {})
            )

            schema["output"]["properties"] = processed_properties
            schema["output"]["required"] = unpacked_ref_def_properties.get(
                "required", []
            )

    return schema
