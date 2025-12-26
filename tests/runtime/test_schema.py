"""Tests for schema utility functions."""

from unittest.mock import MagicMock

from pydantic import BaseModel, Field

from uipath_langchain.runtime.schema import (
    _resolve_refs,
    get_entrypoints_schema,
)


class TestResolveRefs:
    """Tests for the resolve_refs function."""

    def test_simple_schema_without_refs(self):
        """Should return schema unchanged when no $refs exist."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }

        result, has_circular = _resolve_refs(schema)

        assert result == schema
        assert has_circular is False

    def test_simple_ref_resolution(self):
        """Should resolve a simple $ref to its definition."""
        schema = {
            "properties": {"user": {"$ref": "#/$defs/User"}},
            "$defs": {
                "User": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                }
            },
        }

        result, has_circular = _resolve_refs(schema)

        assert result["properties"]["user"]["type"] == "object"
        assert result["properties"]["user"]["properties"]["name"]["type"] == "string"
        assert has_circular is False

    def test_circular_dependency_detection(self):
        """Should detect circular dependencies in schema."""
        schema = {
            "properties": {"node": {"$ref": "#/$defs/Node"}},
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "next": {"$ref": "#/$defs/Node"},
                    },
                }
            },
        }

        result, has_circular = _resolve_refs(schema)

        assert has_circular is True
        # Check that circular ref was replaced with simplified schema
        assert result["properties"]["node"]["properties"]["next"]["type"] == "object"
        assert (
            "Circular reference"
            in result["properties"]["node"]["properties"]["next"]["description"]
        )

    def test_nested_refs_in_properties(self):
        """Should resolve nested $refs in object properties."""
        schema = {
            "properties": {
                "person": {"$ref": "#/$defs/Person"},
                "address": {"$ref": "#/$defs/Address"},
            },
            "$defs": {
                "Person": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
                "Address": {
                    "type": "object",
                    "properties": {"street": {"type": "string"}},
                },
            },
        }

        result, has_circular = _resolve_refs(schema)

        assert result["properties"]["person"]["type"] == "object"
        assert result["properties"]["person"]["properties"]["name"]["type"] == "string"
        assert result["properties"]["address"]["type"] == "object"
        assert (
            result["properties"]["address"]["properties"]["street"]["type"] == "string"
        )
        assert has_circular is False

    def test_refs_in_arrays(self):
        """Should resolve $refs inside array items."""
        schema = {
            "properties": {
                "users": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/User"},
                }
            },
            "$defs": {
                "User": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                }
            },
        }

        result, has_circular = _resolve_refs(schema)

        assert result["properties"]["users"]["items"]["type"] == "object"
        assert (
            result["properties"]["users"]["items"]["properties"]["id"]["type"]
            == "integer"
        )
        assert has_circular is False

    def test_multiple_circular_dependencies(self):
        """Should handle multiple circular dependencies in same schema."""
        schema = {
            "properties": {
                "node1": {"$ref": "#/$defs/Node"},
                "node2": {"$ref": "#/$defs/Node"},
            },
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "next": {"$ref": "#/$defs/Node"},
                    },
                }
            },
        }

        result, has_circular = _resolve_refs(schema)

        assert has_circular is True


class TestGenerateSchemaFromGraph:
    """Tests for the generate_schema_from_graph function."""

    def test_graph_without_schemas(self):
        """Should return empty schemas when graph has no input/output schemas."""
        mock_graph = MagicMock()
        del mock_graph.input_schema
        del mock_graph.output_schema

        result = get_entrypoints_schema(mock_graph)

        assert result.schema["input"] == {
            "type": "object",
            "properties": {},
            "required": [],
        }
        assert result.schema["output"] == {
            "type": "object",
            "properties": {},
            "required": [],
        }
        assert result.has_input_circular_dependency is False
        assert result.has_output_circular_dependency is False

    def test_graph_with_simple_schemas(self):
        """Should extract input and output schemas from graph."""

        class InputModel(BaseModel):
            query: str = Field(description="User query")
            max_results: int = Field(default=10)

        class OutputModel(BaseModel):
            response: str = Field(description="Agent response")

        mock_graph = MagicMock()
        mock_graph.input_schema = InputModel
        mock_graph.output_schema = OutputModel

        result = get_entrypoints_schema(mock_graph)

        assert "query" in result.schema["input"]["properties"]
        assert "max_results" in result.schema["input"]["properties"]
        assert result.schema["input"]["properties"]["query"]["type"] == "string"
        assert "response" in result.schema["output"]["properties"]
        assert result.schema["output"]["properties"]["response"]["type"] == "string"
        assert result.has_input_circular_dependency is False
        assert result.has_output_circular_dependency is False

    def test_graph_with_circular_input_schema(self):
        """Should detect circular dependencies in input schema."""

        class NodeInput(BaseModel):
            value: str
            children: list["NodeInput"] = Field(default_factory=list)

        mock_graph = MagicMock()
        mock_graph.input_schema = NodeInput
        del mock_graph.output_schema

        result = get_entrypoints_schema(mock_graph)

        assert result.has_input_circular_dependency is True
        assert result.has_output_circular_dependency is False
        assert "value" in result.schema["input"]["properties"]

    def test_graph_with_circular_output_schema(self):
        """Should detect circular dependencies in output schema."""

        class TreeOutput(BaseModel):
            name: str
            parent: "TreeOutput | None" = None

        mock_graph = MagicMock()
        del mock_graph.input_schema
        mock_graph.output_schema = TreeOutput

        result = get_entrypoints_schema(mock_graph)

        assert result.has_input_circular_dependency is False
        assert result.has_output_circular_dependency is True
        assert "name" in result.schema["output"]["properties"]

    def test_graph_with_both_circular_schemas(self):
        """Should detect circular dependencies in both input and output schemas."""

        class CircularInput(BaseModel):
            data: str
            ref: "CircularInput | None" = None

        class CircularOutput(BaseModel):
            result: str
            next: "CircularOutput | None" = None

        mock_graph = MagicMock()
        mock_graph.input_schema = CircularInput
        mock_graph.output_schema = CircularOutput

        result = get_entrypoints_schema(mock_graph)

        assert result.has_input_circular_dependency is True
        assert result.has_output_circular_dependency is True
        assert "data" in result.schema["input"]["properties"]
        assert "result" in result.schema["output"]["properties"]

    def test_graph_with_required_fields(self):
        """Should extract required fields from schemas."""

        class StrictModel(BaseModel):
            required_field: str
            optional_field: str | None = None

        mock_graph = MagicMock()
        mock_graph.input_schema = StrictModel
        del mock_graph.output_schema

        result = get_entrypoints_schema(mock_graph)

        assert "required_field" in result.schema["input"]["required"]
        assert "optional_field" not in result.schema["input"]["required"]
