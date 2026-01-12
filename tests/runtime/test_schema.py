"""Tests for schema utility functions."""

from dataclasses import dataclass, field
from typing import Any

import pytest
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from uipath.runtime.schema import transform_references

from uipath_langchain.runtime import UiPathLangGraphRuntime


def generate_simple_langgraph_graph(input_schema: type, output_schema: type):
    # note: type ignore are needed here since mypy can t validate a dynamically created object's type
    def node(state: input_schema) -> input_schema:  # type: ignore
        return state

    builder = StateGraph(
        state_schema=input_schema,
        input_schema=input_schema,
        output_schema=output_schema,
    )  # type: ignore
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    graph = builder.compile()
    return graph


class TestResolveRefs:
    """Tests for the resolve_refs function."""

    def test_simple_schema_without_refs(self):
        """Should return schema unchanged when no $refs exist."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }

        result, has_circular = transform_references(schema)

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

        result, has_circular = transform_references(schema)

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

        result, has_circular = transform_references(schema)

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

        result, has_circular = transform_references(schema)

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

        result, has_circular = transform_references(schema)

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

        result, has_circular = transform_references(schema)

        assert has_circular is True


class TestGenerateSchemaFromGraph:
    """Tests for the generate_schema_from_graph function."""

    async def test_graph_with_simple_schemas(self):
        """Should extract input and output schemas from graph."""

        class InputModel(BaseModel):
            query: str = Field(description="User query")
            max_results: int = Field(default=10)

        class OutputModel(BaseModel):
            response: str = Field(description="Agent response")

        runtime = UiPathLangGraphRuntime(
            graph=generate_simple_langgraph_graph(
                input_schema=InputModel, output_schema=OutputModel
            ),
            entrypoint="test_entrypoint",
        )

        entire_schema = await runtime.get_schema()
        input_schema = entire_schema.input
        output_schema = entire_schema.output

        assert "query" in input_schema["properties"]
        assert "max_results" in input_schema["properties"]
        assert input_schema["properties"]["query"]["type"] == "string"
        assert "response" in output_schema["properties"]
        assert output_schema["properties"]["response"]["type"] == "string"

    async def test_graph_with_complex_pydantic_schemas(self):
        """Should extract complex nested input and output schemas from graph."""

        class Address(BaseModel):
            street: str
            city: str
            zip_code: str | None = None

        class User(BaseModel):
            name: str
            age: int
            email: str | None = None
            addresses: list[Address] = Field(default_factory=list)

        class InputModel(BaseModel):
            user: User
            tags: list[str] = Field(default_factory=list)
            metadata: dict[str, Any] = Field(default_factory=dict)
            priority: int = Field(default=5, ge=1, le=10)

        class ResultItem(BaseModel):
            id: str
            score: float
            data: dict[str, Any]

        class OutputModel(BaseModel):
            results: list[ResultItem]
            total_count: int
            success: bool = True

        runtime = UiPathLangGraphRuntime(
            graph=generate_simple_langgraph_graph(
                input_schema=InputModel, output_schema=OutputModel
            ),
            entrypoint="test_entrypoint",
        )

        entire_schema = await runtime.get_schema()
        input_schema = entire_schema.input
        output_schema = entire_schema.output

        assert "user" in input_schema["properties"]
        assert "tags" in input_schema["properties"]
        assert "metadata" in input_schema["properties"]
        assert "priority" in input_schema["properties"]
        assert input_schema["properties"]["tags"]["type"] == "array"
        assert input_schema["properties"]["metadata"]["type"] == "object"

        assert "results" in output_schema["properties"]
        assert "total_count" in output_schema["properties"]
        assert "success" in output_schema["properties"]
        assert output_schema["properties"]["results"]["type"] == "array"
        assert output_schema["properties"]["total_count"]["type"] == "integer"

    async def test_graph_with_complex_dataclass_schemas(self):
        """Should extract complex nested dataclass input and output schemas from graph."""

        @dataclass
        class Address:
            street: str
            city: str
            zip_code: str | None = None

        @dataclass
        class User:
            name: str
            age: int
            email: str | None = None
            addresses: list[Address] = field(default_factory=list)

        @dataclass
        class InputModel:
            user: User
            tags: list[str] = field(default_factory=list)
            metadata: dict[str, Any] = field(default_factory=dict)
            priority: int = 5

        @dataclass
        class ResultItem:
            id: str
            score: float
            data: dict[str, Any]

        @dataclass
        class OutputModel:
            results: list[ResultItem]
            total_count: int
            success: bool = True

        runtime = UiPathLangGraphRuntime(
            graph=generate_simple_langgraph_graph(
                input_schema=InputModel, output_schema=OutputModel
            ),
            entrypoint="test_entrypoint",
        )

        entire_schema = await runtime.get_schema()
        input_schema = entire_schema.input
        output_schema = entire_schema.output

        assert "user" in input_schema["properties"]
        assert "tags" in input_schema["properties"]
        assert "metadata" in input_schema["properties"]
        assert "priority" in input_schema["properties"]
        assert input_schema["properties"]["tags"]["type"] == "array"
        assert input_schema["properties"]["metadata"]["type"] == "object"

        assert "results" in output_schema["properties"]
        assert "total_count" in output_schema["properties"]
        assert "success" in output_schema["properties"]
        assert output_schema["properties"]["results"]["type"] == "array"
        assert output_schema["properties"]["total_count"]["type"] == "integer"

    async def test_graph_with_complex_typeddict_schemas(self):
        """Should extract complex nested TypedDict input and output schemas from graph."""

        class Address(TypedDict):
            street: str
            city: str
            zip_code: str | None

        class User(TypedDict):
            name: str
            age: int
            email: str | None
            addresses: list[Address]

        class InputModel(TypedDict):
            user: User
            tags: list[str]
            metadata: dict[str, Any]
            priority: int

        class ResultItem(TypedDict):
            id: str
            score: float
            data: dict[str, Any]

        class OutputModel(TypedDict):
            results: list[ResultItem]
            total_count: int
            success: bool

        runtime = UiPathLangGraphRuntime(
            graph=generate_simple_langgraph_graph(
                input_schema=InputModel, output_schema=OutputModel
            ),
            entrypoint="test_entrypoint",
        )

        entire_schema = await runtime.get_schema()
        input_schema = entire_schema.input
        output_schema = entire_schema.output

        assert "user" in input_schema["properties"]
        assert "tags" in input_schema["properties"]
        assert "metadata" in input_schema["properties"]
        assert "priority" in input_schema["properties"]
        assert input_schema["properties"]["tags"]["type"] == "array"
        assert input_schema["properties"]["metadata"]["type"] == "object"

        assert "results" in output_schema["properties"]
        assert "total_count" in output_schema["properties"]
        assert "success" in output_schema["properties"]
        assert output_schema["properties"]["results"]["type"] == "array"
        assert output_schema["properties"]["total_count"]["type"] == "integer"

    async def test_graph_with_required_fields(self):
        """Should extract required fields from schemas."""

        class StrictModel(BaseModel):
            required_field: str
            optional_field: str | None = None

        runtime = UiPathLangGraphRuntime(
            graph=generate_simple_langgraph_graph(
                input_schema=StrictModel, output_schema=StrictModel
            ),
            entrypoint="test_entrypoint",
        )

        entire_schema = await runtime.get_schema()
        input_schema = entire_schema.input
        output_schema = entire_schema.output

        assert "required_field" in input_schema["required"]
        assert "optional_field" not in input_schema["required"]

        assert "required_field" in output_schema["required"]
        assert "optional_field" not in output_schema["required"]


class TestSchemaGeneration:
    @pytest.mark.parametrize(
        "input_model_code",
        [
            """
# pydantic BaseModel

from pydantic import BaseModel, Field
from uipath.platform.attachments import Attachment

class InputModel(BaseModel):
    input_file: Attachment
    other_field: int | None = Field(default=None)""",
            """
# dataclass

from uipath.platform.attachments import Attachment
from dataclasses import dataclass
@dataclass
class InputModel:
    input_file: Attachment
    other_field: int | None = None""",
            """
# TypedDict

from typing_extensions import TypedDict
from typing_extensions import NotRequired
from uipath.platform.attachments import Attachment
class InputModel(TypedDict):
    input_file: Attachment
    other_field: NotRequired[int | None]
    """,
        ],
    )
    async def test_schema_generation_resolves_attachments(
        self, input_model_code: str
    ) -> None:
        """Test that attachments are resolved in runtime schema"""

        # execute model code to get its schema
        exec_globals: dict[str, Any] = {}
        exec(input_model_code, exec_globals)
        InputModel = exec_globals["InputModel"]

        runtime = UiPathLangGraphRuntime(
            graph=generate_simple_langgraph_graph(
                input_schema=InputModel, output_schema=InputModel
            ),
            entrypoint="test_entrypoint",
        )

        def check_attachment_in_schema(schema: dict[str, Any]):
            assert "definitions" in schema
            assert "job-attachment" in schema["definitions"]
            assert schema["definitions"]["job-attachment"]["type"] == "object"
            assert (
                schema["definitions"]["job-attachment"]["x-uipath-resource-kind"]
                == "JobAttachment"
            )
            assert all(
                prop_name in schema["definitions"]["job-attachment"]["properties"]
                for prop_name in ["ID", "FullName", "MimeType", "Metadata"]
            )

            assert len(schema["properties"]) == 2
            assert all(
                prop_name in schema["properties"]
                for prop_name in ["input_file", "other_field"]
            )
            assert schema["required"] == ["input_file"]

        entire_schema = await runtime.get_schema()

        input_schema = entire_schema.input
        output_schema = entire_schema.output
        check_attachment_in_schema(input_schema)
        check_attachment_in_schema(output_schema)
