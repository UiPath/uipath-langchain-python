"""Tests for Data Fabric prompt builder — enriched schemas and prompt structure."""

from types import SimpleNamespace
from unittest.mock import patch

from uipath_langchain.agent.tools.datafabric_tool.datafabric_prompt_builder import (
    build_entity_context,
    build_sql_context,
    format_sql_context,
)


def _sql_type(type_name: str) -> SimpleNamespace:
    """Create a sql_type-like object with a .name attribute."""
    return SimpleNamespace(name=type_name)


def _make_mock_entity(
    name: str = "Customer",
    display_name: str = "Customer",
    description: str = "All customers",
    record_count: int = 1500,
    fields: list[dict] | None = None,
):
    """Create a mock Entity with FieldMetadata-like fields."""
    if fields is None:
        fields = [
            {
                "name": "id",
                "display_name": "ID",
                "is_primary_key": True,
                "is_foreign_key": False,
                "is_external_field": False,
                "is_hidden_field": False,
                "is_system_field": False,
                "is_unique": True,
                "is_required": True,
                "sql_type": _sql_type("int"),
                "description": "Primary identifier",
            },
            {
                "name": "name",
                "display_name": "Customer Name",
                "is_primary_key": False,
                "is_foreign_key": False,
                "is_external_field": False,
                "is_hidden_field": False,
                "is_system_field": False,
                "is_unique": False,
                "is_required": False,
                "sql_type": _sql_type("varchar"),
                "description": None,
            },
            {
                "name": "order_id",
                "display_name": "Order ID",
                "is_primary_key": False,
                "is_foreign_key": True,
                "is_external_field": False,
                "is_hidden_field": False,
                "is_system_field": False,
                "is_unique": False,
                "is_required": False,
                "sql_type": _sql_type("int"),
                "description": "Reference to orders table",
            },
        ]

    mock_fields = [SimpleNamespace(**f) for f in fields]
    return SimpleNamespace(
        id="ent-001",
        name=name,
        display_name=display_name,
        description=description,
        record_count=record_count,
        fields=mock_fields,
    )


class TestBuildEntityContext:
    def test_primary_key_is_extracted(self):
        entity = _make_mock_entity()
        ctx = build_entity_context(entity)
        id_field = next(f for f in ctx.entity_schema.fields if f.name == "id")
        assert id_field.is_primary_key is True

    def test_foreign_key_is_extracted(self):
        entity = _make_mock_entity()
        ctx = build_entity_context(entity)
        fk_field = next(f for f in ctx.entity_schema.fields if f.name == "order_id")
        assert fk_field.is_foreign_key is True

    def test_hidden_and_system_fields_excluded(self):
        entity = _make_mock_entity(
            fields=[
                {
                    "name": "visible",
                    "display_name": "Visible",
                    "is_primary_key": False,
                    "is_foreign_key": False,
                    "is_external_field": False,
                    "is_hidden_field": False,
                    "is_system_field": False,
                    "is_unique": False,
                    "is_required": False,
                    "sql_type": _sql_type("varchar"),
                    "description": None,
                },
                {
                    "name": "hidden",
                    "display_name": "Hidden",
                    "is_primary_key": False,
                    "is_foreign_key": False,
                    "is_external_field": False,
                    "is_hidden_field": True,
                    "is_system_field": False,
                    "is_unique": False,
                    "is_required": False,
                    "sql_type": _sql_type("varchar"),
                    "description": None,
                },
            ]
        )
        ctx = build_entity_context(entity)
        assert len(ctx.entity_schema.fields) == 1
        assert ctx.entity_schema.fields[0].name == "visible"


class TestFormatSqlContext:
    def test_schema_appears_before_sql_guide(self):
        entity = _make_mock_entity()
        ctx = build_sql_context([entity])
        text = format_sql_context(ctx)
        schema_pos = text.find("## Available Data Fabric Entities")
        guide_pos = text.find("## SQL Query Generation Guide")
        assert schema_pos < guide_pos

    def test_enriched_table_headers(self):
        entity = _make_mock_entity()
        ctx = build_sql_context([entity])
        text = format_sql_context(ctx)
        assert "| Field | Type | Key | Description |" in text

    def test_pk_marker_in_output(self):
        entity = _make_mock_entity()
        ctx = build_sql_context([entity])
        text = format_sql_context(ctx)
        assert "| PK |" in text

    def test_fk_marker_in_output(self):
        entity = _make_mock_entity()
        ctx = build_sql_context([entity])
        text = format_sql_context(ctx)
        assert "| FK |" in text

    def test_record_count_in_output(self):
        entity = _make_mock_entity(record_count=1500)
        ctx = build_sql_context([entity])
        text = format_sql_context(ctx)
        assert "1,500 records" in text

    def test_description_in_field_row(self):
        entity = _make_mock_entity()
        ctx = build_sql_context([entity])
        text = format_sql_context(ctx)
        assert "Primary identifier" in text

    def test_agent_instructions_at_end(self):
        entity = _make_mock_entity()
        ctx = build_sql_context([entity], base_system_prompt="Be helpful")
        text = format_sql_context(ctx)
        agent_pos = text.find("## Agent Instructions")
        guide_pos = text.find("## SQL Query Generation Guide")
        assert agent_pos > guide_pos

    @patch(
        "uipath_langchain.agent.tools.datafabric_tool.datafabric_prompt_builder.load_optimized_prompts"
    )
    def test_optimized_prompts_used_when_available(self, mock_load):
        mock_load.return_value = {
            "optimized_instruction": "OPTIMIZED: Generate SQL",
            "few_shot_examples": [
                {"question": "How many?", "sql": "SELECT COUNT(id) FROM T LIMIT 1"}
            ],
            "optimization_metadata": {"val_accuracy": 0.92},
        }
        entity = _make_mock_entity()
        ctx = build_sql_context([entity])
        text = format_sql_context(ctx)
        assert "OPTIMIZED: Generate SQL" in text
        assert "How many?" in text

    @patch(
        "uipath_langchain.agent.tools.datafabric_tool.datafabric_prompt_builder.load_optimized_prompts"
    )
    def test_fallback_when_no_optimized_prompts(self, mock_load):
        mock_load.return_value = None
        entity = _make_mock_entity()
        ctx = build_sql_context([entity])
        text = format_sql_context(ctx)
        assert "QUERY PLANNING" in text  # From SQL_GENERATION_GUIDE
