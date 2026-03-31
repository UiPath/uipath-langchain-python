"""Pydantic models for Data Fabric entity schemas."""

from pydantic import BaseModel, Field

NUMERIC_TYPES = frozenset({"int", "decimal", "float", "double", "bigint"})
TEXT_TYPES = frozenset({"varchar", "nvarchar", "text", "string", "ntext"})


class FieldSchema(BaseModel):
    """Structured representation of a Data Fabric entity field."""

    name: str
    display_name: str | None = None
    type: str
    description: str | None = None
    is_foreign_key: bool = False
    is_required: bool = False
    is_unique: bool = False
    nullable: bool = True

    @property
    def display_type(self) -> str:
        """Type string with modifiers for markdown display."""
        modifiers = []
        if self.is_required:
            modifiers.append("required")
        if modifiers:
            return f"{self.type}, {', '.join(modifiers)}"
        return self.type

    @property
    def is_numeric(self) -> bool:
        return self.type.lower() in NUMERIC_TYPES

    @property
    def is_text(self) -> bool:
        return self.type.lower() in TEXT_TYPES


class EntitySchema(BaseModel):
    """Structured representation of a Data Fabric entity."""

    id: str | None = None
    entity_name: str
    display_name: str
    description: str | None = None
    record_count: int | None = None
    fields: list[FieldSchema]


class QueryPattern(BaseModel):
    """A SQL query pattern example derived from an entity's fields."""

    intent: str
    sql: str


class EntitySQLContext(BaseModel):
    """Entity schema enriched with query patterns for SQL generation."""

    entity_schema: EntitySchema
    query_patterns: list[QueryPattern]


class SQLContext(BaseModel):
    """Top-level container for the full schema context injected into the system prompt."""

    base_system_prompt: str | None = None
    resource_description: str | None = None
    sql_expert_system_prompt: str | None = None
    constraints: str | None = None
    entity_contexts: list[EntitySQLContext]


class DataFabricQueryInput(BaseModel):
    """Input schema for natural language queries against Data Fabric entities."""

    user_query: str = Field(
        ...,
        description=(
            "Natural language question about the data in Data Fabric entities. "
            "The tool will translate this to SQL, execute, and return an answer."
        ),
    )


class DataFabricExecuteSqlInput(BaseModel):
    """Input schema for SQL queries against Data Fabric entities."""

    sql_query: str = Field(
        ...,
        description=(
            "Complete SQL SELECT statement. "
            "Use exact table and column names from the entity schemas."
        ),
    )
