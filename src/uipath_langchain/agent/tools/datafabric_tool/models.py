"""Pydantic models for Data Fabric entity schemas."""

from enum import Enum
from typing import Any, Optional

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


# ---------------------------------------------------------------------------
# Write models
# ---------------------------------------------------------------------------


class EntityWriteOperation(str, Enum):
    """Supported write operations on Data Fabric entities."""

    insert = "insert"
    update = "update"
    delete = "delete"


class DataFabricWriteInput(BaseModel):
    """Input schema for write operations against Data Fabric entities.

    This is the tool args schema presented to the LLM.
    """

    entity_key: str = Field(
        ...,
        description="The entity name (table name) to write to.",
    )
    operation: EntityWriteOperation = Field(
        ...,
        description="The write operation: 'insert', 'update', or 'delete'.",
    )
    record_id: Optional[str] = Field(
        default=None,
        description="The record ID. Required for update and delete operations.",
    )
    fields: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Field name-value pairs for the record. "
            "Required for insert and update operations."
        ),
    )


class WriteResult(BaseModel):
    """Result of a write operation against a Data Fabric entity."""

    success: bool
    operation: str
    entity_key: str
    record_id: Optional[str] = None
    record: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class WritableFieldInfo(BaseModel):
    """Schema information for a writable field on an entity."""

    name: str
    display_name: str
    type_name: str
    is_required: bool
    description: Optional[str] = None
    choiceset_id: Optional[str] = None
    allowed_values: Optional[list[str]] = None
    is_choiceset: bool = False


class EntityWriteSchema(BaseModel):
    """Pre-resolved write schema for a context-derived entity."""

    entity_key: str
    display_name: str
    writable_fields: list[WritableFieldInfo]
