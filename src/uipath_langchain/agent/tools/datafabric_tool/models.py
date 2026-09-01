"""Pydantic models for Data Fabric entity schemas."""

from pydantic import BaseModel, Field

NUMERIC_TYPES = frozenset({"int", "decimal", "float", "double", "bigint"})
TEXT_TYPES = frozenset({"varchar", "nvarchar", "text", "string", "ntext"})


class ChoiceSetValueSchema(BaseModel):
    """A single choice-set value with its display label and stored NumberId."""

    label: str
    number_id: int


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
    is_system_field: bool = False
    # For relationship (foreign-key) fields: the related entity's SQL table and
    # the column to join on. The field itself stores the related record's Id, so
    # the join is always ``related.<ref_join_key> = <this table>.<name>``.
    ref_entity_table: str | None = None
    ref_join_key: str = "Id"
    ref_field_name: str | None = None
    # Choice-set metadata: populated when the field is CHOICE_SET_SINGLE or
    # CHOICE_SET_MULTIPLE so the prompt can include the label↔NumberId mapping.
    choiceset_id: str | None = None
    choiceset_values: list[ChoiceSetValueSchema] = []

    @property
    def is_choice_set(self) -> bool:
        """True when this field is backed by a choice set."""
        return bool(self.choiceset_id)

    @property
    def display_type(self) -> str:
        """Type string with modifiers for markdown display."""
        modifiers = []
        if self.is_required:
            modifiers.append("required")
        if self.is_foreign_key:
            modifiers.append("fk")
        if self.is_system_field:
            modifiers.append("system")
        if self.is_choice_set:
            modifiers.append("choice_set")
        if modifiers:
            return f"{self.type}, {', '.join(modifiers)}"
        return self.type

    @property
    def is_relationship(self) -> bool:
        """True when this field references another entity that can be joined."""
        return self.is_foreign_key and self.ref_entity_table is not None

    @property
    def is_numeric(self) -> bool:
        # Choice-set fields are stored as INT but are categorical, not numeric.
        if self.is_choice_set:
            return False
        return self.type.lower() in NUMERIC_TYPES

    @property
    def is_text(self) -> bool:
        return self.type.lower() in TEXT_TYPES

    @property
    def choiceset_mapping_str(self) -> str:
        """Formatted label=NumberId mapping for prompt injection, or empty."""
        if not self.choiceset_values:
            return ""
        return ", ".join(
            f"{v.label}={v.number_id}" for v in self.choiceset_values
        )


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
    # Cross-entity shared choice sets: choiceset_id → list of "Entity.Field" refs.
    # Used to emit join hints when two entities share the same choice set.
    shared_choicesets: dict[str, list[str]] = {}
    # Complete NumberId→label mapping for result post-processing:
    # { "Entity.Field": { 0: "Critical", 1: "High", ... } }
    choiceset_label_maps: dict[str, dict[int, str]] = {}


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
