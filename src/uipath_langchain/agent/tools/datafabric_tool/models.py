"""Pydantic models for Data Fabric entity schemas."""

import json
import math
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from uipath.platform.entities import EntityOperation

EXECUTE_SQL = "execute_sql"
EXECUTE_OPERATION = "execute_operation"
READ_KIND = "Read"
MUTATION_KIND = "Mutation"

NUMERIC_TYPES = frozenset({"int", "decimal", "float", "double", "bigint"})
TEXT_TYPES = frozenset({"varchar", "nvarchar", "text", "string", "ntext"})
# Operation parameter types whose arguments are sent as JSON numbers or booleans.
NUMBER_PARAMETER_TYPES = frozenset({"int", "bigint", "decimal", "float", "real"})
BOOLEAN_PARAMETER_TYPE = "bit"


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
        if modifiers:
            return f"{self.type}, {', '.join(modifiers)}"
        return self.type

    @property
    def is_relationship(self) -> bool:
        """True when this field references another entity that can be joined."""
        return self.is_foreign_key and self.ref_entity_table is not None

    @property
    def is_numeric(self) -> bool:
        return self.type.lower() in NUMERIC_TYPES

    @property
    def is_text(self) -> bool:
        return self.type.lower() in TEXT_TYPES


class OperationParameterSchema(BaseModel):
    """A parameter an entity operation declares."""

    name: str
    sql_type: str | None = None
    is_required: bool = False
    is_list: bool = False

    @property
    def display(self) -> str:
        """Name with type and modifiers for markdown display."""
        modifiers = [self.sql_type or "unknown"]
        if self.is_required:
            modifiers.append("required")
        if self.is_list:
            modifiers.append("list")
        return f"{self.name} ({', '.join(modifiers)})"


class OperationSchema(BaseModel):
    """An operation an entity declares, as the prompt describes it."""

    name: str
    kind: str
    description: str | None = None
    parameters: list[OperationParameterSchema] = []


class EntitySchema(BaseModel):
    """Structured representation of a Data Fabric entity."""

    id: str | None = None
    entity_name: str
    display_name: str
    description: str | None = None
    record_count: int | None = None
    fields: list[FieldSchema]
    operations: list[OperationSchema] = []


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


class DataFabricQueryV3Input(DataFabricQueryInput):
    """Input schema when the entities may declare operations."""

    allow_changes: bool = Field(
        default=False,
        description=(
            "Set to true only when the user asked to change data. Operations "
            "that change data are refused while it is false."
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


class OperationArgument(BaseModel):
    """One argument of an operation call, as a name and a text value."""

    name: str = Field(..., description="Exact parameter name.")
    value: str = Field(
        ...,
        description=(
            "The value as text, e.g. 42, true or INV-7. For a list parameter, a "
            'JSON array such as ["INV-1", "INV-2"].'
        ),
    )

    @field_validator("value", mode="before")
    @classmethod
    def _as_text(cls, value: Any) -> Any:
        # Models sometimes send a number, boolean or list despite the schema.
        if isinstance(value, (bool, int, float, list)):
            return json.dumps(value)
        return value


class DataFabricExecuteOperationInput(BaseModel):
    """Input schema for running an operation a Data Fabric entity declares."""

    entity_name: str = Field(
        ...,
        description="SQL table name of the entity that declares the operation.",
    )
    operation_name: str = Field(
        ...,
        description="Exact operation name, as listed under the entity's Operations.",
    )
    # A list of pairs rather than an open map: Gemini rejects an object schema
    # with no properties.
    arguments: list[OperationArgument] = Field(
        default_factory=list,
        description="One name and value pair per parameter the operation lists.",
    )


def entity_operations(entity: Any) -> list["EntityOperation"]:
    """Return the operations an entity declares, or none."""
    # Entities from v1 metadata, older SDKs and test doubles carry no list.
    operations = getattr(entity, "operations", None)
    return list(operations) if isinstance(operations, list) else []


def operation_kind(operation: "EntityOperation") -> str:
    """Return Read or Mutation; anything not declared Read is gated as a Mutation."""
    return READ_KIND if (operation.kind or "").lower() == "read" else MUTATION_KIND


def operation_arguments(
    arguments: list[OperationArgument], operation: "EntityOperation"
) -> dict[str, Any]:
    """Fold argument pairs into a dict, typed by the parameters the operation declares.

    Later duplicate names win. A value whose parameter is unknown, or that does
    not parse as its declared type, is kept as text.
    """
    parameters = {p.name: p for p in operation.parameters}
    folded: dict[str, Any] = {}
    for argument in arguments:
        parameter = parameters.get(argument.name)
        if parameter is None:
            folded[argument.name] = argument.value
        elif parameter.is_list:
            folded[argument.name] = [
                _typed_value(item, parameter.sql_type)
                for item in _list_items(argument.value)
            ]
        else:
            folded[argument.name] = _typed_value(argument.value, parameter.sql_type)
    return folded


def _list_items(text: str) -> list[str]:
    """Split a list argument: a JSON array, or else comma-separated text."""
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = None
    if isinstance(parsed, list):
        return [item if isinstance(item, str) else json.dumps(item) for item in parsed]
    return [item.strip() for item in text.split(",") if item.strip()]


def _typed_value(text: str, sql_type: str | None) -> Any:
    """Convert text to a number or boolean when its SQL type calls for one."""
    kind = (sql_type or "").lower()
    if kind in NUMBER_PARAMETER_TYPES:
        try:
            return int(text)
        except ValueError:
            pass
        try:
            number = float(text)
        except ValueError:
            return text
        return number if math.isfinite(number) else text
    if kind == BOOLEAN_PARAMETER_TYPE:
        flag = text.strip().lower()
        if flag in ("true", "1"):
            return True
        if flag in ("false", "0"):
            return False
    return text
