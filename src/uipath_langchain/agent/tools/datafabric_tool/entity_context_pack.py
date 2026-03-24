"""Entity Context Pack — rich metadata for Text2SQL optimization.

Builds ECPs from Data Fabric entity metadata + sample data at INIT time.
No LLM generation — synonyms from field.description, samples from DF API,
type flags from sql_type heuristics.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from uipath.platform.entities import Entity

logger = logging.getLogger(__name__)

# --- Type classification sets ---

_NUMERIC_TYPES = frozenset({
    "int", "integer", "bigint", "smallint", "tinyint",
    "decimal", "numeric", "float", "double", "real", "money", "number",
})

_TEMPORAL_TYPES = frozenset({
    "date", "datetime", "datetime2", "timestamp", "time",
    "datetimeoffset", "smalldatetime",
})

_CATEGORICAL_TYPES = frozenset({
    "varchar", "nvarchar", "text", "ntext", "string", "char", "nchar",
})


# --- Dataclasses ---
# to_dict() methods use sparse serialization (omit falsy fields) to save tokens.


@dataclass
class ColumnContext:
    """Column metadata for ECP."""

    name: str
    type: str
    description: str | None = None
    synonyms: list[str] = field(default_factory=list)
    examples: list[Any] = field(default_factory=list)
    is_numeric: bool = False
    is_temporal: bool = False
    is_categorical: bool = False
    is_primary_key: bool = False
    is_foreign_key: bool = False
    reference_entity: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "type": self.type}
        if self.description:
            d["description"] = self.description
        if self.synonyms:
            d["synonyms"] = self.synonyms
        if self.examples:
            d["examples"] = self.examples
        if self.is_primary_key:
            d["is_primary_key"] = True
        if self.is_foreign_key:
            d["is_foreign_key"] = True
            if self.reference_entity:
                d["reference_entity"] = self.reference_entity
        if self.is_numeric:
            d["is_numeric"] = True
        if self.is_temporal:
            d["is_temporal"] = True
        if self.is_categorical:
            d["is_categorical"] = True
        return d


@dataclass
class QueryCapabilities:
    """Structured SQL capabilities for LLM parsing.

    Intentionally duplicates some sql_constraints.txt content in a
    machine-parseable format alongside the free-text rules.
    """

    allowed_clauses: list[str] = field(default_factory=lambda: [
        "SELECT", "WHERE", "GROUP BY", "HAVING", "ORDER BY",
        "LIMIT", "OFFSET", "DISTINCT", "LEFT JOIN",
    ])
    allowed_aggregations: list[str] = field(default_factory=lambda: [
        "COUNT(column_name)", "SUM", "AVG", "MIN", "MAX",
    ])
    allowed_expressions: list[str] = field(default_factory=lambda: [
        "CASE/WHEN (in SELECT and ORDER BY only)",
        "CAST", "NULLIF",
        "ROUND", "ABS", "LOWER", "UPPER", "TRIM",
        "SUBSTRING", "CHAR_LENGTH",
        "arithmetic (+, -, *, /) in SELECT only",
        "string concat (||)",
    ])
    allowed_predicates: list[str] = field(default_factory=lambda: [
        "=", "<>", ">", "<", ">=", "<=",
        "BETWEEN", "IN", "LIKE", "IS NULL", "IS NOT NULL",
        "AND", "OR",
    ])
    disallowed: list[str] = field(default_factory=lambda: [
        "SELECT *",
        "COUNT(*) — use COUNT(column_name)",
        "COALESCE / IFNULL — use CASE WHEN x IS NULL THEN default ELSE x END",
        "SUBSTR / LENGTH / CONCAT — use SUBSTRING / CHAR_LENGTH / ||",
        "subqueries in WHERE — use derived tables in FROM",
        "CTE (WITH clause)",
        "window functions (ROW_NUMBER, RANK, PARTITION BY)",
        "FULL OUTER JOIN / CROSS JOIN / self-joins",
        "UNION ALL — only UNION (deduplicating) works",
        "more than 4 tables in JOIN chain",
        "CASE WHEN in WHERE — only in SELECT and ORDER BY",
        "arithmetic in WHERE — only in SELECT",
        "date functions — use LIKE on date strings",
        "INSERT / UPDATE / DELETE / DDL — read-only",
    ])
    critical_rules: list[str] = field(default_factory=lambda: [
        "ALWAYS use explicit column names — never SELECT *",
        "Use COUNT(column_name) — never COUNT(*) or COUNT(1)",
        "LIMIT is REQUIRED on every query without a WHERE clause",
        "All non-aggregated columns in SELECT must appear in GROUP BY",
        "Maximum 4 tables in a JOIN chain",
        "Use SUBSTRING not SUBSTR, CHAR_LENGTH not LENGTH, || not CONCAT",
        "Use CASE WHEN for null handling — COALESCE does not work",
        "Verify string values against sample data for exact casing and format",
    ])

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_clauses": self.allowed_clauses,
            "allowed_aggregations": self.allowed_aggregations,
            "allowed_expressions": self.allowed_expressions,
            "allowed_predicates": self.allowed_predicates,
            "disallowed": self.disallowed,
            "critical_rules": self.critical_rules,
        }


@dataclass
class EntityContextPack:
    """Complete context for a single entity."""

    entity_name: str
    display_name: str
    description: str | None = None
    columns: list[ColumnContext] = field(default_factory=list)
    row_count: int | None = None
    query_capabilities: QueryCapabilities = field(default_factory=QueryCapabilities)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "entity_name": self.entity_name,
            "display_name": self.display_name,
        }
        if self.description:
            d["description"] = self.description
        if self.row_count is not None:
            d["row_count"] = self.row_count
        d["columns"] = [c.to_dict() for c in self.columns]
        d["query_capabilities"] = self.query_capabilities.to_dict()
        return d


# --- Helpers ---


def classify_field_type(sql_type_name: str) -> tuple[bool, bool, bool]:
    """Classify a SQL type into (is_numeric, is_temporal, is_categorical)."""
    t = sql_type_name.lower().strip()
    return (t in _NUMERIC_TYPES, t in _TEMPORAL_TYPES, t in _CATEGORICAL_TYPES)


def extract_synonyms(field_name: str, description: str | None) -> list[str]:
    """Extract synonyms from a field description.

    Splits on commas, semicolons, 'or', 'aka', parentheticals.
    Excludes the field name itself and deduplicates.
    """
    if not description:
        return []

    name_lower = field_name.lower()
    synonyms: set[str] = set()

    parens = re.findall(r"\(([^)]+)\)", description)
    for p in parens:
        p_stripped = p.strip()
        if p_stripped and p_stripped.lower() != name_lower:
            synonyms.add(p_stripped)

    parts = re.split(
        r"[,;]|\bor\b|\baka\b|\balso known as\b", description, flags=re.IGNORECASE
    )
    for part in parts:
        token = part.strip().strip(".")
        if (
            token
            and len(token.split()) <= 4
            and token.lower() != name_lower
            and not token.lower().startswith("the ")
        ):
            synonyms.add(token)

    return sorted(synonyms)


async def _fetch_sample_rows(
    entity_key: str, limit: int = 5
) -> list[dict[str, Any]]:
    """Fetch sample rows from Data Fabric using list_records API."""
    from uipath.platform import UiPath

    sdk = UiPath()
    try:
        records = await sdk.entities.list_records_async(entity_key, limit=limit)
        return [record.model_dump(exclude={"id"}) for record in records]
    except Exception:
        logger.warning("Failed to fetch sample rows for '%s'", entity_key, exc_info=True)
        return []


def _extract_column_examples(
    field_name: str, sample_rows: list[dict[str, Any]], max_examples: int = 3
) -> list[Any]:
    """Extract unique example values for a column from sample rows."""
    seen: set[str] = set()
    examples: list[Any] = []
    for row in sample_rows:
        val = row.get(field_name)
        if val is None:
            continue
        val_str = str(val)
        if val_str not in seen:
            seen.add(val_str)
            examples.append(val)
            if len(examples) >= max_examples:
                break
    return examples


# --- Builders ---


async def build_entity_context_pack(entity: Entity) -> EntityContextPack:
    """Build a full ECP from an Entity, including sample data from DF API."""
    sample_rows = await _fetch_sample_rows(entity.id)

    columns: list[ColumnContext] = []
    for f in entity.fields or []:
        if f.is_hidden_field or f.is_system_field:
            continue

        sql_type_name = f.sql_type.name if f.sql_type else "unknown"
        is_num, is_temp, is_cat = classify_field_type(sql_type_name)

        ref_entity = None
        if f.is_foreign_key and f.reference_entity:
            ref_entity = f.reference_entity.display_name or f.reference_entity.name

        columns.append(ColumnContext(
            name=f.name,
            type=sql_type_name,
            description=f.description,
            synonyms=extract_synonyms(f.name, f.description),
            examples=_extract_column_examples(f.name, sample_rows),
            is_numeric=is_num,
            is_temporal=is_temp,
            is_categorical=is_cat,
            is_primary_key=f.is_primary_key,
            is_foreign_key=f.is_foreign_key,
            reference_entity=ref_entity,
        ))

    return EntityContextPack(
        entity_name=entity.name,
        display_name=entity.display_name or entity.name,
        description=entity.description,
        columns=columns,
        row_count=entity.record_count,
    )


async def build_entity_context_packs(
    entities: list[Entity],
) -> list[EntityContextPack]:
    """Build ECPs for all entities concurrently."""
    tasks = [build_entity_context_pack(e) for e in entities]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    packs: list[EntityContextPack] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning("Failed to build ECP for '%s': %s", entities[i].name, result)
        else:
            packs.append(result)
    return packs


# --- Formatting ---


def format_ecp_for_context(context_packs: list[EntityContextPack]) -> str:
    """Format ECPs as JSON for injection into agent system prompt.

    Produces: SQL generation guidelines + SQL constraints + ECP JSON block.
    """
    if not context_packs:
        return ""

    from .datafabric_tool import _load_sql_constraints, _load_system_prompt

    lines: list[str] = []

    system_prompt = _load_system_prompt()
    if system_prompt:
        lines.extend(["## SQL Query Generation Guidelines", "", system_prompt, ""])

    sql_constraints = _load_sql_constraints()
    if sql_constraints:
        lines.extend(["## SQL Constraints", "", sql_constraints, ""])

    ecp_json = json.dumps(
        [pack.to_dict() for pack in context_packs],
        indent=2,
        default=str,
    )
    lines.extend(["## Entity Context Packs", "", "```json", ecp_json, "```"])

    return "\n".join(lines)
