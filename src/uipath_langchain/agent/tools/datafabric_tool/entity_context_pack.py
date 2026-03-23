"""Entity Context Pack - Rich metadata for prod Text2SQL optimization.

Builds ECPs from Data Fabric entity metadata + sample data at INIT time.
No LLM generation — synonyms from field.description, samples from DF API,
type flags from sql_type heuristics.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from uipath.platform.entities import Entity, FieldMetadata

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent

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
        """Serialize to JSON-compatible dict."""
        d: dict[str, Any] = {
            "name": self.name,
            "type": self.type,
        }
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
        d["is_numeric"] = self.is_numeric
        d["is_temporal"] = self.is_temporal
        d["is_categorical"] = self.is_categorical
        return d


@dataclass
class EntityContextPack:
    """Complete context for a single entity."""

    entity_name: str
    display_name: str
    description: str | None = None
    columns: list[ColumnContext] = field(default_factory=list)
    row_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        d: dict[str, Any] = {
            "entity_name": self.entity_name,
            "display_name": self.display_name,
        }
        if self.description:
            d["description"] = self.description
        if self.row_count is not None:
            d["row_count"] = self.row_count
        d["columns"] = [c.to_dict() for c in self.columns]
        return d


# --- Helpers ---


def classify_field_type(sql_type_name: str) -> tuple[bool, bool, bool]:
    """Classify a SQL type into (is_numeric, is_temporal, is_categorical)."""
    t = sql_type_name.lower().strip()
    return (
        t in _NUMERIC_TYPES,
        t in _TEMPORAL_TYPES,
        t in _CATEGORICAL_TYPES,
    )


def extract_synonyms(field_name: str, description: str | None) -> list[str]:
    """Extract synonyms from a field description.

    Splits on commas, semicolons, 'or', 'aka', parentheticals.
    Excludes the field name itself and deduplicates.
    """
    if not description:
        return []

    name_lower = field_name.lower()
    synonyms: set[str] = set()

    # Extract parenthetical content: "Total enrollment (K-12 students)"
    parens = re.findall(r"\(([^)]+)\)", description)
    for p in parens:
        p_stripped = p.strip()
        if p_stripped and p_stripped.lower() != name_lower:
            synonyms.add(p_stripped)

    # Split on delimiters
    parts = re.split(r"[,;]|\bor\b|\baka\b|\balso known as\b", description, flags=re.IGNORECASE)
    for part in parts:
        token = part.strip().strip(".")
        # Only keep short phrases (likely synonyms, not full sentences)
        if (
            token
            and len(token.split()) <= 4
            and token.lower() != name_lower
            and not token.lower().startswith("the ")
        ):
            synonyms.add(token)

    return sorted(synonyms)


async def fetch_sample_rows(
    entity_key: str, limit: int = 5
) -> list[dict[str, Any]]:
    """Fetch sample rows from Data Fabric using list_records API."""
    from uipath.platform import UiPath

    sdk = UiPath()
    try:
        records = await sdk.entities.list_records_async(entity_key, limit=limit)
        return [record.model_dump(exclude={"id"}) for record in records]
    except Exception as e:
        logger.warning(f"Failed to fetch sample rows for '{entity_key}': {e}")
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


async def build_entity_context_pack(entity: Entity) -> EntityContextPack:
    """Build a full ECP from an Entity, including sample data from DF API."""
    # Fetch sample rows concurrently with building column metadata
    sample_rows = await fetch_sample_rows(entity.id)

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
            logger.warning(
                f"Failed to build ECP for '{entities[i].name}': {result}"
            )
        else:
            packs.append(result)
    return packs


# --- Formatting ---


@lru_cache(maxsize=1)
def _load_sql_constraints() -> str:
    """Load SQL constraints from sql_constraints.txt."""
    constraints_path = _PROMPTS_DIR / "sql_constraints.txt"
    try:
        return constraints_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning(f"SQL constraints file not found: {constraints_path}")
        return ""


def format_ecp_for_context(context_packs: list[EntityContextPack]) -> str:
    """Format ECPs as JSON for injection into agent system prompt.

    Produces: SQL constraints + ECP JSON block.
    The system_prompt.txt (SQL expert guidelines) is NOT included here —
    it goes into the Studio Web system message at design time.
    """
    if not context_packs:
        return ""

    lines: list[str] = []

    sql_constraints = _load_sql_constraints()
    if sql_constraints:
        lines.append("## SQL Constraints")
        lines.append("")
        lines.append(sql_constraints)
        lines.append("")

    ecp_json = json.dumps(
        [pack.to_dict() for pack in context_packs],
        indent=2,
        default=str,
    )

    lines.append("## Entity Context Packs")
    lines.append("")
    lines.append("```json")
    lines.append(ecp_json)
    lines.append("```")

    return "\n".join(lines)
