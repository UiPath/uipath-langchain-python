"""Schema context building and formatting for Data Fabric entities.

Converts raw Entity SDK objects into structured Pydantic models (SQLContext),
then formats them as text for system prompt injection.

The SQL strategy section (``sql_expert_system_prompt``) is rendered from a
versioned prompt template via the ``prompts`` package. ``SQL_CONSTRAINTS`` is
appended verbatim — the system prompt should describe strategy only, not
backend deny-lists.
"""

import logging
from collections import defaultdict

from uipath.platform.entities import EntitiesService, Entity

from .datafabric_prompts import SQL_CONSTRAINTS
from .models import (
    ChoiceSetValueSchema,
    EntitySchema,
    EntitySQLContext,
    FieldSchema,
    QueryPattern,
    SQLContext,
)
from .prompts import build_prompt_context, get_prompt_version

logger = logging.getLogger(__name__)


def _resolve_choiceset_values(
    entities_service: EntitiesService | None,
    choiceset_id: str,
) -> list[ChoiceSetValueSchema]:
    """Fetch choice-set values and return as schema objects.

    Falls back to an empty list on any error so prompt building never fails.
    """
    if entities_service is None:
        return []
    try:
        values = entities_service.get_choiceset_values(choiceset_id)
        return [
            ChoiceSetValueSchema(label=v.display_name, number_id=v.number_id)
            for v in values
        ]
    except Exception:
        logger.warning(
            "Failed to fetch choice-set values for %s", choiceset_id, exc_info=True
        )
        return []


def _fetch_entity_choiceset_ids(
    entities_service: EntitiesService | None,
    entity_id: str | None,
) -> dict[str, str]:
    """Fetch the full entity schema to get ChoiceSetId per field.

    The ``resolve_entity_set`` API does not populate ``choiceset_id`` on
    fields, but the entity GET endpoint returns ``choiceSetId`` (camelCase).
    This helper bridges the gap by making a raw HTTP GET to the entity
    endpoint and parsing the response.

    Returns a dict ``{field_name: choiceset_id}``.
    """
    if entities_service is None or not entity_id:
        return {}
    try:
        from uipath.platform.common._models import Endpoint

        response = entities_service._data.request(
            "GET", Endpoint(f"datafabric_/api/Entity/{entity_id}")
        )
        data = response.json()
        result: dict[str, str] = {}
        for field in data.get("fields", []):
            cs_id = field.get("choiceSetId")
            name = field.get("name")
            if cs_id and name:
                result[name] = cs_id
        return result
    except Exception:
        logger.debug(
            "Could not fetch entity metadata for CS IDs: %s", entity_id, exc_info=True
        )
        return {}


def build_entity_context(
    entity: Entity,
    entities_service: EntitiesService | None = None,
) -> EntitySQLContext:
    """Convert an Entity SDK object to schema + derived query patterns.

    Auto-added system/audit fields (Id, CreateTime, UpdateTime, CreatedBy,
    UpdatedBy) are surfaced in the schema, tagged ``system`` via
    :attr:`FieldSchema.is_system_field`, but are always excluded from the
    derived query patterns so the examples reference only business fields.

    When ``entities_service`` is provided, choice-set fields are enriched
    with their label↔NumberId mappings.
    """
    field_schemas: list[FieldSchema] = []
    # Query patterns are derived from business fields only — system fields,
    # even when surfaced in the schema, must never drive an example query.
    business_field_names: list[str] = []
    numeric_field: str | None = None
    text_field: str | None = None
    # Cache choice-set values by choiceset_id to avoid duplicate fetches
    # when multiple fields share the same choice set.
    _cs_cache: dict[str, list[ChoiceSetValueSchema]] = {}
    # Fetch entity-level choice-set IDs from the full entity schema.
    # resolve_entity_set doesn't populate choiceset_id on fields due to
    # a casing mismatch (API returns choiceSetId, model expects choicesetId).
    _entity_cs_ids = _fetch_entity_choiceset_ids(entities_service, entity.id)

    for field in entity.fields or []:
        if field.is_hidden_field:
            continue
        is_system = field.is_system_field
        type_name = field.sql_type.name if field.sql_type else "unknown"
        # A relationship is either a declared foreign key or a Relationship-typed
        # field; use the same condition to tag it and to extract its target, so
        # the two never disagree.
        is_relationship = (
            field.is_foreign_key
            or getattr(field, "field_display_type", None) == "Relationship"
        )
        ref_entity_table: str | None = None
        ref_field_name: str | None = None
        if is_relationship:
            ref_entity = getattr(field, "reference_entity", None)
            ref_entity_table = getattr(ref_entity, "name", None)
            ref_field = getattr(field, "reference_field", None)
            ref_definition = getattr(ref_field, "definition", None)
            ref_field_name = getattr(ref_definition, "name", None)

        # Detect choice-set fields and resolve their choiceset_id.
        # Priority order: (1) SDK field.choiceset_id, (2) entity-level fetch,
        # (3) reference_choiceset, (4) field_display_type detection.
        cs_id = getattr(field, "choiceset_id", None)
        if not cs_id:
            cs_id = _entity_cs_ids.get(field.name)
        if not cs_id:
            ref_cs = getattr(field, "reference_choiceset", None)
            if ref_cs:
                cs_id = getattr(ref_cs, "id", None)

        cs_values: list[ChoiceSetValueSchema] = []
        if cs_id:
            if cs_id not in _cs_cache:
                _cs_cache[cs_id] = _resolve_choiceset_values(
                    entities_service, cs_id
                )
            cs_values = _cs_cache[cs_id]

        fs = FieldSchema(
            name=field.name,
            display_name=field.display_name,
            type=type_name,
            description=field.description,
            is_foreign_key=is_relationship,
            is_required=field.is_required,
            is_unique=field.is_unique,
            nullable=not field.is_required,
            is_system_field=is_system,
            ref_entity_table=ref_entity_table,
            ref_field_name=ref_field_name,
            choiceset_id=cs_id,
            choiceset_values=cs_values,
        )
        field_schemas.append(fs)

        if is_system:
            continue
        business_field_names.append(fs.name)
        if not numeric_field and fs.is_numeric:
            numeric_field = fs.name
        if not text_field and fs.is_text:
            text_field = fs.name

    field_names = business_field_names
    table = entity.name

    # Pick a choice-set field for the filter example if one exists.
    cs_field = next((f for f in field_schemas if f.is_choice_set and f.choiceset_values), None)

    group_field = text_field or (field_names[0] if field_names else "Category")
    agg_field = numeric_field or (field_names[1] if len(field_names) > 1 else "Amount")
    filter_field = text_field or (field_names[0] if field_names else "Name")
    fields_sample = ", ".join(field_names[:5]) if field_names else "*"
    count_col = field_names[0] if field_names else "id"

    query_patterns = [
        QueryPattern(
            intent="Show all",
            sql=f"SELECT {fields_sample} FROM {table} LIMIT 100",
        ),
        QueryPattern(
            intent="Find by X",
            sql=f"SELECT {fields_sample} FROM {table} WHERE {filter_field} = 'value' LIMIT 100",
        ),
        QueryPattern(
            intent="Top N by Y",
            sql=f"SELECT {fields_sample} FROM {table} ORDER BY {agg_field} DESC LIMIT N",
        ),
        QueryPattern(
            intent="Count by X",
            sql=f"SELECT {group_field}, COUNT({count_col}) as count FROM {table} GROUP BY {group_field}",
        ),
        QueryPattern(
            intent="Top N segments",
            sql=f"SELECT {group_field}, COUNT({count_col}) as count FROM {table} GROUP BY {group_field} ORDER BY count DESC LIMIT N",
        ),
        QueryPattern(
            intent="Sum/Avg of Y",
            sql=f"SELECT SUM({agg_field}) as total FROM {table}",
        ),
    ]

    # Add a choice-set filter example if applicable.
    if cs_field and cs_field.choiceset_values:
        first_val = cs_field.choiceset_values[0]
        query_patterns.append(
            QueryPattern(
                intent=f"Filter by {cs_field.name}",
                sql=(
                    f"SELECT {fields_sample} FROM {table} "
                    f"WHERE {cs_field.name} = {first_val.number_id} LIMIT 100"
                    f"  -- {first_val.number_id} = {first_val.label}"
                ),
            )
        )

    schema = EntitySchema(
        id=entity.id,
        entity_name=entity.name,
        display_name=entity.display_name or entity.name,
        description=entity.description,
        record_count=entity.record_count,
        fields=field_schemas,
    )
    return EntitySQLContext(entity_schema=schema, query_patterns=query_patterns)


def _build_shared_choicesets(
    entity_contexts: list[EntitySQLContext],
) -> dict[str, list[str]]:
    """Detect choice-set fields shared across entities.

    Returns a dict from choiceset_id → list of ``"Entity.Field"`` refs.
    Only includes entries where 2+ fields across different entities share
    the same choice set (same-entity duplicates are excluded).
    """
    cs_to_fields: dict[str, list[str]] = defaultdict(list)
    for ectx in entity_contexts:
        entity_name = ectx.entity_schema.entity_name
        for field in ectx.entity_schema.fields:
            if field.choiceset_id:
                cs_to_fields[field.choiceset_id].append(f"{entity_name}.{field.name}")
    # Keep only those shared across entities.
    shared: dict[str, list[str]] = {}
    for cs_id, refs in cs_to_fields.items():
        entity_set = {r.split(".")[0] for r in refs}
        if len(entity_set) >= 2:
            shared[cs_id] = refs
    return shared


def _build_choiceset_label_maps(
    entity_contexts: list[EntitySQLContext],
) -> dict[str, dict[int, str]]:
    """Build NumberId→label maps for all choice-set fields across all entities.

    Returns ``{ "Entity.Field": { 0: "Critical", 1: "High", ... } }``.
    """
    maps: dict[str, dict[int, str]] = {}
    for ectx in entity_contexts:
        entity_name = ectx.entity_schema.entity_name
        for field in ectx.entity_schema.fields:
            if field.choiceset_values:
                maps[f"{entity_name}.{field.name}"] = {
                    v.number_id: v.label for v in field.choiceset_values
                }
    return maps


def build_sql_context(
    entities: list[Entity],
    resource_description: str = "",
    base_system_prompt: str = "",
    prompt_version: str | None = None,
    entities_service: EntitiesService | None = None,
) -> SQLContext:
    """Build the full SQL context from entities, prompts, and constraints.

    Args:
        entities: Resolved Data Fabric entities.
        resource_description: Optional free-text description folded into the
            rendered prompt as ``## Domain Guidance``.
        base_system_prompt: Optional outer-agent system prompt prepended as
            ``## Agent Instructions``.
        prompt_version: Optional version key (e.g. ``"v0"``, ``"v1"``).
            Defaults to the registry's default.
        entities_service: Optional platform service for fetching choice-set
            values. When provided, choice-set fields are enriched with their
            label↔NumberId mappings.
    """
    version = get_prompt_version(prompt_version)
    ctx = build_prompt_context(
        entities=entities,
        resource_description=resource_description,
    )
    rendered_prompt = version.render(ctx)

    entity_contexts = [
        build_entity_context(e, entities_service=entities_service) for e in entities
    ]

    return SQLContext(
        base_system_prompt=base_system_prompt or None,
        resource_description=None,
        sql_expert_system_prompt=rendered_prompt,
        constraints=SQL_CONSTRAINTS,
        entity_contexts=entity_contexts,
        shared_choicesets=_build_shared_choicesets(entity_contexts),
        choiceset_label_maps=_build_choiceset_label_maps(entity_contexts),
    )


def _format_section(heading: str, body: str | None) -> list[str]:
    """Render a heading followed by its body, or nothing when body is empty."""
    if not body:
        return []
    return [heading, "", body, ""]


def _format_relationships(entity: EntitySchema, entity_tables: set[str]) -> list[str]:
    """Render the Relationships subsection for one entity.

    Relationship fields store the related record's Id; the join is spelled out
    so the model doesn't compare the FK column to a human-readable value. Only
    relationships whose target entity is in this set (and thus queryable) are
    surfaced — a dangling reference would produce an unusable join.
    """
    relationships = [
        field
        for field in entity.fields
        if field.is_relationship and field.ref_entity_table in entity_tables
    ]
    if not relationships:
        return []

    lines = [
        f"**Relationships for {entity.entity_name}:**",
        f"_Join on the related entity's Id. Use LEFT JOIN to keep all {entity.entity_name} "
        "rows (relationship may be unset); INNER JOIN when the related record must exist or "
        "you filter on it. Project the specific related column you need — not `*`._",
        "",
    ]
    for field in relationships:
        join = (
            f"LEFT JOIN {field.ref_entity_table} "
            f"ON {field.ref_entity_table}.{field.ref_join_key} = {entity.entity_name}.{field.name}"
        )
        repr_hint = (
            f", representative field `{field.ref_entity_table}.{field.ref_field_name}`"
            if field.ref_field_name
            else ""
        )
        lines.append(
            f"- `{entity.entity_name}.{field.name}` → `{field.ref_entity_table}` "
            f"(`{join}`{repr_hint})"
        )
    lines.append("")
    return lines


def _format_choiceset_values(entity: EntitySchema) -> list[str]:
    """Render the choice-set value mappings for an entity's CS fields."""
    cs_fields = [f for f in entity.fields if f.is_choice_set and f.choiceset_values]
    if not cs_fields:
        return []

    lines = [
        f"**Choice-set value mappings for {entity.entity_name}:**",
        "_These fields store integer NumberIds, not labels. "
        "Use the integer value in WHERE/JOIN clauses._",
        "",
    ]
    for field in cs_fields:
        mapping = field.choiceset_mapping_str
        lines.append(f"- `{field.name}`: {mapping}")
    lines.append("")
    return lines


def _format_shared_choicesets(shared: dict[str, list[str]]) -> list[str]:
    """Render cross-entity shared choice-set join hints."""
    if not shared:
        return []

    lines = [
        "## Shared Choice-Set Join Paths",
        "_The following fields across different entities use the same choice set "
        "and share the same integer value space. They can be directly compared "
        "in JOIN ON or WHERE clauses._",
        "",
    ]
    for cs_id, refs in shared.items():
        lines.append(f"- {' = '.join(f'`{r}`' for r in refs)} (same choice set)")
    lines.append("")
    return lines


def _format_entity(entity_ctx: EntitySQLContext, entity_tables: set[str]) -> list[str]:
    """Render one entity's schema table, relationships, and query patterns."""
    entity = entity_ctx.entity_schema
    lines = [f"### Entity: {entity.display_name} (SQL table: `{entity.entity_name}`)"]
    if entity.description:
        lines.append(f"_{entity.description}_")
    lines.append("")
    lines.append("| Field | Type | Description |")
    lines.append("|-------|------|-------------|")
    for field in entity.fields:
        desc = (field.description or "").replace("|", r"\|").replace("\n", " ")
        # Append choice-set mapping inline in the description column.
        if field.is_choice_set and field.choiceset_mapping_str:
            if desc:
                desc += f" — values: {field.choiceset_mapping_str}"
            else:
                desc = f"values: {field.choiceset_mapping_str}"
        lines.append(f"| {field.name} | {field.display_type} | {desc} |")
    lines.append("")

    lines.extend(_format_choiceset_values(entity))
    lines.extend(_format_relationships(entity, entity_tables))

    lines.append(f"**Query Patterns for {entity.entity_name}:**")
    lines.append("")
    lines.append("| User Intent | SQL Pattern |")
    lines.append("|-------------|-------------|")
    for p in entity_ctx.query_patterns:
        lines.append(f"| '{p.intent}' | `{p.sql}` |")
    lines.append("")
    return lines


def format_sql_context(ctx: SQLContext) -> str:
    """Format a SQLContext as text for system prompt injection."""
    lines: list[str] = []
    lines += _format_section("## Agent Instructions", ctx.base_system_prompt)
    lines += _format_section(
        "## SQL Query Generation Guidelines", ctx.sql_expert_system_prompt
    )
    lines += _format_section("## SQL Constraints", ctx.constraints)
    lines += _format_section("## Entity set description", ctx.resource_description)

    lines.append("## All available Data Fabric Entities")
    lines.append("")

    entity_tables = {ec.entity_schema.entity_name for ec in ctx.entity_contexts}
    for entity_ctx in ctx.entity_contexts:
        lines.extend(_format_entity(entity_ctx, entity_tables))

    # Shared choice-set join hints (cross-entity).
    lines.extend(_format_shared_choicesets(ctx.shared_choicesets))

    return "\n".join(lines)


def build(
    entities: list[Entity],
    resource_description: str = "",
    base_system_prompt: str = "",
    prompt_version: str | None = None,
    entities_service: EntitiesService | None = None,
) -> str:
    """Build the full SQL prompt text for the inner sub-graph LLM.

    Combines agent system prompt, the rendered SQL strategy prompt, the
    Calcite constraint deny-list, and entity schemas + query patterns.

    Args:
        entities: List of Entity objects with schema information.
        resource_description: Optional description of the resource/entity set;
            folded into the rendered prompt as domain guidance.
        base_system_prompt: Optional system prompt from the outer agent.
        prompt_version: Optional version key (e.g. ``"v0"``, ``"v1"``).
            Defaults to the registry's default.
        entities_service: Optional platform service for fetching choice-set
            values. When provided, choice-set fields are enriched.

    Returns:
        Formatted prompt string for the inner LLM system message.
    """
    if not entities:
        return ""

    ctx = build_sql_context(
        entities,
        resource_description,
        base_system_prompt,
        prompt_version=prompt_version,
        entities_service=entities_service,
    )
    return format_sql_context(ctx)
