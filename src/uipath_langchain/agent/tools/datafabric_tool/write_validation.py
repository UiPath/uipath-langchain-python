"""Validation logic for Data Fabric write operations.

Provides entity writability checks, schema-derived field filtering,
and mutation intent validation.  Only native Data Fabric entities are
writable; writes to context-derived writable entities get strict
field-level validation.
"""

from __future__ import annotations

from uipath.platform.entities import Entity

from .models import (
    DataFabricWriteInput,
    EntityWriteOperation,
    EntityWriteSchema,
    WritableFieldInfo,
)


def is_entity_writable(entity: Entity) -> bool:
    """Check if an entity supports writes via EntitiesService CRUD.

    Only native Data Fabric entities are writable. Federated entities
    (with external_fields), ChoiceSets, SystemEntities, and InternalEntities
    are not writable through this path.
    """
    # Only "Entity" type is writable (not ChoiceSet, SystemEntity, InternalEntity)
    if entity.entity_type != "Entity":
        return False
    # Federated entities have external_fields — writes go to source system, not DF
    if entity.external_fields:
        return False
    return True


def derive_writable_fields(entity: Entity) -> list[WritableFieldInfo]:
    """Extract writable fields from an Entity's metadata.

    Filters out system fields, hidden fields, primary keys, and attachment
    fields — these are not user-settable via the write API.

    Returns an empty list for non-writable entities (federated, ChoiceSet, etc.).

    Args:
        entity: A resolved Entity object with field metadata.

    Returns:
        List of WritableFieldInfo for fields the LLM may write to.
    """
    if not is_entity_writable(entity):
        return []

    writable: list[WritableFieldInfo] = []
    for field in entity.fields or []:
        if (
            field.is_system_field
            or field.is_hidden_field
            or field.is_primary_key
            or field.is_attachment
        ):
            continue
        type_name = field.sql_type.name if field.sql_type else "unknown"
        choiceset_id = getattr(field, "choiceset_id", None) or None
        writable.append(
            WritableFieldInfo(
                name=field.name,
                display_name=field.display_name,
                type_name=type_name,
                is_required=field.is_required,
                description=field.description,
                choiceset_id=choiceset_id if choiceset_id else None,
                is_choiceset=bool(choiceset_id),
            )
        )
    return writable


def validate_mutation_intent(
    intent: DataFabricWriteInput,
    write_schemas: dict[str, EntityWriteSchema] | None = None,
) -> list[str]:
    """Validate a write intent before executing.

    v1 only writes to context-derived writable entities.  If the target
    entity is not present in *write_schemas* the request is rejected.

    Args:
        intent: The write operation intent to validate.
        write_schemas: Mapping of entity_key -> EntityWriteSchema
            for writable context-derived entities.

    Returns:
        Empty list if valid; list of human-readable error strings otherwise.
    """
    errors: list[str] = []
    op = intent.operation

    # Entity must be in the writable set
    schemas = write_schemas or {}
    schema = schemas.get(intent.entity_key)
    if schema is None:
        writable_list = sorted(schemas.keys()) if schemas else []
        errors.append(
            f"Entity '{intent.entity_key}' is not configured for writes. "
            f"Writable entities: {writable_list}"
        )
        return errors

    # Structural: DELETE and UPDATE require record_id
    if op in (EntityWriteOperation.delete, EntityWriteOperation.update):
        if not intent.record_id:
            errors.append(
                f"'{op.value}' operation requires 'record_id'. "
                f"Query the entity first to obtain the record ID."
            )

    # Structural: INSERT and UPDATE require fields
    if op in (EntityWriteOperation.insert, EntityWriteOperation.update):
        if not intent.fields:
            errors.append(
                f"'{op.value}' operation requires 'fields' with at least one "
                f"field name-value pair."
            )

    # If structural errors exist, return early — field-level checks need fields
    if errors:
        return errors

    # Strict mode: validate against pre-resolved writable fields
    if intent.fields:
        writable_names = {f.name for f in schema.writable_fields}
        unknown = set(intent.fields.keys()) - writable_names
        if unknown:
            available = ", ".join(sorted(writable_names))
            errors.append(
                f"Unknown field(s) for entity '{intent.entity_key}': "
                f"{', '.join(sorted(unknown))}. "
                f"Available writable fields: {available}"
            )

    # INSERT: enforce required fields from metadata
    if op == EntityWriteOperation.insert and intent.fields:
        required_names = {f.name for f in schema.writable_fields if f.is_required}
        missing = required_names - set(intent.fields.keys())
        if missing:
            errors.append(
                f"INSERT on '{intent.entity_key}' requires field(s): "
                f"{', '.join(sorted(missing))}"
            )

    return errors
