"""Parse an Ontology-service YARRRML mapping into the Data Fabric entities it binds."""

from typing import Any, NamedTuple


class YarrrmlParseError(ValueError):
    """Raised when the YARRRML mapping cannot be read or binds no entity."""


class MappedEntity(NamedTuple):
    name: str
    entity_id: str | None
    folder_key: str | None


def _first(source: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = source.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def parse_yarrrml_entities(
    yarrrml_text: str, default_folder_key: str | None
) -> list[MappedEntity]:
    """Extract the entities a YARRRML mapping binds.

    Args:
        yarrrml_text: The YARRRML mapping document.
        default_folder_key: The ontology's own folder, used for a source that
            names no ``folderId``.

    Returns:
        De-duplicated entities, in mapping order.

    Raises:
        YarrrmlParseError: The document is not YAML, has no ``mappings``, or
            binds no Data Fabric entity.
    """
    import yaml  # type: ignore[import-untyped]

    try:
        document = yaml.safe_load(yarrrml_text)
    except yaml.YAMLError as e:
        raise YarrrmlParseError(f"YARRRML mapping is not valid YAML: {e}") from e

    mappings = document.get("mappings") if isinstance(document, dict) else None
    if not isinstance(mappings, dict) or not mappings:
        raise YarrrmlParseError("No `mappings` found in the YARRRML mapping.")

    entities: list[MappedEntity] = []
    for mapping in mappings.values():
        sources = mapping.get("sources") if isinstance(mapping, dict) else None
        if isinstance(sources, dict):
            sources = [sources]
        for source in sources if isinstance(sources, list) else []:
            if not isinstance(source, dict):
                continue
            table = _first(source, "table")
            if _first(source, "access") is None or table is None:
                continue
            entity = MappedEntity(
                name=table,
                entity_id=_first(source, "entityId", "entity-id", "entity_id"),
                folder_key=_first(
                    source,
                    "folderId",
                    "folder-id",
                    "folder_id",
                    "folderKey",
                    "folder-key",
                )
                or default_folder_key,
            )
            if entity not in entities:
                entities.append(entity)

    if not entities:
        raise YarrrmlParseError(
            "The YARRRML mapping binds no Data Fabric entity (no source with "
            "`access` and `table`)."
        )
    return entities
