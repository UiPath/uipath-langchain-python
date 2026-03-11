"""Fetch sample data from Data Fabric entities for rewrite context."""

import logging
from typing import Any

from uipath.platform.entities import Entity

logger = logging.getLogger(__name__)

SAMPLE_ROWS_PER_ENTITY = 5
MAX_COLUMNS_PER_ENTITY = 10


async def fetch_sample_data(
    entities: list[Entity],
) -> dict[str, list[dict[str, Any]]]:
    """Fetch sample rows from each entity via SQL SELECT ... LIMIT N.

    Called once at graph-build time. Results are cached in the rewrite node closure.

    Args:
        entities: List of Entity objects with schema information.

    Returns:
        Dict mapping entity display_name to list of row dicts.
        Entities that fail to query return empty lists.
    """
    from uipath.platform import UiPath

    sdk = UiPath()
    sample_data: dict[str, list[dict[str, Any]]] = {}

    for entity in entities:
        display_name = entity.display_name or entity.name
        columns = [
            f.name
            for f in (entity.fields or [])
            if not f.is_hidden_field and not f.is_system_field
        ]
        if not columns:
            logger.warning(
                "Entity '%s' has no visible fields, skipping sample fetch",
                display_name,
            )
            sample_data[display_name] = []
            continue

        query_columns = columns[:MAX_COLUMNS_PER_ENTITY]
        col_list = ", ".join(query_columns)
        sql = f"SELECT {col_list} FROM {entity.name} LIMIT {SAMPLE_ROWS_PER_ENTITY}"

        try:
            records = await sdk.entities.query_entity_records_async(sql_query=sql)
            sample_data[display_name] = records[:SAMPLE_ROWS_PER_ENTITY]
            logger.info(
                "Fetched %d sample rows for entity '%s'",
                len(sample_data[display_name]),
                display_name,
            )
        except Exception as e:
            logger.warning(
                "Failed to fetch sample data for '%s': %s", display_name, e
            )
            sample_data[display_name] = []

    return sample_data
