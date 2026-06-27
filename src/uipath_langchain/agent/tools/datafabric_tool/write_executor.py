"""Write executor for Data Fabric entity CRUD operations.

Wraps EntitiesService single-record methods (insert, update, delete)
and translates results/exceptions into WriteResult objects.
"""

from __future__ import annotations

import logging

from uipath.platform.entities import EntitiesService

from .models import DataFabricWriteInput, EntityWriteOperation, WriteResult

logger = logging.getLogger(__name__)


class WriteExecutor:
    """Executes validated write intents against the Data Fabric API.

    Uses single-record methods which fire Data Fabric triggers.
    Batch methods are reserved for a future phase.

    Args:
        entities_service: The resolved EntitiesService instance.
    """

    def __init__(self, entities_service: EntitiesService) -> None:
        self._entities_service = entities_service

    async def execute(self, intent: DataFabricWriteInput) -> WriteResult:
        """Execute a write operation and return the result.

        Args:
            intent: A validated DataFabricWriteInput.

        Returns:
            WriteResult with success/failure info and the affected record.
        """
        op = intent.operation
        try:
            if op == EntityWriteOperation.insert:
                record = await self._entities_service.insert_record_async(
                    intent.entity_key, intent.fields
                )
                return WriteResult(
                    success=True,
                    operation=op.value,
                    entity_key=intent.entity_key,
                    record_id=record.id,
                    record=record.model_dump(by_alias=True),
                )

            elif op == EntityWriteOperation.update:
                assert intent.record_id is not None
                record = await self._entities_service.update_record_async(
                    intent.entity_key, intent.record_id, intent.fields
                )
                return WriteResult(
                    success=True,
                    operation=op.value,
                    entity_key=intent.entity_key,
                    record_id=record.id,
                    record=record.model_dump(by_alias=True),
                )

            elif op == EntityWriteOperation.delete:
                assert intent.record_id is not None
                await self._entities_service.delete_record_async(
                    intent.entity_key, intent.record_id
                )
                return WriteResult(
                    success=True,
                    operation=op.value,
                    entity_key=intent.entity_key,
                    record_id=intent.record_id,
                )

            else:
                return WriteResult(
                    success=False,
                    operation=str(op),
                    entity_key=intent.entity_key,
                    error=f"Unsupported operation: {op}",
                )

        except Exception as exc:
            logger.warning(
                "Data Fabric write failed: entity=%s op=%s error=%s",
                intent.entity_key,
                op.value,
                exc,
                exc_info=True,
            )
            return WriteResult(
                success=False,
                operation=op.value,
                entity_key=intent.entity_key,
                record_id=intent.record_id,
                error=str(exc),
            )
