"""SQLite implementation of UiPathResumableStorageProtocol."""

import json
from typing import cast

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel
from uipath.runtime import (
    UiPathApiTrigger,
    UiPathResumeTrigger,
    UiPathResumeTriggerName,
    UiPathResumeTriggerType,
)


class SqliteResumableStorage:
    """SQLite storage for resume triggers."""

    def __init__(
        self, memory: AsyncSqliteSaver, table_name: str = "__uipath_resume_triggers"
    ):
        self.memory = memory
        self.table_name = table_name
        self._initialized = False

    async def _ensure_table(self) -> None:
        """Create table if needed."""
        if self._initialized:
            return

        await self.memory.setup()
        async with self.memory.lock, self.memory.conn.cursor() as cur:
            await cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    key TEXT,
                    folder_key TEXT,
                    folder_path TEXT,
                    payload TEXT,
                    timestamp DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'utc'))
                )
            """)
            await self.memory.conn.commit()
            self._initialized = True

    async def save_trigger(self, trigger: UiPathResumeTrigger) -> None:
        """Save resume trigger to database."""
        await self._ensure_table()

        trigger_key = (
            trigger.api_resume.inbox_id if trigger.api_resume else trigger.item_key
        )
        payload = trigger.payload
        if payload:
            payload = (
                (
                    payload.model_dump()
                    if isinstance(payload, BaseModel)
                    else json.dumps(payload)
                )
                if isinstance(payload, dict)
                else str(payload)
            )

        async with self.memory.lock, self.memory.conn.cursor() as cur:
            await cur.execute(
                f"INSERT INTO {self.table_name} (type, key, name, payload, folder_path, folder_key) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    trigger.trigger_type.value,
                    trigger_key,
                    trigger.trigger_name.value,
                    payload,
                    trigger.folder_path,
                    trigger.folder_key,
                ),
            )
            await self.memory.conn.commit()

    async def get_latest_trigger(self) -> UiPathResumeTrigger | None:
        """Get most recent trigger from database."""
        await self._ensure_table()

        async with self.memory.lock, self.memory.conn.cursor() as cur:
            await cur.execute(f"""
                SELECT type, key, name, folder_path, folder_key, payload
                FROM {self.table_name}
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            result = await cur.fetchone()

            if not result:
                return None

            trigger_type, key, name, folder_path, folder_key, payload = cast(
                tuple[str, str, str, str, str, str], tuple(result)
            )

            resume_trigger = UiPathResumeTrigger(
                trigger_type=UiPathResumeTriggerType(trigger_type),
                trigger_name=UiPathResumeTriggerName(name),
                item_key=key,
                folder_path=folder_path,
                folder_key=folder_key,
                payload=payload,
            )

            if resume_trigger.trigger_type == UiPathResumeTriggerType.API:
                resume_trigger.api_resume = UiPathApiTrigger(
                    inbox_id=resume_trigger.item_key, request=resume_trigger.payload
                )

            return resume_trigger
