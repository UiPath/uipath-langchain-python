"""SQLite implementation of UiPathResumableStorageProtocol."""

import json
from typing import Any, cast

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel
from uipath.runtime import UiPathResumeTrigger


class SqliteResumableStorage:
    """SQLite storage for resume triggers and arbitrary kv pairs."""

    def __init__(
        self,
        memory: AsyncSqliteSaver,
    ):
        self.memory = memory
        self.rs_table_name = "__uipath_resume_triggers"
        self.kv_table_name = "__uipath_runtime_kv"
        self._initialized = False

    async def _ensure_table(self) -> None:
        """Create tables if needed."""
        if self._initialized:
            return

        await self.memory.setup()
        async with self.memory.lock, self.memory.conn.cursor() as cur:
            # Enable WAL mode for high concurrency
            await cur.execute("PRAGMA journal_mode=WAL")

            await cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.rs_table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    runtime_id TEXT NOT NULL,
                    interrupt_id TEXT NOT NULL,
                    data TEXT NOT NULL,
                    timestamp DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'utc'))
                )
                """
            )

            await cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.rs_table_name}_runtime_id
                ON {self.rs_table_name}(runtime_id)
                """
            )

            await cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.kv_table_name} (
                    runtime_id TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT,
                    timestamp DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'utc')),
                    PRIMARY KEY (runtime_id, namespace, key)
                )
                """
            )

            await self.memory.conn.commit()

        self._initialized = True

    async def save_triggers(
        self, runtime_id: str, triggers: list[UiPathResumeTrigger]
    ) -> None:
        """Save resume triggers to database, replacing all existing triggers for this runtime_id."""
        await self._ensure_table()

        async with self.memory.lock, self.memory.conn.cursor() as cur:
            # Delete all existing triggers for this runtime_id
            await cur.execute(
                f"""
                DELETE FROM {self.rs_table_name}
                WHERE runtime_id = ?
                """,
                (runtime_id,),
            )

            # Insert new triggers
            for trigger in triggers:
                trigger_data = trigger.model_dump()
                trigger_data["payload"] = trigger.payload
                trigger_data["trigger_name"] = trigger.trigger_name

                await cur.execute(
                    f"""
                    INSERT INTO {self.rs_table_name}
                        (runtime_id, interrupt_id, data)
                    VALUES (?, ?, ?)
                    """,
                    (
                        runtime_id,
                        trigger.interrupt_id,
                        json.dumps(trigger_data),
                    ),
                )
            await self.memory.conn.commit()

    async def get_triggers(self, runtime_id: str) -> list[UiPathResumeTrigger] | None:
        """Get all triggers for runtime_id from database."""
        await self._ensure_table()

        async with self.memory.lock, self.memory.conn.cursor() as cur:
            await cur.execute(
                f"""
                SELECT data
                FROM {self.rs_table_name}
                WHERE runtime_id = ?
                ORDER BY timestamp ASC
                """,
                (runtime_id,),
            )
            results = await cur.fetchall()

        if not results:
            return None

        triggers = []
        for result in results:
            data_text = cast(str, result[0])
            trigger = UiPathResumeTrigger.model_validate_json(data_text)
            triggers.append(trigger)

        return triggers

    async def delete_trigger(
        self, runtime_id: str, trigger: UiPathResumeTrigger
    ) -> None:
        """Delete resume trigger from storage."""
        await self._ensure_table()

        async with self.memory.lock, self.memory.conn.cursor() as cur:
            await cur.execute(
                f"""
                DELETE FROM {self.rs_table_name}
                WHERE runtime_id = ? AND interrupt_id = ?
                """,
                (
                    runtime_id,
                    trigger.interrupt_id,
                ),
            )
            await self.memory.conn.commit()

    async def set_value(
        self,
        runtime_id: str,
        namespace: str,
        key: str,
        value: Any,
    ) -> None:
        """Save arbitrary key-value pair to database."""
        if not (
            isinstance(value, str)
            or isinstance(value, dict)
            or isinstance(value, BaseModel)
            or value is None
        ):
            raise TypeError("Value must be str, dict, BaseModel or None.")

        await self._ensure_table()

        value_text = self._dump_value(value)

        async with self.memory.lock, self.memory.conn.cursor() as cur:
            await cur.execute(
                f"""
                INSERT INTO {self.kv_table_name} (runtime_id, namespace, key, value)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(runtime_id, namespace, key)
                DO UPDATE SET
                    value = excluded.value,
                    timestamp = (strftime('%Y-%m-%d %H:%M:%S', 'now', 'utc'))
                """,
                (runtime_id, namespace, key, value_text),
            )
            await self.memory.conn.commit()

    async def get_value(self, runtime_id: str, namespace: str, key: str) -> Any:
        """Get arbitrary key-value pair from database (scoped by runtime_id + namespace)."""
        await self._ensure_table()

        async with self.memory.lock, self.memory.conn.cursor() as cur:
            await cur.execute(
                f"""
                SELECT value
                FROM {self.kv_table_name}
                WHERE runtime_id = ? AND namespace = ? AND key = ?
                LIMIT 1
                """,
                (runtime_id, namespace, key),
            )
            row = await cur.fetchone()

        if not row:
            return None

        return self._load_value(cast(str | None, row[0]))

    def _dump_value(self, value: str | dict[str, Any] | BaseModel | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, BaseModel):
            return "j:" + json.dumps(value.model_dump())
        if isinstance(value, dict):
            return "j:" + json.dumps(value)
        return "s:" + value

    def _load_value(self, raw: str | None) -> Any:
        if raw is None:
            return None
        if raw.startswith("s:"):
            return raw[2:]
        if raw.startswith("j:"):
            return json.loads(raw[2:])
        return raw
