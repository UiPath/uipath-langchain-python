import json
import logging
from enum import Enum
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from uipath._cli._runtime._contracts import UiPathErrorCategory, UiPathResumeTrigger
from uipath._cli._runtime._hitl import HitlProcessor

from ._exception import LangGraphErrorCode, LangGraphRuntimeError

logger = logging.getLogger(__name__)


def serialize_output(output: Any) -> Any:
    """
    Recursively serialize an output object.

    Args:
        output: The object to serialize

    Returns:
        Dict[str, Any]: Serialized output as dictionary
    """
    if output is None:
        return {}

    # Handle Pydantic models
    if hasattr(output, "model_dump"):
        return serialize_output(output.model_dump(by_alias=True))
    elif hasattr(output, "dict"):
        return serialize_output(output.dict())
    elif hasattr(output, "to_dict"):
        return serialize_output(output.to_dict())

    # Handle dictionaries
    elif isinstance(output, dict):
        return {k: serialize_output(v) for k, v in output.items()}

    # Handle lists
    elif isinstance(output, list):
        return [serialize_output(item) for item in output]

    # Handle other iterables (convert to dict first)
    elif hasattr(output, "__iter__") and not isinstance(output, (str, bytes)):
        try:
            return serialize_output(dict(output))
        except (TypeError, ValueError):
            return output

    # Handle Enums
    elif isinstance(output, Enum):
        return output.value

    # Return primitive types as is
    return output


async def create_and_save_resume_trigger(
    interrupt_value: Any,
    memory: AsyncSqliteSaver,
    resume_triggers_table: str = "__uipath_resume_triggers",
) -> UiPathResumeTrigger:
    """
    Create a resume trigger from interrupt value and save it to the database.

    Args:
        interrupt_value: The interrupt value from dynamic interrupt
        memory: The SQLite checkpointer/memory instance
        resume_triggers_table: Name of the resume triggers table

    Returns:
        UiPathResumeTrigger: The created resume trigger

    Raises:
        LangGraphRuntimeError: If database operations fail
    """
    # Create HITL processor
    hitl_processor = HitlProcessor(interrupt_value)

    # Setup database and create table if needed
    await memory.setup()
    async with memory.lock, memory.conn.cursor() as cur:
        try:
            await cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {resume_triggers_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    key TEXT,
                    folder_key TEXT,
                    folder_path TEXT,
                    payload TEXT,
                    timestamp DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'utc'))
                )
            """)
        except Exception as e:
            raise LangGraphRuntimeError(
                LangGraphErrorCode.DB_TABLE_CREATION_FAILED,
                "Failed to create resume triggers table",
                f"Database error while creating table: {str(e)}",
                UiPathErrorCategory.SYSTEM,
            ) from e

        # Create resume trigger
        try:
            resume_trigger = await hitl_processor.create_resume_trigger()
        except Exception as e:
            raise LangGraphRuntimeError(
                LangGraphErrorCode.HITL_EVENT_CREATION_FAILED,
                "Failed to process HITL request",
                f"Error while trying to process HITL request: {str(e)}",
                UiPathErrorCategory.SYSTEM,
            ) from e

        # Save to database
        if resume_trigger.api_resume:
            trigger_key = resume_trigger.api_resume.inbox_id
        else:
            trigger_key = resume_trigger.item_key

        try:
            logger.debug(
                f"ResumeTrigger: {resume_trigger.trigger_type} {resume_trigger.item_key}"
            )

            if isinstance(resume_trigger.payload, dict):
                payload = json.dumps(resume_trigger.payload)
            else:
                payload = str(resume_trigger.payload)

            await cur.execute(
                f"INSERT INTO {resume_triggers_table} (type, key, payload, folder_path, folder_key) VALUES (?, ?, ?, ?, ?)",
                (
                    resume_trigger.trigger_type.value,
                    trigger_key,
                    payload,
                    resume_trigger.folder_path,
                    resume_trigger.folder_key,
                ),
            )
            await memory.conn.commit()
        except Exception as e:
            raise LangGraphRuntimeError(
                LangGraphErrorCode.DB_INSERT_FAILED,
                "Failed to save resume trigger",
                f"Database error while saving resume trigger: {str(e)}",
                UiPathErrorCategory.SYSTEM,
            ) from e

    return resume_trigger
