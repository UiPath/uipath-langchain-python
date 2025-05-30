import logging
from typing import Any, Optional, cast

from langgraph.types import Command
from uipath import UiPath
from uipath._cli._runtime._contracts import (
    UiPathErrorCategory,
    UiPathResumeTriggerType,
    UiPathRuntimeStatus, UiPathResumeTrigger, UiPathApiTrigger,
)

from ._context import LangGraphRuntimeContext
from ._escalation import Escalation
from ._exception import LangGraphRuntimeError
from uipath._cli._runtime._hitl import HitlReader
logger = logging.getLogger(__name__)

class LangGraphInputProcessor:
    """
    Handles input processing for graph execution, including resume scenarios
    where it needs to fetch data from UiPath.
    """

    def __init__(self, context: LangGraphRuntimeContext):
        """
        Initialize the LangGraphInputProcessor.

        Args:
            context: The runtime context for the graph execution.
        """
        self.context = context
        self.escalation = Escalation(self.context.config_path)
        self.uipath = UiPath()

    async def process(self) -> Any:
        """
        Process the input data, handling resume scenarios by fetching
        necessary data from UiPath if needed.
        """
        logger.debug(f"Resumed: {self.context.resume} Input: {self.context.input_json}")

        if not self.context.resume:
            return self.context.input_json

        if self.context.input_json:
            return Command(resume=self.context.input_json)

        trigger = await self._get_latest_trigger()
        if not trigger:
            return Command(resume=self.context.input_json)

        type, key, folder_path, folder_key, payload = trigger
        resume_trigger = UiPathResumeTrigger(
            trigger_type=type,
            item_key=key,
            folder_path=folder_path,
            folder_key=folder_key,
            payload=payload
        )
        logger.debug(f"ResumeTrigger: {type} {key}")
        if resume_trigger.trigger_type == UiPathResumeTriggerType.API:
            resume_trigger.api_resume = UiPathApiTrigger(inbox_id=resume_trigger.item_key, request= resume_trigger.payload)
        return Command(resume = await HitlReader.read(resume_trigger))

    async def _get_latest_trigger(self) -> Optional[tuple[str, str, str, str, str]]:
        """Fetch the most recent trigger from the database."""
        if self.context.memory is None:
            return None
        try:
            await self.context.memory.setup()
            async with (
                self.context.memory.lock,
                self.context.memory.conn.cursor() as cur,
            ):
                await cur.execute(f"""
                    SELECT type, key, folder_path, folder_key, payload
                    FROM {self.context.resume_triggers_table}
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)
                result = await cur.fetchone()
                if result is None:
                    return None
                return cast(tuple[str, str, str, str, str], tuple(result))
        except Exception as e:
            raise LangGraphRuntimeError(
                "DB_QUERY_FAILED",
                "Database query failed",
                f"Error querying resume trigger information: {str(e)}",
                UiPathErrorCategory.SYSTEM,
            ) from e
