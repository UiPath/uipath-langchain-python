"""Consumption licensing for UiPath agents.

Registers per-exchange consumption with the licensing service when
an agent exchange qualifies (tool calls or sufficient user input).
"""

import logging

from uipath.agent.models.agent import AgentDefinition
from uipath.platform.common import UiPathConfig

from uipath_agents._services.licensing_service import (
    register_conversational_licensing_async,
)

logger = logging.getLogger(__name__)

_MIN_USER_MESSAGE_LENGTH = 8


class ConversationalConsumptionHandler:
    """Registers per-exchange licensing consumption for conversational agents.

    An exchange qualifies when the agent is conversational and either a tool
    was invoked or the user message exceeds _MIN_USER_MESSAGE_LENGTH chars.
    """

    def __init__(
        self,
        agent_definition: AgentDefinition | None,
    ) -> None:
        self._agent_definition = agent_definition

    @property
    def _is_byo_execution(self) -> bool:
        if not self._agent_definition:
            return False
        return bool(
            self._agent_definition.settings.byom_properties
            and self._agent_definition.settings.byom_properties.connection_id
        )

    async def register_consumption_if_applicable(
        self,
        *,
        agenthub_config: str,
        had_tool_calls: bool = False,
        user_message_length: int = 0,
    ) -> None:
        """Register consumption if the exchange qualifies. Errors are suppressed."""
        try:
            if not self._should_register_consumption(
                had_tool_calls, user_message_length
            ):
                return
            await register_conversational_licensing_async(
                self._agent_definition,
                agenthub_config=agenthub_config,
                is_byo_execution=self._is_byo_execution,
                job_key=UiPathConfig.job_key,
            )
        except Exception:
            logger.debug("Failed to record conversational consumption", exc_info=True)

    def _should_register_consumption(
        self, had_tool_calls: bool, user_message_length: int
    ) -> bool:
        """Return True if the exchange qualifies for consumption registration."""
        if not (self._agent_definition and self._agent_definition.is_conversational):
            return False
        if had_tool_calls:
            return True
        return user_message_length > _MIN_USER_MESSAGE_LENGTH
