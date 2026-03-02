"""Conversational consumption handler for UiPath agents.

Encapsulates all conversational consumption logic: determining whether an exchange
qualifies for consumption recording and registering it with the licensing service.
"""

import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage
from uipath.agent.models.agent import AgentDefinition
from uipath.platform.common import UiPathConfig
from uipath_langchain.runtime.messages import UiPathChatMessagesMapper

from uipath_agents._services.licensing_service import (
    register_conversational_licensing_async,
)

from .llmops import LlmOpsInstrumentationCallback

logger = logging.getLogger(__name__)


class ConversationalConsumptionHandler:
    """Handles conversational consumption registration for agent exchanges.

    Determines whether an exchange qualifies for recording (conversational agent
    with tool calls or user message > 8 chars) and registers consumption
    with the licensing service.
    """

    def __init__(
        self,
        agent_definition: AgentDefinition | None,
        callback: LlmOpsInstrumentationCallback,
    ) -> None:
        self._agent_definition = agent_definition
        self._callback = callback

    @property
    def _is_byo_execution(self) -> bool:
        if not self._agent_definition:
            return False
        return bool(
            self._agent_definition.settings.byom_properties
            and self._agent_definition.settings.byom_properties.connection_id
        )

    async def register_consumption_if_applicable(
        self, input_data: Dict[str, Any] | None, *, agenthub_config: str
    ) -> None:
        """Record consumption if the exchange qualifies. Silently handles errors."""
        try:
            if not self._should_register_consumption(input_data):
                return
            await register_conversational_licensing_async(
                self._agent_definition,
                agenthub_config=agenthub_config,
                is_byo_execution=self._is_byo_execution,
                job_key=UiPathConfig.job_key,
            )
        except Exception:
            logger.debug("Failed to record conversational consumption", exc_info=True)

    def _should_register_consumption(self, input_data: Dict[str, Any] | None) -> bool:
        """Conversational agent + (tool calls OR user message > 8 chars)."""
        if not (self._agent_definition and self._agent_definition.is_conversational):
            return False
        if self._callback.had_tool_calls:
            return True
        if not input_data:
            return False
        messages = input_data.get("messages")
        if not messages or not isinstance(messages, list):
            return False
        user_text = self._extract_user_message_text(messages)
        return len(user_text) > 8 if user_text else False

    def _extract_user_message_text(self, messages: list[Any]) -> str | None:
        """Extract last user message text using UiPathChatMessagesMapper."""
        mapper = UiPathChatMessagesMapper(runtime_id="", storage=None)
        langchain_messages = mapper.map_messages(messages)

        for msg in reversed(langchain_messages):
            if isinstance(msg, HumanMessage):
                return mapper._extract_text(msg.content) or None
        return None
