"""Licensed runtime wrapper for UiPath agents.

Adds licensing concerns to any UiPathRuntimeProtocol implementation
via composition, keeping licensing separate from tracing/instrumentation.

Handles two licensing modes:
- One-time startup registration on first execution
- Per-exchange conversational consumption after each successful execution
"""

import logging
from typing import Any, AsyncGenerator, Dict, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from uipath.agent.models.agent import AgentDefinition
from uipath.platform.common import UiPathConfig
from uipath.runtime import (
    UiPathExecuteOptions,
    UiPathRuntimeProtocol,
    UiPathRuntimeResult,
    UiPathRuntimeStatus,
    UiPathStreamOptions,
)
from uipath.runtime.events import UiPathRuntimeEvent
from uipath.runtime.schema import UiPathRuntimeSchema
from uipath_langchain.runtime.messages import UiPathChatMessagesMapper

from uipath_agents._licensing.consumption import (
    ConversationalConsumptionHandler,
)
from uipath_agents._services.licensing_service import register_licensing_async
from uipath_agents.agent_graph_builder.config import AgentExecutionType
from uipath_agents.agent_graph_builder.llm_utils import _get_agenthub_config

logger = logging.getLogger(__name__)


class ToolCallTracker(BaseCallbackHandler):
    """LangChain callback that tracks whether any tool was invoked during execution."""

    def __init__(self) -> None:
        super().__init__()
        self._had_tool_calls = False

    @property
    def had_tool_calls(self) -> bool:
        return self._had_tool_calls

    def reset(self) -> None:
        self._had_tool_calls = False

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Any = None,
        metadata: Any = None,
        **kwargs: Any,
    ) -> None:
        self._had_tool_calls = True


def _had_tool_calls_from_result(result: UiPathRuntimeResult) -> bool:
    """Check for tool-type messages in the execution result output."""
    if not isinstance(result.output, dict):
        return False
    messages = result.output.get("messages", [])
    if not isinstance(messages, list):
        return False
    return any(isinstance(m, dict) and m.get("type") == "tool" for m in messages)


def _extract_user_message_length(input_data: Dict[str, Any] | None) -> int:
    """Return the text length of the last user message in the input, or 0."""
    if not input_data:
        return 0
    messages = input_data.get("messages")
    if not messages or not isinstance(messages, list):
        return 0
    mapper = UiPathChatMessagesMapper(runtime_id="", storage=None)
    langchain_messages = mapper.map_messages(messages)
    for msg in reversed(langchain_messages):
        if isinstance(msg, HumanMessage):
            text = mapper._extract_text(msg.content)
            return len(text) if text else 0
    return 0


class LicensedRuntime:
    """Runtime wrapper that registers licensing consumption.

    Wraps any UiPathRuntimeProtocol to add:
    - One-time startup licensing on first execution (skipped on resume)
    - Per-exchange conversational consumption after each successful execution
    """

    def __init__(
        self,
        delegate: UiPathRuntimeProtocol,
        *,
        agent_definition: AgentDefinition | None = None,
        tool_call_tracker: ToolCallTracker | None = None,
        execution_type: AgentExecutionType,
        is_resume: bool = False,
    ) -> None:
        self._delegate = delegate
        self._agent_definition = agent_definition
        self._tool_call_tracker = tool_call_tracker
        self._execution_type = execution_type
        self._startup_licensed = is_resume

        self._log_initialization(agent_definition, is_resume)

        self._consumption_handler: ConversationalConsumptionHandler | None = None
        if agent_definition and agent_definition.is_conversational:
            self._consumption_handler = ConversationalConsumptionHandler(
                agent_definition=agent_definition,
            )

    @property
    def delegate(self) -> UiPathRuntimeProtocol:
        return self._delegate

    async def execute(
        self,
        input: Dict[str, Any] | None = None,
        options: UiPathExecuteOptions | None = None,
    ) -> UiPathRuntimeResult:
        await self._register_startup_licensing()
        self._reset_tool_tracker()

        result = await self._delegate.execute(input, options)

        if result.status == UiPathRuntimeStatus.SUCCESSFUL:
            await self._register_conversational_consumption(input, result)

        return result

    async def stream(
        self,
        input: Dict[str, Any] | None = None,
        options: UiPathStreamOptions | None = None,
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        await self._register_startup_licensing()
        self._reset_tool_tracker()

        final_result: Optional[UiPathRuntimeResult] = None
        async for event in self._delegate.stream(input, options):
            if isinstance(event, UiPathRuntimeResult):
                final_result = event
            yield event

        if final_result and final_result.status == UiPathRuntimeStatus.SUCCESSFUL:
            await self._register_conversational_consumption(input, final_result)

    async def get_schema(self) -> UiPathRuntimeSchema:
        return await self._delegate.get_schema()

    async def dispose(self) -> None:
        await self._delegate.dispose()

    def get_agent_model(self) -> str | None:
        if hasattr(self._delegate, "get_agent_model"):
            return self._delegate.get_agent_model()
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    # --- Internal ---

    @staticmethod
    def _log_initialization(
        agent_definition: AgentDefinition | None,
        is_resume: bool,
    ) -> None:
        model = agent_definition.settings.model if agent_definition else None
        is_conversational = bool(
            agent_definition and agent_definition.is_conversational
        )
        licensing_context = UiPathConfig.licensing_context

        logger.info(
            "LicensedRuntime initialized: model='%s', "
            "is_conversational=%s, is_resume=%s, "
            "licensing_context=%s",
            model,
            is_conversational,
            is_resume,
            licensing_context,
        )

    async def _register_startup_licensing(self) -> None:
        if self._startup_licensed:
            return
        self._startup_licensed = True
        try:
            await register_licensing_async(
                self._agent_definition, job_key=UiPathConfig.job_key
            )
        except Exception:
            logger.debug("Failed to register startup consumption", exc_info=True)

    def _reset_tool_tracker(self) -> None:
        if self._tool_call_tracker:
            self._tool_call_tracker.reset()

    def _detect_tool_calls(self, result: UiPathRuntimeResult) -> bool:
        if self._tool_call_tracker and self._tool_call_tracker.had_tool_calls:
            return True
        return _had_tool_calls_from_result(result)

    async def _register_conversational_consumption(
        self,
        input_data: Dict[str, Any] | None,
        result: UiPathRuntimeResult,
    ) -> None:
        if not self._consumption_handler:
            return
        try:
            agenthub_config = _get_agenthub_config(
                self._execution_type,
                is_conversational=True,
            )
            had_tools = self._detect_tool_calls(result)
            user_msg_len = _extract_user_message_length(input_data)
            await self._consumption_handler.register_consumption_if_applicable(
                agenthub_config=agenthub_config,
                had_tool_calls=had_tools,
                user_message_length=user_msg_len,
            )
        except Exception:
            logger.debug("Failed to register conversational consumption", exc_info=True)
