from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from uipath import UiPath
from uipath._cli._runtime._contracts import UiPathErrorCode
from uipath._services.guardrails_service import GuardrailsService, BuiltInGuardrailValidationResult
from uipath.agent.models.agent import (
    AgentBuiltInValidatorGuardrail,
    AgentGuardrailActionType,
)

from ..exceptions import AgentTerminationException
from ..middleware_types import AgentMiddleware

logger = logging.getLogger(__name__)


def _message_text(msg: AnyMessage) -> str:
    """Extract a printable text representation from a LangChain message."""
    if isinstance(msg, (HumanMessage, SystemMessage)):
        return msg.content if isinstance(msg.content, str) else str(msg.content)
    return str(getattr(msg, "content", "")) if hasattr(msg, "content") else ""


class PiiDetectionMiddleware(AgentMiddleware):
    """PII guardrail middleware invoking UiPath GuardrailsService.

    The middleware implements LangChain-style `before_agent` and `after_agent`
    hooks. It evaluates the aggregate message text using UiPath Guardrails and
    enforces the configured action (block or warn).
    See https://docs.uipath.com for Guardrails reference.
    """

    def __init__(
        self,
        guardrail: AgentBuiltInValidatorGuardrail,
        service: GuardrailsService | None = None,
    ) -> None:
        """Initialize the middleware.

        Args:
            guardrail: Built-in validator guardrail configuration for PII.
            service: Optional GuardrailsService instance to reuse/inject.
        """
        self.guardrail: AgentBuiltInValidatorGuardrail = guardrail
        self._service = service

    async def before_agent(
        self,
        messages: list[SystemMessage | HumanMessage],
        handler: Callable[[list[SystemMessage | HumanMessage]], Any],
    ) -> list[SystemMessage | HumanMessage]:
        """Validate messages before agent execution and proceed or block."""
        self._evaluate_messages(messages, phase="before_agent")
        print("before agent guardrail")
        return await self._maybe_await(handler(messages))

    async def after_agent(
        self, messages: list[AnyMessage], handler: Callable[[list[AnyMessage]], Any]
    ) -> list[AnyMessage]:
        """Validate messages after agent execution and proceed or block."""
        self._evaluate_messages(messages, phase="after_agent")
        print("after agent guardrail")
        return await self._maybe_await(handler(messages))

    def _evaluate_messages(
        self, messages: list[AnyMessage], phase: str
    ) -> None:
        """Call GuardrailsService and enforce the configured action."""
        text = "\n".join([_message_text(m) for m in messages if _message_text(m)])

        try:
            guardrails_service = (
                self._service
                if self._service is not None
                else GuardrailsService(
                    config=UiPath()._config,
                    execution_context=UiPath()._execution_context,
                )
            )
            result = guardrails_service.evaluate_guardrail(text, self.guardrail)
            # result: BuiltInGuardrailValidationResult = BuiltInGuardrailValidationResult(
            #     validation_passed=False, reason="dummy"
            # )
        except Exception as exc:
            logger.error("Failed to evaluate guardrail: %s", exc)
            raise

        if not result.validation_passed:
            default_title = f"Guardrail violation ({phase})"
            action = self.guardrail.action
            if action and action.action_type == AgentGuardrailActionType.BLOCK:
                error_message = action.reason or default_title
                detail = result.reason or "Validation failed by PII guardrail"
                raise AgentTerminationException(
                    code=UiPathErrorCode.EXECUTION_ERROR,
                    title=error_message,
                    detail=detail,
                )

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        """Await value if it's awaitable; otherwise return it as-is."""
        if hasattr(value, "__await__"):
            return await value
        return value
