from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, Literal, Optional, Tuple

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from uipath import UiPath
from uipath._cli._runtime._contracts import UiPathErrorCode
from uipath._services.guardrails_service import GuardrailsService
from uipath.agent.models.agent import (
    AgentGuardrailActionType,
)
from uipath.models.guardrails import (
    BuiltInValidatorGuardrail,
    CustomGuardrail,
    GuardrailScope,
)

from .exceptions import AgentTerminationException
from .types import AgentGraphState

logger = logging.getLogger(__name__)


def _message_text(msg: AnyMessage) -> str:
    if isinstance(msg, (HumanMessage, SystemMessage)):
        return msg.content if isinstance(msg.content, str) else str(msg.content)
    return str(getattr(msg, "content", "")) if hasattr(msg, "content") else ""


def _create_guardrail_node(
    guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
    scope: GuardrailScope,
    hook_type: Literal["before", "after"],
    payload_generator: Callable[[AgentGraphState], str],
) -> Tuple[str, Callable[[AgentGraphState], Any]]:
    """Private factory for guardrail nodes used by public creators."""
    sanitized_name = re.sub(r'\W+', '_', guardrail.name).strip('_').lower()
    node_name = f"{sanitized_name}_{hook_type}_{scope.lower()}"

    async def node(state: AgentGraphState) -> Dict[str, Any]:
        text = payload_generator(state)
        try:
            uipath = UiPath()
            service = GuardrailsService(
                config=uipath._config, execution_context=uipath._execution_context
            )
            result = service.evaluate_guardrail(text, guardrail)
        except Exception as exc:
            logger.error("Failed to evaluate guardrail: %s", exc)
            raise

        if not result.validation_passed:
            action = guardrail.action
            if action and action.action_type == AgentGuardrailActionType.BLOCK:
                raise AgentTerminationException(
                    code=UiPathErrorCode.EXECUTION_ERROR,
                    title=action.reason or "Guardrail violation",
                    detail=result.reason
                    or f"Validation failed for guardrail {guardrail.name} of type {guardrail.guardrail_type}",
                )
        # No message change by default; keep last message type intact
        return {}

    return node_name, node


def create_llm_guardrail_node(
    guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
    hook_type: Literal["before", "after"],
) -> Tuple[str, Callable[[AgentGraphState], Any]]:
    def _payload_generator(state: AgentGraphState) -> str:
        if not state.messages:
            return ""
        return _message_text(state.messages[-1])

    return _create_guardrail_node(
        guardrail, GuardrailScope.LLM, hook_type, _payload_generator
    )


def create_agent_guardrail_node(
    guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
    hook_type: Literal["before", "after"],
) -> Tuple[str, Callable[[AgentGraphState], Any]]:
    def _payload_generator(state: AgentGraphState) -> str:
        if not state.messages:
            return ""
        return _message_text(state.messages[-1])

    return _create_guardrail_node(
        guardrail, GuardrailScope.AGENT, hook_type, _payload_generator
    )


def create_tool_guardrail_node(
    guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
    hook_type: Literal["before", "after"],
) -> Tuple[str, Callable[[AgentGraphState], Any]]:
    def _payload_generator(state: AgentGraphState) -> str:
        if not state.messages:
            return ""
        return _message_text(state.messages[-1])

    return _create_guardrail_node(
        guardrail, GuardrailScope.TOOL, hook_type, _payload_generator
    )
