from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, Literal, Tuple

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.types import Command
from uipath import UiPath
from uipath._services.guardrails_service import GuardrailsService
from uipath.models.guardrails import (
    BuiltInValidatorGuardrail,
    CustomGuardrail,
    GuardrailScope,
)

from .types import AgentGuardrailsGraphState

logger = logging.getLogger(__name__)


def _message_text(msg: AnyMessage) -> str:
    if isinstance(msg, (HumanMessage, SystemMessage)):
        return msg.content if isinstance(msg.content, str) else str(msg.content)
    return str(getattr(msg, "content", "")) if hasattr(msg, "content") else ""


def _create_guardrail_node(
    guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
    scope: GuardrailScope,
    execution_stage: Literal["PreExecution", "PostExecution"],
    payload_generator: Callable[[AgentGuardrailsGraphState], str],
    success_node: str,
    failure_node: str,
) -> Tuple[str, Callable[[AgentGuardrailsGraphState], Any]]:
    """Private factory for guardrail evaluation nodes.

    Returns a node that evaluates the guardrail and routes via Command:
    - goto success_node on validation pass
    - goto failure_node on validation fail
    """
    raw_node_name = f"{scope.lower()}_{execution_stage}_{guardrail.name}"
    node_name = re.sub(r"\W+", "_", raw_node_name).strip("_").lower()

    async def node(state: AgentGuardrailsGraphState) -> Dict[str, Any] | Command:
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
            return Command(
                goto=failure_node, update={"guardrail_validation_result": result.reason}
            )
        return Command(goto=success_node, update={"guardrail_validation_result": None})

    return node_name, node


def create_llm_guardrail_node(
    guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
    execution_stage: Literal["PreExecution", "PostExecution"],
    success_node: str,
    failure_node: str,
) -> Tuple[str, Callable[[AgentGuardrailsGraphState], Any]]:
    def _payload_generator(state: AgentGuardrailsGraphState) -> str:
        if not state.messages:
            return ""
        return _message_text(state.messages[-1])

    return _create_guardrail_node(
        guardrail,
        GuardrailScope.LLM,
        execution_stage,
        _payload_generator,
        success_node,
        failure_node,
    )


def create_agent_guardrail_node(
    guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
    execution_stage: Literal["PreExecution", "PostExecution"],
    success_node: str,
    failure_node: str,
) -> Tuple[str, Callable[[AgentGuardrailsGraphState], Any]]:
    # To be implemented in future PR
    def _payload_generator(state: AgentGuardrailsGraphState) -> str:
        if not state.messages:
            return ""
        return _message_text(state.messages[-1])

    return _create_guardrail_node(
        guardrail,
        GuardrailScope.AGENT,
        execution_stage,
        _payload_generator,
        success_node,
        failure_node,
    )


def create_tool_guardrail_node(
    guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
    execution_stage: Literal["PreExecution", "PostExecution"],
    success_node: str,
    failure_node: str,
) -> Tuple[str, Callable[[AgentGuardrailsGraphState], Any]]:
    # To be implemented in future PR
    def _payload_generator(state: AgentGuardrailsGraphState) -> str:
        if not state.messages:
            return ""
        return _message_text(state.messages[-1])

    return _create_guardrail_node(
        guardrail,
        GuardrailScope.TOOL,
        execution_stage,
        _payload_generator,
        success_node,
        failure_node,
    )
