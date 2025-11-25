from __future__ import annotations

import inspect
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt
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


# ----- Extensible guardrail actions API -----
InlineViolationHandler = Callable[
    [AgentGraphState, CustomGuardrail | BuiltInValidatorGuardrail, Any],
    Dict[str, Any] | None,
]
BranchViolationNode = Tuple[str, Callable[[AgentGraphState], Any]]
ActionApplyResult = Union[BranchViolationNode, InlineViolationHandler]


class GuardrailAction(ABC):
    """Extensible action interface executed when a guardrail fails validation.

    Implementations can either:
    - return an inline handler (BlockAction/LogAction) that runs inside the guardrail node, or
    - return a new node (HitlAction) to be linked into the guardrail subgraph.

    See https://docs.uipath.com for guardrail concepts and best practices.
    """

    @abstractmethod
    def apply(
        self,
        *,
        guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
        scope: GuardrailScope,
        hook_type: Literal["before", "after"],
        payload_generator: Callable[[AgentGraphState], str],
    ) -> ActionApplyResult:
        """Create either an inline violation handler or a node-producing action.

        Args:
            guardrail: The guardrail definition to enforce.
            scope: The scope where this guardrail applies (LLM/AGENT/TOOL).
            hook_type: Whether this is a 'before' or 'after' hook.
            payload_generator: Function to extract the text payload from state.

        Returns:
            Either:
            - (name, node) tuple to be added to the graph, or
            - a callable handler that will run inline if the guardrail fails.
        """
        ...


class BlockAction(GuardrailAction):
    """Inline action that terminates execution when a guardrail fails."""

    def apply(
        self,
        *,
        guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
        scope: GuardrailScope,
        hook_type: Literal["before", "after"],
        payload_generator: Callable[[AgentGraphState], str],
    ) -> ActionApplyResult:
        def _handler(
            state: AgentGraphState,
            gr: CustomGuardrail | BuiltInValidatorGuardrail,
            result: Any,
        ) -> Dict[str, Any] | None:
            raise AgentTerminationException(
                code=UiPathErrorCode.EXECUTION_ERROR,
                title="Guardrail violation",
                detail=result.reason,
            )

        return _handler


class LogAction(GuardrailAction):
    """Inline action that logs guardrail violations and continues."""

    def __init__(self, level: int = logging.WARNING):
        self.level = level

    def apply(
        self,
        *,
        guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
        scope: GuardrailScope,
        hook_type: Literal["before", "after"],
        payload_generator: Callable[[AgentGraphState], str],
    ) -> ActionApplyResult:
        def _handler(
            state: AgentGraphState,
            gr: CustomGuardrail | BuiltInValidatorGuardrail,
            result: Any,
        ) -> Dict[str, Any] | None:
            logger.log(
                self.level,
                "Guardrail '%s' failed at %s %s: %s",
                gr.name,
                hook_type,
                scope.value if hasattr(scope, "value") else str(scope),
                result.reason,
            )
            # No state mutation by default
            return {}

        return _handler


class HitlAction(GuardrailAction):
    """Node-producing action that inserts a HITL interruption node into the graph.

    The returned node triggers a dynamic interrupt for HITL without re-evaluating.
    The runtime will persist a resume trigger and suspend execution.
    """

    def apply(
        self,
        *,
        guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
        scope: GuardrailScope,
        hook_type: Literal["before", "after"],
        payload_generator: Callable[[AgentGraphState], str],
    ) -> ActionApplyResult:
        sanitized = re.sub(r"\W+", "_", guardrail.name).strip("_").lower()
        node_name = f"{sanitized}_hitl_{hook_type}_{scope.lower()}"

        async def _node(state: AgentGraphState) -> Dict[str, Any]:
            text = payload_generator(state)
            payload = {
                "type": "hitl_guardrail_violation",
                "guardrail": getattr(guardrail, "name", "unknown"),
                "scope": scope.value if hasattr(scope, "value") else str(scope),
                "hook": hook_type,
                "payload": text,
            }
            return interrupt(payload)

        return node_name, _node


def _create_guardrail_node(
    guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
    scope: GuardrailScope,
    hook_type: Literal["before", "after"],
    payload_generator: Callable[[AgentGraphState], str],
    action: GuardrailAction,
) -> Tuple[str, Callable[[AgentGraphState], Any]]:
    """Private factory for guardrail nodes used by public creators.

    For inline actions, the returned node will invoke the inline handler when
    a validation fails. For node-producing actions, the inline handler is a no-op
    and the actual action node is expected to be linked by the subgraph builder.
    """
    sanitized_name = re.sub(r"\W+", "_", guardrail.name).strip("_").lower()
    node_name = f"{sanitized_name}_{hook_type}_{scope.lower()}"

    apply_result = action.apply(
        guardrail=guardrail,
        scope=scope,
        hook_type=hook_type,
        payload_generator=payload_generator,
    )
    inline_handler: InlineViolationHandler | None
    if isinstance(apply_result, tuple):
        inline_handler = None  # Node-producing action, do nothing inline
    else:
        inline_handler = apply_result

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
            # Prefer inline handler if provided by action
            if inline_handler is not None:
                maybe = inline_handler(state, guardrail, result)
                if inspect.isawaitable(maybe):
                    await maybe  # type: ignore[func-returns-value]
                # Inline handlers can optionally return a partial state update
                if isinstance(maybe, dict):
                    return maybe
        # No message change by default; keep last message type intact
        return {}

    return node_name, node


def create_llm_guardrail_node(
    guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
    hook_type: Literal["before", "after"],
    action: GuardrailAction,
) -> Tuple[str, Callable[[AgentGraphState], Any]]:
    def _payload_generator(state: AgentGraphState) -> str:
        if not state.messages:
            return ""
        return _message_text(state.messages[-1])

    return _create_guardrail_node(
        guardrail, GuardrailScope.LLM, hook_type, _payload_generator, action
    )


def create_agent_guardrail_node(
    guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
    hook_type: Literal["before", "after"],
    action: GuardrailAction,
) -> Tuple[str, Callable[[AgentGraphState], Any]]:
    def _payload_generator(state: AgentGraphState) -> str:
        if not state.messages:
            return ""
        return _message_text(state.messages[-1])

    return _create_guardrail_node(
        guardrail, GuardrailScope.AGENT, hook_type, _payload_generator, action
    )


def create_tool_guardrail_node(
    guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
    hook_type: Literal["before", "after"],
    action: GuardrailAction,
) -> Tuple[str, Callable[[AgentGraphState], Any]]:
    def _payload_generator(state: AgentGraphState) -> str:
        if not state.messages:
            return ""
        return _message_text(state.messages[-1])

    return _create_guardrail_node(
        guardrail, GuardrailScope.TOOL, hook_type, _payload_generator, action
    )
