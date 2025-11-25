from __future__ import annotations

import inspect
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Literal, Tuple, Union

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt, Command
from uipath import UiPath
from uipath._cli._runtime._contracts import UiPathErrorCode
from uipath._services.guardrails_service import GuardrailsService
from uipath.models import CreateAction
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


def _hook_type_to_execution_stage(hook_type: Literal["before", "after"]) -> str:
    """Convert hook type to execution stage string. """
    return "PreExecution" if hook_type == "before" else "PostExecution"


def _hook_type_to_tool_field(hook_type: Literal["before", "after"]) -> str:
    """Convert hook type to tool field name.

    Args:
        hook_type: The hook type ("before" or "after").

    Returns:
        "ToolInputs" for "before", "ToolOutputs" for "after".
    """
    return "ToolInputs" if hook_type == "before" else "ToolOutputs"


def _extract_escalation_content(
    state: AgentGraphState, scope: GuardrailScope
) -> str:
    """Extract escalation content from state based on guardrail scope.

    Args:
        state: The current agent graph state.
        scope: The guardrail scope (LLM/AGENT/TOOL).

    Returns:
        JSON string in format {"content": "..."} for all scopes.
        For TOOL scope: {"content": ""}.
        For AGENT/LLM scope: {"content": "<message content>"}.
    """
    if scope == GuardrailScope.TOOL:
        return ""
    # For AGENT and LLM scopes, return the content of the last message in JSON format
    if not state.messages:
        return json.dumps({"content": ""})
    content = _message_text(state.messages[-1])
    return json.dumps({"content": content})


def _process_escalation_response(
    state: AgentGraphState,
    escalation_result: Dict[str, Any],
    scope: GuardrailScope,
) -> Dict[str, Any] | Command:
    """Process escalation response and update state based on guardrail scope.

    Args:
        state: The current agent graph state.
        escalation_result: The result from the escalation interrupt.
        scope: The guardrail scope (LLM/AGENT/TOOL).

    Returns:
        For AGENT/LLM scope: Command to update messages with reviewed inputs.
        For TOOL scope: Empty dict (no message alteration).

    Raises:
        AgentTerminationException: If escalation response processing fails.
    """
    if scope == GuardrailScope.TOOL:
        # For TOOL scope, don't alter messages
        return {}

    # For AGENT and LLM scopes, process the escalation response
    try:
        reviewed_inputs_json = json.loads(escalation_result["ReviewedInputs"])
        # Extract content from JSON format {"content": "..."}
        content = reviewed_inputs_json.get("content", "")
        msgs = state.messages.copy()
        if len(msgs) > 1:
            msgs[1].content = content
        return Command(update={"messages": msgs})
    except Exception as e:
        raise AgentTerminationException(
            code=UiPathErrorCode.EXECUTION_ERROR,
            title="Escalation rejected",
            detail="TODO",
        ) from e


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
            input = _extract_escalation_content(state, scope)
            tool_field = _hook_type_to_tool_field(hook_type)
            data = {
                "GuardrailName": guardrail.name,
                "GuardrailDescription": guardrail.description,
                "TenantName": "DefaultTenant",
                "AgentTrace": "https://alpha.uipath.com/f88fa028-ccdd-4b5f-bee4-01ef94d134d8/studio_/designer/48fff406-52e9-4a37-ba66-76c0212d9c6b",
                "Tool": "Create_Issue",
                "ExecutionStage": _hook_type_to_execution_stage(hook_type),
                "GuardrailResult": "Input data matched the guardrail conditions: [fields.project.key] Contains [AL]"
            }
            data[tool_field] = input
            escalation_result = interrupt(
                CreateAction(app_name="Guardrail.Escalation.Action.App", app_folder_path="TestEscalation",
                             title="Agents Guardrail Task CTI",
                             data=data,
                             app_version=1, assignee="cristian.iliescu@uipath.com"))

            return _process_escalation_response(state, escalation_result, scope)

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
