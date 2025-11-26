from __future__ import annotations

import inspect
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.types import Command, interrupt
from uipath import UiPath
from uipath._cli._runtime._contracts import UiPathErrorCategory, UiPathErrorCode
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


def _hook_type_to_tool_field(hook_type: Literal["PreExecution", "PostExecution"]) -> str:
    """Convert hook type to tool field name.

    Args:
        hook_type: The hook type ("PreExecution" or "PostExecution").

    Returns:
        "ToolInputs" for "PreExecution", "ToolOutputs" for "PostExecution".
    """
    return "ToolInputs" if hook_type == "PreExecution" else "ToolOutputs"


def _extract_escalation_content(
    state: AgentGraphState, scope: GuardrailScope, hook_type: Literal["PreExecution", "PostExecution"]
) -> str:
    """Extract escalation content from state based on guardrail scope and hook type.

    Args:
        state: The current agent graph state.
        scope: The guardrail scope (LLM/AGENT/TOOL).
        hook_type: The hook type ("PreExecution" or "PostExecution").

    Returns:
        For non-LLM scope: Empty string.
        For LLM PreExecution: JSON string with message content.
        For LLM PostExecution: JSON array with tool call content and message content.
    """
    if scope != GuardrailScope.LLM:
        return ""

    if not state.messages:
        raise AgentTerminationException(
            code=UiPathErrorCode.EXECUTION_ERROR,
            title="Invalid state message",
        )

    last_message = state.messages[-1]
    if hook_type == "PreExecution":
        content = _message_text(last_message)
        return json.dumps(content) if content else ""

    ai_message: AIMessage = last_message  # type: ignore[assignment]
    content_list: list[str] = []

    if ai_message.tool_calls:
        for tool_call in ai_message.tool_calls:
            args = tool_call["args"] if isinstance(tool_call, dict) else tool_call.args
            if isinstance(args, dict) and "content" in args and args["content"] is not None:
                content_list.append(json.dumps(args["content"]))

    message_content = _message_text(last_message)
    if message_content:
        content_list.append(message_content)

    return json.dumps(content_list)


def _process_escalation_response(
    state: AgentGraphState,
    escalation_result: Dict[str, Any],
    scope: GuardrailScope,
    hook_type: Literal["PreExecution", "PostExecution"],
) -> Dict[str, Any] | Command:
    """Process escalation response and update state based on guardrail scope.

    Args:
        state: The current agent graph state.
        escalation_result: The result from the escalation interrupt.
        scope: The guardrail scope (LLM/AGENT/TOOL).
        hook_type: The hook type ("PreExecution" or "PostExecution").

    Returns:
        For LLM scope: Command to update messages with reviewed inputs/outputs.
        For non-LLM scope: Empty dict (no message alteration).

    Raises:
        AgentTerminationException: If escalation response processing fails.
    """
    if scope != GuardrailScope.LLM:
        return {}

    try:
        reviewed_field = "ReviewedInputs" if hook_type == "PreExecution" else "ReviewedOutputs"

        msgs = state.messages.copy()
        if not msgs or reviewed_field not in escalation_result:
            return {}

        last_message = msgs[-1]

        if hook_type == "PreExecution":
            reviewed_content = escalation_result[reviewed_field]
            if reviewed_content:
                last_message.content = json.loads(reviewed_content)
        else:
            reviewed_outputs_json = escalation_result[reviewed_field]
            if not reviewed_outputs_json:
                return {}

            content_list = json.loads(reviewed_outputs_json)
            if not content_list:
                return {}

            ai_message: AIMessage = last_message  # type: ignore[assignment]
            content_index = 0

            if ai_message.tool_calls:
                tool_calls = list(ai_message.tool_calls)
                for tool_call in tool_calls:
                    args = tool_call["args"] if isinstance(tool_call, dict) else tool_call.args
                    if isinstance(args, dict) and "content" in args and args["content"] is not None:
                        if content_index < len(content_list):
                            updated_content = json.loads(content_list[content_index])
                            args["content"] = updated_content
                            if isinstance(tool_call, dict):
                                tool_call["args"] = args
                            else:
                                tool_call.args = args
                            content_index += 1
                ai_message.tool_calls = tool_calls

            if len(content_list) > content_index:
                ai_message.content = content_list[-1]

        return Command(update={"messages": msgs})
    except Exception as e:
        raise AgentTerminationException(
            code=UiPathErrorCode.EXECUTION_ERROR,
            title="Escalation rejected",
            detail="TODO",
        ) from e


# ----- Extensible guardrail actions API -----
ActionInlineEnforcement = Callable[
    [AgentGraphState, CustomGuardrail | BuiltInValidatorGuardrail, Any],
    Dict[str, Any] | None,
]


@dataclass(frozen=True)
class ActionEnforcementNode:
    """Node-producing action result.

    Attributes:
        name: Unique node name to register in the graph.
        node: The async callable implementing the node.
    """

    name: str
    node: Callable[[AgentGraphState], Any]


ActionEnforcementOutcome = Union[ActionEnforcementNode, ActionInlineEnforcement]


class GuardrailAction(ABC):
    """Extensible action interface executed when a guardrail fails validation.

    Implementations can either:
    - return an inline handler (BlockAction/LogAction) that runs inside the guardrail node, or
    - return a new node (HitlAction) to be linked into the guardrail subgraph.

    See https://docs.uipath.com for guardrail concepts and best practices.
    """

    @abstractmethod
    def enforcement_outcome(
        self,
        *,
        guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
        scope: GuardrailScope,
        hook_type: Literal["PreExecution", "PostExecution"],
    ) -> ActionEnforcementOutcome:
        """Create either an inline violation handler or a node-producing action.

        Args:
            guardrail: The guardrail definition to enforce.
            scope: The scope where this guardrail applies (LLM/AGENT/TOOL).
            hook_type: Whether this is a 'before' or 'after' hook.

        Returns:
            Either:
            - (name, node) tuple to be added to the graph, or
            - a callable handler that will run inline if the guardrail fails.
        """
        ...


class BlockAction(GuardrailAction):
    """Inline action that terminates execution when a guardrail fails.

    Args:
        reason: Optional reason string to include in the raised exception title.
    """

    def __init__(self, reason: Optional[str] = None):
        self.reason = reason

    def enforcement_outcome(
        self,
        *,
        guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
        scope: GuardrailScope,
        hook_type: Literal["PreExecution", "PostExecution"],
    ) -> ActionEnforcementOutcome:
        def _handler(
            state: AgentGraphState,
            gr: CustomGuardrail | BuiltInValidatorGuardrail,
            result: Any,
        ) -> Dict[str, Any] | None:
            raise AgentTerminationException(
                code=UiPathErrorCode.EXECUTION_ERROR,
                title="Guardrail violation",
                detail=self.reason,
                category=UiPathErrorCategory.USER,
            )

        return _handler


class LogAction(GuardrailAction):
    """Inline action that logs guardrail violations and continues."""

    def __init__(self, level: int = logging.WARNING):
        self.level = level

    def enforcement_outcome(
        self,
        *,
        guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
        scope: GuardrailScope,
        hook_type: Literal["PreExecution", "PostExecution"],
    ) -> ActionEnforcementOutcome:
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

    def enforcement_outcome(
        self,
        *,
        guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
        scope: GuardrailScope,
        hook_type: Literal["PreExecution", "PostExecution"],
    ) -> ActionEnforcementOutcome:
        sanitized = re.sub(r"\W+", "_", guardrail.name).strip("_").lower()
        node_name = f"{sanitized}_hitl_{hook_type}_{scope.lower()}"

        async def _node(state: AgentGraphState) -> Dict[str, Any]:
            input = _extract_escalation_content(state, scope, hook_type)
            tool_field = _hook_type_to_tool_field(hook_type)
            data = {
                "GuardrailName": guardrail.name,
                "GuardrailDescription": guardrail.description,
                "TenantName": "DefaultTenant",
                "AgentTrace": "https://alpha.uipath.com/f88fa028-ccdd-4b5f-bee4-01ef94d134d8/studio_/designer/48fff406-52e9-4a37-ba66-76c0212d9c6b",
                "Tool": "Create_Issue",
                "ExecutionStage": hook_type,
                "GuardrailResult": "guardreilResult",
                tool_field: input,
            }
            escalation_result = interrupt(
                CreateAction(
                    app_name="Guardrail.Escalation.Action.App",
                    app_folder_path="TestEscalation",
                    title="Agents Guardrail Task CTI",
                    data=data,
                    app_version=1,
                    assignee="cristian.iliescu@uipath.com",
                )
            )

            return _process_escalation_response(state, escalation_result, scope, hook_type)

        return ActionEnforcementNode(node_name, _node)


def _create_guardrail_node(
    guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
    scope: GuardrailScope,
    hook_type: Literal["PreExecution", "PostExecution"],
    payload_generator: Callable[[AgentGraphState], str],
    action_enforcement_outcome: ActionEnforcementOutcome | None,
) -> Tuple[str, Callable[[AgentGraphState], Any]]:
    """Private factory for guardrail nodes used by public creators.

    For inline actions, the returned node will invoke the inline handler when
    a validation fails. For node-producing actions, the inline handler is a no-op
    and the actual action node is expected to be linked by the subgraph builder.
    """
    sanitized_name = re.sub(r"\W+", "_", guardrail.name).strip("_").lower()
    node_name = f"{sanitized_name}_{hook_type}_{scope.lower()}"

    async def node(state: AgentGraphState) -> Dict[str, Any] | Command:
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
            if isinstance(action_enforcement_outcome, ActionEnforcementNode):
                return Command(goto=action_enforcement_outcome.name)
            if action_enforcement_outcome is not None:
                maybe = action_enforcement_outcome(state, guardrail, result)
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
    hook_type: Literal["PreExecution", "PostExecution"],
    action_enforcement_outcome: ActionEnforcementOutcome | None,
) -> Tuple[str, Callable[[AgentGraphState], Any]]:
    def _payload_generator(state: AgentGraphState) -> str:
        if not state.messages:
            return ""
        return _message_text(state.messages[-1])

    return _create_guardrail_node(
        guardrail,
        GuardrailScope.LLM,
        hook_type,
        _payload_generator,
        action_enforcement_outcome,
    )


def create_agent_guardrail_node(
    guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
    hook_type: Literal["PreExecution", "PostExecution"],
    action_enforcement_outcome: ActionEnforcementOutcome | None,
) -> Tuple[str, Callable[[AgentGraphState], Any]]:
    def _payload_generator(state: AgentGraphState) -> str:
        if not state.messages:
            return ""
        return _message_text(state.messages[-1])

    return _create_guardrail_node(
        guardrail,
        GuardrailScope.AGENT,
        hook_type,
        _payload_generator,
        action_enforcement_outcome,
    )


def create_tool_guardrail_node(
    guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
    hook_type: Literal["PreExecution", "PostExecution"],
    action_enforcement_outcome: ActionEnforcementOutcome | None,
) -> Tuple[str, Callable[[AgentGraphState], Any]]:
    def _payload_generator(state: AgentGraphState) -> str:
        if not state.messages:
            return ""
        return _message_text(state.messages[-1])

    return _create_guardrail_node(
        guardrail,
        GuardrailScope.TOOL,
        hook_type,
        _payload_generator,
        action_enforcement_outcome,
    )
