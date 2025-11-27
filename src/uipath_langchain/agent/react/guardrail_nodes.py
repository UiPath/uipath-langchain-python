from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Literal, Tuple

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
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
from .types import AgentGuardrailsGraphState

logger = logging.getLogger(__name__)


def _message_text(msg: AnyMessage) -> str:
    if isinstance(msg, (HumanMessage, SystemMessage)):
        return msg.content if isinstance(msg.content, str) else str(msg.content)
    return str(getattr(msg, "content", "")) if hasattr(msg, "content") else ""


def _hook_type_to_tool_field(
    hook_type: Literal["PreExecution", "PostExecution"],
) -> str:
    """Convert hook type to tool field name.

    Args:
        hook_type: The hook type ("PreExecution" or "PostExecution").

    Returns:
        "ToolInputs" for "PreExecution", "ToolOutputs" for "PostExecution".
    """
    return "ToolInputs" if hook_type == "PreExecution" else "ToolOutputs"


def _extract_escalation_content(
    state: AgentGuardrailsGraphState,
    scope: GuardrailScope,
    hook_type: Literal["PreExecution", "PostExecution"],
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
            if (
                isinstance(args, dict)
                and "content" in args
                and args["content"] is not None
            ):
                content_list.append(json.dumps(args["content"]))

    message_content = _message_text(last_message)
    if message_content:
        content_list.append(message_content)

    return json.dumps(content_list)


def _process_escalation_response(
    state: AgentGuardrailsGraphState,
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
        reviewed_field = (
            "ReviewedInputs" if hook_type == "PreExecution" else "ReviewedOutputs"
        )

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
                    args = (
                        tool_call["args"]
                        if isinstance(tool_call, dict)
                        else tool_call.args
                    )
                    if (
                        isinstance(args, dict)
                        and "content" in args
                        and args["content"] is not None
                    ):
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


# Graph node tuple (name, callable)
GraphNode = Tuple[str, Callable[[AgentGuardrailsGraphState], Any]]


class GuardrailAction(ABC):
    """Extensible action interface producing a node for validation failure."""

    @abstractmethod
    def enforcement_outcome(
        self,
        *,
        guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
        scope: GuardrailScope,
        execution_stage: Literal["PreExecution", "PostExecution"],
    ) -> GraphNode:
        """Create and return the GraphNode to execute on validation failure."""
        ...


class BlockAction(GuardrailAction):
    """Inline action that terminates execution when a guardrail fails.

    Args:
        reason: Optional reason string to include in the raised exception title.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def enforcement_outcome(
        self,
        *,
        guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
        scope: GuardrailScope,
        execution_stage: Literal["PreExecution", "PostExecution"],
    ) -> GraphNode:
        sanitized = re.sub(r"\W+", "_", getattr(guardrail, "name", "guardrail")).strip(
            "_"
        )
        node_name = f"{sanitized}_{execution_stage.lower()}_{scope.lower()}_block"

        async def _node(_state: AgentGuardrailsGraphState) -> Dict[str, Any]:
            raise AgentTerminationException(
                code=UiPathErrorCode.EXECUTION_ERROR,
                title="Guardrail violation",
                detail=self.reason,
                category=UiPathErrorCategory.USER,
            )

        return node_name, _node


class LogAction(GuardrailAction):
    """Inline action that logs guardrail violations and continues."""

    def __init__(self, level: int = logging.WARNING) -> None:
        self.level = level

    def enforcement_outcome(
        self,
        *,
        guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
        scope: GuardrailScope,
        execution_stage: Literal["PreExecution", "PostExecution"],
    ) -> GraphNode:
        sanitized = re.sub(r"\W+", "_", getattr(guardrail, "name", "guardrail")).strip(
            "_"
        )
        node_name = f"{sanitized}_{execution_stage.lower()}_{scope.lower()}_log"

        async def _node(_state: AgentGuardrailsGraphState) -> Dict[str, Any]:
            logger.log(
                self.level,
                "Guardrail '%s' failed at %s %s: %s",
                guardrail.name,
                execution_stage,
                scope.value if hasattr(scope, "value") else str(scope),
                _state.guardrailResultReason,
            )
            return {}

        return node_name, _node


class HitlAction(GuardrailAction):
    def enforcement_outcome(
        self,
        *,
        guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
        scope: GuardrailScope,
        execution_stage: Literal["PreExecution", "PostExecution"],
    ) -> GraphNode:
        sanitized = re.sub(r"\W+", "_", guardrail.name).strip("_").lower()
        node_name = f"{sanitized}_hitl_{execution_stage}_{scope.lower()}"

        async def _node(state: AgentGuardrailsGraphState) -> Dict[str, Any]:
            input = _extract_escalation_content(state, scope, execution_stage)
            tool_field = _hook_type_to_tool_field(execution_stage)
            data = {
                "GuardrailName": guardrail.name,
                "GuardrailDescription": guardrail.description,
                "TenantName": "AgentsRuntime",
                "AgentTrace": "https://alpha.uipath.com/f88fa028-ccdd-4b5f-bee4-01ef94d134d8/studio_/designer/48fff406-52e9-4a37-ba66-76c0212d9c6b",
                "Tool": "Create_Issue",
                "ExecutionStage": execution_stage,
                "GuardrailResult": state.guardrailResultReason,
                tool_field: input,
            }
            escalation_result = interrupt(
                CreateAction(
                    app_name="Guardrail.Escalation.Action.App",
                    app_folder_path="Shared",
                    title="Agents Guardrail Task VB",
                    data=data,
                    app_version=1,
                    assignee="valentina.bojan@uipath.com",
                )
            )
            return _process_escalation_response(
                state, escalation_result, scope, execution_stage
            )

        return node_name, _node


def _create_guardrail_node(
    guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
    scope: GuardrailScope,
    execution_stage: Literal["PreExecution", "PostExecution"],
    payload_generator: Callable[[AgentGuardrailsGraphState], str],
    success_node: GraphNode,
    failure_node: GraphNode,
) -> Tuple[str, Callable[[AgentGuardrailsGraphState], Any]]:
    """Private factory for guardrail evaluation nodes.

    Returns a node that evaluates the guardrail and routes via Command:
    - goto success_node on validation pass
    - goto failure_node on validation fail
    """
    sanitized_name = re.sub(r"\W+", "_", guardrail.name).strip("_").lower()
    node_name = f"{sanitized_name}_{execution_stage}_{scope.lower()}"

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
            return Command(goto=failure_node[0])
        return Command(goto=success_node[0])

    return node_name, node


def create_llm_guardrail_node(
    guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
    execution_stage: Literal["PreExecution", "PostExecution"],
    success_node: GraphNode,
    failure_node: GraphNode,
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
    success_node: GraphNode,
    failure_node: GraphNode,
) -> Tuple[str, Callable[[AgentGuardrailsGraphState], Any]]:
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
    success_node: GraphNode,
    failure_node: GraphNode,
) -> Tuple[str, Callable[[AgentGuardrailsGraphState], Any]]:
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
