from __future__ import annotations

import json
import re
from typing import Any, Dict, Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command, interrupt
from uipath.platform.common import CreateEscalation
from uipath.platform.guardrails import (
    BaseGuardrail,
    GuardrailScope,
)
from uipath.runtime.errors import UiPathErrorCode

from ...exceptions import AgentTerminationException
from ..guardrail_nodes import _message_text
from ..types import AgentGuardrailsGraphState, ExecutionStage
from .base_action import GuardrailAction, GuardrailActionNode


class EscalateAction(GuardrailAction):
    """Node-producing action that inserts a HITL interruption node into the graph.

    The returned node creates a human-in-the-loop interruption that suspends execution
    and waits for human review. When execution resumes, if the escalation was approved,
    the flow continues with the reviewed content; otherwise, an error is raised.
    """

    def __init__(
        self,
        app_name: str,
        app_folder_path: str,
        version: int,
        assignee: str,
    ):
        self.app_name = app_name
        self.app_folder_path = app_folder_path
        self.version = version
        self.assignee = assignee

    def action_node(
        self,
        *,
        guardrail: BaseGuardrail,
        scope: GuardrailScope,
        execution_stage: ExecutionStage,
    ) -> GuardrailActionNode:
        node_name = _get_node_name(execution_stage, guardrail, scope)

        async def _node(
            state: AgentGuardrailsGraphState,
        ) -> Dict[str, Any] | Command[Any]:
            input = _extract_escalation_content(state, scope, execution_stage)
            escalation_field = _execution_stage_to_escalation_field(execution_stage)

            data = {
                "GuardrailName": guardrail.name,
                "GuardrailDescription": guardrail.description,
                "Component": scope.name.lower(),
                "ExecutionStage": _execution_stage_to_string(execution_stage),
                "GuardrailResult": state.guardrail_validation_result,
                escalation_field: input,
            }

            escalation_result = interrupt(
                CreateEscalation(
                    app_name=self.app_name,
                    app_folder_path=self.app_folder_path,
                    title=self.app_name,
                    data=data,
                    assignee=self.assignee,
                )
            )

            if escalation_result.action == "Approve":
                return _process_escalation_response(
                    state, escalation_result.data, scope, execution_stage
                )

            raise AgentTerminationException(
                code=UiPathErrorCode.EXECUTION_ERROR,
                title="Escalation rejected",
                detail=f"Action was rejected after reviewing the task created by guardrail [{guardrail.name}]. Please contact your administrator.",
            )

        return node_name, _node


def _get_node_name(
    execution_stage: ExecutionStage, guardrail: BaseGuardrail, scope: GuardrailScope
) -> str:
    sanitized = re.sub(r"\W+", "_", guardrail.name).strip("_").lower()
    node_name = f"{sanitized}_hitl_{execution_stage.name.lower()}_{scope.lower()}"
    return node_name


def _execution_stage_to_string(
    execution_stage: ExecutionStage,
) -> Literal["PreExecution", "PostExecution"]:
    """Convert ExecutionStage enum to string literal.

    Args:
        execution_stage: The execution stage enum.

    Returns:
        "PreExecution" for PRE_EXECUTION, "PostExecution" for POST_EXECUTION.
    """
    if execution_stage == ExecutionStage.PRE_EXECUTION:
        return "PreExecution"
    return "PostExecution"


def _process_escalation_response(
    state: AgentGuardrailsGraphState,
    escalation_result: Dict[str, Any],
    scope: GuardrailScope,
    execution_stage: ExecutionStage,
) -> Dict[str, Any] | Command[Any]:
    """Process escalation response and update state based on guardrail scope.

    Args:
        state: The current agent graph state.
        escalation_result: The result from the escalation interrupt.
        scope: The guardrail scope (LLM/AGENT/TOOL).
        execution_stage: The hook type ("PreExecution" or "PostExecution").

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
            "ReviewedInputs"
            if execution_stage == ExecutionStage.PRE_EXECUTION
            else "ReviewedOutputs"
        )

        msgs = state.messages.copy()
        if not msgs or reviewed_field not in escalation_result:
            return {}

        last_message = msgs[-1]

        if execution_stage == ExecutionStage.PRE_EXECUTION:
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

            # For AI messages, process tool calls if present
            if isinstance(last_message, AIMessage):
                ai_message: AIMessage = last_message
                content_index = 0

                if ai_message.tool_calls:
                    tool_calls = list(ai_message.tool_calls)
                    for tool_call in tool_calls:
                        args = tool_call["args"]
                        if (
                            isinstance(args, dict)
                            and "content" in args
                            and args["content"] is not None
                        ):
                            if content_index < len(content_list):
                                updated_content = json.loads(
                                    content_list[content_index]
                                )
                                args["content"] = updated_content
                                tool_call["args"] = args
                                content_index += 1
                    ai_message.tool_calls = tool_calls

                if len(content_list) > content_index:
                    ai_message.content = content_list[-1]
            else:
                # Fallback for other message types
                if content_list:
                    last_message.content = content_list[-1]

        return Command[Any](update={"messages": msgs})
    except Exception as e:
        raise AgentTerminationException(
            code=UiPathErrorCode.EXECUTION_ERROR,
            title="Escalation rejected",
            detail=str(e),
        ) from e


def _extract_escalation_content(
    state: AgentGuardrailsGraphState,
    scope: GuardrailScope,
    execution_stage: ExecutionStage,
) -> str | list[str | Dict[str, Any]]:
    """Extract escalation content from state based on guardrail scope and execution stage.

    Args:
        state: The current agent graph state.
        scope: The guardrail scope (LLM/AGENT/TOOL).
        execution_stage: The execution stage enum.

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
            detail="No messages in state",
        )

    last_message = state.messages[-1]
    if execution_stage == ExecutionStage.PRE_EXECUTION:
        if isinstance(last_message, ToolMessage):
            return last_message.content

        content = _message_text(last_message)
        return json.dumps(content) if content else ""

    # For AI messages, process tool calls if present
    if isinstance(last_message, AIMessage):
        ai_message: AIMessage = last_message
        content_list: list[str] = []

        if ai_message.tool_calls:
            for tool_call in ai_message.tool_calls:
                args = tool_call["args"]
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

    # Fallback for other message types
    return _message_text(last_message)


def _execution_stage_to_escalation_field(
    execution_stage: ExecutionStage,
) -> str:
    """Convert execution stage to escalation data field name.

    Args:
        execution_stage: The execution stage enum.

    Returns:
        "Inputs" for PRE_EXECUTION, "Outputs" for POST_EXECUTION.
    """
    return "Inputs" if execution_stage == ExecutionStage.PRE_EXECUTION else "Outputs"
