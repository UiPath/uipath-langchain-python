from __future__ import annotations

import json
import re
from typing import Any, Dict, Literal

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
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
        """Initialize EscalateAction with escalation app configuration.

        Args:
            app_name: Name of the escalation app.
            app_folder_path: Folder path where the escalation app is located.
            version: Version of the escalation app.
            assignee: User or role assigned to handle the escalation.
        """
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
        guarded_component_name: str,
    ) -> GuardrailActionNode:
        """Create a HITL escalation node for the guardrail.

        Args:
            guardrail: The guardrail that triggered this escalation action.
            scope: The guardrail scope (LLM/AGENT/TOOL).
            execution_stage: The execution stage (PRE_EXECUTION or POST_EXECUTION).

        Returns:
            A tuple of (node_name, node_function) where the node function triggers
            a HITL interruption and processes the escalation response.
        """
        node_name = _get_node_name(execution_stage, guardrail, scope)

        async def _node(
            state: AgentGuardrailsGraphState,
        ) -> Dict[str, Any] | Command[Any]:
            input = _extract_escalation_content(
                state, scope, execution_stage, guarded_component_name
            )
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
                    state,
                    escalation_result.data,
                    scope,
                    execution_stage,
                    guarded_component_name,
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


def _process_escalation_response(
    state: AgentGuardrailsGraphState,
    escalation_result: Dict[str, Any],
    scope: GuardrailScope,
    execution_stage: ExecutionStage,
    guarded_node_name: str,
) -> Dict[str, Any] | Command[Any]:
    """Process escalation response and route to appropriate handler based on scope.

    Args:
        state: The current agent graph state.
        escalation_result: The result from the escalation interrupt containing reviewed inputs/outputs.
        scope: The guardrail scope (LLM/AGENT/TOOL).
        execution_stage: The execution stage (PRE_EXECUTION or POST_EXECUTION).

    Returns:
        For LLM/TOOL scope: Command to update messages with reviewed inputs/outputs, or empty dict.
        For AGENT scope: Empty dict (no message alteration).
    """
    match scope:
        case GuardrailScope.LLM:
            return _process_llm_escalation_response(
                state, escalation_result, execution_stage
            )
        case GuardrailScope.TOOL:
            return _process_tool_escalation_response(
                state, escalation_result, execution_stage, guarded_node_name
            )
        case GuardrailScope.AGENT:
            return {}


def _process_llm_escalation_response(
    state: AgentGuardrailsGraphState,
    escalation_result: Dict[str, Any],
    execution_stage: ExecutionStage,
) -> Dict[str, Any] | Command[Any]:
    """Process escalation response for LLM scope guardrails.

    Updates message content or tool calls based on reviewed inputs/outputs from escalation.

    Args:
        state: The current agent graph state.
        escalation_result: The result from the escalation interrupt containing reviewed inputs/outputs.
        execution_stage: The execution stage (PRE_EXECUTION or POST_EXECUTION).

    Returns:
        Command to update messages with reviewed inputs/outputs, or empty dict if no updates needed.

    Raises:
        AgentTerminationException: If escalation response processing fails.
    """
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


def _process_tool_escalation_response(
    state: AgentGuardrailsGraphState,
    escalation_result: Dict[str, Any],
    execution_stage: ExecutionStage,
    tool_name: str,
) -> Dict[str, Any] | Command[Any]:
    """Process escalation response for TOOL scope guardrails.

    Updates the tool call arguments (PreExecution) or tool message content (PostExecution)
    for the specific tool matching the tool_name. For PreExecution, finds the tool call
    with the matching name and updates only that tool call's args with the reviewed dict.
    For PostExecution, updates the tool message content.

    Args:
        state: The current agent graph state.
        escalation_result: The result from the escalation interrupt containing reviewed inputs/outputs.
        execution_stage: The execution stage (PRE_EXECUTION or POST_EXECUTION).
        tool_name: Name of the tool to update. Only the tool call matching this name will be updated.

    Returns:
        Command to update messages with reviewed tool call args or content, or empty dict if no updates needed.

    Raises:
        AgentTerminationException: If escalation response processing fails.
    """
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
            if not isinstance(last_message, AIMessage):
                return {}

            # Get reviewed tool calls args from escalation result
            reviewed_inputs_json = escalation_result[reviewed_field]
            if not reviewed_inputs_json:
                return {}

            reviewed_tool_calls_args = json.loads(reviewed_inputs_json)
            if not isinstance(reviewed_tool_calls_args, dict):
                return {}

            # Find and update only the tool call with matching name
            if last_message.tool_calls:
                tool_calls = list(last_message.tool_calls)
                for tool_call in tool_calls:
                    call_name = extract_tool_name(tool_call)
                    if call_name == tool_name:
                        # Update args for the matching tool call
                        if isinstance(reviewed_tool_calls_args, dict):
                            if isinstance(tool_call, dict):
                                tool_call["args"] = reviewed_tool_calls_args
                            else:
                                tool_call.args = reviewed_tool_calls_args
                        break
                last_message.tool_calls = tool_calls
        else:
            if not isinstance(last_message, ToolMessage):
                return {}

            # PostExecution: update tool message content
            reviewed_outputs_json = escalation_result[reviewed_field]
            if reviewed_outputs_json:
                last_message.content = reviewed_outputs_json

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
    guarded_node_name: str,
) -> str | list[str | Dict[str, Any]]:
    """Extract escalation content from state based on guardrail scope and execution stage.

    Args:
        state: The current agent graph state.
        scope: The guardrail scope (LLM/AGENT/TOOL).
        execution_stage: The execution stage (PRE_EXECUTION or POST_EXECUTION).

    Returns:
        str or list[str | Dict[str, Any]]: For LLM scope, returns JSON string or list with message/tool call content.
        For AGENT scope, returns empty string. For TOOL scope, returns JSON string or list with tool-specific content.

    Raises:
        AgentTerminationException: If no messages are found in state.
    """
    if not state.messages:
        raise AgentTerminationException(
            code=UiPathErrorCode.EXECUTION_ERROR,
            title="Invalid state message",
            detail="No message found into agent state",
        )

    match scope:
        case GuardrailScope.LLM:
            return _extract_llm_escalation_content(state, execution_stage)
        case GuardrailScope.AGENT:
            return _extract_agent_escalation_content(state, execution_stage)
        case GuardrailScope.TOOL:
            return _extract_tool_escalation_content(
                state, execution_stage, guarded_node_name
            )


def _extract_llm_escalation_content(
    state: AgentGuardrailsGraphState, execution_stage: ExecutionStage
) -> str | list[str | Dict[str, Any]]:
    """Extract escalation content for LLM scope guardrails.

    Args:
        state: The current agent graph state.
        execution_stage: The execution stage (PRE_EXECUTION or POST_EXECUTION).

    Returns:
        str or list[str | Dict[str, Any]]: For PreExecution, returns JSON string with message content or empty string.
        For PostExecution, returns JSON string (array) with tool call content and message content.
        Returns empty string if no content found.
    """
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


def _extract_agent_escalation_content(
    state: AgentGuardrailsGraphState, execution_stage: ExecutionStage
) -> str | list[str | Dict[str, Any]]:
    """Extract escalation content for AGENT scope guardrails.

    Args:
        state: The current agent graph state.
        execution_stage: The execution stage (PRE_EXECUTION or POST_EXECUTION).

    Returns:
        str: Empty string (AGENT scope guardrails do not extract escalation content).
    """
    return ""


def _extract_tool_escalation_content(
    state: AgentGuardrailsGraphState, execution_stage: ExecutionStage, tool_name: str
) -> str | list[str | Dict[str, Any]]:
    """Extract escalation content for TOOL scope guardrails.

    Args:
        state: The current agent graph state.
        execution_stage: The execution stage (PRE_EXECUTION or POST_EXECUTION).
        tool_name: Optional tool name to filter tool calls. If provided, only extracts args for matching tool.

    Returns:
        str or list[str | Dict[str, Any]]: For PreExecution, returns JSON string with tool call arguments
        for the specified tool name, or empty string if not found. For PostExecution, returns string with
        tool message content, or empty string if message type doesn't match.
    """
    last_message = state.messages[-1]
    if execution_stage == ExecutionStage.PRE_EXECUTION:
        if not isinstance(last_message, AIMessage):
            return ""
        if not last_message.tool_calls:
            return ""

        # Find the tool call with matching name
        for tool_call in last_message.tool_calls:
            call_name = extract_tool_name(tool_call)
            if call_name == tool_name:
                # Extract args from the matching tool call
                args = (
                    tool_call.get("args")
                    if isinstance(tool_call, dict)
                    else getattr(tool_call, "args", None)
                )
                if args is not None:
                    return json.dumps(args)
        return ""
    else:
        if not isinstance(last_message, ToolMessage):
            return ""
        return last_message.content


def extract_tool_name(tool_call: ToolCall) -> Any | None:
    return (
        tool_call.get("name")
        if isinstance(tool_call, dict)
        else getattr(tool_call, "name", None)
    )


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
