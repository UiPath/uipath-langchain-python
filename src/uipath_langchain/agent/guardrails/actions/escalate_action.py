from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, Literal, cast

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, ToolMessage
from langgraph.types import Command, interrupt
from uipath.platform.common import CreateEscalation
from uipath.platform.guardrails import (
    BaseGuardrail,
    GuardrailScope,
)
from uipath.runtime.errors import UiPathErrorCode

from ...exceptions import AgentTerminationException
from ...react.types import AgentGuardrailsGraphState
from ..types import ExecutionStage
from ..utils import _extract_tool_args_from_message, get_message_content
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
            # Validate message count based on execution stage
            _validate_message_count(state, execution_stage)

            # Build base data dictionary with common fields
            data: Dict[str, Any] = {
                "GuardrailName": guardrail.name,
                "GuardrailDescription": guardrail.description,
                "Component": scope.name.lower(),
                "ExecutionStage": _execution_stage_to_string(execution_stage),
                "GuardrailResult": state.guardrail_validation_result,
            }

            # Add stage-specific fields
            if execution_stage == ExecutionStage.PRE_EXECUTION:
                # PRE_EXECUTION: Only Inputs field from last message
                input_content = _extract_escalation_content(
                    state.messages[-1],
                    state,
                    scope,
                    execution_stage,
                    guarded_component_name,
                )
                data["Inputs"] = input_content
            else:  # POST_EXECUTION
                if scope == GuardrailScope.AGENT:
                    input_message = state.messages[1]
                else:
                    input_message = state.messages[-2]
                input_content = _extract_escalation_content(
                    input_message,
                    state,
                    scope,
                    ExecutionStage.PRE_EXECUTION,
                    guarded_component_name,
                )

                # Extract Outputs from last message using POST_EXECUTION logic
                output_content = _extract_escalation_content(
                    state.messages[-1],
                    state,
                    scope,
                    execution_stage,
                    guarded_component_name,
                )

                data["Inputs"] = input_content
                data["Outputs"] = output_content

            escalation_result = interrupt(
                CreateEscalation(
                    app_name=self.app_name,
                    app_folder_path=self.app_folder_path,
                    title="Agents Guardrail Task",
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
                detail=f"Please contact your administrator. Action was rejected after reviewing the task created by guardrail [{guardrail.name}], with reason: {escalation_result.data['Reason']}",
            )

        return node_name, _node


def _validate_message_count(
    state: AgentGuardrailsGraphState,
    execution_stage: ExecutionStage,
) -> None:
    """Validate that state has the required number of messages for the execution stage.

    Args:
        state: The current agent graph state.
        execution_stage: The execution stage (PRE_EXECUTION or POST_EXECUTION).

    Raises:
        AgentTerminationException: If the state doesn't have enough messages.
    """
    required_messages = 1 if execution_stage == ExecutionStage.PRE_EXECUTION else 2
    actual_messages = len(state.messages)

    if actual_messages < required_messages:
        stage_name = (
            "PRE_EXECUTION"
            if execution_stage == ExecutionStage.PRE_EXECUTION
            else "POST_EXECUTION"
        )
        detail = f"{stage_name} requires at least {required_messages} message{'s' if required_messages > 1 else ''} in state, but found {actual_messages}."
        if execution_stage == ExecutionStage.POST_EXECUTION:
            detail += " Cannot extract Inputs from previous message."

        raise AgentTerminationException(
            code=UiPathErrorCode.EXECUTION_ERROR,
            title=f"Invalid state for {stage_name}",
            detail=detail,
        )


def _get_node_name(
    execution_stage: ExecutionStage, guardrail: BaseGuardrail, scope: GuardrailScope
) -> str:
    raw_node_name = f"{scope.name}_{execution_stage.name}_{guardrail.name}_hitl"
    node_name = re.sub(r"\W+", "_", raw_node_name.lower()).strip("_")
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
        Command updates for the state (e.g., updating messages / tool calls / agent_result),
        or an empty dict if no update is needed.
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
            return _process_agent_escalation_response(
                state, escalation_result, execution_stage
            )


def _process_agent_escalation_response(
    state: AgentGuardrailsGraphState,
    escalation_result: Dict[str, Any],
    execution_stage: ExecutionStage,
) -> Dict[str, Any] | Command[Any]:
    """Process escalation response for AGENT scope guardrails.

    For AGENT scope:
    - PRE_EXECUTION: updates the last message content using `ReviewedInputs`
    - POST_EXECUTION: updates `agent_result` using `ReviewedOutputs`

    Args:
        state: The current agent graph state.
        escalation_result: The result from the escalation interrupt containing reviewed inputs/outputs.
        execution_stage: The execution stage (PRE_EXECUTION or POST_EXECUTION).

    Returns:
        Command to update state, or empty dict if no updates are needed.

    Raises:
        AgentTerminationException: If escalation response processing fails.
    """
    try:
        reviewed_field = get_reviewed_field_name(execution_stage)
        if reviewed_field not in escalation_result:
            return {}

        reviewed_value = escalation_result.get(reviewed_field)
        if not reviewed_value:
            return {}

        try:
            parsed = json.loads(reviewed_value)
        except json.JSONDecodeError:
            parsed = reviewed_value

        if execution_stage == ExecutionStage.PRE_EXECUTION:
            msgs = state.messages.copy()
            if not msgs:
                return {}
            msgs[-1].content = parsed
            return Command(update={"messages": msgs})

        # POST_EXECUTION: update agent_result
        return Command(update={"agent_result": parsed})
    except Exception as e:
        raise AgentTerminationException(
            code=UiPathErrorCode.EXECUTION_ERROR,
            title="Escalation rejected",
            detail=str(e),
        ) from e


def get_reviewed_field_name(execution_stage):
    return (
        "ReviewedInputs"
        if execution_stage == ExecutionStage.PRE_EXECUTION
        else "ReviewedOutputs"
    )


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
        reviewed_field = get_reviewed_field_name(execution_stage)

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

            reviewed_tool_calls_list = json.loads(reviewed_outputs_json)
            if not reviewed_tool_calls_list:
                return {}

            # Track if tool calls were successfully processed
            tool_calls_processed = False

            # For AI messages, process tool calls if present
            if isinstance(last_message, AIMessage):
                ai_message: AIMessage = last_message

                if ai_message.tool_calls and isinstance(reviewed_tool_calls_list, list):
                    tool_calls = list(ai_message.tool_calls)

                    # Create a name-to-args mapping from reviewed tool call data
                    reviewed_tool_calls_map = {}
                    for reviewed_data in reviewed_tool_calls_list:
                        if (
                            isinstance(reviewed_data, dict)
                            and "name" in reviewed_data
                            and "args" in reviewed_data
                        ):
                            reviewed_tool_calls_map[reviewed_data["name"]] = (
                                reviewed_data["args"]
                            )

                    # Update tool calls with reviewed args by matching name
                    if reviewed_tool_calls_map:
                        for tool_call in tool_calls:
                            tool_name = (
                                tool_call.get("name")
                                if isinstance(tool_call, dict)
                                else getattr(tool_call, "name", None)
                            )
                            if tool_name and tool_name in reviewed_tool_calls_map:
                                if isinstance(tool_call, dict):
                                    tool_call["args"] = reviewed_tool_calls_map[
                                        tool_name
                                    ]
                                else:
                                    tool_call.args = reviewed_tool_calls_map[tool_name]

                        ai_message.tool_calls = tool_calls
                        tool_calls_processed = True

            # Fallback: update message content if tool_calls weren't processed
            if not tool_calls_processed:
                last_message.content = reviewed_outputs_json

        return Command(update={"messages": msgs})
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
        reviewed_field = get_reviewed_field_name(execution_stage)

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
                    call_name = (
                        tool_call.get("name")
                        if isinstance(tool_call, dict)
                        else getattr(tool_call, "name", None)
                    )
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

        return Command(update={"messages": msgs})
    except Exception as e:
        raise AgentTerminationException(
            code=UiPathErrorCode.EXECUTION_ERROR,
            title="Escalation rejected",
            detail=str(e),
        ) from e


def _extract_escalation_content(
    message: BaseMessage,
    state: AgentGuardrailsGraphState,
    scope: GuardrailScope,
    execution_stage: ExecutionStage,
    guarded_node_name: str,
) -> str | list[str | Dict[str, Any]]:
    """Extract escalation content from a message based on guardrail scope and execution stage.

    Args:
        message: The message to extract content from.
        scope: The guardrail scope (LLM/AGENT/TOOL).
        execution_stage: The execution stage (PRE_EXECUTION or POST_EXECUTION).
        guarded_node_name: Name of the guarded component.

    Returns:
        str or list[str | Dict[str, Any]]: For LLM scope, returns JSON string or list with message/tool call content.
        For AGENT scope, returns empty string. For TOOL scope, returns JSON string or list with tool-specific content.
    """
    match scope:
        case GuardrailScope.LLM:
            return _extract_llm_escalation_content(message, execution_stage)
        case GuardrailScope.AGENT:
            return _extract_agent_escalation_content(message, state, execution_stage)
        case GuardrailScope.TOOL:
            return _extract_tool_escalation_content(
                message, execution_stage, guarded_node_name
            )


def _extract_agent_escalation_content(
    message: BaseMessage,
    state: AgentGuardrailsGraphState,
    execution_stage: ExecutionStage,
) -> str | list[str | Dict[str, Any]]:
    """Extract escalation content for AGENT scope guardrails.

    Args:
        message: The message used to extract the agent input content.
        state: The current agent guardrails graph state. Used to read `agent_result` for POST_EXECUTION.
        execution_stage: PRE_EXECUTION or POST_EXECUTION.

    Returns:
        - PRE_EXECUTION: the agent input string (from message content).
        - POST_EXECUTION: a JSON-serialized representation of `state.agent_result`.
    """
    if execution_stage == ExecutionStage.PRE_EXECUTION:
        return get_message_content(cast(AnyMessage, message))

    output_content = state.agent_result or ""
    return json.dumps(output_content)


def _extract_llm_escalation_content(
    message: BaseMessage, execution_stage: ExecutionStage
) -> str | list[str | Dict[str, Any]]:
    """Extract escalation content for LLM scope guardrails.

    Args:
        message: The message to extract content from.
        execution_stage: The execution stage (PRE_EXECUTION or POST_EXECUTION).

    Returns:
        str or list[str | Dict[str, Any]]: For PreExecution, returns JSON string with message content or empty string.
        For PostExecution, returns JSON string (array) with tool call content and message content.
        Returns empty string if no content found.
    """
    if execution_stage == ExecutionStage.PRE_EXECUTION:
        if isinstance(message, ToolMessage):
            return message.content

        return get_message_content(cast(AnyMessage, message))

    # For AI messages, process tool calls if present
    if isinstance(message, AIMessage):
        ai_message: AIMessage = message

        if ai_message.tool_calls:
            content_list: list[Dict[str, Any]] = []
            for tool_call in ai_message.tool_calls:
                tool_call_data = {
                    "name": tool_call.get("name"),
                    "args": tool_call.get("args"),
                }
                content_list.append(tool_call_data)
            return json.dumps(content_list)

    # Fallback for other message types
    return get_message_content(cast(AnyMessage, message))


def _extract_tool_escalation_content(
    message: BaseMessage, execution_stage: ExecutionStage, tool_name: str
) -> str | list[str | Dict[str, Any]]:
    """Extract escalation content for TOOL scope guardrails.

    Args:
        message: The message to extract content from.
        execution_stage: The execution stage (PRE_EXECUTION or POST_EXECUTION).
        tool_name: Optional tool name to filter tool calls. If provided, only extracts args for matching tool.

    Returns:
        str or list[str | Dict[str, Any]]: For PreExecution, returns JSON string with tool call arguments
        for the specified tool name, or empty string if not found. For PostExecution, returns string with
        tool message content, or empty string if message type doesn't match.
    """
    if execution_stage == ExecutionStage.PRE_EXECUTION:
        args = _extract_tool_args_from_message(cast(AnyMessage, message), tool_name)
        if args:
            return json.dumps(args)
        return ""
    else:
        if not isinstance(message, ToolMessage):
            return ""
        content = message.content

        # If content is already dict/list, serialize to JSON
        if isinstance(content, (dict, list)):
            return json.dumps(content)

        # If content is a string that looks like a Python literal, convert to JSON
        if isinstance(content, str):
            try:
                # Try to parse as Python literal and convert to JSON
                parsed_content = ast.literal_eval(content)
                return json.dumps(parsed_content)
            except (ValueError, SyntaxError):
                # If parsing fails, return as-is
                pass

        return content


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
