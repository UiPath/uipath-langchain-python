import re
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command
from uipath.core.guardrails.guardrails import FieldReference, FieldSource
from uipath.platform.guardrails import BaseGuardrail, GuardrailScope
from uipath.runtime.errors import UiPathErrorCategory, UiPathErrorCode

from uipath_langchain.agent.guardrails.types import ExecutionStage

from ...exceptions import AgentTerminationException
from ...react.types import AgentGuardrailsGraphState
from .base_action import GuardrailAction, GuardrailActionNode


class FilterAction(GuardrailAction):
    """Action that filters inputs/outputs on guardrail failure.

    For Tool scope, this action removes specified fields from tool call arguments.
    For AGENT and LLM scopes, this action raises an exception as it's not supported yet.
    """

    def __init__(self, fields: list[FieldReference] | None = None):
        """Initialize FilterAction with fields to filter.

        Args:
            fields: List of FieldReference objects specifying which fields to filter.
        """
        self.fields = fields or []

    def action_node(
        self,
        *,
        guardrail: BaseGuardrail,
        scope: GuardrailScope,
        execution_stage: ExecutionStage,
        guarded_component_name: str,
    ) -> GuardrailActionNode:
        """Create a guardrail action node that performs filtering.

        Args:
            guardrail: The guardrail responsible for the validation.
            scope: The scope in which the guardrail applies.
            execution_stage: Whether this runs before or after execution.
            guarded_component_name: Name of the guarded component.

        Returns:
            A tuple containing the node name and the async node callable.
        """
        raw_node_name = f"{scope.name}_{execution_stage.name}_{guardrail.name}_filter"
        node_name = re.sub(r"\W+", "_", raw_node_name.lower()).strip("_")

        async def _node(
            _state: AgentGuardrailsGraphState,
        ) -> dict[str, Any] | Command[Any]:
            if scope == GuardrailScope.TOOL:
                return _filter_tool_fields(
                    _state,
                    self.fields,
                    execution_stage,
                    guarded_component_name,
                    guardrail.name,
                )

            raise AgentTerminationException(
                code=UiPathErrorCode.EXECUTION_ERROR,
                title="Guardrail filter action not supported",
                detail=f"FilterAction is not supported for scope [{scope.name}] at this time.",
                category=UiPathErrorCategory.USER,
            )

        return node_name, _node


def _filter_tool_fields(
    state: AgentGuardrailsGraphState,
    fields_to_filter: list[FieldReference],
    execution_stage: ExecutionStage,
    tool_name: str,
    guardrail_name: str,
) -> dict[str, Any] | Command[Any]:
    """Filter specified fields from tool call arguments or tool output.

    The filter action filters fields based on the execution stage:
    - PRE_EXECUTION: Only input fields are filtered
    - POST_EXECUTION: Only output fields are filtered

    Args:
        state: The current agent graph state.
        fields_to_filter: List of FieldReference objects specifying which fields to filter.
        execution_stage: The execution stage (PRE_EXECUTION or POST_EXECUTION).
        tool_name: Name of the tool to filter.
        guardrail_name: Name of the guardrail for logging purposes.

    Returns:
        Command to update messages with filtered tool call args or output.

    Raises:
        AgentTerminationException: If filtering fails.
    """
    try:
        if not fields_to_filter:
            return {}

        if execution_stage == ExecutionStage.PRE_EXECUTION:
            return _filter_tool_input_fields(state, fields_to_filter, tool_name)
        else:
            return _filter_tool_output_fields(state, fields_to_filter)

    except Exception as e:
        raise AgentTerminationException(
            code=UiPathErrorCode.EXECUTION_ERROR,
            title="Filter action failed",
            detail=f"Failed to filter tool fields: {str(e)}",
            category=UiPathErrorCategory.USER,
        ) from e


def _filter_tool_input_fields(
    state: AgentGuardrailsGraphState,
    fields_to_filter: list[FieldReference],
    tool_name: str,
) -> dict[str, Any] | Command[Any]:
    """Filter specified input fields from tool call arguments (PRE_EXECUTION only).

    This function is called at PRE_EXECUTION to filter input fields from tool call arguments
    before the tool is executed.

    Args:
        state: The current agent graph state.
        fields_to_filter: List of FieldReference objects specifying which fields to filter.
        tool_name: Name of the tool to filter.

    Returns:
        Command to update messages with filtered tool call args, or empty dict if no input fields to filter.
    """
    # Check if there are any input fields to filter
    has_input_fields = any(
        field_ref.source == FieldSource.INPUT for field_ref in fields_to_filter
    )

    if not has_input_fields:
        return {}

    msgs = state.messages.copy()
    if not msgs:
        return {}

    # Find the AIMessage with tool calls
    # At PRE_EXECUTION, this is always the last message
    ai_message = None
    for i in range(len(msgs) - 1, -1, -1):
        msg = msgs[i]
        if isinstance(msg, AIMessage) and msg.tool_calls:
            ai_message = msg
            break

    if ai_message is None:
        return {}

    # Find and filter the tool call with matching name
    # Type assertion: we know ai_message is AIMessage from the check above
    assert isinstance(ai_message, AIMessage)
    tool_calls = list(ai_message.tool_calls)
    modified = False

    for tool_call in tool_calls:
        call_name = (
            tool_call.get("name")
            if isinstance(tool_call, dict)
            else getattr(tool_call, "name", None)
        )

        if call_name == tool_name:
            # Get the current args
            args = (
                tool_call.get("args")
                if isinstance(tool_call, dict)
                else getattr(tool_call, "args", None)
            )

            if args and isinstance(args, dict):
                # Filter out the specified input fields
                filtered_args = args.copy()
                for field_ref in fields_to_filter:
                    # Only filter input fields
                    if (
                        field_ref.source == FieldSource.INPUT
                        and field_ref.path in filtered_args
                    ):
                        del filtered_args[field_ref.path]
                        modified = True

                # Update the tool call with filtered args
                if isinstance(tool_call, dict):
                    tool_call["args"] = filtered_args
                else:
                    tool_call.args = filtered_args

            break

    if modified:
        ai_message.tool_calls = tool_calls
        return Command(update={"messages": msgs})

    return {}


def _filter_tool_output_fields(
    state: AgentGuardrailsGraphState,
    fields_to_filter: list[FieldReference],
) -> dict[str, Any] | Command[Any]:
    """Filter specified output fields from tool output (POST_EXECUTION only).

    This function is called at POST_EXECUTION to filter output fields from tool results
    after the tool has been executed.

    Args:
        state: The current agent graph state.
        fields_to_filter: List of FieldReference objects specifying which fields to filter.

    Returns:
        Command to update messages with filtered tool output, or empty dict if no output fields to filter.
    """
    # Check if there are any output fields to filter
    has_output_fields = any(
        field_ref.source == FieldSource.OUTPUT for field_ref in fields_to_filter
    )

    if not has_output_fields:
        return {}

    msgs = state.messages.copy()
    if not msgs:
        return {}

    last_message = msgs[-1]
    if not isinstance(last_message, ToolMessage):
        return {}

    # Parse the tool output content
    import json

    content = last_message.content
    if not content:
        return {}

    # Try to parse the content as JSON or dict
    try:
        if isinstance(content, dict):
            output_data = content
        elif isinstance(content, str):
            try:
                output_data = json.loads(content)
            except json.JSONDecodeError:
                # Try to parse as Python literal (dict representation)
                import ast

                try:
                    output_data = ast.literal_eval(content)
                    if not isinstance(output_data, dict):
                        return {}
                except (ValueError, SyntaxError):
                    return {}
        else:
            # Content is not JSON-parseable, can't filter specific fields
            return {}
    except Exception:
        return {}

    if not isinstance(output_data, dict):
        return {}

    # Filter out the specified fields
    filtered_output = output_data.copy()
    modified = False

    for field_ref in fields_to_filter:
        # Only filter output fields
        if field_ref.source == FieldSource.OUTPUT and field_ref.path in filtered_output:
            del filtered_output[field_ref.path]
            modified = True

    if modified:
        # Update the tool message content with filtered output
        last_message.content = json.dumps(filtered_output)
        return Command(update={"messages": msgs})

    return {}
