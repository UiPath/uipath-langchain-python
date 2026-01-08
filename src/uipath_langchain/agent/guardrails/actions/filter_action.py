import re
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.types import Command
from uipath.core.guardrails.guardrails import FieldReference, FieldSource
from uipath.platform.guardrails import BaseGuardrail, GuardrailScope
from uipath.runtime.errors import UiPathErrorCategory, UiPathErrorCode

from uipath_langchain.agent.guardrails.types import ExecutionStage

from ...exceptions import AgentTerminationException
from ...react.types import AgentGuardrailsGraphState
from ...react.utils import extract_tool_call_from_state
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

    tool_call_id = getattr(state, "tool_call_id", None)
    tool_call, message = extract_tool_call_from_state(
        state, tool_name, tool_call_id, return_message=True
    )

    if tool_call is None:
        return {}

    args = tool_call["args"]
    if not args or not isinstance(args, dict):
        return {}

    # Filter out the specified input fields
    filtered_args = args.copy()
    modified = False

    for field_ref in fields_to_filter:
        # Only filter input fields
        if field_ref.source == FieldSource.INPUT and field_ref.path in filtered_args:
            del filtered_args[field_ref.path]
            modified = True

        if modified:
            tool_call["args"] = filtered_args
            message.tool_calls = [
                tool_call if tool_call["id"] == tc["id"] else tc
                for tc in message.tool_calls
            ]

        return Command(update={"messages": [message]})

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
