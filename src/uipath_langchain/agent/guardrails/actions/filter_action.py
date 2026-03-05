import json
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command
from uipath.core.guardrails.guardrails import FieldReference, FieldSource
from uipath.platform.guardrails import BaseGuardrail, GuardrailScope
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.guardrails.types import ExecutionStage
from uipath_langchain.agent.react.utils import (
    extract_current_tool_call_index,
    find_latest_ai_message,
)

from ...exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from ...messages.message_utils import replace_tool_calls
from ...react.types import AgentGuardrailsGraphState
from .base_action import GuardrailAction, GuardrailActionNode


@dataclass
class FilterResult:
    """Result of a filter operation."""

    command: dict[str, Any] | Command[Any]
    updated_input: dict[str, Any] | None = None
    updated_output: dict[str, Any] | None = None


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

    @property
    def action_type(self) -> str:
        return "Filter"

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

        metadata: dict[str, Any] = {
            "guardrail": guardrail,
            "scope": scope,
            "execution_stage": execution_stage,
            "excluded_fields": [field.path for field in self.fields]
            if self.fields
            else [],
            "updated_data": {"input": None, "output": None},
        }

        async def _node(
            _state: AgentGuardrailsGraphState,
        ) -> dict[str, Any] | Command[Any]:
            if scope == GuardrailScope.TOOL:
                result = _filter_tool_fields(
                    _state,
                    self.fields,
                    execution_stage,
                    guarded_component_name,
                )
                # Update metadata with filter results
                metadata["updated_data"]["input"] = result.updated_input
                metadata["updated_data"]["output"] = result.updated_output
                return result.command

            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.TERMINATION_GUARDRAIL_ERROR,
                title="Guardrail filter action not supported",
                detail=f"FilterAction is not supported for scope [{scope.name}] at this time.",
                category=UiPathErrorCategory.SYSTEM,
            )

        _node.__metadata__ = metadata  # type: ignore[attr-defined]

        return node_name, _node


def _filter_tool_fields(
    state: AgentGuardrailsGraphState,
    fields_to_filter: list[FieldReference],
    execution_stage: ExecutionStage,
    tool_name: str,
) -> FilterResult:
    """Filter specified fields from tool call arguments or tool output.

    The filter action filters fields based on the execution stage:
    - PRE_EXECUTION: Only input fields are filtered
    - POST_EXECUTION: Only output fields are filtered

    Args:
        state: The current agent graph state.
        fields_to_filter: List of FieldReference objects specifying which fields to filter.
        execution_stage: The execution stage (PRE_EXECUTION or POST_EXECUTION).
        tool_name: Name of the tool to filter.

    Returns:
        FilterResult containing the command and updated input/output data.

    Raises:
        AgentRuntimeError: If filtering fails.
    """
    try:
        if not fields_to_filter:
            return FilterResult(command={})

        if execution_stage == ExecutionStage.PRE_EXECUTION:
            return _filter_tool_input_fields(state, fields_to_filter, tool_name)
        else:
            return _filter_tool_output_fields(state, fields_to_filter)

    except Exception as e:
        raise AgentRuntimeError(
            code=AgentRuntimeErrorCode.TERMINATION_GUARDRAIL_ERROR,
            title="Filter action failed",
            detail=f"Failed to filter tool fields: {str(e)}",
            category=UiPathErrorCategory.SYSTEM,
        ) from e


def _filter_tool_input_fields(
    state: AgentGuardrailsGraphState,
    fields_to_filter: list[FieldReference],
    tool_name: str,
) -> FilterResult:
    """Filter specified input fields from tool call arguments (PRE_EXECUTION only).

    This function is called at PRE_EXECUTION to filter input fields from tool call arguments
    before the tool is executed.

    Args:
        state: The current agent graph state.
        fields_to_filter: List of FieldReference objects specifying which fields to filter.
        tool_name: Name of the tool to filter.

    Returns:
        FilterResult with command to update messages and updated_input data.
    """
    # Check if there are any input fields to filter
    has_input_fields = any(
        field_ref.source == FieldSource.INPUT for field_ref in fields_to_filter
    )

    if not has_input_fields:
        return FilterResult(command={})

    msgs = state.messages.copy()
    if not msgs:
        return FilterResult(command={})

    # Find the AIMessage with tool calls
    # At PRE_EXECUTION, this is always the last message
    ai_message = find_latest_ai_message(msgs)
    if ai_message is None or not ai_message.tool_calls:
        return FilterResult(command={})

    # Find and filter the tool call with matching name
    # Type assertion: we know ai_message is AIMessage from the check above
    assert isinstance(ai_message, AIMessage)
    tool_calls = list(ai_message.tool_calls)
    modified = False
    filtered_args: dict[str, Any] = {}

    current_tool_call_index = extract_current_tool_call_index(msgs, tool_name)
    if current_tool_call_index is None:
        return FilterResult(command={})

    tool_call = tool_calls[current_tool_call_index]

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
    else:
        raise AgentRuntimeError(
            code=AgentRuntimeErrorCode.STATE_ERROR,
            title=f"Tool call name [{call_name}] does not match expected tool name [{tool_name}].",
            detail=f"Expected tool call for '{tool_name}' but found '{call_name}' at the current index.",
            category=UiPathErrorCategory.SYSTEM,
        )

    if modified:
        ai_message = replace_tool_calls(ai_message, tool_calls)
        return FilterResult(
            command=Command(update={"messages": [ai_message]}),
            updated_input=filtered_args,
        )

    return FilterResult(command={})


def _filter_tool_output_fields(
    state: AgentGuardrailsGraphState,
    fields_to_filter: list[FieldReference],
) -> FilterResult:
    """Filter specified output fields from tool output (POST_EXECUTION only).

    This function is called at POST_EXECUTION to filter output fields from tool results
    after the tool has been executed.

    Args:
        state: The current agent graph state.
        fields_to_filter: List of FieldReference objects specifying which fields to filter.

    Returns:
        FilterResult with command to update messages and updated_output data.
    """
    import ast

    # Check if there are any output fields to filter
    has_output_fields = any(
        field_ref.source == FieldSource.OUTPUT for field_ref in fields_to_filter
    )

    if not has_output_fields:
        return FilterResult(command={})

    msgs = state.messages.copy()
    if not msgs:
        return FilterResult(command={})

    last_message = msgs[-1]
    if not isinstance(last_message, ToolMessage):
        return FilterResult(command={})

    content = last_message.content
    if not content:
        return FilterResult(command={})

    # Try to parse the content as JSON or dict
    try:
        if isinstance(content, dict):
            output_data = content
        elif isinstance(content, str):
            try:
                output_data = json.loads(content)
            except json.JSONDecodeError:
                # Try to parse as Python literal (dict representation)
                try:
                    output_data = ast.literal_eval(content)
                    if not isinstance(output_data, dict):
                        return FilterResult(command={})
                except (ValueError, SyntaxError):
                    return FilterResult(command={})
        else:
            # Content is not JSON-parseable, can't filter specific fields
            return FilterResult(command={})
    except Exception:
        return FilterResult(command={})

    if not isinstance(output_data, dict):
        return FilterResult(command={})

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
        return FilterResult(
            command=Command(update={"messages": msgs}),
            updated_output=filtered_output,
        )

    return FilterResult(command={})
