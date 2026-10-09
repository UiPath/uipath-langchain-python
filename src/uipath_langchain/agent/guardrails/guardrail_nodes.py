import asyncio
import json
import logging
import re
from typing import Any, Callable

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command
from uipath.core.guardrails import (
    DeterministicGuardrail,
    DeterministicGuardrailsService,
    GuardrailValidationResult,
    GuardrailValidationResultType,
)
from uipath.platform import UiPath
from uipath.platform.errors import EnrichedException
from uipath.platform.guardrails import (
    BaseGuardrail,
    BuiltInValidatorGuardrail,
    GuardrailAttachment,
    GuardrailScope,
    GuardrailTerminationMode,
)
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.guardrails.attachment_refs import (
    resolve_guardrail_attachments,
    resolve_referenced_attachments,
)
from uipath_langchain.agent.guardrails.types import ExecutionStage
from uipath_langchain.agent.guardrails.utils import (
    _extract_tool_args_from_message,
    _extract_tool_output_data,
    _extract_tools_args_from_message,
    get_message_content,
)
from uipath_langchain.agent.react.types import AgentGuardrailsGraphState
from uipath_langchain.agent.react.utils import (
    extract_current_tool_call_index,
    find_latest_ai_message,
)

from ..exceptions import AgentRuntimeError, AgentRuntimeErrorCode

logger = logging.getLogger(__name__)


def _evaluate_deterministic_guardrail(
    state: AgentGuardrailsGraphState,
    guardrail: DeterministicGuardrail,
    execution_stage: ExecutionStage,
    input_data_extractor: Callable[[AgentGuardrailsGraphState], dict[str, Any]],
    output_data_extractor: Callable[[AgentGuardrailsGraphState], dict[str, Any]] | None,
):
    """Evaluate deterministic guardrail.

    Args:
        state: The current agent graph state.
        guardrail: The deterministic guardrail to evaluate.
        execution_stage: The execution stage (PRE_EXECUTION or POST_EXECUTION).
        input_data_extractor: Function to extract input data from state.
        output_data_extractor: Function to extract output data from state (optional).

    Returns:
        The guardrail evaluation result.
    """
    service = DeterministicGuardrailsService()
    input_data = input_data_extractor(state)

    if execution_stage == ExecutionStage.PRE_EXECUTION:
        return service.evaluate_pre_deterministic_guardrail(
            input_data=input_data, guardrail=guardrail
        )
    else:  # POST_EXECUTION
        output_data = output_data_extractor(state) if output_data_extractor else {}
        return service.evaluate_post_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=guardrail,
        )


async def _evaluate_builtin_guardrail(
    guardrail: BuiltInValidatorGuardrail,
    text: str,
    attachments: list[GuardrailAttachment] | None = None,
    termination_mode: GuardrailTerminationMode | None = None,
):
    """Evaluate built-in validator guardrail.

    Args:
        guardrail: The built-in validator guardrail to evaluate.
        text: The payload text to validate.
        attachments: Resolved attachment references the validator may inspect.
        termination_mode: When the service may stop scanning; ``FAIL_FAST`` stops
            at the first violation.

    Returns:
        The guardrail evaluation result.
    """
    uipath = UiPath()
    try:
        return await asyncio.to_thread(
            uipath.guardrails.evaluate_guardrail,
            text,
            guardrail,
            attachments=attachments,
            termination_mode=termination_mode,
        )
    except EnrichedException as exc:
        # A 400 with attachments means the references were rejected; a file must never
        # fail the run, so evaluate the text alone.
        if not attachments or exc.status_code != 400:
            raise
        logger.warning(
            "Guardrail '%s' rejected the attachment references (HTTP 400); "
            "re-evaluating without attachments.",
            guardrail.name,
        )
        return await asyncio.to_thread(
            uipath.guardrails.evaluate_guardrail,
            text,
            guardrail,
            attachments=None,
            termination_mode=termination_mode,
        )


def _flagged_file_names(
    guardrail_result: GuardrailValidationResult,
    attachments: list[GuardrailAttachment] | None,
) -> list[str] | None:
    """Names of the sent attachments the guardrail service flagged, in the order sent."""
    flagged_ids = {
        attachment_id.lower()
        for attachment_id in guardrail_result.flagged_attachment_ids or []
    }
    names = [
        attachment.file_name
        for attachment in attachments or []
        if attachment.id.lower() in flagged_ids
    ]
    return names or None


def _create_validation_command(
    guardrail_result: GuardrailValidationResult,
    success_node: str,
    failure_node: str,
    attachments: list[GuardrailAttachment] | None = None,
) -> Command[Any]:
    """Create command based on validation result.

    ``guardrail_flagged_file_names`` is written on every outcome so that a value from
    an earlier guardrail never survives the inner-state merge.

    Args:
        guardrail_result: The guardrail evaluation result.
        success_node: Node to route to on validation pass.
        failure_node: Node to route to on validation fail.
        attachments: Attachment references sent with the evaluation.

    Returns:
        Command to update state and route to appropriate node.

    Raises:
        AgentRuntimeError: If the result is neither PASSED nor VALIDATION_FAILED.
    """
    span_id = getattr(guardrail_result, "span_id", None)
    flagged_file_names = _flagged_file_names(guardrail_result, attachments)

    if guardrail_result.result == GuardrailValidationResultType.PASSED:
        inner_state: dict[str, Any] = {
            "guardrail_validation_result": True,
            "guardrail_validation_details": guardrail_result.reason,
            "guardrail_flagged_file_names": flagged_file_names,
        }
        if span_id:
            inner_state["guardrail_span_id"] = span_id
        return Command(
            goto=success_node,
            update={"inner_state": inner_state},
        )

    if guardrail_result.result == GuardrailValidationResultType.VALIDATION_FAILED:
        inner_state = {
            "guardrail_validation_result": False,
            "guardrail_validation_details": guardrail_result.reason,
            "guardrail_flagged_file_names": flagged_file_names,
        }
        if span_id:
            inner_state["guardrail_span_id"] = span_id
        return Command(
            goto=failure_node,
            update={"inner_state": inner_state},
        )

    # For other results (FEATURE_DISABLED, ENTITLEMENTS_MISSING, etc.), interrupt execution
    raise AgentRuntimeError(
        code=AgentRuntimeErrorCode.TERMINATION_GUARDRAIL_VIOLATION,
        title="Guardrail validation error",
        detail=guardrail_result.reason
        or f"Guardrail validation returned unexpected result: {guardrail_result.result.value}",
        category=UiPathErrorCategory.DEPLOYMENT,
    )


AttachmentSource = Callable[[AgentGuardrailsGraphState], Any]
"""Reads, from the state, the payload whose file mentions a guardrail inspects."""


async def _resolve_attachments(
    state: AgentGuardrailsGraphState,
    guardrail: BuiltInValidatorGuardrail,
    attachment_source: AttachmentSource | None,
) -> list[GuardrailAttachment] | None:
    """Attachment references for one built-in guardrail evaluation.

    Without a source the guardrail judges the conversation so far, so it reads the run's
    whole attachment registry. With one it judges a single payload, so it reads only the
    files that payload mentions. ``None`` lets the backend fall back to the text; a
    source that raises gets that too. Never raises.
    """
    registry = state.inner_state.job_attachments
    if attachment_source is None:
        return await resolve_guardrail_attachments(registry, guardrail)
    try:
        source = attachment_source(state)
    except Exception:
        logger.warning(
            "Could not read the payload of guardrail '%s' for files; evaluating "
            "without them.",
            guardrail.name,
            exc_info=True,
        )
        return None
    return resolve_referenced_attachments(source, registry, guardrail)


def _create_guardrail_node(
    guardrail: BaseGuardrail,
    scope: GuardrailScope,
    execution_stage: ExecutionStage,
    payload_generator: Callable[[AgentGuardrailsGraphState], str],
    success_node: str,
    failure_node: str,
    input_data_extractor: Callable[[AgentGuardrailsGraphState], dict[str, Any]]
    | None = None,
    output_data_extractor: Callable[[AgentGuardrailsGraphState], dict[str, Any]]
    | None = None,
    tool_name: str | None = None,
    tool_type: str | None = None,
    termination_mode: GuardrailTerminationMode | None = None,
    attachment_source: AttachmentSource | None = None,
) -> tuple[str, Callable[[AgentGuardrailsGraphState], Any]]:
    """Private factory for guardrail evaluation nodes.

    Returns a node with observability metadata attached as __metadata__ attribute:
    - goto success_node on validation pass
    - goto failure_node on validation fail

    ``attachment_source`` picks the files a built-in guardrail inspects; see
    :func:`_resolve_attachments`.
    """
    raw_node_name = f"{scope.name}_{execution_stage.name}_{guardrail.name}"
    node_name = re.sub(r"\W+", "_", raw_node_name.lower()).strip("_")

    metadata: dict[str, Any] = {
        "tool_name": tool_name,
        "tool_type": tool_type,
        "guardrail": guardrail,
        "scope": scope,
        "execution_stage": execution_stage,
        "payload": {"input": None, "output": None},
    }

    async def node(
        state: AgentGuardrailsGraphState,
    ):
        try:
            attachments: list[GuardrailAttachment] | None = None
            # Route to appropriate evaluation service based on guardrail type and scope
            if (
                isinstance(guardrail, DeterministicGuardrail)
                and scope == GuardrailScope.TOOL
                and input_data_extractor is not None
            ):
                # Extract and store input/output data for observability
                input_data = input_data_extractor(state)
                metadata["payload"]["input"] = input_data
                if (
                    output_data_extractor
                    and execution_stage == ExecutionStage.POST_EXECUTION
                ):
                    output_data = output_data_extractor(state)
                    metadata["payload"]["output"] = output_data

                result = _evaluate_deterministic_guardrail(
                    state,
                    guardrail,
                    execution_stage,
                    input_data_extractor,
                    output_data_extractor,
                )
            elif isinstance(guardrail, BuiltInValidatorGuardrail):
                # Generate and store payload for observability
                payload = payload_generator(state)
                if execution_stage == ExecutionStage.PRE_EXECUTION:
                    metadata["payload"]["input"] = payload
                else:
                    metadata["payload"]["output"] = payload

                attachments = await _resolve_attachments(
                    state, guardrail, attachment_source
                )

                result = await _evaluate_builtin_guardrail(
                    guardrail, payload, attachments, termination_mode
                )
            else:
                # Provide specific error message for DeterministicGuardrails with wrong scope
                if isinstance(guardrail, DeterministicGuardrail):
                    raise AgentRuntimeError(
                        code=AgentRuntimeErrorCode.TERMINATION_GUARDRAIL_ERROR,
                        title="Invalid guardrail scope",
                        detail=f"DeterministicGuardrail '{guardrail.name}' can only be used with TOOL scope. "
                        f"Current scope: {scope.name}. "
                        f"Please configure this guardrail to use only TOOL scope.",
                        category=UiPathErrorCategory.USER,
                    )
                else:
                    raise AgentRuntimeError(
                        code=AgentRuntimeErrorCode.TERMINATION_GUARDRAIL_ERROR,
                        title="Unsupported guardrail type",
                        detail=f"Guardrail type '{type(guardrail).__name__}' is not supported. "
                        f"Expected DeterministicGuardrail (TOOL scope only) or BuiltInValidatorGuardrail.",
                        category=UiPathErrorCategory.USER,
                    )

            return _create_validation_command(
                result, success_node, failure_node, attachments
            )

        except Exception as exc:
            logger.error(
                "Failed to evaluate guardrail '%s': %s",
                guardrail.name,
                exc,
            )
            raise

    node.__metadata__ = metadata  # type: ignore[attr-defined]

    return node_name, node


def create_llm_guardrail_node(
    guardrail: BaseGuardrail,
    execution_stage: ExecutionStage,
    success_node: str,
    failure_node: str,
    *,
    termination_mode: GuardrailTerminationMode | None = None,
) -> tuple[str, Callable[[AgentGuardrailsGraphState], Any]]:
    def _payload_generator(state: AgentGuardrailsGraphState) -> str:
        if not state.messages:
            return ""
        match execution_stage:
            case ExecutionStage.PRE_EXECUTION:
                return get_message_content(state.messages[-1])
            case ExecutionStage.POST_EXECUTION:
                return json.dumps(_extract_tools_args_from_message(state.messages[-1]))

    def _tool_call_args(state: AgentGuardrailsGraphState) -> Any:
        if not state.messages:
            return None
        return _extract_tools_args_from_message(state.messages[-1])

    return _create_guardrail_node(
        guardrail,
        GuardrailScope.LLM,
        execution_stage,
        _payload_generator,
        success_node,
        failure_node,
        termination_mode=termination_mode,
        attachment_source=(
            _tool_call_args
            if execution_stage == ExecutionStage.POST_EXECUTION
            else None
        ),
    )


def create_agent_init_guardrail_node(
    guardrail: BaseGuardrail,
    execution_stage: ExecutionStage,
    success_node: str,
    failure_node: str,
    *,
    termination_mode: GuardrailTerminationMode | None = None,
) -> tuple[str, Callable[[AgentGuardrailsGraphState], Any]]:
    def _payload_generator(state: AgentGuardrailsGraphState) -> str:
        if not state.messages:
            return ""
        return get_message_content(state.messages[-1])

    return _create_guardrail_node(
        guardrail,
        GuardrailScope.AGENT,
        execution_stage,
        _payload_generator,
        success_node,
        failure_node,
        termination_mode=termination_mode,
    )


def create_agent_terminate_guardrail_node(
    guardrail: BaseGuardrail,
    execution_stage: ExecutionStage,
    success_node: str,
    failure_node: str,
    *,
    termination_mode: GuardrailTerminationMode | None = None,
) -> tuple[str, Callable[[AgentGuardrailsGraphState], Any]]:
    def _payload_generator(state: AgentGuardrailsGraphState) -> str:
        return str(state.inner_state.agent_result)

    def _agent_result(state: AgentGuardrailsGraphState) -> Any:
        return state.inner_state.agent_result

    return _create_guardrail_node(
        guardrail,
        GuardrailScope.AGENT,
        execution_stage,
        _payload_generator,
        success_node,
        failure_node,
        termination_mode=termination_mode,
        attachment_source=_agent_result,
    )


def _tool_call_field(tool_call: Any, field: str) -> Any:
    """Read ``field`` from a tool call given as a dict or an object."""
    if isinstance(tool_call, dict):
        return tool_call.get(field)
    return getattr(tool_call, field, None)


def create_tool_guardrail_node(
    guardrail: BaseGuardrail,
    execution_stage: ExecutionStage,
    success_node: str,
    failure_node: str,
    tool_name: str,
    tool_type: str | None = None,
    termination_mode: GuardrailTerminationMode | None = None,
) -> tuple[str, Callable[[AgentGuardrailsGraphState], Any]]:
    """Create a guardrail node for TOOL scope guardrails.

    Args:
        guardrail: The guardrail to evaluate.
        execution_stage: The execution stage (PRE_EXECUTION or POST_EXECUTION).
        success_node: Node to route to on validation pass.
        failure_node: Node to route to on validation fail.
        tool_name: Name of the tool to extract arguments from.
        tool_type: Optional type of the tool (e.g., "process", "escalation", "mcp").
        termination_mode: When the guardrail service may stop scanning.

    Returns:
        A tuple of (node_name, node_function) for the guardrail evaluation node.
    """

    def _current_call_args(state: AgentGuardrailsGraphState) -> dict[str, Any]:
        """Arguments of the tool call this evaluation is about.

        One AI message can carry several calls to the same tool, executed one after
        another with a ToolMessage appended after each. The call under evaluation is
        therefore not "the first call named ``tool_name``" but, before execution, the
        first one without a ToolMessage yet (the same selection the tool node makes),
        and after execution, the one the last ToolMessage answers. Falls back to the
        first matching call when the history has no ToolMessage bookkeeping.
        """
        messages = state.messages
        if not messages:
            return {}

        ai_message = find_latest_ai_message(messages)
        if ai_message is None or not ai_message.tool_calls:
            return {}

        selected: Any = None
        if execution_stage == ExecutionStage.PRE_EXECUTION:
            try:
                index = extract_current_tool_call_index(messages, tool_name)
            except AgentRuntimeError:
                index = None
            if index is not None and index < len(ai_message.tool_calls):
                selected = ai_message.tool_calls[index]
        else:  # POST_EXECUTION
            last_message = messages[-1]
            if isinstance(last_message, ToolMessage):
                selected = next(
                    (
                        call
                        for call in ai_message.tool_calls
                        if _tool_call_field(call, "id") == last_message.tool_call_id
                    ),
                    None,
                )

        if selected is None:
            return _extract_tool_args_from_message(ai_message, tool_name)

        return _extract_tool_args_from_message(
            AIMessage(content="", tool_calls=[selected]), tool_name
        )

    def _payload_generator(state: AgentGuardrailsGraphState) -> str:
        """Extract tool call arguments for the specified tool name.

        Args:
            state: The current agent graph state.

        Returns:
            JSON string of the tool call arguments, or empty string if not found.
        """
        if not state.messages:
            return ""

        if execution_stage == ExecutionStage.PRE_EXECUTION:
            return json.dumps(_current_call_args(state))

        return get_message_content(state.messages[-1])

    # Create closures for input/output data extraction (for deterministic guardrails)
    def _input_data_extractor(state: AgentGuardrailsGraphState) -> dict[str, Any]:
        return _current_call_args(state)

    def _output_data_extractor(state: AgentGuardrailsGraphState) -> dict[str, Any]:
        return _extract_tool_output_data(state)

    def _tool_result(state: AgentGuardrailsGraphState) -> Any:
        # A mention always carries the literal ``ID`` key; skip parsing plain-text
        # results, which the output extractor would otherwise warn about.
        if not state.messages or "ID" not in get_message_content(state.messages[-1]):
            return None
        return _extract_tool_output_data(state)

    return _create_guardrail_node(
        guardrail,
        GuardrailScope.TOOL,
        execution_stage,
        _payload_generator,
        success_node,
        failure_node,
        _input_data_extractor,
        _output_data_extractor,
        tool_name,
        tool_type,
        termination_mode=termination_mode,
        attachment_source=(
            _current_call_args
            if execution_stage == ExecutionStage.PRE_EXECUTION
            else _tool_result
        ),
    )
