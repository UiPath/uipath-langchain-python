import json
import logging
import re
from typing import Any, Callable

from langgraph.types import Command
from uipath.core.guardrails import (
    DeterministicGuardrail,
    DeterministicGuardrailsService,
)
from uipath.platform import UiPath
from uipath.platform.guardrails import (
    BaseGuardrail,
    BuiltInValidatorGuardrail,
    GuardrailScope,
)
from uipath.runtime.errors import UiPathErrorCode

from uipath_langchain.agent.guardrails.types import ExecutionStage
from uipath_langchain.agent.guardrails.utils import (
    _extract_tool_args_from_message,
    _extract_tool_output_data,
    _extract_tools_args_from_message,
    get_message_content,
)
from uipath_langchain.agent.react.types import AgentGuardrailsGraphState

from ..exceptions import AgentTerminationException

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


def _evaluate_builtin_guardrail(
    state: AgentGuardrailsGraphState,
    guardrail: BuiltInValidatorGuardrail,
    payload_generator: Callable[[AgentGuardrailsGraphState], str],
):
    """Evaluate built-in validator guardrail.

    Args:
        state: The current agent graph state.
        guardrail: The built-in validator guardrail to evaluate.
        payload_generator: Function to generate payload text from state.

    Returns:
        The guardrail evaluation result.
    """
    text = payload_generator(state)
    uipath = UiPath()
    return uipath.guardrails.evaluate_guardrail(text, guardrail)


def _create_validation_command(
    result,
    success_node: str,
    failure_node: str,
) -> Command[Any]:
    """Create command based on validation result.

    Args:
        result: The guardrail evaluation result.
        success_node: Node to route to on validation pass.
        failure_node: Node to route to on validation fail.

    Returns:
        Command to update state and route to appropriate node.
    """
    if not result.validation_passed:
        return Command(
            goto=failure_node, update={"guardrail_validation_result": result.reason}
        )
    return Command(goto=success_node, update={"guardrail_validation_result": None})


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
) -> tuple[str, Callable[[AgentGuardrailsGraphState], Any]]:
    """Private factory for guardrail evaluation nodes.

    Returns a node that evaluates the guardrail and routes via Command:
    - goto success_node on validation pass
    - goto failure_node on validation fail
    """
    raw_node_name = f"{scope.name}_{execution_stage.name}_{guardrail.name}"
    node_name = re.sub(r"\W+", "_", raw_node_name.lower()).strip("_")

    async def node(
        state: AgentGuardrailsGraphState,
    ):
        try:
            # Route to appropriate evaluation service based on guardrail type and scope
            if (
                isinstance(guardrail, DeterministicGuardrail)
                and scope == GuardrailScope.TOOL
                and input_data_extractor is not None
            ):
                result = _evaluate_deterministic_guardrail(
                    state,
                    guardrail,
                    execution_stage,
                    input_data_extractor,
                    output_data_extractor,
                )
            elif isinstance(guardrail, BuiltInValidatorGuardrail):
                result = _evaluate_builtin_guardrail(
                    state, guardrail, payload_generator
                )
            else:
                # Provide specific error message for DeterministicGuardrails with wrong scope
                if isinstance(guardrail, DeterministicGuardrail):
                    raise AgentTerminationException(
                        code=UiPathErrorCode.EXECUTION_ERROR,
                        title="Invalid guardrail scope",
                        detail=f"DeterministicGuardrail '{guardrail.name}' can only be used with TOOL scope. "
                        f"Current scope: {scope.name}. "
                        f"Please configure this guardrail to use only TOOL scope.",
                    )
                else:
                    raise AgentTerminationException(
                        code=UiPathErrorCode.EXECUTION_ERROR,
                        title="Unsupported guardrail type",
                        detail=f"Guardrail type '{type(guardrail).__name__}' is not supported. "
                        f"Expected DeterministicGuardrail (TOOL scope only) or BuiltInValidatorGuardrail.",
                    )

            return _create_validation_command(result, success_node, failure_node)

        except Exception as exc:
            logger.error(
                "Failed to evaluate guardrail '%s': %s",
                guardrail.name,
                exc,
            )
            raise

    return node_name, node


def create_llm_guardrail_node(
    guardrail: BaseGuardrail,
    execution_stage: ExecutionStage,
    success_node: str,
    failure_node: str,
) -> tuple[str, Callable[[AgentGuardrailsGraphState], Any]]:
    def _payload_generator(state: AgentGuardrailsGraphState) -> str:
        if not state.messages:
            return ""
        match execution_stage:
            case ExecutionStage.PRE_EXECUTION:
                return get_message_content(state.messages[-1])
            case ExecutionStage.POST_EXECUTION:
                return json.dumps(_extract_tools_args_from_message(state.messages[-1]))

    return _create_guardrail_node(
        guardrail,
        GuardrailScope.LLM,
        execution_stage,
        _payload_generator,
        success_node,
        failure_node,
    )


def create_agent_init_guardrail_node(
    guardrail: BaseGuardrail,
    execution_stage: ExecutionStage,
    success_node: str,
    failure_node: str,
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
    )


def create_agent_terminate_guardrail_node(
    guardrail: BaseGuardrail,
    execution_stage: ExecutionStage,
    success_node: str,
    failure_node: str,
) -> tuple[str, Callable[[AgentGuardrailsGraphState], Any]]:
    def _payload_generator(state: AgentGuardrailsGraphState) -> str:
        return str(state.agent_result)

    return _create_guardrail_node(
        guardrail,
        GuardrailScope.AGENT,
        execution_stage,
        _payload_generator,
        success_node,
        failure_node,
    )


def create_tool_guardrail_node(
    guardrail: BaseGuardrail,
    execution_stage: ExecutionStage,
    success_node: str,
    failure_node: str,
    tool_name: str,
) -> tuple[str, Callable[[AgentGuardrailsGraphState], Any]]:
    """Create a guardrail node for TOOL scope guardrails.

    Args:
        guardrail: The guardrail to evaluate.
        execution_stage: The execution stage (PRE_EXECUTION or POST_EXECUTION).
        success_node: Node to route to on validation pass.
        failure_node: Node to route to on validation fail.
        tool_name: Name of the tool to extract arguments from.

    Returns:
        A tuple of (node_name, node_function) for the guardrail evaluation node.
    """

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
            last_message = state.messages[-1]
            args_dict = _extract_tool_args_from_message(last_message, tool_name)
            if args_dict:
                return json.dumps(args_dict)

        return get_message_content(state.messages[-1])

    # Create closures for input/output data extraction (for deterministic guardrails)
    def _input_data_extractor(state: AgentGuardrailsGraphState) -> dict[str, Any]:
        if execution_stage == ExecutionStage.PRE_EXECUTION:
            if len(state.messages) < 1:
                return {}
            message = state.messages[-1]
        else:  # POST_EXECUTION
            if len(state.messages) < 2:
                return {}
            message = state.messages[-2]

        return _extract_tool_args_from_message(message, tool_name)

    def _output_data_extractor(state: AgentGuardrailsGraphState) -> dict[str, Any]:
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
    )
