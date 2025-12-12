import json
import logging
import re
from typing import Any, Callable

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.types import Command
from uipath.platform import UiPath
from uipath.platform.guardrails import (
    BaseGuardrail,
    GuardrailScope,
)

from uipath_langchain.agent.guardrails.types import ExecutionStage

from .types import AgentGuardrailsGraphState

logger = logging.getLogger(__name__)


def _message_text(msg: AnyMessage) -> str:
    if isinstance(msg, (HumanMessage, SystemMessage)):
        return msg.content if isinstance(msg.content, str) else str(msg.content)
    return str(getattr(msg, "content", "")) if hasattr(msg, "content") else ""


def _create_guardrail_node(
    guardrail: BaseGuardrail,
    scope: GuardrailScope,
    execution_stage: ExecutionStage,
    payload_generator: Callable[[AgentGuardrailsGraphState], str],
    success_node: str,
    failure_node: str,
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
        text = payload_generator(state)
        try:
            uipath = UiPath()
            result = uipath.guardrails.evaluate_guardrail(text, guardrail)
        except Exception as exc:
            logger.error("Failed to evaluate guardrail: %s", exc)
            raise

        if not result.validation_passed:
            return Command(
                goto=failure_node, update={"guardrail_validation_result": result.reason}
            )
        return Command(goto=success_node, update={"guardrail_validation_result": None})

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
    guardrail: BaseGuardrail,
    execution_stage: ExecutionStage,
    success_node: str,
    failure_node: str,
) -> tuple[str, Callable[[AgentGuardrailsGraphState], Any]]:
    # To be implemented in future PR
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
            if not isinstance(state.messages[-1], AIMessage):
                return ""
            message = state.messages[-1]

            if not message.tool_calls:
                return ""

            # Find the first tool call with matching name
            for tool_call in message.tool_calls:
                call_name = (
                    tool_call.get("name")
                    if isinstance(tool_call, dict)
                    else getattr(tool_call, "name", None)
                )
                if call_name == tool_name:
                    # Extract args from the tool call
                    args = (
                        tool_call.get("args")
                        if isinstance(tool_call, dict)
                        else getattr(tool_call, "args", None)
                    )
                    if args is not None:
                        return json.dumps(args)

        return _message_text(state.messages[-1])

    return _create_guardrail_node(
        guardrail,
        GuardrailScope.TOOL,
        execution_stage,
        _payload_generator,
        success_node,
        failure_node,
    )
