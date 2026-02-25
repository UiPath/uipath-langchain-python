"""Agent graph construction - wrapper delegating to uipath_langchain.agent.graph."""

from typing import Any, Sequence

from langgraph.graph.state import CompiledStateGraph, StateGraph
from uipath.agent.models.agent import (
    LowCodeAgentDefinition,
)
from uipath.platform.guardrails import BaseGuardrail
from uipath.runtime.base import UiPathDisposableProtocol
from uipath_langchain.agent.guardrails import build_guardrails_with_actions
from uipath_langchain.agent.guardrails.actions.base_action import GuardrailAction
from uipath_langchain.agent.react import (
    AgentGraphConfig,
    create_agent,
    resolve_input_model,
    resolve_output_model,
)
from uipath_langchain.agent.tools import create_tools_from_resources
from uipath_langchain.agent.tools.mcp import create_mcp_tools_from_agent

from uipath_agents.agent_graph_builder.version import supports_parallel_tool_calls

from .config import AgentExecutionType, get_thinking_messages_limit
from .llm_utils import create_llm
from .message_utils import create_message_factory
from .session_info_debug_state import SessionInfoDebugStateFactory

AGENT_MAX_ITERATIONS_DEFAULT = 25


def _filter_guardrails_for_evals(
    guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] | None,
    execution_type: AgentExecutionType,
) -> list[tuple[BaseGuardrail, GuardrailAction]]:
    """Filter out guardrails disabled for evaluations.

    When running inside an evaluation, guardrails with
    ``enabled_for_evals=False`` are excluded.
    """
    if execution_type != AgentExecutionType.EVAL:
        return list(guardrails or [])
    return [
        (guardrail, action)
        for guardrail, action in (guardrails or [])
        if guardrail.enabled_for_evals
    ]


async def build_agent_graph(
    agent_definition: LowCodeAgentDefinition,
    execution_type: AgentExecutionType = AgentExecutionType.RUNTIME,
) -> tuple[
    StateGraph[Any, Any, Any, Any] | CompiledStateGraph[Any, Any, Any, Any],
    list[UiPathDisposableProtocol],
]:
    """Build LangGraph agent from agent.json configuration.

    Args:
        agent_definition: Agent definition model
        execution_type: Execution mode of the agent: playground | runtime | eval

    Returns:
        Tuple of (compiled graph, disposables list).
    """
    byo_connection_id = (
        agent_definition.settings.byom_properties.connection_id
        if agent_definition.settings.byom_properties
        else None
    )
    llm = create_llm(
        model=agent_definition.settings.model,
        temperature=agent_definition.settings.temperature,
        max_tokens=agent_definition.settings.max_tokens,
        execution_type=execution_type,
        byo_connection_id=byo_connection_id,
        disable_streaming=(not agent_definition.is_conversational),
    )
    tools = await create_tools_from_resources(agent_definition, llm)
    session_info_factory = (
        SessionInfoDebugStateFactory()
        if execution_type == AgentExecutionType.PLAYGROUND
        else None
    )
    mcp_tools, mcp_clients = await create_mcp_tools_from_agent(
        agent_definition,
        session_info_factory=session_info_factory,
        terminate_on_close=execution_type != AgentExecutionType.PLAYGROUND,
    )
    tools.extend(mcp_tools)
    input_model = resolve_input_model(agent_definition.input_schema)
    output_model = resolve_output_model(agent_definition.output_schema)

    messages = create_message_factory(agent_definition, input_model)

    guardrails = build_guardrails_with_actions(agent_definition.guardrails, tools)
    guardrails = _filter_guardrails_for_evals(guardrails, execution_type)

    agent_config = AgentGraphConfig(
        llm_messages_limit=agent_definition.settings.max_iterations
        or AGENT_MAX_ITERATIONS_DEFAULT,
        thinking_messages_limit=get_thinking_messages_limit(
            agent_definition.settings.model
        ),
        is_conversational=agent_definition.is_conversational,
        parallel_tool_calls=supports_parallel_tool_calls(
            agent_definition.version, agent_definition.settings.model
        ),
    )

    graph = create_agent(
        model=llm,
        tools=tools,
        messages=messages,
        input_schema=input_model,
        output_schema=output_model,
        config=agent_config,
        guardrails=guardrails,
    )
    return graph, list(mcp_clients)
