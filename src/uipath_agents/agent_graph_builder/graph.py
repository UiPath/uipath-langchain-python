"""Agent graph construction - wrapper delegating to uipath_langchain.agent.graph."""

import asyncio
from functools import partial
from typing import Any, Sequence

from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph, StateGraph
from uipath.agent.models.agent import (
    AgentMcpResourceConfig,
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
from uipath_langchain.agent.tools.mcp import create_mcp_tools_and_clients

from uipath_agents.agent_graph_builder.version import supports_parallel_tool_calls

from .._config import get_flags
from .config import (
    _FF_MODEL_SETTINGS,
    AgentExecutionType,
    get_thinking_messages_limit,
    is_deep_agent_enabled,
)
from .deep import create_deep_agent_graph
from .deep_agent_prompts import get_deep_agent_meta_prompt
from .llm_utils import create_llm
from .message_utils import (
    build_user_message,
    create_message_factory,
    extract_system_prompt,
)
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

    flags_task = asyncio.get_running_loop().run_in_executor(
        None, get_flags, [_FF_MODEL_SETTINGS]
    )

    try:
        llm = create_llm(
            model=agent_definition.settings.model,
            temperature=agent_definition.settings.temperature,
            max_tokens=agent_definition.settings.max_tokens,
            execution_type=execution_type,
            byo_connection_id=byo_connection_id,
            disable_streaming=(not agent_definition.is_conversational),
            is_conversational=bool(agent_definition.is_conversational),
        )
    finally:
        try:
            await flags_task
        except Exception:
            pass

    tools = await create_tools_from_resources(agent_definition, llm)
    session_info_factory = (
        SessionInfoDebugStateFactory()
        if execution_type == AgentExecutionType.PLAYGROUND
        else None
    )
    mcp_tools, mcp_clients = await create_mcp_tools_and_clients(
        [
            resource
            for resource in agent_definition.resources
            if isinstance(resource, AgentMcpResourceConfig)
        ],
        session_info_factory=session_info_factory,
        terminate_on_close=execution_type != AgentExecutionType.PLAYGROUND,
    )
    tools.extend(mcp_tools)

    if is_deep_agent_enabled(agent_definition):
        return _build_deep_agent(agent_definition, llm, tools), list(mcp_clients)

    return _build_shallow_agent(agent_definition, llm, tools, execution_type), list(
        mcp_clients
    )


def _build_shallow_agent(
    agent_definition: LowCodeAgentDefinition,
    llm: BaseChatModel,
    tools: list[BaseTool],
    execution_type: AgentExecutionType,
) -> StateGraph[Any, Any, Any, Any]:
    """Build a standard (shallow) ReAct agent."""
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

    return create_agent(
        model=llm,
        tools=tools,
        messages=messages,
        input_schema=input_model,
        output_schema=output_model,
        config=agent_config,
        guardrails=guardrails,
    )


def _build_deep_agent(
    agent_definition: LowCodeAgentDefinition,
    llm: BaseChatModel,
    tools: list[BaseTool],
) -> StateGraph[Any, Any, Any, Any]:
    """Build a deep agent graph from agent definition."""
    if agent_definition.guardrails:
        raise NotImplementedError("Guardrails are not yet supported for deep agents.")

    meta_prompt = get_deep_agent_meta_prompt()
    system_prompt = extract_system_prompt(agent_definition)
    system_prompt = system_prompt + "\n\n" + meta_prompt

    response_format: ToolStrategy[Any] = ToolStrategy(agent_definition.output_schema)

    input_model = resolve_input_model(agent_definition.input_schema)
    output_model = resolve_output_model(agent_definition.output_schema)

    return create_deep_agent_graph(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        backend=None,
        response_format=response_format,
        input_schema=input_model,
        output_schema=output_model,
        build_user_message=partial(build_user_message, agent_definition),
    )
