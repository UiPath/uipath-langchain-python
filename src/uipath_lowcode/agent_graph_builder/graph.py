"""Agent graph construction - wrapper delegating to uipath_langchain.agent.graph."""

from typing import Any, Optional, Union

from langgraph.graph import StateGraph
from uipath.agent.models.agent import LowCodeAgentDefinition
from uipath_langchain.agent.react import (
    AgentGraphConfig,
    AgentGraphState,
    create_agent,
    resolve_output_model,
)
from uipath_langchain.agent.tools import create_tools_from_resources

from .llm_utils import create_llm
from .message_utils import build_agent_messages

AGENT_LOOP_RECURSION_LIMIT = 50


async def build_agent_graph(
    agent_definition: LowCodeAgentDefinition,
    input_data: Optional[Union[str, dict[str, Any]]] = None,
) -> StateGraph[AgentGraphState]:
    """Build LangGraph agent from agent.json configuration and optional input data.

    Args:
        agent_definition: Agent definition model
        input_data: Optional input data for the agent

    Returns:
        StateGraph configured with the agent definition and feature flags.
    """

    agent_input_args = input_data if isinstance(input_data, dict) else {}

    tools = await create_tools_from_resources(agent_definition)
    llm = create_llm(
        model=agent_definition.settings.model,
        temperature=agent_definition.settings.temperature,
        max_tokens=agent_definition.settings.max_tokens,
    )

    agent_messages = build_agent_messages(
        agent_definition.messages, agent_input_args, agent_definition.name
    )
    output_model = resolve_output_model(agent_definition.output_schema)

    # Create agent config with feature flags
    agent_config = AgentGraphConfig(
        recursion_limit=AGENT_LOOP_RECURSION_LIMIT,
    )

    return create_agent(
        model=llm,
        tools=tools,
        messages=agent_messages,
        state_schema=AgentGraphState,
        response_format=output_model,
        config=agent_config,
    )
