"""Agent graph construction - wrapper delegating to uipath_langchain.agent.graph."""

from typing import Any

from uipath.agent.models.agent import (
    LowCodeAgentDefinition,
)
from uipath_langchain.agent.guardrails import build_guardrails_with_actions
from uipath_langchain.agent.react import (
    AgentGraphConfig,
    create_agent,
    resolve_input_model,
    resolve_output_model,
)
from uipath_langchain.agent.tools import create_tools_from_resources

from .llm_utils import create_llm
from .message_utils import build_agent_messages

# Maximum number of agent loop iterations before termination
# Set to 50 to prevent infinite loops while allowing complex reasoning chains
AGENT_LOOP_RECURSION_LIMIT = 50


async def build_agent_graph(
    agent_definition: LowCodeAgentDefinition,
    input_data: dict[str, Any],
):
    """Build LangGraph agent from agent.json configuration and optional input data.

    Args:
        agent_definition: Agent definition model
        input_data: Optional input data for the agent

    Returns:
        StateGraph configured with the agent definition and feature flags.
    """

    tools = await create_tools_from_resources(agent_definition)
    llm = create_llm(
        model=agent_definition.settings.model,
        temperature=agent_definition.settings.temperature,
        max_tokens=agent_definition.settings.max_tokens,
    )

    agent_messages = build_agent_messages(
        agent_definition.messages, input_data, agent_definition.name or ""
    )
    input_model = resolve_input_model(agent_definition.input_schema)
    output_model = resolve_output_model(agent_definition.output_schema)

    guardrails = build_guardrails_with_actions(agent_definition.guardrails)

    # Create agent config with feature flags
    agent_config = AgentGraphConfig(
        recursion_limit=AGENT_LOOP_RECURSION_LIMIT,
    )

    return create_agent(
        model=llm,
        tools=tools,
        messages=agent_messages,
        input_schema=input_model,
        output_schema=output_model,
        config=agent_config,
        guardrails=guardrails,
    )
