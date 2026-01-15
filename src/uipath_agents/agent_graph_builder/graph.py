"""Agent graph construction - wrapper delegating to uipath_langchain.agent.graph."""

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

from .config import AgentExecutionType, get_thinking_messages_limit
from .llm_utils import create_llm
from .message_utils import create_message_factory

AGENT_LOOP_RECURSION_LIMIT = 50


async def build_agent_graph(
    agent_definition: LowCodeAgentDefinition,
    execution_type: AgentExecutionType = AgentExecutionType.RUNTIME,
):
    """Build LangGraph agent from agent.json configuration.

    Args:
        agent_definition: Agent definition model

    Returns:
        StateGraph configured with the agent definition and feature flags.
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
    )
    tools = await create_tools_from_resources(agent_definition, llm)
    input_model = resolve_input_model(agent_definition.input_schema)
    output_model = resolve_output_model(agent_definition.output_schema)

    messages = create_message_factory(agent_definition, input_model)

    guardrails = build_guardrails_with_actions(agent_definition.guardrails)

    agent_config = AgentGraphConfig(
        recursion_limit=AGENT_LOOP_RECURSION_LIMIT,
        thinking_messages_limit=get_thinking_messages_limit(
            agent_definition.settings.model
        ),
        is_conversational=agent_definition.is_conversational,
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
