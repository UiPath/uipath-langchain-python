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

from .input_utils import validate_input_data
from .llm_utils import create_llm
from .message_utils import build_agent_messages

AGENT_LOOP_RECURSION_LIMIT = 50


async def build_agent_graph(
    agent_definition: LowCodeAgentDefinition,
    input_data: Optional[Union[str, dict[str, Any]]] = None,
) -> StateGraph[AgentGraphState]:
    """Build LangGraph agent from agent.json configuration and optional input data."""

    input_arguments = validate_input_data(agent_definition.input_schema, input_data)

    tools = await create_tools_from_resources(agent_definition)
    llm = create_llm(
        model=agent_definition.settings.model,
        temperature=agent_definition.settings.temperature,
        max_tokens=agent_definition.settings.max_tokens,
    )

    initial_messages = build_agent_messages(
        agent_definition.messages, input_arguments, agent_definition.name
    )
    output_model = resolve_output_model(agent_definition.output_schema)

    agent_config = AgentGraphConfig(recursion_limit=AGENT_LOOP_RECURSION_LIMIT)

    return create_agent(
        model=llm,
        tools=tools,
        messages=initial_messages,
        state_schema=AgentGraphState,
        response_format=output_model,
        config=agent_config,
    )
