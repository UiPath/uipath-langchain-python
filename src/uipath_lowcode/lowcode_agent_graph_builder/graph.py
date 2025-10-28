"""LowCode Agent graph construction - wrapper delegating to uipath_langchain.agent.graph."""

from pathlib import Path
from typing import Any, Optional, Union

from langgraph.graph import StateGraph
from uipath_langchain.agent.react import (
    AgentGraphConfig,
    AgentGraphState,
    create_agent,
    resolve_output_model,
)
from uipath_langchain.agent.tools import create_tools_from_resources

from .constants import (
    AGENT_CONFIG_FILENAME,
    AGENT_LOOP_RECURSION_LIMIT,
)
from .input_loader import load_agent_configuration, load_input_arguments
from .llm_utils import create_llm
from .message_utils import build_agent_messages


async def build_lowcode_agent_graph(
    input_data: Optional[Union[str, dict[str, Any]]] = None,
) -> StateGraph[AgentGraphState]:
    """Build LangGraph agent from agent.json configuration and optional input data."""

    agent_json_path = Path.cwd() / AGENT_CONFIG_FILENAME
    agent_definition = load_agent_configuration(agent_json_path)
    input_arguments = load_input_arguments(
        agent_definition.input_schema, input_data=input_data
    )

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
