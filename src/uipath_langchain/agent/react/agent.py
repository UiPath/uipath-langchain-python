import os
from typing import Callable, Sequence, Type, TypeVar, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pydantic import BaseModel
from uipath.platform.guardrails import BaseGuardrail

from ..guardrails.actions import GuardrailAction
from ..tools import create_tool_node
from .guardrails.guardrails_subgraph import (
    create_agent_init_guardrails_subgraph,
    create_agent_terminate_guardrails_subgraph,
    create_llm_guardrails_subgraph,
    create_tools_guardrails_subgraph,
)
from .init_node import (
    create_init_node,
)
from .llm_node import (
    create_llm_node,
)
from .router import (
    create_route_agent,
)
from .terminate_node import (
    create_terminate_node,
)
from .tools import create_flow_control_tools
from .types import AgentGraphConfig, AgentGraphNode, AgentGraphState

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


def create_state_with_input(input_schema: Type[InputT]):
    InnerAgentGraphState = type(
        "InnerAgentGraphState",
        (AgentGraphState, input_schema),
        {},
    )

    cast(type[BaseModel], InnerAgentGraphState).model_rebuild()
    return InnerAgentGraphState


def create_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    messages: Sequence[SystemMessage | HumanMessage]
    | Callable[[InputT], Sequence[SystemMessage | HumanMessage]],
    *,
    input_schema: Type[InputT] | None = None,
    output_schema: Type[OutputT] | None = None,
    config: AgentGraphConfig | None = None,
    guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] | None = None,
) -> StateGraph[AgentGraphState, None, InputT, OutputT]:
    """Build agent graph with INIT -> AGENT (subgraph) <-> TOOLS loop, terminated by control flow tools.

    The AGENT node is a subgraph that runs:
    - before-agent guardrail middlewares
    - the LLM tool-executing node
    - after-agent guardrail middlewares

    Control flow tools (end_execution, raise_error) are auto-injected alongside regular tools.
    """
    if config is None:
        config = AgentGraphConfig()

    os.environ["LANGCHAIN_RECURSION_LIMIT"] = str(config.recursion_limit)

    agent_tools = list(tools)
    flow_control_tools: list[BaseTool] = create_flow_control_tools(output_schema)
    llm_tools: list[BaseTool] = [*agent_tools, *flow_control_tools]

    init_node = create_init_node(messages, input_schema)

    tool_nodes = create_tool_node(agent_tools)
    tool_nodes_with_guardrails = create_tools_guardrails_subgraph(
        tool_nodes, guardrails
    )
    terminate_node = create_terminate_node(output_schema)

    InnerAgentGraphState = create_state_with_input(
        input_schema if input_schema is not None else BaseModel
    )

    builder: StateGraph[AgentGraphState, None, InputT, OutputT] = StateGraph(
        InnerAgentGraphState, input_schema=input_schema, output_schema=output_schema
    )
    init_with_guardrails_subgraph = create_agent_init_guardrails_subgraph(
        (AgentGraphNode.GUARDED_INIT, init_node),
        guardrails,
    )
    builder.add_node(AgentGraphNode.INIT, init_with_guardrails_subgraph)

    for tool_name, tool_node in tool_nodes_with_guardrails.items():
        builder.add_node(tool_name, tool_node)

    terminate_with_guardrails_subgraph = create_agent_terminate_guardrails_subgraph(
        (AgentGraphNode.GUARDED_TERMINATE, terminate_node),
        guardrails,
    )
    builder.add_node(AgentGraphNode.TERMINATE, terminate_with_guardrails_subgraph)

    builder.add_edge(START, AgentGraphNode.INIT)

    llm_node = create_llm_node(model, llm_tools, config.thinking_messages_limit)
    llm_with_guardrails_subgraph = create_llm_guardrails_subgraph(
        (AgentGraphNode.LLM, llm_node), guardrails
    )
    builder.add_node(AgentGraphNode.AGENT, llm_with_guardrails_subgraph)
    builder.add_edge(AgentGraphNode.INIT, AgentGraphNode.AGENT)

    tool_node_names = list(tool_nodes_with_guardrails.keys())
    route_agent = create_route_agent(config.thinking_messages_limit)
    builder.add_conditional_edges(
        AgentGraphNode.AGENT,
        route_agent,
        [AgentGraphNode.AGENT, *tool_node_names, AgentGraphNode.TERMINATE],
    )

    for tool_name in tool_node_names:
        builder.add_edge(tool_name, AgentGraphNode.AGENT)

    builder.add_edge(AgentGraphNode.TERMINATE, END)

    return builder
