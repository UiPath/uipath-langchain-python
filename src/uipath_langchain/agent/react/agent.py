import os
from typing import Callable, Sequence, Type, TypeVar, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pydantic import BaseModel

from ..tools import create_tool_node
from .init_node import (
    create_init_node,
)
from .llm_node import (
    create_llm_node,
)
from .middleware_nodes import create_middleware_nodes
from .middleware_types import AgentMiddleware
from .router import (
    route_agent,
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
    middlewares: Sequence[AgentMiddleware] | None = None,
) -> StateGraph[AgentGraphState, None, InputT, OutputT]:
    """Build agent graph with INIT -> AGENT <-> TOOLS loop, terminated by control flow tools.

    Control flow tools (end_execution, raise_error) are auto-injected alongside regular tools.
    """
    if config is None:
        config = AgentGraphConfig()

    os.environ["LANGCHAIN_RECURSION_LIMIT"] = str(config.recursion_limit)

    agent_tools = list(tools)
    flow_control_tools: list[BaseTool] = create_flow_control_tools(output_schema)
    llm_tools: list[BaseTool] = [*agent_tools, *flow_control_tools]

    init_node = create_init_node(messages)
    agent_node = create_llm_node(model, llm_tools)
    tool_nodes = create_tool_node(agent_tools)
    terminate_node = create_terminate_node(output_schema)

    InnerAgentGraphState = create_state_with_input(
        input_schema if input_schema is not None else BaseModel
    )

    builder: StateGraph[AgentGraphState, None, InputT, OutputT] = StateGraph(
        InnerAgentGraphState, input_schema=input_schema, output_schema=output_schema
    )
    builder.add_node(AgentGraphNode.INIT, init_node)
    builder.add_node(AgentGraphNode.AGENT, agent_node)

    for tool_name, tool_node in tool_nodes.items():
        builder.add_node(tool_name, tool_node)

    builder.add_node(AgentGraphNode.TERMINATE, terminate_node)

    builder.add_edge(START, AgentGraphNode.INIT)

    # Build before_agent chain (INIT -> before... -> AGENT)
    before_agent_middleware_nodes = create_middleware_nodes(
        middlewares, "before_agent"
    )
    for node_name, node_callable in before_agent_middleware_nodes.items():
        builder.add_node(node_name, node_callable)

    before_agent_middleware_node_names = list(before_agent_middleware_nodes.keys())
    if before_agent_middleware_node_names:
        builder.add_edge(AgentGraphNode.INIT, before_agent_middleware_node_names[0])
        for cur, nxt in zip(before_agent_middleware_node_names, before_agent_middleware_node_names[1:], strict=False):
            builder.add_edge(cur, nxt)
        builder.add_edge(before_agent_middleware_node_names[-1], AgentGraphNode.AGENT)
    else:
        builder.add_edge(AgentGraphNode.INIT, AgentGraphNode.AGENT)

    # Build after_agent chain to run BEFORE TERMINATE
    after_agent_middleware_nodes = create_middleware_nodes(middlewares, "after_agent")
    for node_name, node_callable in after_agent_middleware_nodes.items():
        builder.add_node(node_name, node_callable)
    after_agent_names = list(after_agent_middleware_nodes.keys())

    tool_node_names = list(tool_nodes.keys())
    # Route AGENT to after chain (if present) instead of directly to TERMINATE
    post_agent_destination = after_agent_names[0] if after_agent_names else AgentGraphNode.TERMINATE
    destinations = [AgentGraphNode.AGENT, *tool_node_names, post_agent_destination]
    builder.add_conditional_edges(AgentGraphNode.AGENT, route_agent, destinations)

    for tool_name in tool_node_names:
        builder.add_edge(tool_name, AgentGraphNode.AGENT)

    # Chain after nodes to TERMINATE -> END
    if after_agent_names:
        for cur, nxt in zip(after_agent_names, after_agent_names[1:], strict=False):
            builder.add_edge(cur, nxt)
        builder.add_edge(after_agent_names [-1], AgentGraphNode.TERMINATE)
    builder.add_edge(AgentGraphNode.TERMINATE, END)

    return builder
