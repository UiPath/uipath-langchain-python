import os
from typing import Callable, Sequence, Type, TypeVar, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pydantic import BaseModel
from uipath.agent.models.agent import AgentGuardrail
from uipath.models.guardrails import GuardrailScope

from ..tools import create_tool_node
from .guardrail_nodes import create_llm_guardrail_node
from .init_node import (
    create_init_node,
)
from .llm_node import (
    create_llm_node,
)
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
    guardrails: Sequence[AgentGuardrail] | None = None,
) -> StateGraph[AgentGraphState, None, InputT, OutputT]:
    """Build agent graph with INIT -> AGENT(subgraph) <-> TOOLS loop, terminated by control flow tools.

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

    for tool_name, tool_node in tool_nodes.items():
        builder.add_node(tool_name, tool_node)

    builder.add_node(AgentGraphNode.TERMINATE, terminate_node)

    builder.add_edge(START, AgentGraphNode.INIT)

    # Build AGENT subgraph: before-agent guardrail nodes -> LLM -> after-agent guardrail nodes
    agent_subgraph = StateGraph(AgentGraphState)
    # Build guardrail nodes directly from guardrails (no middleware abstraction)
    before_nodes: dict[str, Callable] = {}
    after_nodes: dict[str, Callable] = {}
    llm_guardrails = [
        gr for gr in (guardrails or []) if GuardrailScope.LLM in gr.selector.scopes
    ]
    for gr in llm_guardrails:
        name, fn = create_llm_guardrail_node(gr, "before")
        before_nodes[name] = fn
    for gr in llm_guardrails:
        name, fn = create_llm_guardrail_node(gr, "after")
        after_nodes[name] = fn

    # Register middleware nodes inside subgraph
    for node_name, node_callable in before_nodes.items():
        agent_subgraph.add_node(node_name, node_callable)
    for node_name, node_callable in after_nodes.items():
        agent_subgraph.add_node(node_name, node_callable)
    # Register LLM node inside subgraph
    AGENT_LLM_NODE_NAME = "AGENT_LLM"
    agent_subgraph.add_node(AGENT_LLM_NODE_NAME, agent_node)
    # Wire subgraph: START -> before... -> LLM -> after... -> END
    before_names = list(before_nodes.keys())
    after_names = list(after_nodes.keys())
    if before_names:
        agent_subgraph.add_edge(START, before_names[0])
        for cur, nxt in zip(before_names, before_names[1:], strict=False):
            agent_subgraph.add_edge(cur, nxt)
        agent_subgraph.add_edge(before_names[-1], AGENT_LLM_NODE_NAME)
    else:
        agent_subgraph.add_edge(START, AGENT_LLM_NODE_NAME)

    if after_names:
        agent_subgraph.add_edge(AGENT_LLM_NODE_NAME, after_names[0])
        for cur, nxt in zip(after_names, after_names[1:], strict=False):
            agent_subgraph.add_edge(cur, nxt)
        agent_subgraph.add_edge(after_names[-1], END)
    else:
        agent_subgraph.add_edge(AGENT_LLM_NODE_NAME, END)

    # Add compiled subgraph as the AGENT node in the outer graph
    builder.add_node(AgentGraphNode.AGENT, agent_subgraph.compile())
    builder.add_edge(AgentGraphNode.INIT, AgentGraphNode.AGENT)

    tool_node_names = list(tool_nodes.keys())
    builder.add_conditional_edges(
        AgentGraphNode.AGENT,
        route_agent,
        [AgentGraphNode.AGENT, *tool_node_names, AgentGraphNode.TERMINATE],
    )

    for tool_name in tool_node_names:
        builder.add_edge(tool_name, AgentGraphNode.AGENT)

    builder.add_edge(AgentGraphNode.TERMINATE, END)

    return builder
