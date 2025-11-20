import os
from typing import Callable, Sequence, Type, TypeVar, cast, Any

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
    middlewares: Sequence[Any] | None = None,
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

    # Optional: before_agent middleware nodes
    before_nodes: list[str] = []
    after_middlewares: list[Any] = []
    if middlewares:
        for idx, mw in enumerate(middlewares):
            before_hook = getattr(mw, "before_agent", None)
            if not callable(before_hook):
                continue
            node_name = f"agent_before_{idx}"

            async def before_node(state: AgentGraphState, _mw: Any = mw):
                before = getattr(_mw, "before_agent", None)
                if callable(before):
                    result = before(state.messages, lambda msgs: msgs)
                    if hasattr(result, "__await__"):
                        result = await result
                    if isinstance(result, list):
                        return {"messages": result}
                return {"messages": state.messages}

            builder.add_node(node_name, before_node)
            before_nodes.append(node_name)
            after_middlewares.append(mw)

    if before_nodes:
        builder.add_edge(AgentGraphNode.INIT, before_nodes[0])
        for cur, nxt in zip(before_nodes, before_nodes[1:], strict=False):
            builder.add_edge(cur, nxt)
        builder.add_edge(before_nodes[-1], AgentGraphNode.AGENT)
    else:
        builder.add_edge(AgentGraphNode.INIT, AgentGraphNode.AGENT)

    tool_node_names = list(tool_nodes.keys())
    builder.add_conditional_edges(
        AgentGraphNode.AGENT,
        route_agent,
        [AgentGraphNode.AGENT, *tool_node_names, AgentGraphNode.TERMINATE],
    )

    for tool_name in tool_node_names:
        builder.add_edge(tool_name, AgentGraphNode.AGENT)

    # Optional: after_agent middleware nodes
    after_nodes: list[str] = []
    if after_middlewares:
        for idx, mw in enumerate(after_middlewares):
            node_name = f"agent_after_{idx}"

            async def after_node(state: AgentGraphState, _mw: Any = mw):
                after = getattr(_mw, "after_agent", None)
                if callable(after):
                    result = after(state.messages, lambda msgs: msgs)
                    if hasattr(result, "__await__"):
                        result = await result
                    if isinstance(result, list):
                        return {"messages": result}
                return {"messages": state.messages}

            builder.add_node(node_name, after_node)
            after_nodes.append(node_name)

    if after_nodes:
        builder.add_edge(AgentGraphNode.TERMINATE, after_nodes[0])
        for cur, nxt in zip(after_nodes, after_nodes[1:], strict=False):
            builder.add_edge(cur, nxt)
        builder.add_edge(after_nodes[-1], END)
    else:
        builder.add_edge(AgentGraphNode.TERMINATE, END)

    return builder
