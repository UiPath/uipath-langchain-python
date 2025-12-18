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
    attach_post_agent_guardrails,
    attach_pre_agent_guardrails,
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
    route_agent,
)
from .terminate_node import (
    create_terminate_node,
)
from .tools import create_flow_control_tools
from .types import AgentGraphConfig, AgentGraphNode, AgentGraphState, AgentGuardrailsGraphState

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)
StateT = TypeVar("StateT", bound=BaseModel)


def create_state_with_input(input_schema: Type[InputT],
    *,
    base_state: Type[StateT] = AgentGraphState,
) -> Type[BaseModel]:
    """Create a dynamic Agent graph state model that includes user input fields.

    LangGraph requires a concrete Pydantic model for state. We create one dynamically
    by mixing the chosen base state with the user-provided `input_schema`.

    Args:
        input_schema: Pydantic model defining the agent input shape.
        base_state: Base state model to extend. Use `AgentGuardrailsGraphState` when
            guardrail nodes are attached at the parent graph level.

    Returns:
        A dynamically created Pydantic model type.
    """
    InnerAgentGraphState = type("InnerAgentGraphState", (base_state, input_schema), {})

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
    tool_nodes = create_tool_node(agent_tools)
    tool_nodes_with_guardrails = create_tools_guardrails_subgraph(
        tool_nodes, guardrails
    )
    terminate_node = create_terminate_node(output_schema)

    base_state: type[BaseModel]
    if guardrails:
        base_state = AgentGuardrailsGraphState
    else:
        base_state = AgentGraphState

    InnerAgentGraphState = create_state_with_input(
        input_schema if input_schema is not None else BaseModel,
        base_state=cast(type[BaseModel], base_state),
    )

    builder: StateGraph[AgentGraphState, None, InputT, OutputT] = StateGraph(
        InnerAgentGraphState, input_schema=input_schema, output_schema=output_schema
    )
    builder.add_node(AgentGraphNode.INIT, init_node)

    for tool_name, tool_node in tool_nodes_with_guardrails.items():
        builder.add_node(tool_name, tool_node)

    builder.add_edge(START, AgentGraphNode.INIT)

    llm_node = create_llm_node(model, llm_tools)
    llm_with_guardrails_subgraph = create_llm_guardrails_subgraph(
        (AgentGraphNode.LLM, llm_node), guardrails
    )
    builder.add_node(AgentGraphNode.AGENT, llm_with_guardrails_subgraph)

    init_next = attach_pre_agent_guardrails(
        cast(StateGraph[AgentGuardrailsGraphState], builder),
        guardrails,
        init_node_name=AgentGraphNode.INIT,
        next_node_name=AgentGraphNode.AGENT,
    )
    builder.add_edge(AgentGraphNode.INIT, init_next)

    terminate_end_node = attach_post_agent_guardrails(
        cast(StateGraph[AgentGuardrailsGraphState], builder),
        terminate_node,
        guardrails,
        terminate_node_name=AgentGraphNode.TERMINATE,
        next_node_name=END,
    )
    builder.add_edge(AgentGraphNode.TERMINATE, terminate_end_node)

    tool_node_names = list(tool_nodes_with_guardrails.keys())
    builder.add_conditional_edges(
        AgentGraphNode.AGENT,
        route_agent,
        [AgentGraphNode.AGENT, *tool_node_names, AgentGraphNode.TERMINATE],
    )

    for tool_name in tool_node_names:
        builder.add_edge(tool_name, AgentGraphNode.AGENT)

    return builder
