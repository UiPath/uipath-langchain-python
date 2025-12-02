from __future__ import annotations

from typing import Any, Callable, Literal, Sequence

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from uipath.agent.models.agent import AgentGuardrail
from uipath.models.guardrails import GuardrailScope

from .guardrail_nodes import (
    create_agent_guardrail_node,
    create_llm_guardrail_node,
    create_tool_guardrail_node,
)
from .actions.base_action import GuardrailAction, GuardrailActionNode
from .types import AgentGuardrailsGraphState


def create_guardrails_subgraph(
    main_inner_node: tuple[str, Any],
    guardrails: Sequence[tuple[AgentGuardrail, GuardrailAction]] | None,
    scope: GuardrailScope,
    node_factory: Callable[
        [
            AgentGuardrail,
            Literal["PreExecution", "PostExecution"],
            str,  # success node name
            str,  # fail node name
        ],
        GuardrailActionNode,
    ] = create_llm_guardrail_node,
) -> Any:
    """Build a subgraph that enforces guardrails around an inner node.

    START -> pre-eval nodes (dynamic goto) -> inner -> post-eval nodes (dynamic goto) -> END

    No static edges are added between guardrail nodes; each eval decides via Command.
    Failure nodes are added but not chained; they are expected to route via Command.
    """
    inner_name, inner_node = main_inner_node

    subgraph = StateGraph(AgentGuardrailsGraphState)

    # Add pre execution guardrail nodes
    first_pre_exec_guardrail_node = _build_guardrail_node_chain(
        subgraph, guardrails, scope, "PreExecution", node_factory, inner_name
    )
    subgraph.add_edge(START, first_pre_exec_guardrail_node)

    # Add post execution guardrail nodes
    first_post_exec_guardrail_node = _build_guardrail_node_chain(
        subgraph, guardrails, scope, "PostExecution", node_factory, END
    )
    subgraph.add_node(inner_name, inner_node)
    subgraph.add_edge(inner_name, first_post_exec_guardrail_node)

    return subgraph.compile()


def _build_guardrail_node_chain(
    subgraph: StateGraph,
    guardrails: Sequence[tuple[AgentGuardrail, GuardrailAction]] | None,
    scope: GuardrailScope,
    execution_stage: Literal["PreExecution", "PostExecution"],
    node_factory: Callable[
        [
            AgentGuardrail,
            Literal["PreExecution", "PostExecution"],
            str,  # success node name
            str,  # fail node name
        ],
        GuardrailActionNode,
    ],
    next_node: str,
) -> str:
    """Recursively build a chain of guardrail nodes in reverse order.

    This function processes guardrails from last to first, creating a chain where:
    - Each guardrail node evaluates the guardrail condition
    - On success, it routes to the next guardrail node (or the final next_node)
    - On failure, it routes to a failure node that either throws an error or continues to next_node

    Args:
        subgraph: The StateGraph to add nodes and edges to.
        guardrails: Sequence of (guardrail, action) tuples to process. Processed in reverse.
        scope: The scope of the guardrails (LLM, AGENT, or TOOL).
        execution_stage: Whether this is "PreExecution" or "PostExecution" guardrails.
        node_factory: Factory function to create guardrail evaluation nodes.
        next_node: The node name to route to after all guardrails pass.

    Returns:
        The name of the first guardrail node in the chain (or next_node if no guardrails).
    """
    # Base case: no guardrails to process, return the next node directly
    if not guardrails:
        return next_node

    guardrail, action = guardrails[-1]
    remaining_guardrails = guardrails[:-1]

    fail_node_name, fail_node = action.action_node(
        guardrail=guardrail, scope=scope, execution_stage=execution_stage
    )

    # Create the guardrail evaluation node.
    guardrail_node_name, guardrail_node = node_factory(
        guardrail, execution_stage, next_node, fail_node_name
    )

    # Add both nodes to the subgraph
    subgraph.add_node(guardrail_node_name, guardrail_node)
    subgraph.add_node(fail_node_name, fail_node)

    # Failure path route to the next node
    subgraph.add_edge(fail_node_name, next_node)

    previous_node_name = _build_guardrail_node_chain(
        subgraph,
        remaining_guardrails,
        scope,
        execution_stage,
        node_factory,
        guardrail_node_name,
    )

    return previous_node_name


def create_llm_guardrails_subgraph(
    llm_node: tuple[str, Any],
    guardrails: Sequence[tuple[AgentGuardrail, GuardrailAction]] | None,
) -> Any:
    applicable_guardrails = [
        (guardrail, _)
        for (guardrail, _) in (guardrails or [])
        if GuardrailScope.LLM in guardrail.selector.scopes
    ]
    return create_guardrails_subgraph(
        main_inner_node=llm_node,
        guardrails=applicable_guardrails,
        scope=GuardrailScope.LLM,
        node_factory=create_llm_guardrail_node,
    )


def create_agent_guardrails_subgraph(
    agent_node: tuple[str, Any],
    guardrails: Sequence[tuple[AgentGuardrail, GuardrailAction]] | None,
) -> Any:
    applicable_guardrails = [
        (guardrail, _)
        for (guardrail, _) in (guardrails or [])
        if GuardrailScope.AGENT in guardrail.selector.scopes
    ]
    return create_guardrails_subgraph(
        main_inner_node=agent_node,
        guardrails=applicable_guardrails,
        scope=GuardrailScope.AGENT,
        node_factory=create_agent_guardrail_node,
    )


def create_tool_guardrails_subgraph(
    tool_node: tuple[str, Any],
    guardrails: Sequence[tuple[AgentGuardrail, GuardrailAction]] | None,
) -> Any:
    tool_name, _ = tool_node
    applicable_guardrails = [
        (guardrail, action)
        for (guardrail, action) in (guardrails or [])
        if GuardrailScope.TOOL in guardrail.selector.scopes
        and tool_name in guardrail.selector.match_names
    ]
    return create_guardrails_subgraph(
        main_inner_node=tool_node,
        guardrails=applicable_guardrails,
        scope=GuardrailScope.TOOL,
        node_factory=create_tool_guardrail_node,
    )
