from functools import partial
from typing import Any, Callable, Sequence

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from uipath.platform.guardrails import (
    BaseGuardrail,
    BuiltInValidatorGuardrail,
    GuardrailScope,
)

from uipath_langchain.agent.guardrails.types import ExecutionStage

from .actions.base_action import GuardrailAction, GuardrailActionNode
from .guardrail_nodes import (
    create_agent_guardrail_node,
    create_llm_guardrail_node,
    create_tool_guardrail_node,
)
from .types import AgentGuardrailsGraphState

_VALIDATOR_ALLOWED_STAGES = {
    "prompt_injection": {ExecutionStage.PRE_EXECUTION},
    "pii_detection": {ExecutionStage.PRE_EXECUTION, ExecutionStage.POST_EXECUTION},
}


def _filter_guardrails_by_stage(
    guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] | None,
    stage: ExecutionStage,
) -> list[tuple[BaseGuardrail, GuardrailAction]]:
    """Filter guardrails that apply to a specific execution stage."""
    filtered_guardrails = []
    for guardrail, action in guardrails or []:
        # Internal knowledge: Check against configured allowed stages
        if (
            isinstance(guardrail, BuiltInValidatorGuardrail)
            and guardrail.validator_type in _VALIDATOR_ALLOWED_STAGES
            and stage not in _VALIDATOR_ALLOWED_STAGES[guardrail.validator_type]
        ):
            continue
        filtered_guardrails.append((guardrail, action))
    return filtered_guardrails


def _create_guardrails_subgraph(
    main_inner_node: tuple[str, Any],
    guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] | None,
    scope: GuardrailScope,
    execution_stages: Sequence[ExecutionStage],
    node_factory: Callable[
        [
            BaseGuardrail,
            ExecutionStage,
            str,  # success node name
            str,  # fail node name
        ],
        GuardrailActionNode,
    ] = create_llm_guardrail_node,
):
    """Build a subgraph that enforces guardrails around an inner node.

    The constructed graph conditionally includes pre- and/or post-execution guardrail
    chains based on ``execution_stages``:
    - If ``ExecutionStage.PRE_EXECUTION`` is included, the graph links
      START -> first pre-guardrail node -> ... -> inner.
      Otherwise, it directly links START -> inner.
    - If ``ExecutionStage.POST_EXECUTION`` is included, the graph links
      inner -> first post-guardrail node -> ... -> END.
      Otherwise, it directly links inner -> END.

    No static edges are added between guardrail nodes; each evaluation node routes
    dynamically to its configured success/failure targets. Failure nodes are added
    but not chained; they are expected to route via Command to the provided next node.
    """
    inner_name, inner_node = main_inner_node

    subgraph = StateGraph(AgentGuardrailsGraphState)

    subgraph.add_node(inner_name, inner_node)

    # Add pre execution guardrail nodes
    if ExecutionStage.PRE_EXECUTION in execution_stages:
        pre_guardrails = _filter_guardrails_by_stage(
            guardrails, ExecutionStage.PRE_EXECUTION
        )
        first_pre_exec_guardrail_node = _build_guardrail_node_chain(
            subgraph,
            pre_guardrails,
            scope,
            ExecutionStage.PRE_EXECUTION,
            node_factory,
            inner_name,
            inner_name,
        )
        subgraph.add_edge(START, first_pre_exec_guardrail_node)
    else:
        subgraph.add_edge(START, inner_name)

    # Add post execution guardrail nodes
    if ExecutionStage.POST_EXECUTION in execution_stages:
        post_guardrails = _filter_guardrails_by_stage(
            guardrails, ExecutionStage.POST_EXECUTION
        )
        first_post_exec_guardrail_node = _build_guardrail_node_chain(
            subgraph,
            post_guardrails,
            scope,
            ExecutionStage.POST_EXECUTION,
            node_factory,
            END,
            inner_node,
        )
        subgraph.add_edge(inner_name, first_post_exec_guardrail_node)
    else:
        subgraph.add_edge(inner_name, END)

    return subgraph.compile()


def _build_guardrail_node_chain(
    subgraph: StateGraph[AgentGuardrailsGraphState],
    guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] | None,
    scope: GuardrailScope,
    execution_stage: ExecutionStage,
    node_factory: Callable[
        [
            BaseGuardrail,
            ExecutionStage,
            str,  # success node name
            str,  # fail node name
        ],
        GuardrailActionNode,
    ],
    next_node: str,
    guarded_node_name: str,
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
        guardrail=guardrail,
        scope=scope,
        execution_stage=execution_stage,
        guarded_component_name=guarded_node_name,
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
        guarded_node_name,
    )

    return previous_node_name


def create_llm_guardrails_subgraph(
    llm_node: tuple[str, Any],
    guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] | None,
):
    applicable_guardrails = [
        (guardrail, _)
        for (guardrail, _) in (guardrails or [])
        if GuardrailScope.LLM in guardrail.selector.scopes
    ]
    return _create_guardrails_subgraph(
        main_inner_node=llm_node,
        guardrails=applicable_guardrails,
        scope=GuardrailScope.LLM,
        execution_stages=[ExecutionStage.PRE_EXECUTION, ExecutionStage.POST_EXECUTION],
        node_factory=create_llm_guardrail_node,
    )


def create_tools_guardrails_subgraph(
    tool_nodes: dict[str, ToolNode],
    guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] | None,
) -> dict[str, ToolNode]:
    """Create tool nodes with guardrails.
    Args:
    """
    result: dict[str, ToolNode] = {}
    for tool_name, tool_node in tool_nodes.items():
        subgraph = create_tool_guardrails_subgraph(
            (tool_name, tool_node),
            guardrails,
        )
        result[tool_name] = subgraph

    return result


def create_agent_guardrails_subgraph(
    agent_node: tuple[str, Any],
    guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] | None,
    execution_stage: ExecutionStage,
):
    """Create a subgraph for AGENT-scoped guardrails that applies checks at the specified stage.

    This is intended for wrapping nodes like INIT or TERMINATE, where guardrails should run
    either before (pre-execution) or after (post-execution) the node logic.
    """
    applicable_guardrails = [
        (guardrail, _)
        for (guardrail, _) in (guardrails or [])
        if GuardrailScope.AGENT in guardrail.selector.scopes
    ]
    return _create_guardrails_subgraph(
        main_inner_node=agent_node,
        guardrails=applicable_guardrails,
        scope=GuardrailScope.AGENT,
        execution_stages=[execution_stage],
        node_factory=create_agent_guardrail_node,
    )


def create_tool_guardrails_subgraph(
    tool_node: tuple[str, Any],
    guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] | None,
):
    tool_name, _ = tool_node
    applicable_guardrails = [
        (guardrail, action)
        for (guardrail, action) in (guardrails or [])
        if GuardrailScope.TOOL in guardrail.selector.scopes
        and guardrail.selector.match_names is not None
        and tool_name in guardrail.selector.match_names
    ]
    return _create_guardrails_subgraph(
        main_inner_node=tool_node,
        guardrails=applicable_guardrails,
        scope=GuardrailScope.TOOL,
        execution_stages=[ExecutionStage.PRE_EXECUTION, ExecutionStage.POST_EXECUTION],
        node_factory=partial(create_tool_guardrail_node, tool_name=tool_name),
    )
