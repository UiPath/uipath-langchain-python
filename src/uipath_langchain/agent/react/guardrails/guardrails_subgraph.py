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

from uipath_langchain.agent.guardrails.actions.base_action import (
    GuardrailAction,
    GuardrailActionNode,
)
from uipath_langchain.agent.guardrails.guardrail_nodes import (
    create_agent_init_guardrail_node,
    create_agent_terminate_guardrail_node,
    create_llm_guardrail_node,
    create_tool_guardrail_node,
)
from uipath_langchain.agent.guardrails.types import ExecutionStage
from uipath_langchain.agent.react.types import (
    AgentGraphState,
    AgentGuardrailsGraphState,
)

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
        guarded_node_name: Name of the component being guarded (used for action context).

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
    if applicable_guardrails is None or len(applicable_guardrails) == 0:
        return llm_node[1]

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


def attach_pre_agent_guardrails(
    builder: StateGraph[AgentGuardrailsGraphState],
    guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] | None,
    *,
    init_node_name: str,
    next_node_name: str,
) -> str:
    """Attach AGENT-scoped guardrails after INIT at the parent graph level.

    Args:
        builder: The parent `StateGraph` to attach nodes to.
        guardrails: All configured (guardrail, action) tuples.
        init_node_name: The node name of the INIT node in the parent graph.
        next_node_name: The node name to route to after the guardrails pass.

    Returns:
        The name of the first attached guardrail node, or `next_node_name` if no
        applicable guardrails exist.
    """
    applicable_guardrails = [
        (guardrail, action)
        for (guardrail, action) in (guardrails or [])
        if GuardrailScope.AGENT in guardrail.selector.scopes
    ]
    applicable_guardrails = _filter_guardrails_by_stage(
        applicable_guardrails, ExecutionStage.PRE_EXECUTION
    )
    if not applicable_guardrails:
        return next_node_name

    return _build_guardrail_node_chain(
        builder,
        applicable_guardrails,
        GuardrailScope.AGENT,
        ExecutionStage.PRE_EXECUTION,
        create_agent_init_guardrail_node,
        next_node_name,
        init_node_name,
    )


def attach_post_agent_guardrails(
    builder: StateGraph[AgentGuardrailsGraphState],
    terminate_node: Callable[[AgentGraphState], dict[str, Any]],
    guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] | None,
    *,
    terminate_node_name: str,
    next_node_name: str,
) -> str:
    """Attach POST_EXECUTION AGENT guardrails after TERMINATE at the parent graph level.

    Args:
        builder: The parent `StateGraph` to attach nodes to.
        terminate_node: The underlying terminate callable producing the agent output.
        guardrails: All configured (guardrail, action) tuples.
        terminate_node_name: The node name of the TERMINATE node in the parent graph.
        next_node_name: The node name to route to after the guardrails pass.

            Unlike INIT (where the "next" node is a real functional step like `AGENT`),
            TERMINATE needs a *final output node* to return the already-computed result.
            That is why callers typically pass `AgentGraphNode.GUARDED_TERMINATE` here:
            it is a small node that returns `state.agent_result` after guardrails pass.

    Returns:
        The node name that the caller should connect to END:
        - `terminate_node_name` when no applicable guardrails exist
        - `next_node_name` when POST_EXECUTION guardrails are attached
    """
    applicable_guardrails = [
        (guardrail, action)
        for (guardrail, action) in (guardrails or [])
        if GuardrailScope.AGENT in guardrail.selector.scopes
    ]
    applicable_guardrails = _filter_guardrails_by_stage(
        applicable_guardrails, ExecutionStage.POST_EXECUTION
    )

    # Fast path: no guardrails (or none applicable) -> keep the graph simple.
    if not applicable_guardrails:
        builder.add_node(terminate_node_name, terminate_node)
        return terminate_node_name

    def _terminate_store_result(state: AgentGraphState) -> dict[str, Any]:
        """Store terminate output in state so post-execution guardrails can validate it."""
        result = terminate_node(state)
        return {"agent_result": result}

    def _terminate_output(state: AgentGuardrailsGraphState) -> dict[str, Any]:
        """Return the terminate output as the graph output after guardrails passed."""
        if state.agent_result is None:
            raise ValueError("Missing `agent_result` in terminate output node.")
        return state.agent_result

    builder.add_node(terminate_node_name, _terminate_store_result)
    builder.add_node(next_node_name, _terminate_output)

    first_guardrail_node = _build_guardrail_node_chain(
        builder,
        applicable_guardrails,
        GuardrailScope.AGENT,
        ExecutionStage.POST_EXECUTION,
        create_agent_terminate_guardrail_node,
        next_node_name,
        terminate_node_name,
    )
    builder.add_edge(terminate_node_name, first_guardrail_node)
    return next_node_name


def create_agent_terminate_guardrails_subgraph(
    terminate_node: tuple[str, Any],
    guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] | None,
):
    """Create a subgraph for TERMINATE node that applies guardrails on the agent result."""
    node_name, node_func = terminate_node

    def terminate_wrapper(state: Any) -> dict[str, Any]:
        # Call original terminate node
        result = node_func(state)
        # Store result in state
        return {"agent_result": result, "messages": state.messages}

    applicable_guardrails = [
        (guardrail, _)
        for (guardrail, _) in (guardrails or [])
        if GuardrailScope.AGENT in guardrail.selector.scopes
    ]
    if applicable_guardrails is None or len(applicable_guardrails) == 0:
        return terminate_node[1]

    subgraph = _create_guardrails_subgraph(
        main_inner_node=(node_name, terminate_wrapper),
        guardrails=applicable_guardrails,
        scope=GuardrailScope.AGENT,
        execution_stages=[ExecutionStage.POST_EXECUTION],
        node_factory=create_agent_terminate_guardrail_node,
    )

    async def run_terminate_subgraph(
        state: AgentGraphState,
    ) -> dict[str, Any]:
        result_state = await subgraph.ainvoke(state)
        return result_state["agent_result"]

    return run_terminate_subgraph


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
    if applicable_guardrails is None or len(applicable_guardrails) == 0:
        return tool_node[1]

    return _create_guardrails_subgraph(
        main_inner_node=tool_node,
        guardrails=applicable_guardrails,
        scope=GuardrailScope.TOOL,
        execution_stages=[ExecutionStage.PRE_EXECUTION, ExecutionStage.POST_EXECUTION],
        node_factory=partial(create_tool_guardrail_node, tool_name=tool_name),
    )
