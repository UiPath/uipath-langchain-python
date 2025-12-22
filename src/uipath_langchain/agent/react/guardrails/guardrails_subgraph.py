from functools import partial
from typing import Any, Callable, Mapping, Sequence

from langgraph._internal._runnable import RunnableCallable
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from uipath.core.guardrails import DeterministicGuardrail
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
            inner_name,
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
    """Create a guarded LLM node.

    Args:
        llm_node: Tuple of (node_name, node_callable) for the LLM node.
        guardrails: Optional sequence of (guardrail, action) tuples.

    Returns:
        Either the original node callable (if no applicable guardrails) or a compiled
        LangGraph subgraph that enforces the configured guardrails.
    """
    applicable_guardrails = [
        (guardrail, _)
        for (guardrail, _) in (guardrails or [])
        if GuardrailScope.LLM in guardrail.selector.scopes
        and not isinstance(guardrail, DeterministicGuardrail)
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
    tool_nodes: Mapping[str, RunnableCallable],
    guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] | None,
) -> dict[str, RunnableCallable]:
    """Create tool nodes with guardrails applied.
    Args:
        tool_nodes: Mapping of tool name to a LangGraph `ToolNode`.
        guardrails: Optional sequence of (guardrail, action) tuples.

    Returns:
        A mapping of tool name to either the original `ToolNode` or a compiled subgraph
        that enforces the matching tool guardrails.
    """
    result: dict[str, RunnableCallable] = {}
    for tool_name, tool_node in tool_nodes.items():
        subgraph = create_tool_guardrails_subgraph(
            (tool_name, tool_node),
            guardrails,
        )
        result[tool_name] = subgraph

    return result


def create_agent_init_guardrails_subgraph(
    init_node: tuple[str, Any],
    guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] | None,
) -> Any:
    """Create a subgraph for the INIT node and apply AGENT guardrails after INIT.

    This subgraph intentionally **runs the INIT node first** (so it can seed/normalize
    the agent state), and then evaluates guardrails as **PRE_EXECUTION**. This lets
    guardrails intended to run "before agent execution" validate the post-init state.

    Args:
        init_node: Tuple of (node_name, node_callable) for the INIT node.
        guardrails: Optional sequence of (guardrail, action) tuples.

    Returns:
        Either the original node callable (if no applicable guardrails) or a compiled
        LangGraph subgraph that runs INIT then enforces PRE_EXECUTION AGENT guardrails.
    """
    applicable_guardrails = [
        (guardrail, _)
        for (guardrail, _) in (guardrails or [])
        if GuardrailScope.AGENT in guardrail.selector.scopes
        and not isinstance(guardrail, DeterministicGuardrail)
    ]
    applicable_guardrails = _filter_guardrails_by_stage(
        applicable_guardrails, ExecutionStage.PRE_EXECUTION
    )
    if applicable_guardrails is None or len(applicable_guardrails) == 0:
        return init_node[1]

    inner_name, inner_node = init_node
    subgraph = StateGraph(AgentGuardrailsGraphState)
    subgraph.add_node(inner_name, inner_node)
    subgraph.add_edge(START, inner_name)

    first_guardrail_node = _build_guardrail_node_chain(
        subgraph=subgraph,
        guardrails=applicable_guardrails,
        scope=GuardrailScope.AGENT,
        execution_stage=ExecutionStage.PRE_EXECUTION,
        node_factory=create_agent_init_guardrail_node,
        next_node=END,
        guarded_node_name=inner_name,
    )
    subgraph.add_edge(inner_name, first_guardrail_node)
    return subgraph.compile()


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
        and not isinstance(guardrail, DeterministicGuardrail)
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
    """Create a guarded tool node.

    Args:
        tool_node: Tuple of (tool_name, tool_node_callable).
        guardrails: Optional sequence of (guardrail, action) tuples.

    Returns:
        Either the original tool node callable (if no matching guardrails) or a compiled
        LangGraph subgraph that enforces the matching tool guardrails.
    """
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
