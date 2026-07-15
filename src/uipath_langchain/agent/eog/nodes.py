"""Node functions for the EoG (Explanations over Graphs) agent graph."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .ontology_client import OntologyClient
from .types import Belief, EoGState, ExplanatoryEdge, InvestigationConfig, LedgerEntry

# Type alias for async node functions
_NodeFn = Callable[[EoGState], Coroutine[Any, Any, dict[str, Any]]]

logger = logging.getLogger(__name__)

_POLICY_PROMPT_TEMPLATE = """\
You are investigating entity "{entity_id}" in an ontology-based investigation.

Available labels: {label_vocabulary}

Evidence collected for this entity:
{context_data}

Messages from neighboring entities:
{inbox_messages}

Based on the evidence, assign ONE label from the vocabulary to this entity.
Explain your reasoning concisely.
If you believe other entities should be re-examined based on your findings, \
list them.

Respond in JSON: {{"label": "...", "evidence": "...", \
"propagations": [{{"entity": "...", "reason": "..."}}]}}"""


def _make_bootstrap(
    client: OntologyClient,
    ontology: str,
    config: InvestigationConfig | None,
) -> _NodeFn:
    """Create bootstrap node with injected dependencies."""

    async def _bootstrap(state: EoGState) -> dict[str, Any]:
        """Initialize: discover graph, set up beliefs for seed entities."""
        cfg = config or state.investigation_config or InvestigationConfig(
            label_vocabulary=["Defer"]
        )

        graph_meta, functions = await asyncio.gather(
            client.discover(ontology),
            client.list_functions(ontology),
        )

        beliefs: dict[str, Belief] = {}
        for entity_id in cfg.seed_entities:
            beliefs[entity_id] = Belief(
                label=cfg.default_label,
                evidence="Initial seed entity.",
            )

        return {
            "ontology_graph": {
                "metadata": graph_meta,
                "functions": functions,
            },
            "beliefs": beliefs,
            "active_set": list(cfg.seed_entities),
            "steps_taken": 0,
            "investigation_config": cfg,
        }

    return _bootstrap


async def pop_node(state: EoGState) -> dict[str, Any]:
    """Dequeue next entity from active_set (FIFO).

    Returns:
        Partial state update with current_entity and remaining active_set.
    """
    active = list(state.active_set)
    entity = active.pop(0) if active else ""
    return {"current_entity": entity, "active_set": active}


def should_continue(state: EoGState) -> str:
    """Route: continue traversal or compute frontier.

    Checks whether the current entity (just popped) is non-empty and the
    step budget has not been exhausted.

    Returns:
        ``"fetch_context"`` if there is an entity to process and budget
        remains, ``"frontier"`` otherwise.
    """
    cfg = state.investigation_config
    max_steps = cfg.max_steps if cfg else 50

    if state.current_entity and state.steps_taken < max_steps:
        return "fetch_context"
    return "frontier"


def _make_fetch_context(
    client: OntologyClient,
    ontology: str,
) -> _NodeFn:
    """Create fetch_context node with injected dependencies."""

    async def _fetch_context(state: EoGState) -> dict[str, Any]:
        """Build bounded context packet for the current entity."""
        entity_id = state.current_entity
        cfg = state.investigation_config
        max_results = cfg.max_results_per_function if cfg else 50

        current_belief = state.beliefs.get(entity_id)
        inbox_messages = state.inbox.get(entity_id, [])

        # Gather neighbor beliefs from ontology_graph edges if available
        neighbor_beliefs: dict[str, dict[str, Any]] = {}
        graph_meta = state.ontology_graph.get("metadata", {})
        edges = graph_meta.get("edges", [])
        for edge in edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            neighbour = None
            rel = edge.get("relationship", "")
            if src == entity_id:
                neighbour = tgt
            elif tgt == entity_id:
                neighbour = src
            if neighbour and neighbour in state.beliefs:
                b = state.beliefs[neighbour]
                neighbor_beliefs[neighbour] = {
                    "label": b.label,
                    "evidence": b.evidence,
                    "relationship": rel,
                }

        # Invoke relevant functions
        functions = state.ontology_graph.get("functions", [])
        function_results: list[dict[str, Any]] = []
        for fn in functions:
            fn_name = fn.get("name", "")
            fn_desc = fn.get("description", "")
            if entity_id.lower() in fn_desc.lower() or not fn_desc:
                try:
                    result = await client.invoke_function(
                        ontology, fn_name, {"entity_id": entity_id}
                    )
                    rows = result.get("rows", [])
                    function_results.append({
                        "function": fn_name,
                        "rows": rows[:max_results],
                    })
                except Exception:
                    logger.warning(
                        "Function %s failed for entity %s",
                        fn_name,
                        entity_id,
                        exc_info=True,
                    )

        context_packet: dict[str, Any] = {
            "entity_id": entity_id,
            "current_belief": current_belief.model_dump() if current_belief else None,
            "inbox_messages": inbox_messages,
            "neighbor_beliefs": neighbor_beliefs,
            "function_results": function_results,
        }
        return {"context_packet": context_packet}

    return _fetch_context


def _make_policy(model: BaseChatModel) -> _NodeFn:
    """Create policy node with injected LLM."""

    async def _policy(state: EoGState) -> dict[str, Any]:
        """LLM call: label entity and identify propagation claims."""
        cfg = state.investigation_config
        label_vocab = cfg.label_vocabulary if cfg else ["Defer"]

        ctx = state.context_packet
        entity_id = ctx.get("entity_id", state.current_entity)

        context_data = json.dumps(
            {
                k: v
                for k, v in ctx.items()
                if k not in ("entity_id", "inbox_messages")
            },
            indent=2,
            default=str,
        )
        inbox_str = json.dumps(
            ctx.get("inbox_messages", []), indent=2, default=str
        )

        prompt = _POLICY_PROMPT_TEMPLATE.format(
            entity_id=entity_id,
            label_vocabulary=", ".join(label_vocab),
            context_data=context_data,
            inbox_messages=inbox_str,
        )

        messages = [
            SystemMessage(content="You are an ontology investigation agent."),
            HumanMessage(content=prompt),
        ]

        try:
            response = await model.ainvoke(messages)
            content = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            parsed = json.loads(content)
        except Exception:
            logger.warning(
                "Policy node: LLM output could not be parsed for entity %s",
                entity_id,
            )
            parsed = {
                "label": "Defer",
                "evidence": "LLM output could not be parsed",
                "propagations": [],
            }

        return {"policy_result": parsed}

    return _policy


async def update_node(state: EoGState) -> dict[str, Any]:
    """Write belief to ledger and track flips.

    Returns:
        Partial state update with updated belief, ledger entry, and
        incremented steps_taken.
    """
    entity_id = state.current_entity
    new_label = state.policy_result.get("label", "Defer")
    evidence = state.policy_result.get("evidence", "")

    existing = state.beliefs.get(entity_id)
    old_label = existing.label if existing else None
    old_flip_count = existing.flip_count if existing else 0

    flip_count = old_flip_count
    if old_label is not None and old_label != new_label:
        flip_count += 1

    updated_belief = Belief(
        label=new_label,
        evidence=evidence,
        flip_count=flip_count,
    )

    entry = LedgerEntry(
        timestamp=time.time(),
        entity_id=entity_id,
        old_label=old_label,
        new_label=new_label,
        evidence=evidence,
    )

    return {
        "beliefs": {entity_id: updated_belief},
        "ledger": [entry],
        "steps_taken": state.steps_taken + 1,
    }


async def propagate_node(state: EoGState) -> dict[str, Any]:
    """Broadcast beliefs to neighbours and re-activate if changed.

    Returns:
        Partial state update with updated inbox and active_set.
    """
    cfg = state.investigation_config
    max_flips = cfg.max_flips if cfg else 3

    propagations = state.policy_result.get("propagations", [])
    entity_id = state.current_entity
    current_belief = state.beliefs.get(entity_id)

    updated_inbox: dict[str, list[dict[str, Any]]] = {}
    new_active: list[str] = []
    explanatory_edges: list[ExplanatoryEdge] = []

    for prop in propagations:
        target = prop.get("entity", "")
        reason = prop.get("reason", "")
        if not target:
            continue

        message: dict[str, Any] = {
            "from": entity_id,
            "label": current_belief.label if current_belief else "Defer",
            "evidence": current_belief.evidence if current_belief else "",
            "reason": reason,
        }

        existing_messages = list(state.inbox.get(target, []))
        existing_messages.append(message)
        updated_inbox[target] = existing_messages

        target_belief = state.beliefs.get(target)
        target_flips = target_belief.flip_count if target_belief else 0

        if (
            target_flips < max_flips
            and target not in state.active_set
            and target not in new_active
        ):
            new_active.append(target)

        explanatory_edges.append(
            ExplanatoryEdge(
                source=entity_id,
                target=target,
                relationship="propagation",
                evidence=reason,
            )
        )

    return {
        "inbox": updated_inbox,
        "active_set": list(state.active_set) + new_active,
        "explanatory_edges": explanatory_edges,
    }


async def frontier_node(state: EoGState) -> dict[str, Any]:
    """Compute the minimal explanatory frontier.

    Filters beliefs to non-Defer labels and builds a summary list.

    Returns:
        Partial state update with the frontier list.
    """
    cfg = state.investigation_config
    default_label = cfg.default_label if cfg else "Defer"

    frontier_list: list[dict[str, Any]] = []
    for entity_id, belief in state.beliefs.items():
        if belief.label != default_label:
            frontier_list.append({
                "entity": entity_id,
                "label": belief.label,
                "evidence": belief.evidence,
            })

    return {"frontier": frontier_list}
