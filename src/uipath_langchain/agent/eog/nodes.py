"""Node functions for the EoG (Explanations over Graphs) agent graph.

Implements the EoG algorithm from arXiv:2601.17915:
- Bootstrap: fetch graph topology, seed entities, initialize beliefs
- Pop: BFS dequeue from active set
- Fetch context (Context Contract): bounded per-entity evidence via
  graph-aware function dispatch + 1-hop neighbor beliefs + inbox
- Policy (Abductive Policy π_abd): stateless LLM labels one entity
- Update: write belief to immutable ledger, track flips, apply damping
- Propagate: broadcast along GRAPH EDGES (not LLM suggestions)
- Frontier: compute minimal explanatory set (irreducible origins)
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .graph_topology import OntologyGraph
from .ontology_client import OntologyClient
from .types import Belief, EoGState, ExplanatoryEdge, InvestigationConfig, LedgerEntry

_NodeFn = Callable[[EoGState], Coroutine[Any, Any, dict[str, Any]]]

logger = logging.getLogger(__name__)

_POLICY_PROMPT_TEMPLATE = """\
You are investigating entity "{entity_id}" (type: {entity_type}) in an \
ontology-based investigation.

Available labels: {label_vocabulary}

Entity data (from ontology functions):
{function_data}

Graph neighbors and their current beliefs:
{neighbor_context}

Messages received from neighbors (belief propagation):
{inbox_messages}

Based on the evidence, assign ONE label from the vocabulary.
Explain your reasoning concisely.

Respond in valid JSON only:
{{"label": "...", "evidence": "..."}}"""


# ── Bootstrap ──────────────────────────────────────────────────────

def _make_bootstrap(
    client: OntologyClient,
    ontology: str,
    config: InvestigationConfig | None,
) -> _NodeFn:
    """Create bootstrap node: fetch graph topology, seed beliefs."""

    async def _bootstrap(state: EoGState) -> dict[str, Any]:
        cfg = config or state.investigation_config or InvestigationConfig(
            label_vocabulary=["Defer"]
        )

        # Fetch the ACTUAL graph topology from OWL + functions
        graph = await client.fetch_graph(ontology)

        beliefs: dict[str, Belief] = {}
        for entity_id in cfg.seed_entities:
            beliefs[entity_id] = Belief(
                label=cfg.default_label,
                evidence="Initial seed entity.",
            )

        return {
            "ontology_graph": graph.to_dict(),
            "beliefs": beliefs,
            "active_set": list(cfg.seed_entities),
            "steps_taken": 0,
            "investigation_config": cfg,
        }

    return _bootstrap


# ── Pop ────────────────────────────────────────────────────────────

async def pop_node(state: EoGState) -> dict[str, Any]:
    """Dequeue next entity from active_set (FIFO BFS order)."""
    active = list(state.active_set)
    entity = active.pop(0) if active else ""
    return {"current_entity": entity, "active_set": active}


def should_continue(state: EoGState) -> str:
    """Route: continue traversal or compute frontier."""
    cfg = state.investigation_config
    max_steps = cfg.max_steps if cfg else 50

    if state.current_entity and state.steps_taken < max_steps:
        return "fetch_context"
    return "frontier"


# ── Context Contract (CxC) ─────────────────────────────────────────

def _make_fetch_context(
    client: OntologyClient,
    ontology: str,
) -> _NodeFn:
    """Create fetch_context node: graph-aware bounded evidence per entity."""

    async def _fetch_context(state: EoGState) -> dict[str, Any]:
        entity_id = state.current_entity
        cfg = state.investigation_config
        max_results = cfg.max_results_per_function if cfg else 50

        # Reconstruct graph from state
        graph = OntologyGraph.from_dict(state.ontology_graph)
        entity_type = graph.entity_for_id(entity_id)

        # 1. Topological context: 1-hop neighbors from the GRAPH
        neighbor_beliefs: dict[str, dict[str, Any]] = {}
        if entity_type:
            for edge in graph.edges_of(entity_type):
                # Find instance-level neighbors from beliefs
                other_type = edge.target if edge.source == entity_type else edge.source
                for eid, belief in state.beliefs.items():
                    if eid != entity_id and graph.entity_for_id(eid) == other_type:
                        neighbor_beliefs[eid] = {
                            "label": belief.label,
                            "evidence": belief.evidence,
                            "entity_type": other_type,
                            "relationship": edge.label,
                        }

        # 2. Inbox messages (from belief propagation)
        inbox_messages = state.inbox.get(entity_id, [])

        # 3. Invoke relevant functions using graph-aware matching
        function_results: list[dict[str, Any]] = []
        if entity_type:
            fns = graph.functions_for(entity_type)
            for fn in fns:
                fn_name = fn.get("name", "")
                params = fn.get("params", [])

                # Build params: match function param names to entity ID
                call_params = _bind_params(params, entity_id, entity_type)
                if call_params is None and params:
                    # Function has required params we can't bind — skip
                    continue

                try:
                    result = await client.invoke_function(
                        ontology, fn_name, call_params
                    )
                    rows = result.get("rows", [])
                    function_results.append({
                        "function": fn_name,
                        "rows": rows[:max_results],
                        "row_count": len(rows),
                    })
                except Exception:
                    logger.debug(
                        "Function %s failed for %s", fn_name, entity_id,
                        exc_info=True,
                    )

        context_packet: dict[str, Any] = {
            "entity_id": entity_id,
            "entity_type": entity_type or "Unknown",
            "function_results": function_results,
            "neighbor_beliefs": neighbor_beliefs,
            "inbox_messages": inbox_messages,
        }
        return {"context_packet": context_packet}

    return _fetch_context


def _bind_params(
    params: list[dict[str, Any]],
    entity_id: str,
    entity_type: str,
) -> dict[str, str] | None:
    """Bind function params to the current entity ID.

    Maps param names to the entity ID using naming conventions:
    - ``invoiceId`` for entity type ``Invoice`` with ID ``INV-2004``
    - ``supplierId`` for entity type ``Supplier`` with ID ``SUP-001``

    Returns None if a required param can't be bound.
    """
    if not params:
        return None

    bound: dict[str, str] = {}
    for param in params:
        pname = param.get("name", "")
        required = param.get("required", False)

        # Try: paramName matches entityType + "Id" pattern
        expected_key = entity_type[0].lower() + entity_type[1:] + "Id"
        alt_key = entity_type.lower() + "Id"
        parent_key = "parent" + entity_type + "Id"

        if pname == expected_key or pname == alt_key or pname == parent_key:
            bound[pname] = entity_id
        elif pname.endswith("Id") and pname[:-2].lower() == entity_type.lower():
            bound[pname] = entity_id
        elif required:
            # Required param we can't bind — this function doesn't
            # apply to a single instance of this entity type
            return None

    return bound if bound else None


# ── Abductive Policy (π_abd) ───────────────────────────────────────

def _make_policy(model: BaseChatModel) -> _NodeFn:
    """Create policy node: stateless LLM labels one entity."""

    async def _policy(state: EoGState) -> dict[str, Any]:
        cfg = state.investigation_config
        label_vocab = cfg.label_vocabulary if cfg else ["Defer"]

        ctx = state.context_packet
        entity_id = ctx.get("entity_id", state.current_entity)
        entity_type = ctx.get("entity_type", "Unknown")

        function_data = json.dumps(
            ctx.get("function_results", []),
            indent=2, default=str,
        )[:3000]  # Bounded context

        neighbor_ctx = json.dumps(
            ctx.get("neighbor_beliefs", {}),
            indent=2, default=str,
        )[:1000]

        inbox_str = json.dumps(
            ctx.get("inbox_messages", []),
            indent=2, default=str,
        )[:1000]

        prompt = _POLICY_PROMPT_TEMPLATE.format(
            entity_id=entity_id,
            entity_type=entity_type,
            label_vocabulary=", ".join(label_vocab),
            function_data=function_data,
            neighbor_context=neighbor_ctx,
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
            # Strip markdown fences if present
            content = content.strip()
            if content.startswith("```"):
                content = "\n".join(content.split("\n")[1:])
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
            parsed = json.loads(content)

            # Enforce label vocabulary
            label = parsed.get("label", "Defer")
            if label not in label_vocab:
                label = "Defer"
            parsed["label"] = label

        except Exception:
            logger.warning(
                "Policy: LLM parse failed for %s", entity_id,
            )
            parsed = {
                "label": "Defer",
                "evidence": "LLM output could not be parsed",
            }

        return {"policy_result": parsed}

    return _policy


# ── Update (Ledger + Damping) ──────────────────────────────────────

async def update_node(state: EoGState) -> dict[str, Any]:
    """Write belief to ledger, track flips, apply damping.

    Damping: if flip_count exceeds max_flips, force label to Defer
    (absorbing state) to prevent oscillation.
    """
    entity_id = state.current_entity
    cfg = state.investigation_config
    max_flips = cfg.max_flips if cfg else 3

    new_label = state.policy_result.get("label", "Defer")
    evidence = state.policy_result.get("evidence", "")

    existing = state.beliefs.get(entity_id)
    old_label = existing.label if existing else None
    old_flip_count = existing.flip_count if existing else 0

    flip_count = old_flip_count
    if old_label is not None and old_label != new_label:
        flip_count += 1

    # Damping: force Defer if oscillating too much
    if flip_count > max_flips:
        new_label = cfg.default_label if cfg else "Defer"
        evidence = f"Damped after {flip_count} flips (was: {new_label})"

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


# ── Propagate (follows GRAPH EDGES, not LLM) ──────────────────────

async def propagate_node(state: EoGState) -> dict[str, Any]:
    """Broadcast belief change to graph neighbors.

    Key difference from the broken version: propagation follows the
    ontology's relationship edges, NOT the LLM's suggestions. The
    controller decides who to re-examine based on graph topology.

    Only re-activates neighbors when the current entity's label CHANGED.
    """
    cfg = state.investigation_config
    max_flips = cfg.max_flips if cfg else 3

    entity_id = state.current_entity
    current_belief = state.beliefs.get(entity_id)

    # Check if belief actually changed
    old_label = None
    for entry in reversed(state.ledger):
        if entry.entity_id == entity_id:
            old_label = entry.old_label
            break

    belief_changed = (old_label is not None and old_label != current_belief.label) \
        if current_belief else False

    if not belief_changed:
        return {}

    # Reconstruct graph to find neighbors
    graph = OntologyGraph.from_dict(state.ontology_graph)
    entity_type = graph.entity_for_id(entity_id)

    updated_inbox: dict[str, list[dict[str, Any]]] = {}
    new_active: list[str] = []
    explanatory_edges: list[ExplanatoryEdge] = []

    if entity_type:
        # Find all instance-level neighbors via graph edges
        neighbor_types: set[str] = set()
        edge_map: dict[str, str] = {}  # neighbor_type → relationship label
        for edge in graph.edges_of(entity_type):
            other = edge.target if edge.source == entity_type else edge.source
            neighbor_types.add(other)
            edge_map[other] = edge.label

        # Match neighbor types to actual entity instances in beliefs
        for neighbor_id, neighbor_belief in state.beliefs.items():
            if neighbor_id == entity_id:
                continue
            neighbor_type = graph.entity_for_id(neighbor_id)
            if neighbor_type not in neighbor_types:
                continue

            rel_label = edge_map.get(neighbor_type, "related")

            # Send message to neighbor
            message: dict[str, Any] = {
                "from": entity_id,
                "from_type": entity_type,
                "label": current_belief.label,
                "evidence": current_belief.evidence[:200],
                "relationship": rel_label,
            }

            existing_messages = list(state.inbox.get(neighbor_id, []))
            existing_messages.append(message)
            updated_inbox[neighbor_id] = existing_messages

            # Re-activate if under flip limit
            if (
                neighbor_belief.flip_count < max_flips
                and neighbor_id not in state.active_set
                and neighbor_id not in new_active
            ):
                new_active.append(neighbor_id)

            explanatory_edges.append(ExplanatoryEdge(
                source=entity_id,
                target=neighbor_id,
                relationship=rel_label,
                evidence=current_belief.evidence[:100],
            ))

    return {
        "inbox": updated_inbox,
        "active_set": list(state.active_set) + new_active,
        "explanatory_edges": explanatory_edges,
    }


# ── Frontier (minimal explanatory set) ─────────────────────────────

async def frontier_node(state: EoGState) -> dict[str, Any]:
    """Compute the minimal explanatory frontier.

    The frontier is the set of entities whose label is NOT the default,
    filtered for irreducibility: if entity A (Source) has an explanatory
    edge to entity B (also Source), and A explains B, then B is removed
    from the frontier (it's explained by A).

    This implements F = {v ∈ V_S : L_v ≠ Defer ∧ ¬∃u (L_u is Source ∧ u→v)}
    from the paper.
    """
    cfg = state.investigation_config
    default_label = cfg.default_label if cfg else "Defer"

    # All non-default beliefs
    findings: dict[str, dict[str, Any]] = {}
    for entity_id, belief in state.beliefs.items():
        if belief.label != default_label:
            graph = OntologyGraph.from_dict(state.ontology_graph)
            findings[entity_id] = {
                "entity": entity_id,
                "entity_type": graph.entity_for_id(entity_id) or "Unknown",
                "label": belief.label,
                "evidence": belief.evidence,
                "flip_count": belief.flip_count,
            }

    # Compute irreducibility: remove entities explained by another
    # entity via an explanatory edge
    explained: set[str] = set()
    for edge in state.explanatory_edges:
        src_belief = state.beliefs.get(edge.source)
        tgt_belief = state.beliefs.get(edge.target)
        if src_belief and tgt_belief:
            # If source is a "Source" and target is "DerivedEffect",
            # the target is explained by the source
            if src_belief.label == "Source" and tgt_belief.label == "DerivedEffect":
                explained.add(edge.target)

    # Frontier = findings minus explained entities
    frontier_list = [
        item for eid, item in findings.items()
        if eid not in explained
    ]

    return {"frontier": frontier_list}
