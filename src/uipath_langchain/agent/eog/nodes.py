"""Node functions for the EoG (Explanations over Graphs) agent.

Redesigned algorithm — no upfront graph fetch. Function definitions are
the navigation contract: ``touches`` = adjacency, ``outputs`` = evidence
schema, ``params`` = what you need to get there.

Flow: seed → pop → discover → gather → label → update → propagate → pop
      └─ synthesize (when active_set empty or budget exhausted)
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .ontology_client import OntologyClient
from .types import (
    Belief,
    EoGState,
    ExplanatoryEdge,
    FunctionSpec,
    InvestigationConfig,
    LedgerEntry,
)

_NodeFn = Callable[[EoGState], Coroutine[Any, Any, dict[str, Any]]]

logger = logging.getLogger(__name__)

# ── ID prefix → entity type mapping ──────────────────────────────

_ID_PREFIXES: dict[str, str] = {
    "INV-": "Invoice",
    "SUP-": "Supplier",
    "PO-": "PurchaseOrder",
    "EXC-": "ToleranceException",
    "COM-": "Commodity",
    "RULE-": "ExceptionRule",
    "CTR-": "Contract",
    "PR-": "Requisition",
    "SPD-": "SpendRecord",
    "FX-": "FXRate",
    "SCP-": "ContractScope",
    "CC-": "CostCenter",
    "BUD-": "Budget",
    "ITM-": "Item",
}


def resolve_entity_type(entity_id: str) -> str | None:
    """Resolve an entity ID to its type via prefix convention."""
    for prefix, entity_type in _ID_PREFIXES.items():
        if entity_id.startswith(prefix):
            return entity_type
    return None


# ── Seed ──────────────────────────────────────────────────────────

def _make_seed(config: InvestigationConfig | None) -> _NodeFn:
    """Create seed node: initialize beliefs, no graph fetch."""

    async def _seed(state: EoGState) -> dict[str, Any]:
        cfg = config or state.investigation_config or InvestigationConfig(
            label_vocabulary=["Defer"]
        )

        beliefs: dict[str, Belief] = {}
        discovered: dict[str, str] = {}
        for entity_id in cfg.seed_records:
            beliefs[entity_id] = Belief(
                label=cfg.default_label,
                evidence="Initial seed record.",
            )
            entity_type = resolve_entity_type(entity_id)
            if entity_type:
                discovered[entity_id] = entity_type

        return {
            "beliefs": beliefs,
            "discovered_records": discovered,
            "active_set": list(cfg.seed_records),
            "steps_taken": 0,
            "investigation_config": cfg,
        }

    return _seed


# ── Pop ───────────────────────────────────────────────────────────

async def pop_node(state: EoGState) -> dict[str, Any]:
    """Dequeue next record from active_set using priority ordering.

    Priority: records with non-default labels (Source, PolicyViolation)
    are visited before Defer records. Within the same priority,
    FIFO order is preserved.
    """
    active = list(state.active_set)
    if not active:
        return {"current_record": "", "active_set": []}

    cfg = state.investigation_config
    default_label = cfg.default_label if cfg else "Defer"

    # Sort: non-default labels first (they have evidence worth propagating),
    # then records with inbox messages, then the rest
    def _priority(eid: str) -> int:
        belief = state.beliefs.get(eid)
        if not belief:
            return 2  # unknown — low priority
        if belief.label != default_label:
            return 0  # active finding — high priority
        if state.inbox.get(eid):
            return 1  # has inbox messages — medium
        return 2  # default label, no messages — low

    active.sort(key=_priority)
    entity = active.pop(0)
    return {"current_record": entity, "active_set": active}


def should_continue(state: EoGState) -> str:
    """Route: continue traversal or synthesize.

    Stops when any of:
    - active_set is empty (no more records to visit)
    - step budget exhausted
    - token budget exhausted
    - convergence: N consecutive visits with no belief change
    """
    if not state.current_record:
        return "synthesize"

    cfg = state.investigation_config
    max_steps = cfg.max_steps if cfg else 50
    max_tokens = cfg.max_tokens if cfg else 0
    convergence_window = cfg.convergence_window if cfg else 5

    # Step budget
    if state.steps_taken >= max_steps:
        logger.info("Budget: step limit %d reached", max_steps)
        return "synthesize"

    # Token budget
    total_tokens = state.total_input_tokens + state.total_output_tokens
    if max_tokens > 0 and total_tokens >= max_tokens:
        logger.info("Budget: token limit %d reached (%d used)", max_tokens, total_tokens)
        return "synthesize"

    # Convergence: no belief changes in last N visits
    if convergence_window > 0 and state.consecutive_no_change >= convergence_window:
        logger.info(
            "Converged: %d consecutive visits with no belief change",
            state.consecutive_no_change,
        )
        return "synthesize"

    return "discover"


# ── Discover ──────────────────────────────────────────────────────

def _make_discover(
    client: OntologyClient,
    ontology: str,
) -> _NodeFn:
    """Create discover node: fetch functions touching this entity type.

    Results are cached in ``function_cache`` per entity type so
    subsequent visits to the same type skip the network call.
    """

    async def _discover(state: EoGState) -> dict[str, Any]:
        entity_id = state.current_record
        entity_type = (
            state.discovered_records.get(entity_id)
            or resolve_entity_type(entity_id)
        )

        if not entity_type:
            return {"context_packet": {"entity_id": entity_id, "entity_type": "Unknown", "functions": [], "evidence": []}}

        # Cache hit — skip network call
        if entity_type in state.function_cache:
            fn_dicts = state.function_cache[entity_type]
        else:
            fn_dicts = await client.list_functions(
                ontology, touches=entity_type,
            )

        return {
            "function_cache": {entity_type: fn_dicts},
            "context_packet": {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "functions": fn_dicts,
                "evidence": [],
            },
        }

    return _discover


# ── Gather ────────────────────────────────────────────────────────

def _make_gather(
    client: OntologyClient,
    ontology: str,
) -> _NodeFn:
    """Create gather node: invoke functions, chain params from results.

    Two-phase binding:
    1. Primary: bind params matching the current record's key.
    2. Secondary: bind params from values discovered in primary results.
    """

    async def _gather(state: EoGState) -> dict[str, Any]:
        ctx = state.context_packet
        entity_id = ctx.get("entity_id", state.current_record)
        entity_type = ctx.get("entity_type", "Unknown")
        fn_dicts = ctx.get("functions", [])
        cfg = state.investigation_config
        max_results = cfg.max_results_per_function if cfg else 50

        functions = [FunctionSpec(**fd) for fd in fn_dicts]

        # Phase 1: invoke functions whose key param matches entity ID
        primary_results: list[dict[str, Any]] = []
        deferred: list[FunctionSpec] = []
        discovered_values: dict[str, str] = {}  # param_name → value

        for fn in functions:
            bound = _bind_primary(fn, entity_id, entity_type)
            if bound is not None:
                result = await _safe_invoke(
                    client, ontology, fn.name, bound, max_results,
                )
                if result:
                    primary_results.append(result)
                    # Extract values for secondary binding
                    _extract_values(result, fn, discovered_values)
            elif fn.required_params:
                deferred.append(fn)
            else:
                # No params required — invoke directly
                result = await _safe_invoke(
                    client, ontology, fn.name, None, max_results,
                )
                if result:
                    primary_results.append(result)
                    _extract_values(result, fn, discovered_values)

        # Phase 2: invoke deferred functions using discovered values
        secondary_results: list[dict[str, Any]] = []
        for fn in deferred:
            bound = _bind_secondary(fn, discovered_values)
            if bound is not None:
                result = await _safe_invoke(
                    client, ontology, fn.name, bound, max_results,
                )
                if result:
                    secondary_results.append(result)
                    _extract_values(result, fn, discovered_values)

        all_results = primary_results + secondary_results

        # Discover new records from result values
        new_records: dict[str, str] = {}
        for param_name, value in discovered_values.items():
            if not value or value == entity_id:
                continue
            discovered_type = _type_from_param_name(param_name)
            if discovered_type and value not in state.beliefs:
                new_records[value] = discovered_type

        # Build neighbor beliefs from existing beliefs
        neighbor_beliefs: dict[str, dict[str, Any]] = {}
        for eid, belief in state.beliefs.items():
            if eid != entity_id and eid in new_records:
                neighbor_beliefs[eid] = {
                    "label": belief.label,
                    "evidence": belief.evidence,
                    "entity_type": new_records.get(eid, "Unknown"),
                }
        # Also include already-known neighbors
        for eid, belief in state.beliefs.items():
            if eid != entity_id and eid not in neighbor_beliefs:
                eid_type = state.discovered_records.get(eid)
                if eid_type and eid_type in {
                    t for fn in functions for t in fn.touches
                }:
                    neighbor_beliefs[eid] = {
                        "label": belief.label,
                        "evidence": belief.evidence,
                        "entity_type": eid_type,
                    }

        inbox_messages = state.inbox.get(entity_id, [])

        # Initialize beliefs for newly discovered records
        new_beliefs: dict[str, Belief] = {}
        default_label = cfg.default_label if cfg else "Defer"
        for eid, etype in new_records.items():
            if eid not in state.beliefs:
                new_beliefs[eid] = Belief(
                    label=default_label,
                    evidence=f"Discovered from {entity_id} via function results.",
                )

        return {
            "context_packet": {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "evidence": all_results,
                "neighbor_beliefs": neighbor_beliefs,
                "inbox_messages": inbox_messages,
                "discovered_values": discovered_values,
            },
            "beliefs": new_beliefs,
            "discovered_records": new_records,
        }

    return _gather


def _bind_primary(
    fn: FunctionSpec,
    entity_id: str,
    entity_type: str,
) -> dict[str, str] | None:
    """Bind function params from the current entity's key.

    Matches param names against the entity type's conventional key:
    e.g., entity_type="ToleranceException" → key param "exceptionId".
    """
    if not fn.params:
        return None

    bound: dict[str, str] = {}
    for p in fn.params:
        pname = p.get("name", "")
        required = p.get("required", False)

        if _param_matches_entity(pname, entity_type):
            bound[pname] = entity_id
        elif required:
            return None  # Can't bind a required param

    return bound if bound else None


def _bind_secondary(
    fn: FunctionSpec,
    discovered_values: dict[str, str],
) -> dict[str, str] | None:
    """Bind function params from values discovered in prior results."""
    bound: dict[str, str] = {}
    for p in fn.params:
        pname = p.get("name", "")
        required = p.get("required", False)

        if pname in discovered_values:
            bound[pname] = discovered_values[pname]
        elif required:
            return None

    return bound if bound else None


def _param_matches_entity(param_name: str, entity_type: str) -> bool:
    """Check if a param name corresponds to an entity type's key."""
    # exceptionId → ToleranceException? No direct match.
    # Use the prefix mapping in reverse.
    expected_type = _type_from_param_name(param_name)
    return expected_type == entity_type if expected_type else False


def _type_from_param_name(param_name: str) -> str | None:
    """Infer entity type from a parameter/column name.

    Convention: param names like ``invoiceId``, ``supplierId``, ``poId``
    map to entity types via a known suffix table.
    """
    name_lower = param_name.lower()
    for entity_type in _PARAM_TO_TYPE:
        if name_lower == _PARAM_TO_TYPE[entity_type]:
            return entity_type
    return None


_PARAM_TO_TYPE: dict[str, str] = {
    "Invoice": "invoiceid",
    "Supplier": "supplierid",
    "PurchaseOrder": "poid",
    "ToleranceException": "exceptionid",
    "Commodity": "commodityid",
    "ExceptionRule": "ruleid",
    "Contract": "contractid",
    "Requisition": "prid",
    "SpendRecord": "spendid",
    "FXRate": "rateid",
    "ContractScope": "scopeid",
    "CostCenter": "costcenterid",
    "Budget": "budgetid",
    "Item": "itemid",
}


def _extract_values(
    result: dict[str, Any],
    fn: FunctionSpec,
    discovered: dict[str, str],
) -> None:
    """Extract entity ID values from function results.

    Scans the first row's columns for names that look like entity keys
    (ending in ``Id``) and records their values for secondary binding.
    """
    rows = result.get("rows", [])
    if not rows:
        return
    first_row = rows[0]
    for col_name, value in first_row.items():
        if not isinstance(value, str):
            continue
        # Only extract columns whose names map to known entity types
        if _type_from_param_name(col_name) is not None:
            if col_name not in discovered:
                discovered[col_name] = value


async def _safe_invoke(
    client: OntologyClient,
    ontology: str,
    fn_name: str,
    params: dict[str, str] | None,
    max_results: int,
) -> dict[str, Any] | None:
    """Invoke a function, returning None on failure."""
    try:
        result = await client.invoke_function(ontology, fn_name, params)
        rows = result.get("rows", [])
        return {
            "function": fn_name,
            "params": params,
            "rows": rows[:max_results],
            "row_count": len(rows),
        }
    except Exception:
        logger.debug(
            "Function %s failed with params %s",
            fn_name, params, exc_info=True,
        )
        return None


# ── Label (Abductive Policy π_abd) ───────────────────────────────

_POLICY_PROMPT = """\
You are investigating entity "{entity_id}" (type: {entity_type}) in an \
ontology-based investigation.

Available labels: {label_vocabulary}

Evidence gathered from ontology functions:
{evidence}

Neighbor entities and their current beliefs:
{neighbor_context}

Messages received from neighbors (belief propagation):
{inbox_messages}

Investigation findings so far (from prior entity visits):
{ledger_summary}

Based on ALL available evidence — this entity's function results, \
neighbor beliefs, inbox messages, AND the accumulated investigation \
findings — assign ONE label from the vocabulary. Consider cross-entity \
patterns: are multiple entities affected by the same root cause? Is this \
entity part of a systemic issue or an isolated incident?

Explain your reasoning concisely.

Respond in valid JSON only:
{{"label": "...", "evidence": "..."}}"""


def _build_ledger_summary(state: EoGState) -> str:
    """Summarize accumulated investigation findings from the ledger.

    Groups by entity type, highlights label changes, and surfaces
    cross-entity patterns for the policy to reason about.
    """
    cfg = state.investigation_config
    default_label = cfg.default_label if cfg else "Defer"

    # Current beliefs (non-default only — these are findings)
    findings: list[str] = []
    by_label: dict[str, list[str]] = {}
    for eid, belief in state.beliefs.items():
        if belief.label != default_label:
            etype = state.discovered_records.get(eid, "?")
            entry = f"{eid} ({etype}): {belief.label}"
            findings.append(entry)
            by_label.setdefault(belief.label, []).append(f"{eid} ({etype})")

    if not findings:
        return "(No findings yet — this is an early visit)"

    lines = [f"Records labeled so far: {len(findings)} of {len(state.beliefs)} visited"]
    for label, entities in sorted(by_label.items()):
        lines.append(f"  {label}: {', '.join(entities)}")

    # Surface label flips (oscillation patterns)
    flipped = [
        f"{eid} (flipped {b.flip_count}x)"
        for eid, b in state.beliefs.items()
        if b.flip_count > 0
    ]
    if flipped:
        lines.append(f"Oscillating entities: {', '.join(flipped)}")

    # Surface damped entities
    max_flips = cfg.max_flips if cfg else 3
    damped = [
        eid for eid, b in state.beliefs.items()
        if b.flip_count > max_flips
    ]
    if damped:
        lines.append(f"Damped (absorbed): {', '.join(damped)}")

    return "\n".join(lines)


def _make_label(model: BaseChatModel) -> _NodeFn:
    """Create label node: ledger-aware LLM labels one record.

    The prompt includes accumulated findings from prior visits so the
    LLM can identify cross-entity patterns (e.g., multiple exceptions
    against the same contract = systemic issue).
    """

    async def _label(state: EoGState) -> dict[str, Any]:
        cfg = state.investigation_config
        label_vocab = cfg.label_vocabulary if cfg else ["Defer"]

        ctx = state.context_packet
        entity_id = ctx.get("entity_id", state.current_record)
        entity_type = ctx.get("entity_type", "Unknown")

        evidence_str = json.dumps(
            ctx.get("evidence", []),
            indent=2, default=str,
        )[:4000]

        neighbor_str = json.dumps(
            ctx.get("neighbor_beliefs", {}),
            indent=2, default=str,
        )[:1000]

        inbox_str = json.dumps(
            ctx.get("inbox_messages", []),
            indent=2, default=str,
        )[:1000]

        ledger_summary = _build_ledger_summary(state)

        prompt = _POLICY_PROMPT.format(
            entity_id=entity_id,
            entity_type=entity_type,
            label_vocabulary=", ".join(label_vocab),
            evidence=evidence_str,
            neighbor_context=neighbor_str,
            inbox_messages=inbox_str,
            ledger_summary=ledger_summary,
        )

        messages = [
            SystemMessage(content="You are an ontology investigation agent."),
            HumanMessage(content=prompt),
        ]

        try:
            response = await model.ainvoke(messages)
            raw = response.content
            if isinstance(raw, str):
                content = raw
            elif isinstance(raw, list):
                # Structured content blocks (e.g., UiPath LLM Gateway)
                content = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in raw
                )
            else:
                content = str(raw)
            content = content.strip()
            if content.startswith("```"):
                content = "\n".join(content.split("\n")[1:])
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
            parsed = json.loads(content)

            label = parsed.get("label", "Defer")
            if label not in label_vocab:
                label = "Defer"
            parsed["label"] = label

            # Capture token usage for budget tracking
            usage = getattr(response, "usage_metadata", None)
            if usage and isinstance(usage, dict):
                parsed["_input_tokens"] = usage.get("input_tokens", 0)
                parsed["_output_tokens"] = usage.get("output_tokens", 0)
            elif usage and hasattr(usage, "input_tokens"):
                parsed["_input_tokens"] = getattr(usage, "input_tokens", 0)
                parsed["_output_tokens"] = getattr(usage, "output_tokens", 0)
        except Exception:
            logger.warning("Label: LLM parse failed for %s", entity_id)
            parsed = {
                "label": "Defer",
                "evidence": "LLM output could not be parsed",
            }

        return {"policy_result": parsed}

    return _label


# ── Update (Ledger + Damping) ────────────────────────────────────

async def update_node(state: EoGState) -> dict[str, Any]:
    """Write belief to ledger, track flips, apply damping, update budget counters."""
    entity_id = state.current_record
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

    # Damping: absorb if oscillating too much
    if flip_count > max_flips:
        damped_label = cfg.default_label if cfg else "Defer"
        logger.info(
            "Damping %s: %d flips, forcing %s (was %s)",
            entity_id, flip_count, damped_label, new_label,
        )
        new_label = damped_label
        evidence = f"Damped after {flip_count} flips"

    belief_changed = old_label is not None and old_label != new_label

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

    # Token tracking from policy_result (set by label node)
    input_tokens = state.policy_result.get("_input_tokens", 0)
    output_tokens = state.policy_result.get("_output_tokens", 0)
    if not isinstance(input_tokens, int):
        input_tokens = 0
    if not isinstance(output_tokens, int):
        output_tokens = 0

    # Convergence tracking
    consecutive = 0 if belief_changed else state.consecutive_no_change + 1

    return {
        "beliefs": {entity_id: updated_belief},
        "ledger": [entry],
        "steps_taken": state.steps_taken + 1,
        "total_input_tokens": state.total_input_tokens + input_tokens,
        "total_output_tokens": state.total_output_tokens + output_tokens,
        "consecutive_no_change": consecutive,
    }


# ── Propagate ────────────────────────────────────────────────────

async def propagate_node(state: EoGState) -> dict[str, Any]:
    """Broadcast belief change to discovered neighbors.

    Propagation follows records discovered in gather results,
    NOT a pre-loaded graph. Only re-activates when belief CHANGED.
    """
    cfg = state.investigation_config
    max_flips = cfg.max_flips if cfg else 3

    entity_id = state.current_record
    current_belief = state.beliefs.get(entity_id)

    # Check if belief actually changed
    old_label = None
    for entry in reversed(state.ledger):
        if entry.entity_id == entity_id:
            old_label = entry.old_label
            break

    belief_changed = (
        old_label is not None and current_belief is not None
        and old_label != current_belief.label
    )

    if not belief_changed:
        return {}

    entity_type = state.discovered_records.get(entity_id, "Unknown")

    # Find neighbors: all discovered entities that share a function's touches
    ctx = state.context_packet
    discovered_values = ctx.get("discovered_values", {})

    updated_inbox: dict[str, list[dict[str, Any]]] = {}
    new_active: list[str] = []
    explanatory_edges: list[ExplanatoryEdge] = []

    for param_name, neighbor_id in discovered_values.items():
        if not neighbor_id or neighbor_id == entity_id:
            continue
        if not isinstance(neighbor_id, str):
            continue

        neighbor_type = _type_from_param_name(param_name)
        if not neighbor_type:
            continue

        neighbor_belief = state.beliefs.get(neighbor_id)
        if not neighbor_belief:
            continue

        # Send inbox message
        message: dict[str, Any] = {
            "from": entity_id,
            "from_type": entity_type,
            "label": current_belief.label,
            "evidence": current_belief.evidence[:200],
            "via": param_name,
        }

        existing_messages = list(state.inbox.get(neighbor_id, []))
        existing_messages.append(message)
        updated_inbox[neighbor_id] = existing_messages

        # Re-activate if under flip limit and not already queued
        if (
            neighbor_belief.flip_count < max_flips
            and neighbor_id not in state.active_set
            and neighbor_id not in new_active
        ):
            new_active.append(neighbor_id)

        explanatory_edges.append(ExplanatoryEdge(
            source=entity_id,
            target=neighbor_id,
            relationship=param_name,
            evidence=current_belief.evidence[:100],
        ))

    return {
        "inbox": updated_inbox,
        "active_set": list(state.active_set) + new_active,
        "explanatory_edges": explanatory_edges,
    }


# ── Synthesize ───────────────────────────────────────────────────

async def synthesize_node(state: EoGState) -> dict[str, Any]:
    """Compute the minimal explanatory frontier.

    Frontier = non-default beliefs minus entities explained by another
    entity with a Source label.
    """
    cfg = state.investigation_config
    default_label = cfg.default_label if cfg else "Defer"

    findings: dict[str, dict[str, Any]] = {}
    for entity_id, belief in state.beliefs.items():
        if belief.label != default_label:
            findings[entity_id] = {
                "entity": entity_id,
                "entity_type": state.discovered_records.get(
                    entity_id, "Unknown"
                ),
                "label": belief.label,
                "evidence": belief.evidence,
                "flip_count": belief.flip_count,
            }

    # Compute irreducibility
    explained: set[str] = set()
    for edge in state.explanatory_edges:
        src_belief = state.beliefs.get(edge.source)
        tgt_belief = state.beliefs.get(edge.target)
        if src_belief and tgt_belief:
            if (
                src_belief.label == "Source"
                and tgt_belief.label == "DerivedEffect"
            ):
                explained.add(edge.target)

    frontier_list = [
        item for eid, item in findings.items()
        if eid not in explained
    ]

    return {"frontier": frontier_list}
