"""S2P EoG Investigation Agent — UiPath Coded Agent.

Investigates procurement anomalies using the EoG (Explanations over Graphs)
pattern: deterministic traversal over the S2P ontology, bounded LLM inference
at each entity, belief propagation along relationships.

Connects to a local ontology-runtime at http://localhost:5002.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from uipath_langchain.chat import UiPathChat


# ── I/O schemas ────────────────────────────────────────────────────────

class Input(BaseModel):
    question: str = Field(description="The procurement question or investigation request")


class Output(BaseModel):
    frontier: str = Field(description="Investigation findings as structured summary")
    steps_taken: int = Field(description="Number of investigation steps executed")


# ── Internal state ─────────────────────────────────────────────────────

class InvestigationState(BaseModel):
    question: str = ""
    active_set: list[str] = Field(default_factory=list)
    current_entity: str = ""
    beliefs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    inbox: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    ledger: list[dict[str, Any]] = Field(default_factory=list)
    context_data: dict[str, Any] = Field(default_factory=dict)
    steps_taken: int = 0
    frontier: str = ""


# ── Ontology client (inline — no external dependency) ──────────────────

ONTOLOGY_BASE = os.getenv("ONTOLOGY_BASE_URL", "http://localhost:5002")
ONTOLOGY_ACCOUNT = os.getenv("ONTOLOGY_ACCOUNT", "datafabric")
ONTOLOGY_TENANT = os.getenv("ONTOLOGY_TENANT", "DefaultTenant")
ONTOLOGY_NAME = os.getenv("ONTOLOGY_NAME", "s2p")

API_BASE = f"{ONTOLOGY_BASE}/{ONTOLOGY_ACCOUNT}/{ONTOLOGY_TENANT}/datafabric_/api"

ENTITY_FUNCTIONS: dict[str, list[dict[str, str]]] = {
    "Invoice": [{"fn": "invoiceDetail", "param": "invoiceId"}],
    "Supplier": [{"fn": "supplierProfile", "param": "supplierId"}],
    "ToleranceException": [{"fn": "exceptionContext", "param": "exceptionId"}],
    "PurchaseOrder": [{"fn": "poDetail", "param": "poId"}],
}

PREFIXES = {
    "INV-": "Invoice", "SUP-": "Supplier", "EXC-": "ToleranceException",
    "PO-": "PurchaseOrder", "COM-": "Commodity", "RULE-": "ExceptionRule",
}

S2P_LABELS = [
    "Source", "DerivedEffect", "PolicyViolation",
    "CandidateMatch", "SupportingEvidence", "Contradiction", "Defer",
]

_llm: UiPathChat | None = None


def _get_llm() -> UiPathChat:
    global _llm
    if _llm is None:
        _llm = UiPathChat(model="gpt-4.1-mini-2025-04-14")
    return _llm


SYSTEM_PROMPT = """\
You are an S2P procurement investigation agent using the EoG pattern.
You examine entities one at a time, label each with a role, and identify \
which neighbors to re-examine.

Labels: Source (root cause), DerivedEffect (consequence), PolicyViolation \
(rule breach), CandidateMatch (resolution), SupportingEvidence (corroborating), \
Contradiction (conflicting), Defer (insufficient evidence).

Respond in valid JSON ONLY:
{"label": "...", "evidence": "...", "propagations": [{"entity": "...", "reason": "..."}]}
Use specific record IDs (e.g. INV-2004, SUP-004) not type names."""


def _infer_type(entity_id: str) -> str:
    for prefix, etype in PREFIXES.items():
        if entity_id.startswith(prefix):
            return etype
    return "Unknown"


async def _invoke_fn(fn_name: str, params: dict[str, Any] | None = None) -> dict:
    url = f"{API_BASE}/ontology/{ONTOLOGY_NAME}/functions/{fn_name}/invoke"
    body = {"params": params} if params else {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json=body)
        resp.raise_for_status()
        return resp.json()


async def _fetch_entity_data(entity_id: str) -> list[dict]:
    etype = _infer_type(entity_id)
    results = []
    for spec in ENTITY_FUNCTIONS.get(etype, []):
        try:
            r = await _invoke_fn(spec["fn"], {spec["param"]: entity_id})
            results.append({"function": spec["fn"], "rows": r.get("rows", [])[:10]})
        except Exception:
            pass
    return results


# ── Graph nodes ────────────────────────────────────────────────────────

async def bootstrap(state: InvestigationState) -> dict[str, Any]:
    """Seed the investigation: call discovery functions, extract entity IDs."""
    seed_data = []

    # Call openExceptions to discover entities with open tolerance exceptions
    try:
        r = await _invoke_fn("openExceptions")
        seed_data.append({"function": "openExceptions", "rows": r.get("rows", [])})
    except Exception:
        pass

    # Call spend overview functions for landscape context
    for fn, params in [
        ("maverickSpendRate", {"period": "2026-Q2"}),
        ("spendByCategory", {"period": "2026-Q2"}),
    ]:
        try:
            r = await _invoke_fn(fn, params)
            seed_data.append({"function": fn, "rows": r.get("rows", [])})
        except Exception:
            pass

    # Extract entity IDs from seed data — scan all ID-shaped fields
    entity_ids: list[str] = []
    for sd in seed_data:
        for row in sd.get("rows", []):
            for key, val in row.items():
                if isinstance(val, str) and any(
                    val.startswith(p) for p in PREFIXES
                ) and val not in entity_ids:
                    entity_ids.append(val)

    entity_ids = entity_ids[:8]

    beliefs = {eid: {"label": "Defer", "evidence": "Seed entity", "flips": 0} for eid in entity_ids}

    return {
        "active_set": entity_ids,
        "beliefs": beliefs,
        "context_data": {"seed_data": seed_data},
        "steps_taken": 0,
        "question": state.question,
    }


async def pop_entity(state: InvestigationState) -> dict[str, Any]:
    """Dequeue next entity from active set."""
    active = list(state.active_set)
    entity = active.pop(0) if active else ""
    return {"current_entity": entity, "active_set": active}


def should_continue(state: InvestigationState) -> str:
    """Route: continue or compute frontier."""
    if state.current_entity and state.steps_taken < 10:
        return "fetch"
    return "frontier"


async def fetch_context(state: InvestigationState) -> dict[str, Any]:
    """Fetch data for the current entity via ontology functions."""
    entity_id = state.current_entity
    data = await _fetch_entity_data(entity_id)
    inbox_msgs = state.inbox.get(entity_id, [])
    neighbor_beliefs = {
        eid: b for eid, b in state.beliefs.items()
        if eid != entity_id and b.get("label") != "Defer"
    }
    return {
        "context_data": {
            "entity_id": entity_id,
            "entity_type": _infer_type(entity_id),
            "function_results": data,
            "inbox": inbox_msgs,
            "neighbors": neighbor_beliefs,
        }
    }


async def policy(state: InvestigationState) -> dict[str, Any]:
    """LLM call: label the entity and identify propagations."""
    llm = _get_llm()

    ctx = state.context_data
    prompt = f"""Investigating: {ctx.get('entity_id')} (type: {ctx.get('entity_type')})
User question: {state.question}

Data:
{json.dumps(ctx.get('function_results', []), indent=2, default=str)[:3000]}

Inbox messages:
{json.dumps(ctx.get('inbox', []), indent=2, default=str)[:500]}

Neighbor beliefs:
{json.dumps(ctx.get('neighbors', {}), indent=2, default=str)[:500]}

Assign a label and identify propagations."""

    try:
        response = await llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        content = response.content if isinstance(response.content, str) else str(response.content)
        content = content.strip()
        if content.startswith("```"):
            content = "\n".join(content.split("\n")[1:])
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        parsed = json.loads(content)
    except Exception as e:
        parsed = {"label": "Defer", "evidence": f"Parse error: {e}", "propagations": []}

    # Update belief
    entity_id = state.current_entity
    old = state.beliefs.get(entity_id, {})
    old_label = old.get("label", "Defer")
    new_label = parsed.get("label", "Defer")
    flips = old.get("flips", 0) + (1 if old_label != new_label and old_label != "Defer" else 0)

    updated_beliefs = dict(state.beliefs)
    updated_beliefs[entity_id] = {
        "label": new_label,
        "evidence": parsed.get("evidence", ""),
        "flips": flips,
    }

    # Ledger
    entry = {
        "step": state.steps_taken + 1,
        "entity": entity_id,
        "type": _infer_type(entity_id),
        "old_label": old_label,
        "new_label": new_label,
        "evidence": parsed.get("evidence", "")[:120],
    }
    ledger = list(state.ledger) + [entry]

    # Propagate
    updated_inbox = dict(state.inbox)
    new_active = list(state.active_set)
    for prop in parsed.get("propagations", []):
        target = prop.get("entity", "")
        if not target:
            continue
        if target not in updated_inbox:
            updated_inbox[target] = []
        updated_inbox[target] = list(updated_inbox.get(target, [])) + [{
            "from": entity_id, "label": new_label,
            "reason": prop.get("reason", ""),
        }]
        if target not in updated_beliefs:
            updated_beliefs[target] = {"label": "Defer", "evidence": "Via propagation", "flips": 0}
        if updated_beliefs[target].get("flips", 0) < 3 and target not in new_active:
            new_active.append(target)

    return {
        "beliefs": updated_beliefs,
        "ledger": ledger,
        "inbox": updated_inbox,
        "active_set": new_active,
        "steps_taken": state.steps_taken + 1,
    }


async def compute_frontier(state: InvestigationState) -> dict[str, Any]:
    """Summarize findings as the frontier."""
    findings = []
    for eid, belief in state.beliefs.items():
        if belief.get("label") != "Defer":
            findings.append(
                f"[{belief['label']}] {eid} ({_infer_type(eid)}): {belief.get('evidence', '')}"
            )

    ledger_summary = "\n".join(
        f"  Step {e['step']}: {e['entity']} {e['old_label']}→{e['new_label']}"
        for e in state.ledger
    )

    frontier = (
        f"Investigation: {state.question}\n"
        f"Steps: {state.steps_taken}\n"
        f"Entities examined: {len(state.beliefs)}\n\n"
        f"Findings ({len(findings)}):\n"
        + "\n".join(f"  {f}" for f in findings)
        + f"\n\nLedger:\n{ledger_summary}"
    )

    return {"frontier": frontier, "steps_taken": state.steps_taken}


# ── Build graph ────────────────────────────────────────────────────────

builder = StateGraph(InvestigationState, input=Input, output=Output)

builder.add_node("bootstrap", bootstrap)
builder.add_node("pop", pop_entity)
builder.add_node("fetch", fetch_context)
builder.add_node("policy", policy)
builder.add_node("frontier", compute_frontier)

builder.add_edge(START, "bootstrap")
builder.add_edge("bootstrap", "pop")
builder.add_conditional_edges("pop", should_continue, {"fetch": "fetch", "frontier": "frontier"})
builder.add_edge("fetch", "policy")
builder.add_edge("policy", "pop")
builder.add_edge("frontier", END)

graph = builder.compile()
