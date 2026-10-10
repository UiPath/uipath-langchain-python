"""Run the EoG investigation agent against the local S2P ontology stack.

Prerequisites:
  1. FQS mock running on :9099 with S2P data
  2. Ontology runtime running on :5002 with S2P ontology DEPLOYED
  3. OPENAI_API_KEY set in environment

Usage:
  cd examples/s2p-ontology
  uv run python run_eog.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time

# Add src to path for dev mode
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from uipath_langchain.agent.eog.ontology_client import OntologyClient
from uipath_langchain.agent.eog.types import (
    Belief,
    EoGState,
    ExplanatoryEdge,
    InvestigationConfig,
    LedgerEntry,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("eog")

# ── S2P function → entity mapping ──────────────────────────────────────
# Which functions to call per entity type, with how to build their params.
ENTITY_FUNCTIONS: dict[str, list[dict]] = {
    "Invoice": [
        {"fn": "invoiceDetail", "params": lambda eid: {"invoiceId": eid}},
    ],
    "Supplier": [
        {"fn": "supplierProfile", "params": lambda eid: {"supplierId": eid}},
    ],
    "ToleranceException": [
        {"fn": "exceptionContext", "params": lambda eid: {"exceptionId": eid}},
    ],
    "PurchaseOrder": [
        {"fn": "poDetail", "params": lambda eid: {"poId": eid}},
    ],
    "Commodity": [
        {"fn": "commodityDescendants", "params": lambda eid: {"parentCommodityId": eid}},
    ],
    "ExceptionRule": [
        {"fn": "matchingExceptionRules", "params": lambda eid: {"commodityId": eid}},
    ],
    # Seed-level functions (no entity-specific param)
    "_seed": [
        {"fn": "openExceptions", "params": lambda _: None},
        {"fn": "maverickSpendRate", "params": lambda _: {"period": "2026-Q2"}},
        {"fn": "spendByCategory", "params": lambda _: {"period": "2026-Q2"}},
    ],
}

S2P_LABELS = [
    "Source",
    "DerivedEffect",
    "PolicyViolation",
    "CandidateMatch",
    "SupportingEvidence",
    "Contradiction",
    "Defer",
]

SYSTEM_PROMPT = """\
You are an S2P (Source-to-Pay) procurement investigation agent using the \
EoG (Explanations over Graphs) pattern.

You investigate procurement anomalies by examining entities one at a time, \
labeling each with a role in the explanation, and identifying which neighbor \
entities should be re-examined.

Label vocabulary:
- Source: This entity is the root cause / origin of the issue
- DerivedEffect: This entity's anomaly is caused by something upstream
- PolicyViolation: This entity violates a business rule or SHACL constraint
- CandidateMatch: This entity is a potential resolution
- SupportingEvidence: This entity provides corroborating data
- Contradiction: This entity's data conflicts with the emerging explanation
- Defer: Insufficient evidence to classify

You MUST respond in valid JSON only, no markdown, no explanation outside JSON:
{"label": "<one label>", "evidence": "<concise reasoning>", "propagations": [{"entity": "<entity_id>", "reason": "<why re-examine>"}]}

Entity IDs for propagation should be specific record IDs from the data \
(e.g., "INV-2004", "SUP-004", "EXC-002"), not abstract type names."""


async def fetch_seed_data(client: OntologyClient, ontology: str) -> list[dict]:
    """Call seed-level functions to discover initial entities."""
    results = []
    for fn_spec in ENTITY_FUNCTIONS["_seed"]:
        try:
            r = await client.invoke_function(ontology, fn_spec["fn"], fn_spec["params"](""))
            results.append({"function": fn_spec["fn"], "rows": r.get("rows", [])})
        except Exception as e:
            log.warning("Seed function %s failed: %s", fn_spec["fn"], e)
    return results


async def fetch_entity_data(
    client: OntologyClient, ontology: str, entity_id: str, entity_type: str
) -> list[dict]:
    """Call functions relevant to an entity type with the specific entity ID."""
    results = []
    fn_specs = ENTITY_FUNCTIONS.get(entity_type, [])
    for fn_spec in fn_specs:
        try:
            params = fn_spec["params"](entity_id)
            r = await client.invoke_function(ontology, fn_spec["fn"], params)
            results.append({"function": fn_spec["fn"], "rows": r.get("rows", [])[:20]})
        except Exception as e:
            log.warning("  Function %s(%s) failed: %s", fn_spec["fn"], entity_id, e)
    return results


def infer_entity_type(entity_id: str) -> str:
    """Infer entity type from ID prefix."""
    prefixes = {
        "INV-": "Invoice",
        "SUP-": "Supplier",
        "EXC-": "ToleranceException",
        "PO-": "PurchaseOrder",
        "COM-": "Commodity",
        "RULE-": "ExceptionRule",
        "CTR-": "Contract",
        "PR-": "Requisition",
        "SPD-": "SpendRecord",
    }
    for prefix, etype in prefixes.items():
        if entity_id.startswith(prefix):
            return etype
    return "Unknown"


async def run_eog(
    llm: BaseChatModel,
    client: OntologyClient,
    ontology: str = "s2p",
    max_steps: int = 10,
) -> None:
    """Run a full EoG investigation loop."""

    # ── 1. Bootstrap: discover seed entities ───────────────────────────
    print("\n" + "=" * 70)
    print("EoG INVESTIGATION — S2P Procurement Ontology")
    print("=" * 70)

    seed_data = await fetch_seed_data(client, ontology)
    print(f"\nSeed data collected from {len(seed_data)} functions:")
    for sd in seed_data:
        print(f"  {sd['function']}: {len(sd['rows'])} rows")

    # Extract initial entity IDs from seed data
    seed_entities: list[str] = []
    for sd in seed_data:
        for row in sd["rows"]:
            for key in ["exceptionId", "invoiceId", "supplierId"]:
                val = row.get(key)
                if val and val not in seed_entities:
                    seed_entities.append(val)
    seed_entities = seed_entities[:8]  # Cap initial seeds

    print(f"\nSeed entities: {seed_entities}")

    # ── 2. Initialize state ────────────────────────────────────────────
    beliefs: dict[str, Belief] = {}
    for eid in seed_entities:
        beliefs[eid] = Belief(label="Defer", evidence="Initial seed entity.")

    active_set = list(seed_entities)
    ledger: list[dict] = []
    explanatory_edges: list[dict] = []
    inbox: dict[str, list[dict]] = {}
    steps = 0

    # ── 3. EoG Loop ───────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("INVESTIGATION LOOP")
    print(f"{'─' * 70}")

    while active_set and steps < max_steps:
        # Pop
        entity_id = active_set.pop(0)
        entity_type = infer_entity_type(entity_id)
        steps += 1

        print(f"\n[Step {steps}] Processing: {entity_id} (type: {entity_type})")

        # Fetch context
        entity_data = await fetch_entity_data(client, ontology, entity_id, entity_type)
        entity_inbox = inbox.get(entity_id, [])

        context = {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "current_belief": beliefs[entity_id].model_dump(),
            "data": entity_data,
            "inbox_messages": entity_inbox,
            "neighbor_beliefs": {
                eid: b.model_dump()
                for eid, b in beliefs.items()
                if eid != entity_id and b.label != "Defer"
            },
        }

        # Policy (LLM call)
        prompt = f"""\
Investigating entity: {entity_id} (type: {entity_type})

Data collected:
{json.dumps(entity_data, indent=2, default=str)[:3000]}

Messages from neighbors:
{json.dumps(entity_inbox, indent=2, default=str)[:1000]}

Current beliefs about other entities:
{json.dumps({eid: b.model_dump() for eid, b in beliefs.items() if eid != entity_id and b.label != "Defer"}, indent=2, default=str)[:1000]}

Assign a label and identify propagations."""

        try:
            response = await llm.ainvoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            content = response.content if isinstance(response.content, str) else str(response.content)
            # Strip markdown code fences if present
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
            parsed = json.loads(content)
        except Exception as e:
            log.warning("  LLM parse failed: %s", e)
            parsed = {"label": "Defer", "evidence": f"LLM parse error: {e}", "propagations": []}

        new_label = parsed.get("label", "Defer")
        evidence = parsed.get("evidence", "")
        propagations = parsed.get("propagations", [])

        # Update belief
        old_belief = beliefs[entity_id]
        old_label = old_belief.label
        flip_count = old_belief.flip_count + (1 if old_label != new_label and old_label != "Defer" else 0)

        beliefs[entity_id] = Belief(label=new_label, evidence=evidence, flip_count=flip_count)
        ledger.append({
            "step": steps,
            "entity": entity_id,
            "type": entity_type,
            "old_label": old_label,
            "new_label": new_label,
            "evidence": evidence[:120],
        })

        status = "CHANGED" if old_label != new_label else "confirmed"
        print(f"  Label: {old_label} → {new_label} ({status})")
        print(f"  Evidence: {evidence[:100]}")

        # Propagate
        for prop in propagations:
            target = prop.get("entity", "")
            reason = prop.get("reason", "")
            if not target:
                continue

            # Add to inbox
            if target not in inbox:
                inbox[target] = []
            inbox[target].append({
                "from": entity_id,
                "label": new_label,
                "evidence": evidence[:80],
                "reason": reason,
            })

            # Initialize belief if new
            if target not in beliefs:
                beliefs[target] = Belief(label="Defer", evidence="Discovered via propagation.")

            # Re-activate if under flip limit
            target_flips = beliefs[target].flip_count
            if target_flips < 3 and target not in active_set:
                active_set.append(target)
                print(f"  → Propagate to {target}: {reason[:60]}")

            explanatory_edges.append({
                "source": entity_id,
                "target": target,
                "evidence": reason,
            })

    # ── 4. Compute Frontier ────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("INVESTIGATION RESULTS")
    print(f"{'=' * 70}")

    print(f"\nSteps taken: {steps}")
    print(f"Entities examined: {len(beliefs)}")

    # Frontier: non-Defer beliefs
    frontier = [
        {"entity": eid, "type": infer_entity_type(eid), "label": b.label, "evidence": b.evidence}
        for eid, b in beliefs.items()
        if b.label != "Defer"
    ]

    print(f"\nFrontier ({len(frontier)} findings):")
    for item in frontier:
        print(f"  [{item['label']}] {item['entity']} ({item['type']})")
        print(f"    {item['evidence'][:120]}")

    print(f"\nLedger ({len(ledger)} entries):")
    for entry in ledger:
        print(f"  Step {entry['step']}: {entry['entity']} ({entry['type']}) "
              f"{entry['old_label']} → {entry['new_label']}")

    if explanatory_edges:
        print(f"\nExplanatory edges ({len(explanatory_edges)}):")
        for edge in explanatory_edges:
            print(f"  {edge['source']} → {edge['target']}: {edge['evidence'][:80]}")


async def main() -> None:
    # LLM setup — try Anthropic first, fall back to OpenAI
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if anthropic_key:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001"),
            temperature=0.0,
            api_key=anthropic_key,
        )
        print(f"Using Anthropic: {llm.model}")
    elif openai_key:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            temperature=0.0,
            api_key=openai_key,
        )
        print(f"Using OpenAI: {llm.model}")
    else:
        print("Error: Set ANTHROPIC_API_KEY or OPENAI_API_KEY")
        sys.exit(1)

    client = OntologyClient(
        base_url=os.environ.get("ONTOLOGY_BASE_URL", "http://localhost:5002"),
    )

    # Verify server is up
    try:
        meta = await client.discover("s2p")
        if meta.get("state") != "DEPLOYED":
            print(f"Error: Ontology s2p is {meta.get('state')}, expected DEPLOYED")
            sys.exit(1)
        print(f"Connected to ontology: {meta['name']} [{meta['state']}]")
    except Exception as e:
        print(f"Error connecting to ontology-runtime: {e}")
        sys.exit(1)

    await run_eog(
        llm=llm,
        client=client,
        ontology="s2p",
        max_steps=12,
    )


if __name__ == "__main__":
    asyncio.run(main())
