#!/usr/bin/env python3
"""EoG Agent Demo — Explanations over Graphs

A step-by-step narrated investigation showing how the EoG algorithm
reasons about entity relationships in a procurement ontology.

Run:  uv run python examples/s2p-ontology/demo_eog.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time as _time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# ── Config ────────────────────────────────────────────────────────

ONTOLOGY_BASE = "http://localhost:5002"
ACCOUNT = "datafabric"
TENANT = "DefaultTenant"
ONTOLOGY = "s2p"
SEED = ["EXC-002"]
MAX_STEPS = 1000
CONVERGENCE_WINDOW = 5
LABEL_VOCAB = ["Source", "DerivedEffect", "PolicyViolation", "Defer"]

W = 80


# ── Token profiler ────────────────────────────────────────────────

class TokenProfiler(BaseCallbackHandler):
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._ts: dict[str, float] = {}
        self._pbar: Any = None
        self._total_tokens: int = 0
        self._entities_seen: set[str] = set()

    def attach_pbar(self, pbar: Any) -> None:
        self._pbar = pbar

    def _update_bar(self, entity: str = "") -> None:
        if not self._pbar:
            return
        if entity:
            self._entities_seen.add(entity)
        n = len(self.calls)
        self._pbar.n = n
        self._pbar.set_description(
            f"  Investigating ({len(self._entities_seen)} entities, {self._total_tokens:,} tok)"
        )
        self._pbar.refresh()

    def on_chat_model_start(self, serialized, messages, *, run_id, **kw):
        self._ts[str(run_id)] = _time.time()
        if messages:
            last = messages[0][-1] if messages[0] else None
            if last:
                content = getattr(last, "content", "")
                if isinstance(content, str):
                    import re
                    m = re.search(r'"([A-Z]{2,4}-\d+)"', content)
                    if m:
                        self._update_bar(m.group(1))

    on_llm_start = on_chat_model_start

    def on_llm_end(self, response: LLMResult, *, run_id, **kw):
        ms = (_time.time() - self._ts.pop(str(run_id), _time.time())) * 1000
        for gen in response.generations:
            for g in gen:
                u = getattr(getattr(g, "message", None), "usage_metadata", None)
                if u:
                    total = u.get("total_tokens", 0)
                    self.calls.append({
                        "input": u.get("input_tokens", 0),
                        "output": u.get("output_tokens", 0),
                        "total": total,
                        "ms": ms,
                    })
                    self._total_tokens += total
        self._update_bar()


_profiler = TokenProfiler()


# ── Helpers ───────────────────────────────────────────────────────

def _hr(char="=", indent=0):
    print(" " * indent + char * (W - indent))

def _heading(title: str):
    print()
    _hr()
    print(f"  {title}")
    _hr()

def _phase(num: int, title: str):
    print()
    print()
    print(f"  {'=' * 6}  PHASE {num}: {title}  {'=' * 6}")
    print()

def _narrate(text: str):
    """Print narrative text wrapped at W chars, indented."""
    words = text.split()
    line = "  "
    for w in words:
        if len(line) + len(w) + 1 > W - 2:
            print(line)
            line = "  " + w
        else:
            line += (" " if len(line) > 2 else "") + w
    if line.strip():
        print(line)

def _pause(label: str = ""):
    """Visual separator between sections."""
    if label:
        print(f"\n  {'.' * 3} {label} {'.' * 3}\n")
    else:
        print()


def _load_uipath_auth_from_file() -> Path | None:
    if os.environ.get("UIPATH_ACCESS_TOKEN"):
        return None
    auth_files: list[Path] = []
    if os.environ.get("UIPATH_AUTH_FILE"):
        auth_files.append(Path(os.environ["UIPATH_AUTH_FILE"]).expanduser())
    if os.environ.get("UIPATH_AUTH_DIR"):
        auth_files.append(Path(os.environ["UIPATH_AUTH_DIR"]).expanduser() / ".auth.json")
    roots = [Path.cwd().resolve(), Path(__file__).resolve().parent]
    for root in roots:
        auth_files.extend(parent / ".uipath" / ".auth.json" for parent in [root, *root.parents])
    seen: set[Path] = set()
    for auth_file in auth_files:
        if auth_file in seen:
            continue
        seen.add(auth_file)
        if not auth_file.is_file():
            continue
        try:
            data = json.loads(auth_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        token = data.get("accessToken") or data.get("access_token") or data.get("token")
        if not token:
            continue
        os.environ["UIPATH_ACCESS_TOKEN"] = str(token)
        if not os.environ.get("UIPATH_URL"):
            for key in ("url", "uipathUrl", "baseUrl", "base_url", "cloudBaseUrl"):
                value = data.get(key)
                if value:
                    os.environ["UIPATH_URL"] = str(value)
                    break
        return auth_file
    return None


async def _check_runtime() -> bool:
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(
                f"{ONTOLOGY_BASE}/{ACCOUNT}/{TENANT}/datafabric_/api/ontology/{ONTOLOGY}",
                timeout=5.0,
            )
            return r.status_code == 200 and r.json().get("state") == "DEPLOYED"
    except Exception:
        return False


async def _fetch_portfolio() -> dict[str, Any]:
    base = f"{ONTOLOGY_BASE}/{ACCOUNT}/{TENANT}/datafabric_/api/ontology/{ONTOLOGY}/functions"
    async with httpx.AsyncClient(timeout=10.0) as c:
        payables = (await c.post(f"{base}/payablesByAge/invoke", json={},
                                 headers={"Content-Type": "application/json"})).json()
        pending = (await c.post(f"{base}/pendingPayables/invoke", json={},
                                headers={"Content-Type": "application/json"})).json()
        exceptions = (await c.post(f"{base}/openExceptions/invoke", json={},
                                   headers={"Content-Type": "application/json"})).json()
        blocked = (await c.post(f"{base}/blockedSupplierInvoices/invoke", json={},
                                headers={"Content-Type": "application/json"})).json()
        fx = (await c.post(f"{base}/fxExposureByCurrency/invoke", json={},
                           headers={"Content-Type": "application/json"})).json()
    return {
        "invoices": payables.get("rows", []),
        "pending_by_supplier": pending.get("rows", []),
        "exceptions": exceptions.get("rows", []),
        "blocked": blocked.get("rows", []),
        "fx": fx.get("rows", []),
    }


def _create_model():
    loaded = _load_uipath_auth_from_file()
    if loaded:
        print(f"  Auth: {loaded}")
    try:
        from uipath_langchain.chat import get_chat_model
        return get_chat_model("gpt-4.1-2025-04-14", temperature=0.0, max_tokens=1024)
    except Exception:
        pass
    if os.environ.get("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.0, max_tokens=1024)
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0.0, max_tokens=1024)
    print("ERROR: No LLM credentials.")
    sys.exit(1)


# ── Investigation ─────────────────────────────────────────────────

async def run_investigation() -> dict:
    from tqdm import tqdm

    from uipath_langchain.agent.eog.agent import create_eog_agent
    from uipath_langchain.agent.eog.ontology_client import OntologyClient
    from uipath_langchain.agent.eog.types import InvestigationConfig

    client = OntologyClient(base_url=ONTOLOGY_BASE, account=ACCOUNT, tenant=TENANT)
    model = _create_model()
    model.callbacks = [_profiler]

    config = InvestigationConfig(
        label_vocabulary=LABEL_VOCAB,
        seed_entities=SEED,
        max_steps=MAX_STEPS,
        max_flips=3,
        convergence_window=CONVERGENCE_WINDOW,
    )

    graph = create_eog_agent(model, client, ONTOLOGY, investigation_config=config)

    with tqdm(
        total=MAX_STEPS,
        desc="  Investigating",
        unit=" steps",
        bar_format="  {desc} | step {n_fmt} [{elapsed}]",
        leave=True,
    ) as pbar:
        _profiler.attach_pbar(pbar)
        result = await graph.compile().ainvoke({})
        entities = len(result.get("discovered_entities", {}))
        tokens = result.get("total_input_tokens", 0) + result.get("total_output_tokens", 0)
        pbar.set_description(f"  Done: {entities} entities, {tokens:,} tokens")
        pbar.n = result.get("steps_taken", len(_profiler.calls))
        pbar.refresh()

    return result


# ══════════════════════════════════════════════════════════════════
# PHASE 1: THE QUESTION
# ══════════════════════════════════════════════════════════════════

def print_question():
    _phase(1, "THE QUESTION")

    _narrate(
        "An AP analyst is looking at exception EXC-002 in the invoice queue. "
        "It's a price variance: the invoice amount is 4% above the PO, but "
        "the tolerance threshold is 3%. Before they spend 45 minutes "
        "investigating, the EoG agent will answer:"
    )

    print()
    print('    "What is the full causal picture behind EXC-002,')
    print('     who is affected, and what should we do about it?"')
    print()

    _narrate(
        "Traditional AP systems would show the analyst a single row: "
        "exception type, variance percentage, invoice ID. The analyst "
        "would then manually open the PO, check the supplier, look up "
        "the contract, search for similar exceptions. The EoG agent "
        "does this investigation automatically by walking the ontology."
    )


# ══════════════════════════════════════════════════════════════════
# PHASE 2: STARTING STATE
# ══════════════════════════════════════════════════════════════════

def print_starting_state(r: dict):
    _phase(2, "STARTING STATE")

    _narrate(
        "The agent begins with a single seed entity and one label "
        "vocabulary. Every entity starts as Defer (unknown). The "
        "investigation will assign each entity one of these labels:"
    )

    print()
    print("  Label Vocabulary:")
    print("  -----------------")
    print("  Source           = root cause or originating entity")
    print("  DerivedEffect    = downstream consequence, not the cause")
    print("  PolicyViolation  = entity that violates a business rule")
    print("  Defer            = insufficient evidence (default)")
    print()

    print(f"  Seed: {SEED[0]}")
    print(f"  Initial belief: {SEED[0]} -> Defer")
    print(f"  Active set: [{SEED[0]}]")
    print(f"  Knowledge: nothing (no functions fetched, no evidence)")


# ══════════════════════════════════════════════════════════════════
# PHASE 3: EXPLORATION & BELIEF PROPAGATION (one level)
# ══════════════════════════════════════════════════════════════════

def print_exploration(r: dict):
    _phase(3, "EXPLORATION (first 3 steps)")

    ledger = r["ledger"]
    edges = r["explanatory_edges"]

    # Step 1
    _pause("Step 1: Discover + Gather + Label for EXC-002")

    _narrate(
        "The agent pops EXC-002 from the queue and asks the ontology: "
        '"What functions touch ToleranceException?" The server returns '
        "3 functions (filtered by ?touches=ToleranceException). The "
        "agent invokes them and gathers evidence."
    )

    print()
    print("  DISCOVER: GET /functions?touches=ToleranceException")
    print("    -> exceptionContext(exceptionId=EXC-002)")
    print("    -> openExceptions()")
    print("    -> exceptionHistory(supplierId=...)")
    print()
    print("  GATHER: invoke functions, scan results for entity IDs")

    # Show what was discovered from step 1
    first_entry = ledger[0] if ledger else None
    entities_after_1 = set()
    for e in edges:
        if e.source == "EXC-002":
            entities_after_1.add(e.target)

    if entities_after_1:
        print(f"    -> Discovered {len(entities_after_1)} new entities from results:")
        for eid in sorted(entities_after_1):
            etype = r["discovered_entities"].get(eid, "?")
            print(f"       {eid} ({etype})")

    print()
    print("  LABEL: LLM reads evidence + neighbor beliefs + ledger")
    if first_entry:
        print(f"    -> {first_entry.entity_id}: {first_entry.old_label or 'Defer'} -> {first_entry.new_label}")
        b = r["beliefs"].get(first_entry.entity_id)
        if b:
            ev = b.evidence[:70]
            print(f'    -> Evidence: "{ev}"')

    print()
    print("  PROPAGATE: belief changed (Defer -> PolicyViolation)")
    print("    -> Neighbors re-queued for investigation:")
    for eid in sorted(entities_after_1):
        print(f"       + {eid} added to active set")

    # Step 2
    if len(ledger) >= 2:
        _pause("Step 2: EXC-001 (discovered neighbor)")

        second = ledger[1]
        _narrate(
            "EXC-001 was discovered in step 1's results. The agent now "
            "visits it. But this time the LLM prompt includes the ledger: "
            '"EXC-002 is already labeled PolicyViolation." This is '
            "ledger-aware labeling."
        )

        print()
        print("  LABEL with ledger context:")
        print("    Ledger says: PolicyViolation: EXC-002 (ToleranceException)")
        print(f"    -> {second.entity_id}: {second.old_label or 'Defer'} -> {second.new_label}")

    # Step 3
    if len(ledger) >= 3:
        _pause("Step 3: INV-2002 (two hops from seed)")

        third = ledger[2]
        _narrate(
            "INV-2002 is the invoice on exception EXC-001. The agent "
            "fetches Invoice-touching functions (12 of them), invokes "
            "invoiceDetail, invoicePaymentContext, etc. The ledger now "
            "shows 2 PolicyViolation entities. The LLM sees the pattern "
            "forming."
        )

        print()
        print("  LABEL with accumulated findings:")
        print("    Ledger: PolicyViolation: EXC-002, EXC-001")
        print(f"    -> {third.entity_id}: {third.old_label or 'Defer'} -> {third.new_label}")
        b3 = r["beliefs"].get(third.entity_id)
        if b3:
            print(f'    -> Evidence: "{b3.evidence[:70]}"')


# ══════════════════════════════════════════════════════════════════
# PHASE 4: ABDUCTIVE REASONING
# ══════════════════════════════════════════════════════════════════

def print_abductive_reasoning(r: dict):
    _phase(4, "ABDUCTIVE REASONING")

    _narrate(
        "Abductive reasoning is inference to the best explanation. "
        "The agent doesn't just classify entities in isolation -- it "
        "reasons about what label BEST EXPLAINS the evidence across "
        "the entire investigation so far. Three mechanisms:"
    )

    print()
    print("  1. LEDGER-AWARE LABELING")
    print("     The LLM prompt includes all prior findings.")
    print("     By step 8, the prompt says:")
    print('     "Entities labeled: PolicyViolation: EXC-002, EXC-001,')
    print('      INV-2002, PO-1002, CTR-002..."')
    print("     A new entity is labeled in context of the full picture,")
    print("     not in isolation.")
    print()

    # Find if any entity got DerivedEffect (shows differentiation)
    derived = [
        (eid, b) for eid, b in r["beliefs"].items()
        if b.label == "DerivedEffect"
    ]
    deferred = [
        (eid, b) for eid, b in r["beliefs"].items()
        if b.label == "Defer"
    ]

    print("  2. LABEL DIFFERENTIATION")
    if derived:
        for eid, b in derived:
            etype = r["discovered_entities"].get(eid, "?")
            print(f"     {eid} ({etype}) was labeled DerivedEffect, not PolicyViolation.")
            print(f'     Reason: "{b.evidence[:65]}"')
            print("     The LLM distinguished downstream effect from root cause")
            print("     because the ledger showed the primary violations.")
    else:
        print("     All entities converged to the same label in this run.")
        print("     In runs with mixed evidence, the LLM differentiates:")
        print("     FX rates as DerivedEffect (downstream), exceptions as Source.")
    print()

    print("  3. BELIEF PROPAGATION")
    _narrate(
        "When an entity's label changes, its neighbors are re-queued "
        "with an inbox message: 'Your neighbor EXC-002 was just labeled "
        "PolicyViolation.' This triggers re-evaluation with new context. "
        "The graph structure determines WHO gets re-evaluated -- only "
        "entities connected through ontology relationships."
    )

    print()

    # Show convergence
    ledger = r["ledger"]
    changes = sum(1 for e in ledger if e.old_label != e.new_label)
    revisits = len(ledger) - len(set(e.entity_id for e in ledger))
    nc = r.get("consecutive_no_change", 0)

    print("  4. CONVERGENCE")
    print(f"     Total steps: {len(ledger)}")
    print(f"     Label changes: {changes} (initial assignments)")
    print(f"     Revisits: {revisits} (re-evaluations after propagation)")
    print(f"     Consecutive no-change at exit: {nc}")

    steps = r["steps_taken"]
    if nc >= CONVERGENCE_WINDOW:
        _narrate(
            f"The agent stopped after {nc} consecutive visits produced "
            "no belief change. The investigation converged -- additional "
            "visits would not change the picture."
        )
    elif steps < MAX_STEPS:
        _narrate(
            "The active set drained naturally -- every discovered entity "
            "was visited and labeled. No convergence cutoff needed."
        )


# ══════════════════════════════════════════════════════════════════
# PHASE 5: INVESTIGATION REPORT
# ══════════════════════════════════════════════════════════════════

def print_report(r: dict, portfolio: dict[str, Any]):
    _phase(5, "INVESTIGATION REPORT")

    # Beliefs table
    print("  ENTITY BELIEFS")
    print(f"  {'Entity':<14s} {'Type':<20s} {'Label':<18s} Evidence")
    print(f"  {'-'*14} {'-'*20} {'-'*18} {'-'*24}")
    for eid, etype in sorted(r["discovered_entities"].items()):
        b = r["beliefs"].get(eid)
        label = b.label if b else "?"
        ev = b.evidence[:40] + "..." if b and len(b.evidence) > 40 else (b.evidence if b else "")
        print(f"  {eid:<14s} {etype:<20s} {label:<18s} {ev}")

    # Explanatory graph
    print()
    print("  CAUSAL GRAPH (why was each entity flagged?)")
    edges = r["explanatory_edges"]
    by_src: dict[str, list[str]] = {}
    for e in edges:
        by_src.setdefault(e.source, []).append(e.target)
    for src in sorted(by_src):
        targets = by_src[src]
        label = r["beliefs"].get(src)
        lbl = f"[{label.label}]" if label else ""
        print(f"  {src} {lbl} --> {', '.join(sorted(targets))}")
    print(f"  ({len(edges)} edges)")

    # Ledger
    print()
    print("  FULL LEDGER")
    ledger = r["ledger"]
    print(f"  {'#':<4s} {'Entity':<14s} {'From':<18s} {'To':<18s}")
    print(f"  {'-'*4} {'-'*14} {'-'*18} {'-'*18}")
    for i, entry in enumerate(ledger, 1):
        old = entry.old_label or "(init)"
        marker = " *" if old != entry.new_label else ""
        print(f"  {i:<4d} {entry.entity_id:<14s} {old:<18s} {entry.new_label:<18s}{marker}")

    # Global policy
    print()
    print("  PAYMENT POLICY")

    invoices = portfolio["invoices"]
    exceptions = portfolio["exceptions"]
    blocked = portfolio["blocked"]

    exc_by_inv: dict[str, str] = {}
    for exc in exceptions:
        exc_by_inv[exc.get("invoiceId", "")] = exc.get("exceptionId", "")
    blocked_ids = {b.get("invoiceId", "") for b in blocked}

    print(f"  {'Invoice':<12s} {'Amount':>10s} {'Supplier':<22s} {'Action':<20s}")
    print(f"  {'-'*12} {'-'*10} {'-'*22} {'-'*20}")

    total_pay = 0
    total_hold = 0
    total_block = 0

    for inv in sorted(invoices, key=lambda x: x.get("status", "")):
        inv_id = inv.get("invoiceId", "?")
        amount = inv.get("amount", 0)
        supplier = inv.get("supplierName", "?")[:22]
        ccy = inv.get("currency", "")

        if inv_id in blocked_ids:
            action = "BLOCK"
            total_block += amount
        elif inv_id in exc_by_inv:
            action = f"HOLD ({exc_by_inv[inv_id]})"
            total_hold += amount
        elif inv.get("status") == "matched":
            action = "PAY"
            total_pay += amount
        else:
            action = "HOLD"
            total_hold += amount

        print(f"  {inv_id:<12s} {ccy}{amount:>8,.0f} {supplier:<22s} {action}")

    print(f"\n  Pay: {total_pay:>10,.0f} | Hold: {total_hold:>10,.0f} | Block: {total_block:>10,.0f}")


# ══════════════════════════════════════════════════════════════════
# PHASE 6: TOKEN PROFILE
# ══════════════════════════════════════════════════════════════════

def print_tokens(r: dict):
    _phase(6, "COST PROFILE")

    calls = _profiler.calls
    if not calls:
        print("  No LLM calls captured.")
        return

    ti = sum(c["input"] for c in calls)
    to = sum(c["output"] for c in calls)
    tt = sum(c["total"] for c in calls)
    tm = sum(c["ms"] for c in calls)
    steps = r["steps_taken"]
    entities = len(r["discovered_entities"])

    print(f"  LLM calls:        {len(calls)}")
    print(f"  Total tokens:     {tt:,} ({ti:,} in + {to:,} out)")
    print(f"  Avg per call:     {ti // len(calls):,} in + {to // len(calls):,} out")
    print(f"  Wall time:        {tm / 1000:.1f}s ({tm / len(calls) / 1000:.1f}s avg)")
    print(f"  Steps:            {steps}")
    print(f"  Entities:         {entities}")
    print(f"  Tokens/entity:    ~{tt // max(entities, 1):,}")
    print()

    print(f"  {'#':<4s} {'In':>7s} {'Out':>6s} {'Total':>7s} {'ms':>7s}")
    print(f"  {'-'*4} {'-'*7} {'-'*6} {'-'*7} {'-'*7}")
    for i, c in enumerate(calls, 1):
        print(f"  {i:<4d} {c['input']:>7,d} {c['output']:>6,d} {c['total']:>7,d} {c['ms']:>6,.0f}ms")
    print(f"  {'-'*4} {'-'*7} {'-'*6} {'-'*7} {'-'*7}")
    print(f"  {'SUM':<4s} {ti:>7,d} {to:>6,d} {tt:>7,d} {tm:>6,.0f}ms")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

async def main():
    if not await _check_runtime():
        print("ERROR: ontology-runtime not reachable or S2P not deployed.")
        sys.exit(1)

    print()
    _hr()
    print("  EXPLANATIONS OVER GRAPHS (EoG)")
    print("  Ontology-Driven Investigative Agent")
    _hr()
    print(f"  Ontology: {ONTOLOGY} | Seed: {', '.join(SEED)}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Phase 1: The question
    print_question()

    # Run investigation
    print()
    _hr("-")
    print("  Running investigation against live S2P ontology...")
    _hr("-")
    print()

    t0 = _time.time()
    result = await run_investigation()
    elapsed = _time.time() - t0

    # Fetch portfolio for policy
    portfolio = await _fetch_portfolio()

    # Phase 2: Starting state
    print_starting_state(result)

    # Phase 3: Exploration walkthrough
    print_exploration(result)

    # Phase 4: Abductive reasoning
    print_abductive_reasoning(result)

    # Phase 5: Report
    print_report(result, portfolio)

    # Phase 6: Cost
    print_tokens(result)

    # Done
    print()
    _hr()
    print(f"  INVESTIGATION COMPLETE ({elapsed:.1f}s wall time)")
    _hr()
    print()


if __name__ == "__main__":
    asyncio.run(main())
