"""Live integration test: EoG agent against local ontology-runtime.

Uses the UiPath coded agent pattern — ``uipath_langchain.chat.get_chat_model``
routes through the UiPath LLM Gateway (or falls back to direct vendor API).
Token profiling via LLMOPS_TRACE_FILE.

Requires:
  - ontology-runtime running on localhost:5002
  - S2P ontology deployed
  - FQS mock running on localhost:9099
  - UiPath platform credentials (UIPATH_URL, UIPATH_ACCESS_TOKEN)
    OR direct vendor key (OPENAI_API_KEY / ANTHROPIC_API_KEY)

Run: uv run python tests/agent/eog/test_live_eog.py
"""

from __future__ import annotations

import asyncio
import os
import sys

import httpx
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class TokenProfiler(BaseCallbackHandler):
    """Captures token usage from every LLM call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._start_times: dict[str, float] = {}

    def on_llm_start(self, serialized: dict, prompts: list, *, run_id, **kwargs) -> None:
        import time
        self._start_times[str(run_id)] = time.time()

    def on_chat_model_start(self, serialized: dict, messages: list, *, run_id, **kwargs) -> None:
        import time
        self._start_times[str(run_id)] = time.time()

    def on_llm_end(self, response: LLMResult, *, run_id, **kwargs) -> None:
        import time
        elapsed = (time.time() - self._start_times.pop(str(run_id), time.time())) * 1000
        model = (response.llm_output or {}).get("model_name", "")

        for gen in response.generations:
            for g in gen:
                usage = getattr(g, "message", None)
                usage = getattr(usage, "usage_metadata", None) if usage else None
                if usage:
                    self.calls.append({
                        "model": model,
                        "input": usage.get("input_tokens", 0),
                        "output": usage.get("output_tokens", 0),
                        "total": usage.get("total_tokens", 0),
                        "duration_ms": elapsed,
                    })

    @property
    def total_input(self) -> int:
        return sum(c["input"] for c in self.calls)

    @property
    def total_output(self) -> int:
        return sum(c["output"] for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return sum(c["total"] for c in self.calls)


_profiler = TokenProfiler()


async def check_runtime() -> bool:
    """Check if ontology-runtime is reachable."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "http://localhost:5002/datafabric/DefaultTenant"
                "/datafabric_/api/ontology/s2p",
                timeout=5.0,
            )
            return resp.status_code == 200 and resp.json().get("state") == "DEPLOYED"
    except Exception:
        return False


def _create_model():
    """Create an LLM via UiPath coded agent pattern, with fallbacks."""
    # Try UiPath LLM Gateway first
    try:
        from uipath_langchain.chat import get_chat_model

        return get_chat_model(
            "gpt-4.1-2025-04-14",
            temperature=0.0,
            max_tokens=1024,
        )
    except Exception:
        pass

    # Fallback: direct Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0.0,
            max_tokens=1024,
        )

    # Fallback: direct OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model="gpt-4.1-2025-04-14",
            temperature=0.0,
            max_tokens=1024,
        )

    print("ERROR: No LLM credentials found.")
    print("Set UIPATH_URL + UIPATH_ACCESS_TOKEN for LLM Gateway,")
    print("or ANTHROPIC_API_KEY / OPENAI_API_KEY for direct vendor API.")
    sys.exit(1)


async def run_eog_investigation() -> dict:
    """Run the EoG agent against the live S2P ontology."""
    from uipath_langchain.agent.eog.agent import create_eog_agent
    from uipath_langchain.agent.eog.ontology_client import OntologyClient
    from uipath_langchain.agent.eog.types import InvestigationConfig

    client = OntologyClient(
        base_url="http://localhost:5002",
        account="datafabric",
        tenant="DefaultTenant",
    )

    model = _create_model()
    model.callbacks = [_profiler]
    print(f"Using LLM: {type(model).__name__}")

    config = InvestigationConfig(
        label_vocabulary=["Source", "DerivedEffect", "PolicyViolation", "Defer"],
        seed_records=["EXC-002"],
        max_steps=1000,
        max_flips=3,
        convergence_window=5,
    )

    graph = create_eog_agent(
        model, client, "s2p", investigation_config=config,
    )
    compiled = graph.compile()
    result = await compiled.ainvoke({})

    return result


def _print_token_profile() -> None:
    """Print token rollup from the callback profiler."""
    calls = _profiler.calls

    print(f"\n{'='*60}")
    print(f"TOKEN PROFILE ({len(calls)} LLM calls)")
    print(f"{'='*60}")

    if calls:
        print(f"\n  {'#':<4s} {'Model':<25s} {'In':>8s} {'Out':>8s} {'Total':>8s} {'ms':>8s}")
        print(f"  {'-'*4} {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for i, c in enumerate(calls, 1):
            model_name = (c["model"] or "?")[:25]
            print(
                f"  {i:<4d} "
                f"{model_name:<25s} "
                f"{c['input']:>8d} {c['output']:>8d} {c['total']:>8d} "
                f"{c['duration_ms']:>8.0f}"
            )
        print(f"  {'-'*4} {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
        total_in = _profiler.total_input
        total_out = _profiler.total_output
        total_all = _profiler.total_tokens
        print(
            f"  {'SUM':<4s} {'':<25s} "
            f"{total_in:>8d} {total_out:>8d} {total_all:>8d}"
        )

        avg_in = total_in / len(calls)
        avg_out = total_out / len(calls)
        print(f"\n  Avg per call:  input={avg_in:.0f}  output={avg_out:.0f}")
        print(f"  Output/input ratio: {total_out / max(total_in, 1):.2f}")
    else:
        print("  No LLM calls captured.")


async def main() -> None:
    """Entry point."""
    if not await check_runtime():
        print("ERROR: ontology-runtime not reachable or S2P not deployed")
        print(
            "Start with:\n"
            "  ONTOLOGY_DATAFABRIC_BASE_URL=http://localhost:9099 \\\n"
            "  ONTOLOGY_METADATA_DB_URL=jdbc:sqlite:./data/metadata.db \\\n"
            "  java -jar ontology-app-1.0.0.jar --spring.profiles.active=dev"
        )
        sys.exit(1)

    print("=== EoG Investigation: seed=EXC-002 (S2P Procurement) ===\n")

    result = await run_eog_investigation()

    # ── Investigation Report ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"INVESTIGATION REPORT")
    print(f"{'='*60}")
    print(f"Steps taken: {result['steps_taken']}")
    print(f"Records discovered: {len(result['discovered_records'])}")
    print(f"Consecutive no-change (at exit): {result.get('consecutive_no_change', '?')}")
    print(f"Token usage (from state): input={result.get('total_input_tokens', 0)}, output={result.get('total_output_tokens', 0)}")
    stop_reason = "budget" if result['steps_taken'] >= 1000 else "convergence" if result.get('consecutive_no_change', 0) >= 5 else "active_set empty"
    print(f"Stop reason: {stop_reason}")

    print(f"\nBeliefs:")
    for eid, etype in sorted(result["discovered_records"].items()):
        belief = result["beliefs"].get(eid)
        label = belief.label if belief else "?"
        evidence = belief.evidence[:80] if belief else ""
        print(f"  {eid:20s} ({etype:25s}) -> {label:20s}  {evidence}")

    print(f"\nFunction cache: {sorted(result['function_cache'].keys())}")

    print(f"\nLedger ({len(result['ledger'])} entries):")
    for entry in result["ledger"]:
        print(f"  {entry.entity_id}: {entry.old_label} -> {entry.new_label}")

    print(f"\nFrontier ({len(result['frontier'])} items):")
    for item in result["frontier"]:
        print(
            f"  {item['entity']} ({item['entity_type']}): "
            f"{item['label']} -- {item.get('evidence', '')[:100]}"
        )

    print(f"\nExplanatory edges ({len(result['explanatory_edges'])}):")
    for edge in result["explanatory_edges"]:
        print(f"  {edge.source} -> {edge.target} via {edge.relationship}")

    # ── Token Profile ─────────────────────────────────────────
    _print_token_profile()


if __name__ == "__main__":
    asyncio.run(main())
