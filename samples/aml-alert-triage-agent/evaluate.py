"""Run the eval set against the agent and report cost, time and accuracy.

  1. Schema validity    - HARD GATE. A well-reasoned answer in the wrong shape is an outage.
  2. Disposition match  - against hand-authored ground truth.
  3. Evidence grounding - are the quoted spans actually in the narrative, or invented?
  4. Cost and latency   - per decider.

Usage:
  python evaluate.py                      # jev (default)
  python evaluate.py --decider llm        # the gateway LLM makes the decisions instead
  python evaluate.py --compare            # run BOTH and print the side-by-side table
  python evaluate.py --limit 3            # quick smoke run
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

load_dotenv(".env", override=True)

from main import (  # noqa: E402  (import after env is loaded)
    GATEWAY_MODEL,
    JEV_MODEL,
    Output,
    graph,
)

BAR_SCHEMA = 1.00
BAR_DISPOSITION = 0.80
INPUT_KEYS = ("alert_id", "customer", "narrative", "account_age_days", "prior_alerts")


def normalise(s: str) -> str:
    return " ".join(s.lower().replace("’", "'").split())


def grounded(evidence: list[str], narrative: str) -> tuple[int, int]:
    """How many evidence quotes actually appear in the source text."""
    hay = normalise(narrative)
    return sum(1 for e in evidence if normalise(e) and normalise(e) in hay), len(evidence)


def run_set(alerts: list[dict], decider: str, quiet: bool = False) -> dict:
    """Run every alert with one decider and collect the numbers."""
    rows = []
    schema_ok = disp_ok = ev_hits = ev_total = 0
    lat, cost_usd, units, in_tok, out_tok = [], 0.0, 0.0, 0, 0

    model = JEV_MODEL if decider == "jev" else GATEWAY_MODEL
    if not quiet:
        print(f"\n--- decider: {decider.upper()} ({model}) on {len(alerts)} alerts ---")

    for a in alerts:
        payload = {k: a[k] for k in INPUT_KEYS}
        payload.update(decider=decider, expected_disposition=a["expected_disposition"],
                       benchmark=False)
        t0 = time.perf_counter()
        try:
            raw = graph.invoke(payload)
        except Exception as exc:
            print(f"  {a['alert_id']}  EXCEPTION  {type(exc).__name__}: {exc}")
            rows.append({"alert_id": a["alert_id"], "error": str(exc)[:200]})
            continue
        wall = int((time.perf_counter() - t0) * 1000)

        try:
            out = Output.model_validate(raw)
            schema_ok += 1
        except ValidationError as exc:
            print(f"  {a['alert_id']}  SCHEMA FAIL  {exc.error_count()} errors")
            rows.append({"alert_id": a["alert_id"], "schema": False})
            continue

        hit = out.disposition == a["expected_disposition"]
        disp_ok += hit
        h, t = grounded(out.evidence, a["narrative"])
        ev_hits, ev_total = ev_hits + h, ev_total + t

        m = out.metrics
        if m:
            lat.append(m.decision_latency_ms)
            cost_usd += m.decision_cost_usd or 0.0
            units += m.decision_platform_units or 0.0
            in_tok += m.decision_input_tokens
            out_tok += m.decision_output_tokens

        if not quiet:
            print(f"  {a['alert_id']}  {'OK  ' if hit else 'MISS'}  got={out.disposition:<9}"
                  f" want={a['expected_disposition']:<9} conf={out.disposition_confidence:.2f}"
                  f" decision={m.decision_latency_ms if m else '?'}ms"
                  f" ev={h}/{t}  ({a['difficulty']}, total {wall}ms)")

        rows.append({
            "alert_id": a["alert_id"], "decider": decider, "schema": True, "hit": hit,
            "got": out.disposition, "want": a["expected_disposition"],
            "difficulty": a["difficulty"], "confidence": out.disposition_confidence,
            "evidence_grounded": [h, t], "metrics": m.model_dump() if m else None,
        })

    n = len(alerts)
    return {
        "decider": decider, "model": model, "n": n, "rows": rows,
        "schema_ok": schema_ok, "disp_ok": disp_ok,
        "schema_rate": schema_ok / n, "disp_rate": disp_ok / n,
        "ev_hits": ev_hits, "ev_total": ev_total,
        "median_latency": statistics.median(lat) if lat else 0,
        "total_latency": sum(lat),
        "cost_usd": cost_usd, "units": units,
        "in_tok": in_tok, "out_tok": out_tok,
    }


def money(r: dict) -> str:
    return f"${r['cost_usd']:.6f}" if r["cost_usd"] else f"{r['units']:.1f} platform units"


def report(r: dict) -> None:
    print("\n" + "=" * 70)
    print(f"  decider            {r['decider'].upper()}  ({r['model']})")
    print(f"  schema validity    {r['schema_ok']}/{r['n']}  {r['schema_rate']:6.1%}   "
          f"bar {BAR_SCHEMA:.0%}   {'PASS' if r['schema_rate'] >= BAR_SCHEMA else 'FAIL'}")
    print(f"  disposition match  {r['disp_ok']}/{r['n']}  {r['disp_rate']:6.1%}   "
          f"bar {BAR_DISPOSITION:.0%}   {'PASS' if r['disp_rate'] >= BAR_DISPOSITION else 'FAIL'}")
    if r["ev_total"]:
        print(f"  evidence grounded  {r['ev_hits']}/{r['ev_total']}  "
              f"{r['ev_hits']/r['ev_total']:6.1%}")
    by = {}
    for row in r["rows"]:
        if "hit" in row:
            by.setdefault(row["difficulty"], []).append(row["hit"])
    for d, hits in sorted(by.items()):
        print(f"    {d:<10} {sum(hits)}/{len(hits)}")
    print(f"  decision latency   median {r['median_latency']:.0f} ms   total {r['total_latency']} ms")
    print(f"  decision cost      {money(r)}   ({r['in_tok']} in / {r['out_tok']} out tokens)")
    print("=" * 70)


def compare(a: dict, b: dict) -> None:
    """a = jev, b = llm."""
    print("\n" + "=" * 70)
    print(f"  {'':<22}{'JEV':>14}{'GATEWAY LLM':>18}")
    print("  " + "-" * 66)
    print(f"  {'accuracy':<22}{a['disp_ok']}/{a['n']} ({a['disp_rate']:.0%})".ljust(38)
          + f"{b['disp_ok']}/{b['n']} ({b['disp_rate']:.0%})".rjust(30))
    print(f"  {'schema valid':<22}{a['schema_ok']}/{a['n']}".ljust(38)
          + f"{b['schema_ok']}/{b['n']}".rjust(30))
    print(f"  {'median decision':<22}{a['median_latency']:.0f} ms".ljust(38)
          + f"{b['median_latency']:.0f} ms".rjust(30))
    print(f"  {'total decision time':<22}{a['total_latency']/1000:.1f} s".ljust(38)
          + f"{b['total_latency']/1000:.1f} s".rjust(30))
    print(f"  {'cost for the set':<22}{money(a)}".ljust(38) + f"{money(b)}".rjust(30))
    print("  " + "-" * 66)
    if a["median_latency"]:
        print(f"  => Jev is {b['median_latency']/a['median_latency']:.1f}x faster per decision")
    agree = sum(1 for x, y in zip(a["rows"], b["rows"])
                if x.get("got") and x.get("got") == y.get("got"))
    print(f"  => the two models agree on {agree}/{a['n']} alerts")
    delta = a["disp_ok"] - b["disp_ok"]
    verdict = ("same accuracy" if delta == 0 else
               f"Jev {'ahead' if delta > 0 else 'behind'} by {abs(delta)} alert(s)")
    print(f"  => accuracy: {verdict} (n={a['n']}, not statistically meaningful)")
    print("=" * 70)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decider", choices=["jev", "llm"], default="jev")
    ap.add_argument("--compare", action="store_true", help="Run both deciders side by side.")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    alerts = json.loads(Path("evals/alerts.json").read_text(encoding="utf-8"))
    if args.limit:
        alerts = alerts[: args.limit]

    if args.compare:
        jev = run_set(alerts, "jev")
        llm = run_set(alerts, "llm")
        report(jev)
        report(llm)
        compare(jev, llm)
        Path("evals/results.json").write_text(
            json.dumps({"jev": jev["rows"], "llm": llm["rows"]}, indent=2), encoding="utf-8")
        print("\n  wrote evals/results.json")
        return 0 if jev["schema_rate"] >= BAR_SCHEMA else 1

    r = run_set(alerts, args.decider)
    report(r)
    Path("evals/results.json").write_text(json.dumps(r["rows"], indent=2), encoding="utf-8")
    print("\n  wrote evals/results.json")
    return 0 if (r["schema_rate"] >= BAR_SCHEMA and r["disp_rate"] >= BAR_DISPOSITION) else 1


if __name__ == "__main__":
    sys.exit(main())
