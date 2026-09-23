"""AML alert triage — a UiPath coded agent whose decisions are made by TypeSafe's Jev.

Split of responsibilities:
  UiPath LLM Gateway  -> language work: read the alert, extract entities, do the arithmetic,
                         and afterwards write the rationale prose.
  Jev (System One)    -> every decision: red flags (Noul), risk level (Score),
                         disposition (Choice). One API call, all questions in parallel.

Jev returns typed, calibrated values and cannot emit free text, which is precisely why the
generation and the judgement are separated rather than asked of one model.
"""

import json
import logging
import os
import time
from typing import Literal, Optional

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typesafe_sdk import TypeSafeClient
from uipath_langchain.chat.models import UiPathChat

from rubric import DISPOSITIONS, RED_FLAGS, RISK_LEVELS, RUBRIC_VERSION, build_questions

log = logging.getLogger("aml_triage")
log.setLevel(logging.INFO)

GATEWAY_MODEL = "gpt-5-mini-2025-08-07"
JEV_MODEL = "jev-latest"
JEV_USD_PER_M_INPUT = 0.042  # documented; Jev output tokens are free
PLATFORM_UNITS_PER_LLM_CALL = 0.2  # documented Standard tier, per LLM call

# UiPath bills gateway LLM calls in platform units, not dollars. Set this to your
# contracted rate to get a like-for-like USD figure in the logs; leave it unset and the
# summary reports platform units instead of inventing a price.
USD_PER_PLATFORM_UNIT = float(os.getenv("USD_PER_PLATFORM_UNIT", "0") or 0)


# --------------------------------------------------------------------------- I/O

class Input(BaseModel):
    alert_id: str
    customer: str
    narrative: str
    account_age_days: Optional[int] = None
    prior_alerts: Optional[int] = None
    decider: Literal["jev", "llm"] = Field(
        "jev", description="Which model makes the decisions: TypeSafe Jev, or the gateway LLM.")
    expected_disposition: Optional[Literal["escalate", "close", "need_info"]] = Field(
        None, description="Ground truth, if known. Enables the CORRECT/WRONG line in the log.")
    benchmark: bool = Field(False, description="Also run the other arm for comparison.")


class Metrics(BaseModel):
    """Cost, time and correctness for the decider that ran - and the other arm if asked."""
    decider: str
    decision_latency_ms: int
    decision_input_tokens: int = 0
    decision_output_tokens: int = 0
    decision_cost_usd: Optional[float] = None
    decision_platform_units: Optional[float] = None
    total_latency_ms: int = 0
    correct: Optional[bool] = None
    # comparison arm, present only when benchmark=true
    other_decider: Optional[str] = None
    other_disposition: Optional[str] = None
    other_latency_ms: Optional[int] = None
    other_cost_usd: Optional[float] = None
    other_platform_units: Optional[float] = None
    other_correct: Optional[bool] = None
    agreement: Optional[bool] = None
    speedup: Optional[float] = None
    cost_ratio: Optional[float] = None


class Output(BaseModel):
    alert_id: str
    disposition: Literal["escalate", "close", "need_info"]
    disposition_confidence: float
    risk_level: str
    risk_score: float
    red_flags: dict[str, float]
    parties_extracted: list[str]
    rationale: str
    evidence: list[str]
    rubric_version: str
    decider: str
    metrics: Optional[Metrics] = None


class State(BaseModel):
    """Input fields + node-derived work + the Output fields the final node fills in.

    Output fields live at the top level because LangGraph's output_schema selects by
    key from the state, not from a nested object.
    """
    # input
    alert_id: str = ""
    customer: str = ""
    narrative: str = ""
    account_age_days: Optional[int] = None
    prior_alerts: Optional[int] = None
    decider: str = "jev"
    expected_disposition: Optional[str] = None
    benchmark: bool = False
    started_at: float = 0.0
    # intermediate
    digest: dict = Field(default_factory=dict)
    parties: list[str] = Field(default_factory=list)
    answers: dict = Field(default_factory=dict)
    timings: dict = Field(default_factory=dict)
    # output
    disposition: str = ""
    disposition_confidence: float = 0.0
    risk_level: str = ""
    risk_score: float = 0.0
    red_flags: dict[str, float] = Field(default_factory=dict)
    parties_extracted: list[str] = Field(default_factory=list)
    rationale: str = ""
    evidence: list[str] = Field(default_factory=list)
    rubric_version: str = RUBRIC_VERSION
    metrics: Optional[Metrics] = None


# ----------------------------------------------------------------------- helpers

def _jev_key() -> str:
    """Local dev reads .env; the serverless run reads the Orchestrator asset."""
    key = os.getenv("JEV_API_KEY") or os.getenv("TYPESAFE_API_KEY")
    if key:
        return key
    from uipath.platform import UiPath  # imported lazily: only needed in the cloud
    asset = UiPath().assets.retrieve(name="JevApiKey")
    return getattr(asset, "value", None) or getattr(asset, "string_value", "")


def _llm(model: str = GATEWAY_MODEL) -> UiPathChat:
    return UiPathChat(model=model)


def _json_from(text: str) -> dict:
    """Gateway models occasionally fence their JSON; tolerate it."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        t = t[4:] if t.lower().startswith("json") else t
    return json.loads(t.strip())


# ------------------------------------------------------------------------- nodes

EXTRACT_PROMPT = """You are preparing an AML alert for a decision model that is poor at \
arithmetic and cannot compare dates. Do ALL counting and arithmetic yourself.

Return ONLY JSON with these keys:
  parties            list of named legal/natural persons in the narrative
  transaction_count  integer, or null if not determinable
  total_amount       number, or null
  largest_amount     number, or null
  smallest_amount    number, or null
  span_days          integer number of days the activity covers, or null
  amounts_cluster_just_below  one of "yes" | "no" | "unknown" - whether the individual \
amounts sit consistently just under a round reporting threshold such as 10,000
  outflow_within_days integer days between credit and onward transfer, or null
  stated_purpose     the customer's stated business purpose, or null
  documentation      what supporting documentation exists, or "none on file"

ALERT:
{narrative}

CUSTOMER: {customer}
ACCOUNT AGE (days): {account_age_days}
PRIOR ALERTS: {prior_alerts}"""


def extract(state: State) -> dict:
    t0 = time.perf_counter()
    started = state.started_at or t0
    raw = _llm().invoke(EXTRACT_PROMPT.format(
        narrative=state.narrative, customer=state.customer,
        account_age_days=state.account_age_days, prior_alerts=state.prior_alerts,
    )).content
    facts = _json_from(raw)
    dt = int((time.perf_counter() - t0) * 1000)

    # The tight digest Jev sees - derived facts only, no raw prose padding.
    digest = {
        "customer": state.customer,
        "account_age_days": state.account_age_days,
        "prior_alerts": state.prior_alerts,
        "narrative": state.narrative,
        **{k: v for k, v in facts.items() if k != "parties"},
    }
    log.info("EXTRACT ok in %sms parties=%s", dt, facts.get("parties"))
    return {"digest": digest, "parties": facts.get("parties") or [], "started_at": started,
            "timings": {**state.timings, "extract_ms": dt}}


LLM_DECIDE_PROMPT = """You are an AML triage decision engine. Judge the alert state below against the rubric. Return ONLY JSON - no prose, no explanation, no markdown.

RED FLAGS - for each, give the probability between 0.0 and 1.0 that the flag is present:
{flags}

RISK LEVEL - return the integer index of the level that applies:
{levels}

DISPOSITION - return exactly one of: {dispositions}

Return ONLY this JSON shape:
{{"red_flags": {{"flag_name": 0.0}}, "risk_level_index": 0, "risk_confidence": 0.0,
  "disposition": "escalate", "disposition_confidence": 0.0}}

ALERT STATE:
{state}"""


def _decide_with_jev(digest: dict) -> tuple[dict, dict]:
    """One System One call answering all seven questions in parallel."""
    client = TypeSafeClient(api_key=_jev_key())
    t0 = time.perf_counter()
    r = client.system_one(state=digest, questions=build_questions(), model=JEV_MODEL)
    dt = int((time.perf_counter() - t0) * 1000)

    answers = {}
    for name, a in r.answers.items():
        if a.type == "noul":
            answers[name] = {"type": "noul", "value": a.noul}
        elif a.type == "score":
            answers[name] = {"type": "score", "value": a.score,
                             "confidence": a.confidence, "legend": a.legend}
        else:
            answers[name] = {"type": "choice", "value": a.choice,
                             "confidence": a.confidence, "probabilities": a.probabilities}

    meta = {
        "decider": "jev",
        "latency_ms": dt,
        "input_tokens": r.usage.input_tokens,
        "output_tokens": r.usage.output_tokens,
        "cost_usd": round(r.usage.input_tokens / 1e6 * JEV_USD_PER_M_INPUT, 10),
        "platform_units": None,
    }
    return answers, meta


def _decide_with_llm(digest: dict) -> tuple[dict, dict]:
    """The same seven judgements, asked of the gateway LLM. Equal work, fair comparison."""
    prompt = LLM_DECIDE_PROMPT.format(
        flags="\n".join(f"  {k}: {v}" for k, v in RED_FLAGS.items()),
        levels="\n".join(f"  {i}: {lv}" for i, lv in enumerate(RISK_LEVELS)),
        dispositions=", ".join(DISPOSITIONS),
        state=json.dumps(digest, indent=2),
    )
    t0 = time.perf_counter()
    resp = _llm().invoke(prompt)
    dt = int((time.perf_counter() - t0) * 1000)
    parsed = _json_from(resp.content)

    idx = max(0, min(int(parsed.get("risk_level_index", 0)), len(RISK_LEVELS) - 1))
    answers = {k: {"type": "noul", "value": float(parsed.get("red_flags", {}).get(k, 0.0))}
               for k in RED_FLAGS}
    answers["risk_level"] = {
        "type": "score", "value": float(idx),
        "confidence": float(parsed.get("risk_confidence", 0.0)),
        "legend": {i: lv for i, lv in enumerate(RISK_LEVELS)},
    }
    disp = str(parsed.get("disposition", "")).strip().lower()
    answers["disposition"] = {
        "type": "choice",
        "value": disp if disp in DISPOSITIONS else "need_info",
        "confidence": float(parsed.get("disposition_confidence", 0.0)),
        "probabilities": {},
    }

    u = resp.usage_metadata or {}
    units = PLATFORM_UNITS_PER_LLM_CALL
    meta = {
        "decider": "llm",
        "latency_ms": dt,
        "input_tokens": u.get("input_tokens", 0),
        "output_tokens": u.get("output_tokens", 0),
        "cost_usd": round(units * USD_PER_PLATFORM_UNIT, 10) if USD_PER_PLATFORM_UNIT else None,
        "platform_units": units,
    }
    return answers, meta


DECIDERS = {"jev": _decide_with_jev, "llm": _decide_with_llm}


def decide(state: State) -> dict:
    """Every decision in this agent is made here, by whichever decider was selected."""
    primary = state.decider if state.decider in DECIDERS else "jev"
    answers, meta = DECIDERS[primary](state.digest)

    cost = (f"${meta['cost_usd']:.8f}" if meta["cost_usd"] is not None
            else f"{meta['platform_units']} platform units")
    log.info("DECIDE  decider=%s  latency=%sms  tokens_in=%s  cost=%s  disposition=%s",
             primary, meta["latency_ms"], meta["input_tokens"], cost,
             answers["disposition"]["value"])

    other_meta, other_answers = None, None
    if state.benchmark:
        alt = "llm" if primary == "jev" else "jev"
        try:
            other_answers, other_meta = DECIDERS[alt](state.digest)
            ocost = (f"${other_meta['cost_usd']:.8f}" if other_meta["cost_usd"] is not None
                     else f"{other_meta['platform_units']} platform units")
            log.info("COMPARE decider=%s  latency=%sms  tokens_in=%s  cost=%s  disposition=%s",
                     alt, other_meta["latency_ms"], other_meta["input_tokens"], ocost,
                     other_answers["disposition"]["value"])
        except Exception as exc:  # the comparison arm must never fail the run
            log.info("COMPARE decider=%s FAILED %s: %s", alt, type(exc).__name__, exc)

    return {"answers": answers,
            "timings": {**state.timings, "decide_ms": meta["latency_ms"],
                        "meta": meta, "other_meta": other_meta,
                        "other_disposition": (other_answers or {}).get(
                            "disposition", {}).get("value")}}


EXPLAIN_PROMPT = """Write the analyst-facing justification for a triage decision that has \
already been made by a decision model. Do not second-guess it; explain it.

DECISION: {disposition}
RISK LEVEL: {risk_level}
RED FLAG SCORES (0-1): {flags}

ALERT NARRATIVE:
{narrative}

Return ONLY JSON:
  rationale  2-4 sentences explaining why this disposition follows from the flags above
  evidence   list of 2-5 SHORT VERBATIM quotes from the narrative that support it"""


def explain(state: State) -> dict:
    disp = state.answers["disposition"]
    score = state.answers["risk_level"]
    level = score["legend"].get(int(score["value"]), RISK_LEVELS[int(score["value"])])
    flags = {k: round(v["value"], 3) for k, v in state.answers.items() if v["type"] == "noul"}

    t0 = time.perf_counter()
    raw = _llm().invoke(EXPLAIN_PROMPT.format(
        disposition=disp["value"], risk_level=level, flags=flags, narrative=state.narrative,
    )).content
    parsed = _json_from(raw)
    dt = int((time.perf_counter() - t0) * 1000)

    meta = state.timings.get("meta") or {}
    other = state.timings.get("other_meta")
    other_disp = state.timings.get("other_disposition")
    total_ms = int((time.perf_counter() - state.started_at) * 1000) if state.started_at else 0

    correct = None
    if state.expected_disposition:
        correct = disp["value"] == state.expected_disposition

    m = Metrics(
        decider=meta.get("decider", state.decider),
        decision_latency_ms=meta.get("latency_ms", -1),
        decision_input_tokens=meta.get("input_tokens", 0),
        decision_output_tokens=meta.get("output_tokens", 0),
        decision_cost_usd=meta.get("cost_usd"),
        decision_platform_units=meta.get("platform_units"),
        total_latency_ms=total_ms,
        correct=correct,
    )
    if other:
        m.other_decider = other.get("decider")
        m.other_disposition = other_disp
        m.other_latency_ms = other.get("latency_ms")
        m.other_cost_usd = other.get("cost_usd")
        m.other_platform_units = other.get("platform_units")
        m.agreement = other_disp == disp["value"]
        if state.expected_disposition and other_disp:
            m.other_correct = other_disp == state.expected_disposition
        if m.decision_latency_ms > 0 and m.other_latency_ms:
            m.speedup = round(m.other_latency_ms / m.decision_latency_ms, 2)
        if m.decision_cost_usd and m.other_cost_usd:
            m.cost_ratio = round(m.other_cost_usd / m.decision_cost_usd, 1)

    result = Output(
        alert_id=state.alert_id,
        disposition=disp["value"],
        disposition_confidence=round(disp["confidence"], 4),
        risk_level=level.split(":")[0],
        risk_score=round(score["value"], 3),
        red_flags=flags,
        parties_extracted=state.parties,
        rationale=parsed.get("rationale", ""),
        evidence=parsed.get("evidence", []) or [],
        rubric_version=RUBRIC_VERSION,
        decider=m.decider,
        metrics=m,
    )
    _log_summary(state, result, m)
    return {**result.model_dump(), "timings": {**state.timings, "explain_ms": dt}}


def _money(cost_usd, units) -> str:
    if cost_usd is not None:
        return f"${cost_usd:.8f}"
    return f"{units} platform units (set USD_PER_PLATFORM_UNIT for a $ figure)"


def _log_summary(state: State, r: Output, m: Metrics) -> None:
    """The block to point a camera at."""
    L = []
    L.append("=" * 62)
    L.append(f"  ALERT            {r.alert_id}")
    L.append(f"  DECIDER          {m.decider.upper()}"
             + (f"  ({JEV_MODEL})" if m.decider == "jev" else f"  ({GATEWAY_MODEL})"))
    L.append(f"  DISPOSITION      {r.disposition}   confidence {r.disposition_confidence:.2f}")
    L.append(f"  RISK             {r.risk_level}  ({r.risk_score})")
    if m.correct is not None:
        L.append(f"  ACCURACY         {'CORRECT' if m.correct else 'WRONG'}"
                 f"   expected={state.expected_disposition}")
    L.append(f"  DECISION TIME    {m.decision_latency_ms} ms")
    L.append(f"  DECISION COST    {_money(m.decision_cost_usd, m.decision_platform_units)}"
             f"   ({m.decision_input_tokens} in / {m.decision_output_tokens} out)")
    L.append(f"  TOTAL RUN TIME   {m.total_latency_ms} ms  (incl. extract + explain LLM calls)")
    if m.other_decider:
        L.append("  " + "-" * 58)
        L.append(f"  COMPARED TO      {m.other_decider.upper()}")
        L.append(f"    disposition    {m.other_disposition}"
                 + (f"   {'CORRECT' if m.other_correct else 'WRONG'}"
                    if m.other_correct is not None else ""))
        L.append(f"    time           {m.other_latency_ms} ms")
        L.append(f"    cost           {_money(m.other_cost_usd, m.other_platform_units)}")
        if m.speedup:
            faster = f"{m.speedup}x faster" if m.speedup >= 1 else f"{1/m.speedup:.2f}x slower"
            L.append(f"    => {m.decider} is {faster}")
        if m.cost_ratio:
            L.append(f"    => {m.decider} is {m.cost_ratio}x cheaper")
        L.append(f"    agreement      {'yes' if m.agreement else 'NO - they disagree'}")
    L.append("=" * 62)
    # Orchestrator splits multi-line records and its RobotLogs timestamps collide at
    # sub-millisecond resolution, so the block comes back scrambled in the job log either
    # way. Numbering each line makes it sortable and makes scrambling obvious on screen.
    for i, line in enumerate(L, 1):
        log.info("SUMMARY[%02d] %s", i, line)
    # One single-line record that always survives intact, for anything parsing the log.
    log.info("SUMMARY_JSON %s", m.model_dump_json())


# ------------------------------------------------------------------------- graph

builder = StateGraph(State, input_schema=Input, output_schema=Output)
builder.add_node("extract", extract)
builder.add_node("decide", decide)
builder.add_node("explain", explain)
builder.add_edge(START, "extract")
builder.add_edge("extract", "decide")
builder.add_edge("decide", "explain")
builder.add_edge("explain", END)
graph = builder.compile()
