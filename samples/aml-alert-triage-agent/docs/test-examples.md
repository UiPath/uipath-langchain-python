# Test Examples

Copy-paste inputs for testing the agent, plus what each one should return.

- [Three ways to run it](#three-ways-to-run-it)
- [Example 1 — clear escalate](#example-1--clear-escalate)
- [Example 2 — clear close](#example-2--clear-close)
- [Example 3 — ambiguous, low confidence](#example-3--ambiguous-low-confidence)
- [Example 4 — same alert, LLM deciding](#example-4--same-alert-llm-deciding)
- [Writing your own alert](#writing-your-own-alert)
- [The full eval set](#the-full-eval-set)
- [What the log looks like](#what-the-log-looks-like)

---

## Three ways to run it

**1. Orchestrator UI** — Processes → `aml-triage-agent` → **Start a job**, then paste the
values into the arguments panel field by field.

**2. CLI, deployed** (runs on serverless, ~45s):

```bash
uipath invoke agent -f evals/sample-escalate.json
```

**3. CLI, local** (no deploy, ~20s, uses `.env`):

```bash
uipath run agent -f evals/sample-escalate.json
python evaluate.py --compare          # the whole 12-alert set, both deciders
```

> The CLI needs a valid UiPath token. Run `uipath auth` once for your own session.

---

## Example 1 — clear escalate

Layering with a refusal to document. The easy win, and the best opener for a demo.

| Argument | Value |
|---|---|
| `alert_id` | `AML-2026-0144` |
| `customer` | `Aurelia Holdings SA (Panama, nominee directors)` |
| `narrative` | `Single inbound transfer of USD 2,400,000 from a Latvian bank, followed 6 hours later by six outbound transfers of USD 395,000-405,000 each to accounts in three different jurisdictions. No commercial rationale stated. Customer declined to provide source-of-funds documentation when contacted.` |
| `account_age_days` | `47` |
| `prior_alerts` | `2` |
| `decider` | `jev` |
| `expected_disposition` | `escalate` |
| `benchmark` | `true` |

```json
{
  "alert_id": "AML-2026-0144",
  "customer": "Aurelia Holdings SA (Panama, nominee directors)",
  "narrative": "Single inbound transfer of USD 2,400,000 from a Latvian bank, followed 6 hours later by six outbound transfers of USD 395,000-405,000 each to accounts in three different jurisdictions. No commercial rationale stated. Customer declined to provide source-of-funds documentation when contacted.",
  "account_age_days": 47,
  "prior_alerts": 2,
  "decider": "jev",
  "expected_disposition": "escalate",
  "benchmark": true
}
```

**Expect:** `escalate` · risk `Critical` · confidence **1.00** · all five red flags between
0.72 and 0.97 · `ACCURACY CORRECT`.

File: `evals/sample-escalate.json`

---

## Example 2 — clear close

A fully documented inheritance that fired the rule only because it was 60× the average
balance. This is the one that proves the agent isn't just escalating everything.

| Argument | Value |
|---|---|
| `alert_id` | `AML-2026-0152` |
| `customer` | `Priya Raghunathan (personal account, retired teacher)` |
| `narrative` | `Received INR 4,200,000 from the sale of an inherited property. Sale deed, probate documentation and buyer identity are all on file and consistent. Funds subsequently moved to a fixed deposit at the same institution. The monitoring rule fired because the amount is 60x the customer's average balance.` |
| `account_age_days` | `5400` |
| `prior_alerts` | `0` |
| `decider` | `jev` |
| `expected_disposition` | `close` |
| `benchmark` | `true` |

```json
{
  "alert_id": "AML-2026-0152",
  "customer": "Priya Raghunathan (personal account, retired teacher)",
  "narrative": "Received INR 4,200,000 from the sale of an inherited property. Sale deed, probate documentation and buyer identity are all on file and consistent. Funds subsequently moved to a fixed deposit at the same institution. The monitoring rule fired because the amount is 60x the customer's average balance.",
  "account_age_days": 5400,
  "prior_alerts": 0,
  "decider": "jev",
  "expected_disposition": "close",
  "benchmark": true
}
```

**Expect:** `close` · risk `Low` · confidence **0.99** · `ACCURACY CORRECT`.

File: `evals/sample-close.json`

---

## Example 3 — ambiguous, low confidence

**This is a known miss, and you should show it anyway.** Ground truth is `need_info`; Jev
returns `escalate` — but at **0.31 confidence**, against 0.89–1.00 on every call it gets
right. The model is declaring its own uncertainty. A confidence gate at 0.50 routes this to
a human and catches the miss without touching a single correct decision.

| Argument | Value |
|---|---|
| `alert_id` | `AML-2026-0150` |
| `customer` | `Nordvik Marine AS (Norway, established 2016)` |
| `narrative` | `Outbound payment of USD 680,000 to a shipyard in a jurisdiction subject to enhanced due diligence requirements. Customer provided a vessel repair contract and photographs of the vessel in dry dock. The shipyard entity is not on any sanctions list but shares a registered address with two entities that were delisted from sanctions in 2024.` |
| `account_age_days` | `3600` |
| `prior_alerts` | `0` |
| `decider` | `jev` |
| `expected_disposition` | `need_info` |
| `benchmark` | `true` |

```json
{
  "alert_id": "AML-2026-0150",
  "customer": "Nordvik Marine AS (Norway, established 2016)",
  "narrative": "Outbound payment of USD 680,000 to a shipyard in a jurisdiction subject to enhanced due diligence requirements. Customer provided a vessel repair contract and photographs of the vessel in dry dock. The shipyard entity is not on any sanctions list but shares a registered address with two entities that were delisted from sanctions in 2024.",
  "account_age_days": 3600,
  "prior_alerts": 0,
  "decider": "jev",
  "expected_disposition": "need_info",
  "benchmark": true
}
```

**Expect:** `escalate` · risk `Medium` · confidence **0.31** · `ACCURACY WRONG`.

File: `evals/sample-ambiguous.json`

---

## Example 4 — same alert, LLM deciding

Identical to Example 1 except `decider` is `llm`. Run them back to back to show the same
agent, same alert, same seven questions — different decision engine.

| Argument | Value |
|---|---|
| `alert_id` | `AML-2026-0144` |
| `customer` | `Aurelia Holdings SA (Panama, nominee directors)` |
| `narrative` | *(same as Example 1)* |
| `account_age_days` | `47` |
| `prior_alerts` | `2` |
| `decider` | **`llm`** |
| `expected_disposition` | `escalate` |
| `benchmark` | `true` |

```json
{
  "alert_id": "AML-2026-0144",
  "customer": "Aurelia Holdings SA (Panama, nominee directors)",
  "narrative": "Single inbound transfer of USD 2,400,000 from a Latvian bank, followed 6 hours later by six outbound transfers of USD 395,000-405,000 each to accounts in three different jurisdictions. No commercial rationale stated. Customer declined to provide source-of-funds documentation when contacted.",
  "account_age_days": 47,
  "prior_alerts": 2,
  "decider": "llm",
  "expected_disposition": "escalate",
  "benchmark": true
}
```

**Expect:** `escalate` (same answer) but a decision time around **6000 ms** instead of
~400 ms, and the summary block reading `=> llm is ~10x slower`.

File: `evals/sample-llm.json`

---

## Writing your own alert

Only three fields are required: `alert_id`, `customer`, `narrative`. Everything else is
optional.

```json
{
  "alert_id": "TEST-001",
  "customer": "Your Entity Ltd (jurisdiction, incorporated when)",
  "narrative": "What the monitoring system saw: amounts, counts, timing, jurisdictions, and what documentation does or does not exist.",
  "account_age_days": 90,
  "prior_alerts": 0,
  "decider": "jev",
  "benchmark": true
}
```

The narrative does the real work. Write it the way a transaction-monitoring system would —
concrete amounts, how many transactions, over what period, to and from where, what the
customer says the money is for, and whether there's paperwork. `account_age_days` and
`prior_alerts` only sharpen the shell-company and history judgements.

Leave `expected_disposition` out if you don't know the answer; the `ACCURACY` line is simply
omitted.

---

## The full eval set

All twelve live in `evals/alerts.json` with ground truth and a note explaining why each one
is there. Four are deliberately ambiguous.

| Alert | Customer | Expected | Difficulty | Why it's in the set |
|---|---|---|---|---|
| `AML-2026-0142` | Meridian Trading Ltd | `escalate` | clear | Textbook structuring plus pass-through plus shell indicators |
| `AML-2026-0143` | Westbrook Dental Practice LLP | `close` | clear | Fully explained increase with documentation |
| `AML-2026-0144` | Aurelia Holdings SA | `escalate` | clear | Classic layering; refusal to document is aggravating |
| `AML-2026-0145` | Tomas Riedel | `close` | clear | Own-account transfer with matching invoice |
| `AML-2026-0146` | Sunrise Textiles Pvt Ltd | `need_info` | **ambiguous** | Established relationship, but invoice mismatch on new routing |
| `AML-2026-0147` | Kestrel Consulting Ltd | `need_info` | **ambiguous** | Looks like structuring, may be ordinary consultancy invoicing |
| `AML-2026-0148` | Halcyon Freight Services | `escalate` | clear | Funnel account into crypto, no operating substance |
| `AML-2026-0149` | Greenfield Agricultural Co-op | `close` | clear | Seasonal pattern matching four years of history |
| `AML-2026-0150` | Nordvik Marine AS | `need_info` | **ambiguous** | Strong docs; shared-address link needs pulling, not concluding |
| `AML-2026-0151` | Delacroix Fine Art Sarl | `escalate` | clear | Third-party payment, BVI co-registration, high-value art |
| `AML-2026-0152` | Priya Raghunathan | `close` | clear | Documented one-off life event; escalating it would be trigger-happy |
| `AML-2026-0153` | Bluepeak Logistics Ltd | `need_info` | **ambiguous** | Real business, undocumented intercompany flows |

Run the whole set:

```bash
python evaluate.py                  # jev
python evaluate.py --decider llm    # the gateway LLM deciding instead
python evaluate.py --compare        # both, side by side
python evaluate.py --limit 3        # quick smoke run
```

---

## What the log looks like

Every run emits this block. In Orchestrator the lines are prefixed `SUMMARY[01]`…`SUMMARY[16]`
and arrive **out of order** — sort by the number, or read the single-line `SUMMARY_JSON`
record instead.

```
==============================================================
  ALERT            AML-2026-0144
  DECIDER          JEV  (jev-latest)
  DISPOSITION      escalate   confidence 1.00
  RISK             Critical  (3.0)
  ACCURACY         CORRECT   expected=escalate
  DECISION TIME    625 ms
  DECISION COST    $0.00003482   (829 in / 161 out)
  TOTAL RUN TIME   27854 ms  (incl. extract + explain LLM calls)
  --------------------------------------------------------
  COMPARED TO      LLM
    disposition    escalate   CORRECT
    time           6224 ms
    cost           0.2 platform units
    => jev is 9.96x faster
    agreement      yes
==============================================================
```

And `--compare` across the whole set:

```
                      JEV       GATEWAY LLM
  ------------------------------------------------------
  accuracy        10/12 (83%)   10/12 (83%)
  schema valid    12/12         12/12
  median decision   398 ms         6160 ms
  total decision    4.8 s           74.2 s
  cost for the set  $0.000426      2.4 platform units
  ------------------------------------------------------
  => Jev is 15.5x faster per decision
  => the two models agree on 10/12 alerts
  => accuracy: same accuracy (n=12, not statistically meaningful)
```
