# Joke Agent — Bring Your Own Guardrail (BYOG)

A minimal starter for using a **customer-managed guardrail** in a coded agent — with **both
guardrail flavors in one agent**. The agent generates a family-friendly joke; agent
input/output and the tool input are validated by a **BYOG configuration** — your own
guardrail vendor connected through Integration Service, referenced purely by validator name
+ connection id. Credentials never appear in this project.

The guardrail can be backed by any vendor your admin configured: a cloud subscription
(e.g. a content-safety service), a vendor validation platform, or a custom Integration
Service connector wrapping an internal classifier. The validator name and connection id in
[graph.py](graph.py) are **placeholders** — substitute your own configuration's values.

## How the two flavors are wired

One agent, one BYOG configuration, each flavor guarding a different scope:

| Flavor | Guards | Stage | Action |
|---|---|---|---|
| `UiPathByoGuardrailMiddleware` (passed to `create_agent(middleware=[...])`) | AGENT — topic in, joke out | PRE_AND_POST | `LogAction` records the verdict, the run continues |
| `@guardrail(validator=ByoValidator(...))` on the LLM factory | LLM — every prompt before it reaches the model | PRE | `BlockAction` aborts the run |

On a harmful topic you see both actions in sequence: the agent-scope guardrail logs the
violation, then the LLM-scope guardrail blocks the run before the model is invoked.

Use **middleware** when guardrail policy should live in one place next to `create_agent()`;
use the **decorator** to guard individual targets (scope is inferred from what you decorate)
and to reuse one validator across many of them. See
[docs/guardrails.md](../../docs/guardrails.md) for the full comparison.

## Prerequisites (one-time, Org Admin)

1. Create the BYOG configuration under **Admin → AI Trust Layer → Guardrails
   Configurations**: pick a guardrail connector, create an Integration Service connection
   with your vendor credentials, and save the configuration with a validator name.
   (If the Guardrails Configurations page is not available on your tenant, Bring Your Own
   Guardrail is not enabled for you yet.)
2. Find the values this sample needs:

   ```bash
   uip agent guardrails list --byo --output json
   ```

   Copy your configuration's `ByoValidatorName` → `BYOG_VALIDATOR_NAME` and
   `ByoConnectionId` → `BYOG_CONNECTION_ID` in [graph.py](graph.py) (both ship as
   placeholders).

> Always pass the connection id: validator names are unique **per connection** only.

## Run it

```bash
uv sync
uv run uipath auth          # or populate .env
uv run uipath run agent '{"topic": "banana"}'
```

Expected outcomes — the verdicts come from **your configured guardrail vendor**
(with fallback to the UiPath-managed validator disabled in the configuration used here):

| Input topic | Result |
|---|---|
| a topic your guardrail passes (e.g. `"banana"` for a harmful-content validator) | agent PRE, LLM PRE (once per model call) and agent POST all evaluate → the agent returns a joke |
| a topic your guardrail flags | agent PRE **logs** the violation (`[GUARDRAIL] ... Failed`), then LLM PRE **blocks** the run with the vendor's verdict details, before the model is invoked |

What counts as a violation is entirely defined by the validator your admin configured —
a harmful-content validator flags violent or hateful topics, a PII validator flags personal
data, a custom classifier applies your own rules.

## Tuning the validator — parameters

Both flavors accept the same connector-defined parameter values, forwarded to the guardrail
on every evaluation: `validator_parameters=` on the middleware, `parameters=` on
`ByoValidator`.

The parameter **ids, types and allowed values are defined by the guardrail connector**, not
by this SDK — read them from your validator's `Parameters` array in
`uip agent guardrails list --byo` and pass the values through as-is. Each parameter there
carries its `Id`, `Type`, whether it is `Required`, its `DefaultValue`, and the applicable
`Options` / `KeySource` / `Min` / `Max` / `Step`. Omit the argument to fall back to those
defaults.

[graph.py](graph.py) carries a commented-out example for the Azure Content Safety
connector's `harmful_content` validator: it selects the four harm categories and sets a
per-category severity threshold. That connector flags a category when
`severity >= threshold` on Azure's 0/2/4/6 scale, so a threshold of `4` lets low-severity
(2) content through while still blocking 4 and 6, and categories that are not selected are
ignored. Verified end to end against a live configuration:

| Threshold | low-severity topic (2) | high-severity topic (4) |
|---|---|---|
| omitted | blocked | blocked |
| `4` | passes | blocked |
| `6` | passes | passes |

## How it works

- Both flavors build a guardrail with `validator_type="byo"` plus
  `byoValidatorName`/`byoConnectionId`, evaluated via
  `POST /agentsruntime_/api/execution/guardrails/validate`.
- The Guardrails Service resolves your configuration and invokes the Integration Service
  connector against your vendor, which returns the verdict.
- The middleware registers agent hooks for its configured scopes; the decorator infers the
  scope from the decorated target (here: a factory returning a `BaseChatModel` → LLM).
- No solution binding is needed: the connection is resolved server-side from the BYOG
  configuration — the agent only references it by validator name + connection id.

## Notes

- **You choose the scopes and stages** — BYOG validators are available on AGENT, LLM and TOOL
  scope for both PRE and POST, exactly like the built-in ones; this sample guards AGENT via
  middleware and LLM via the decorator.
- `ByoValidator` instances are reusable — declare once, stack on multiple targets.
- A removed or disabled configuration also returns an error result; both flavors fail open
  and log the details.

See [docs/guardrails.md](../../docs/guardrails.md) for the full guardrails guide.
