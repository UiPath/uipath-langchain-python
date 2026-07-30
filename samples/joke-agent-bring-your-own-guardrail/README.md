# Joke Agent — Bring Your Own Guardrail (BYOG)

A minimal starter for using a **customer-managed guardrail** in a coded agent. The agent
generates a family-friendly joke; every agent input/output (and the tool input) is validated
by a **BYOG configuration** — your own guardrail vendor connected through Integration
Service, referenced here purely by validator name + connection id. Credentials never appear
in this project.

The guardrail can be backed by any vendor your admin configured: a cloud subscription
(e.g. a content-safety service), a vendor validation platform, or a custom Integration
Service connector wrapping an internal classifier. The validator name and connection id in
[graph.py](graph.py) are **placeholders** — substitute your own configuration's values.

## Prerequisites (one-time, Org Admin)

1. Create the BYOG configuration under **Admin → AI Trust Layer → Guardrails
   Configurations**: pick a guardrail connector, create an Integration Service connection
   with your vendor credentials, and save the configuration with a validator name.
   (If the Guardrails Configurations page is not available on your tenant, Bring Your Own
   Guardrail is not enabled for you yet.)
2. Find the values this sample needs:

   ```bash
   uip agent guardrails list --output json
   ```

   BYOG configurations are listed alongside the built-in validators, so the same validator
   type may appear more than once — pick your configuration's entry. Copy its validator name
   → `BYOG_VALIDATOR_NAME` and its connection id → `BYOG_CONNECTION_ID` in [graph.py](graph.py)
   (both ship as placeholders), and update the `key` / `defaultValue` / `connector` fields in
   [bindings.json](bindings.json) to match your connection.

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
| a topic your guardrail passes (e.g. `"banana"` for a harmful-content validator) | `PASSED` → the agent returns a joke |
| a topic your guardrail flags | `VALIDATION_FAILED` → `BlockAction` aborts the run with the vendor's verdict details |

What counts as a violation is entirely defined by the validator your admin configured —
a harmful-content validator flags violent or hateful topics, a PII validator flags personal
data, a custom classifier applies your own rules.

## Tuning the validator — `validator_parameters`

The middleware accepts an optional `validator_parameters` argument that is forwarded to the
guardrail on every evaluation:

```python
*UiPathByoGuardrailMiddleware(
    validator_name=BYOG_VALIDATOR_NAME,
    scopes=[GuardrailScope.AGENT],
    action=BlockAction(),
    connection_id=BYOG_CONNECTION_ID,
    validator_parameters=BYOG_VALIDATOR_PARAMETERS,
)
```

The parameter **ids, types and allowed values are defined by the guardrail connector**, not
by this SDK — read them from your validator's `Parameters` array in
`uip agent guardrails list` and pass the values through as-is. Each parameter there carries
its `Id`, `Type`, whether it is `Required`, its `DefaultValue`, and the applicable
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

- `UiPathByoGuardrailMiddleware` builds a guardrail with `validator_type="byo"` plus
  `byoValidatorName`/`byoConnectionId`, evaluated via
  `POST /agentsruntime_/api/execution/guardrails/validate`.
- The Guardrails Service resolves your configuration and invokes the Integration Service
  connector against your vendor, which returns the verdict.
- [bindings.json](bindings.json) declares the connection as a solution binding so the
  connection id can be rebound per environment at deploy time.

## Notes

- **You choose the scopes and stages** — BYOG validators are available on AGENT, LLM and TOOL
  scope for both PRE and POST, exactly like the built-in ones; this sample registers AGENT
  and TOOL.
- A removed or disabled configuration also returns an error result; the middleware fails
  open and logs the details.

See [docs/guardrails.md](../../docs/guardrails.md) for the full guardrails guide.
