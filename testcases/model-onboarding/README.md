# model-onboarding testcase

Runs the coded **[`file_processing` agent](src/agents/README.md)** — Studio
Web's "Clone as Coded Agent" of the low-code FileProcessingAgent — against
**one runtime-specified model**, once per `api_flavor × file`.

Each file is uploaded as a real platform attachment and read by the agent's
*Analyze Files* tool, then asked a question whose answer is **unguessable** —
see [`fixtures/README.md`](fixtures/README.md):

| file | question | answer |
|---|---|---|
| `image` ([`fixtures/shape.png`](fixtures/shape.png), a purple square) | "What colour is the large shape in the centre of this image?" | `purple` |
| `pdf` ([`fixtures/document.pdf`](fixtures/document.pdf), one line of text) | "What is the verification code written in this document?" | `PDF-CODE-74915` |

Unguessable is the point: a random code and an arbitrary colour give a model
that never opened the file nothing to fall back on. Earlier fixtures asked
"what animal is in this image?" over `dog.jpg` — "dog" is the most likely
answer to that question with no image at all, and the file name is visible in
the prompt, so a model that skipped the file still scored correct.

A wrong or missing answer fails the cell and flips the single `success`
boolean, asserted alongside the emitted traces.

There is no fallback: if the attachment can't be created (the CI principal
needs permission to `POST /odata/Attachments`), the cell fails, because the
agent path was not exercised.

Unlike `multimodal-invoke` (which hardcodes its model matrix), the model here is
**input**. To onboard a model, edit `input.json` — no code change.

## The one file you edit: `input.json`

```json
{
  "model_spec": {
    "model_name": "gpt-5.2-2025-12-11",
    "api_flavors": ["openai:responses", "openai:chat-completions"],
    "agenthub_config": "agentsplayground"
  }
}
```

- **`model_name`** — the vendor-qualified model ID. Note a single logical model
  may need a *different* ID per vendor family.
- **`api_flavors`** — `vendor_type:api_flavor` pairs passed straight to
  `get_chat_model` (which accepts them as strings), e.g. `openai:responses`,
  `openai:chat-completions`, `awsbedrock:converse`, `awsbedrock:invoke`,
  `awsbedrock:AnthropicMessages`, `vertexai:generate-content`.
  `vendor_type:` alone lets the factory autodetect the flavor. The agent runs
  with the model built for each flavor.

  List only flavors the model actually ships on — a model ID sent to a vendor
  it doesn't exist on is a guaranteed (and misleading) failure.
- **`agenthub_config`** — AgentHub config header value; must exist in the tenant
  behind your `BASE_URL`. Defaults to `agentsplayground`.

Every file in `FILE_REGISTRY` (`src/main.py`) is exercised — add a case there
to cover another format.

## The judge probe (`judge_guardrail`) — needs a PAT, skips without one

After the file cells, each flavor also runs the model under test in the
**judge** role: a ReAct agent runs behind a real LLM-as-judge guardrail whose
judge model is the one being onboarded
([`src/agents/judge_guardrail/agent.py`](src/agents/judge_guardrail/agent.py)).
Two prompts run per flavor — one steered to violate the rule (the judge must
block) and one compliant (the judge must stay quiet) — each sampled 3 times and
decided by majority; the observed counts are always in the cell, e.g.
`judge_guardrail: ✓ judge discriminated (violating blocked 3/3, compliant allowed 3/3)`.

Note the guardrail is evaluated on **both** the incoming request and the
answer (that is how `create_agent` wires AGENT-scope guardrails — there is no
POST-only option), which is why the rule is phrased about "the text" rather
than "the answer".

The hosted validator lives on `agentsruntime_`, which client-credentials (S2S)
tokens cannot reach, so the probe needs a **user-identity PAT**:

- **Without `UIPATH_PAT`** (the default CI path): the cell records
  `judge_guardrail: – skipped (no UIPATH_PAT; needs user identity)` and does
  not fail the run.
- **With `UIPATH_PAT`** exported before `run.sh`: the script swaps the PAT into
  `.env` as `UIPATH_ACCESS_TOKEN` (it must go into the file — the CLI loads
  `.env` with `override=True`, so an exported env var would be silently
  ignored). The PAT then authenticates **everything** in the run, so use a
  dedicated service-user PAT with minimal scope and short expiry. In CI only
  the alpha leg has one (`ALPHA_TEST_PAT`).

## Prerequisites (external to the repo)

- Model IDs per flavor you list.
- Credentials for the target env: alpha (`ALPHA_TEST_CLIENT_ID` /
  `ALPHA_TEST_CLIENT_SECRET` / `ALPHA_BASE_URL`), staging (`STAGING_*`), or
  prod (`CLOUD_*` — prod is named `cloud` in this repo).
- The `agenthub_config` must exist in the target tenant.
- The vendor account behind the model must have it enabled in that tenant
  (Bedrock region, Vertex project, Azure deployment) — otherwise a `✗` cell is
  about provisioning, not the SDK.

## Mechanism A — local run (fast iteration)

From inside this directory:

```bash
export CLIENT_ID=...        # alpha or staging pair
export CLIENT_SECRET=...
export BASE_URL=...
export UIPATH_JOB_KEY=3a03d5cb-fa21-4021-894d-a8e2eda0afe0
export UIPATH_TRACING_ENABLED=false

bash run.sh                       # sync -> auth -> init -> pack -> run x2
bash ../common/validate_output.sh # prints output.json, runs src/assert.py
```

The answer is `assert.py`'s exit code plus the `result_summary` it prints —
including the model's actual answer per file. Exit 0 = model good on that env.
Non-zero = the summary names the failing cell and the error.

## Mechanism B — on-demand CI run (no commit, pick the model at start)

`.github/workflows/model_onboarding.yml` runs this testcase against **any model
you name at dispatch time**. The model spec comes from the workflow inputs and
is written into `input.json` at runtime — you never edit or commit a file to
change the model.

**From the GitHub UI:** Actions → "Model onboarding" → "Run workflow", fill in
`model_name` and `api_flavors`, pick the environment(s), then Run.

**From the CLI:**

```bash
gh workflow run model_onboarding.yml \
  -f model_name="anthropic.claude-sonnet-4-5-20250929-v1:0" \
  -f api_flavors="awsbedrock:converse,awsbedrock:invoke" \
  -f environments="alpha,staging,cloud"

gh run watch $(gh run list --workflow=model_onboarding.yml --limit 1 --json databaseId -q '.[0].databaseId')
# on failure:
gh run view <run-id> --log | grep -A30 "Test Results"
```

Environments are selectable (`alpha`, `staging`, `cloud`, or combinations);
each leg uses its own `ALPHA_*` / `STAGING_*` / `CLOUD_*` secrets. The `cloud`
leg is prod.

> Prod note: a failing `cloud` leg for a brand-new model usually means the model
> is not rolled out to the prod tenant yet — a provisioning signal, not a test
> bug.

## Not run on push/PR

This testcase is **deliberately excluded** from the push/PR integration matrix
(`integration_tests.yml` filters `model-onboarding` out of its discovery). It
runs **only on demand** via Mechanism A (local) or Mechanism B (the
`model_onboarding.yml` dispatch workflow). The committed `input.json` is just a
default/example spec — dispatch inputs override it at runtime.

## What gets asserted (`src/assert.py`)

1. A `.nupkg` was produced.
2. `status == "successful"` and the `output` block exists.
3. `success is True` and `result_summary` is non-empty.
4. `"Successful execution."` appears in `local_run_output.log` (the second,
   empty-`UIPATH_JOB_KEY` run).
5. Traces contain the `probe_file_processing` CHAIN span, the
   `attachments_upload` and `attachments_get_blob_uri` spans (the file really
   was uploaded and its bytes fetched), the `agent` AGENT span, and at least
   one `LLM` span from a reachable client class (`expected_traces.json`).

   Note there is no `Analyze Files` TOOL span: internal tools do not emit one
   (span kinds are AGENT/CHAIN/LLM only), and the span name would be sanitized
   to `Analyze_Files` in any case. Proof that the file was read comes from the
   answer itself — unguessable, per the table above — plus the attachment
   spans.
