# model-onboarding testcase

Runs the coded **[`file_processing` agent](src/agents/README.md)** — Studio
Web's "Clone as Coded Agent" of the low-code FileProcessingAgent — against
**one runtime-specified model**, once per `api_flavor × file`.

Each cell records the model's **actual answer** about the file — uploaded as a
real platform attachment and read by the agent's *Analyze Files* tool — so
`result_summary` is evidence rather than a bare tick. Any failing cell flips
the single `success` boolean, asserted alongside the emitted traces.

There is no fallback: if the attachment can't be created (the CI principal
needs permission to `POST /odata/Attachments`), the cell fails, because the
agent path was not exercised.

Unlike `multimodal-invoke` (which hardcodes its model matrix), the model here is
**input**. To onboard a model, edit `input.json` — no code change.

## The one file you edit: `input.json`

```json
{
  "prompt": "Describe the content of this file in one sentence.",
  "model_spec": {
    "model_name": "gpt-5.2-2025-12-11",
    "api_flavors": ["azure_responses", "azure_chat_completions"],
    "agenthub_config": "agentsplayground",
    "files": ["image", "pdf"]
  }
}
```

- **`model_name`** — the vendor-qualified model ID. Note a single logical model
  may need a *different* ID per vendor family.
- **`api_flavors`** — which API flavors to exercise. Two forms, freely mixed:
  - shorthands: `azure_responses`, `azure_chat_completions`, `vertex`,
    `bedrock_converse`, `bedrock_invoke`;
  - **`vendor_type:api_flavor` pairs passed straight to `get_chat_model`**
    (which accepts them as strings), e.g. `awsbedrock:converse`,
    `openai:responses`, `vertexai:generate-content`,
    `awsbedrock:AnthropicMessages`. `vendor_type:` alone lets the factory
    autodetect the flavor. The agent runs with the model built for that
    flavor.

  List only flavors the model actually ships on — a model ID sent to a vendor
  it doesn't exist on is a guaranteed (and misleading) failure.
- **`agenthub_config`** — AgentHub config header value; must exist in the tenant
  behind your `BASE_URL`. Defaults to `agentsplayground`.
- **`files`** — files the agent processes, one cell each. Valid keys: `image`,
  `pdf`. Defaults to both; at least one is required (the agent needs a file).
- **`prompt`** — the task the agent answers about each file.

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
`model_name`, `api_flavors`, `files`, and pick the environment(s), then Run.

**From the CLI:**

```bash
gh workflow run model_onboarding.yml \
  -f model_name="anthropic.claude-sonnet-4-5-20250929-v1:0" \
  -f api_flavors="bedrock_converse,bedrock_invoke" \
  -f files="image,pdf" \
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
5. Traces contain the `probe_file_processing` CHAIN span and at least one `LLM`
   span from a reachable client class (`expected_traces.json`).
