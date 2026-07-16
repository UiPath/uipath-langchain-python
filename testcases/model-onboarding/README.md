# model-onboarding testcase

Exercises **one runtime-specified model** across the distinct `get_chat_model`
code paths it is expected to support, plus optional file attachments. Rolls
every `path × file` cell up into a single `success` boolean and asserts on both
the output and the emitted traces.

Unlike `multimodal-invoke` (which hardcodes its model matrix), the model here is
**input**. To onboard a model, edit `input.json` — no code change.

## The one file you edit: `input.json`

```json
{
  "prompt": "Describe the content of this file in one sentence.",
  "model_spec": {
    "model_name": "gpt-5.2-2025-12-11",
    "paths": ["azure_responses", "azure_chat_completions"],
    "agenthub_config": "agentsplayground",
    "files": ["image", "pdf"]
  }
}
```

- **`model_name`** — the vendor-qualified model ID. Note a single logical model
  may need a *different* ID per vendor family.
- **`paths`** — which `get_chat_model` code paths to exercise. Valid keys:
  `azure_responses`, `azure_chat_completions`, `vertex`, `bedrock_converse`,
  `bedrock_invoke`, `anthropic_sdk`. List only the paths the model actually
  ships on — a model ID sent to a vendor it doesn't exist on is a guaranteed
  (and misleading) failure.
- **`agenthub_config`** — AgentHub config header value; must exist in the tenant
  behind your `BASE_URL`. Defaults to `agentsplayground`.
- **`files`** — file attachments to test. Valid keys: `image`, `pdf`. Use `[]`
  for a **text-only** model — an empty list runs a plain reachability check via
  `ainvoke` instead of a multimodal call.

## Prerequisites (external to the repo)

- Model IDs per path you list.
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

The answer is `assert.py`'s exit code plus the `result_summary` grid it prints.
Exit 0 = model good on that env. Non-zero = the summary names the failing cell
and the truncated error.

## Mechanism B — CI dispatch (authoritative, all environments)

`.github/workflows/integration_tests.yml` auto-discovers this directory (the
hyphen in `model-onboarding` is required) and runs it across the **full
environment matrix — `alpha`, `staging`, and `cloud` (prod)** — with secrets
wired per environment (`ALPHA_*`, `STAGING_*`, `CLOUD_*`). You get the prod
(`cloud`) leg automatically; it is gated only on the `CLOUD_*` secrets existing
and the model being enabled in the prod tenant. To get the canonical answer:

```bash
git checkout -b onboard/<model>          # edit input.json first
git commit -am "onboard <model>"
git push -u origin onboard/<model>
gh pr create --fill
gh run watch <run-id>                     # model-onboarding/alpha + /staging + /cloud
# on failure:
gh run view <run-id> --log | grep -A30 "Test Results"
```

> Prod note: a failing `cloud` leg for a brand-new model usually means the model
> is not rolled out to the prod tenant yet — a provisioning signal, not a test
> bug.

Optional enhancement (not wired): add a `workflow_dispatch:` trigger with a
`model_spec` input so a run can be launched without a commit.

## What gets asserted (`src/assert.py`)

1. A `.nupkg` was produced.
2. `status == "successful"` and the `output` block exists.
3. `success is True` and `result_summary` is non-empty.
4. `"Successful execution."` appears in `local_run_output.log` (the second,
   empty-`UIPATH_JOB_KEY` run).
5. Traces contain the `run_model_onboarding` CHAIN span and at least one `LLM`
   span from a reachable client class (`expected_traces.json`).
