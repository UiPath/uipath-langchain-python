#!/bin/bash
set -e

echo "Syncing dependencies..."
uv sync

echo "Authenticating with UiPath..."
# Traces.Api reads the Gateway spans from LLMOps. The folder key comes from GATEWAY_FOLDER_KEY.
uv run uipath auth --client-id="$CLIENT_ID" --client-secret="$CLIENT_SECRET" --base-url="$BASE_URL" \
    --scope "OR.Execution Traces.Api"

echo "Initializing the project..."
uv run uipath init

# Keep the job key: as a job, the Gateway model's client-side LLM span is skipped.
# A local run (empty job key) keeps it and would fail forbidden_span_names.
echo "Running agent as a job (UIPATH_JOB_KEY set)..."
export UIPATH_JOB_KEY="${UIPATH_JOB_KEY:-3a03d5cb-fa21-4021-894d-a8e2eda0afe0}"
source ../common/gateway_trace_env.sh
uv run uipath run agent --trace-file .uipath/traces.jsonl --output-file __uipath/output.json \
    --file input.json >> local_run_output.log
