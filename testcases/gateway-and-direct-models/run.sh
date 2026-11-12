#!/bin/bash
set -e

echo "Syncing dependencies..."
uv sync

echo "Authenticating with UiPath..."
uv run uipath auth --client-id="$CLIENT_ID" --client-secret="$CLIENT_SECRET" --base-url="$BASE_URL"

echo "Initializing the project..."
uv run uipath init

echo "Running agent with empty UIPATH_JOB_KEY..."
export UIPATH_JOB_KEY=""
source ../common/gateway_trace_env.sh
uv run uipath run agent --trace-file .uipath/traces.jsonl --output-file __uipath/output.json \
    --file input.json >> local_run_output.log
