#!/bin/bash
set -e

echo "Syncing dependencies..."
uv sync

echo "Authenticating with UiPath..."
uv run uipath auth --client-id="$CLIENT_ID" --client-secret="$CLIENT_SECRET" --base-url="$BASE_URL"

echo "Initializing the project..."
uv run uipath init

echo "Packing agent..."
uv run uipath pack

echo "Input from input.json file"
uv run uipath run agent --file input.json

echo "Resuming agent run by default with {'Answer': true}..."
uv run uipath run agent '{"Answer": true}' --resume;

echo "Running agent again with empty UIPATH_JOB_KEY..."
export UIPATH_JOB_KEY=""
uv run uipath run agent --trace-file .uipath/traces.jsonl --file input.json >> local_run_output.log
uv run uipath run agent --trace-file .uipath/traces.jsonl '{"Answer": true}' --resume >> local_run_output.log
