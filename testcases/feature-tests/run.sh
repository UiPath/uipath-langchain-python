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

echo "Running agent with input from input.json file..."
uv run uipath run agent --file input.json

echo "Running agent again with empty UIPATH_JOB_KEY for local trace collection..."
export UIPATH_JOB_KEY=""
uv run uipath run agent --trace-file .uipath/traces.jsonl --file input.json >> local_run_output.log

echo "Feature tests completed successfully!"
