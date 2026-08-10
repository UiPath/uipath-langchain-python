#!/bin/bash
set -e

echo "Syncing dependencies..."
uv sync

echo "Authenticating with UiPath..."
# `uipath auth --scope` defaults to 'OR.Execution' alone, which cannot create
# attachments (POST /odata/Attachments -> 403) even though the external app
# grants far more. Request what the agent actually needs.
uv run uipath auth --client-id="$CLIENT_ID" --client-secret="$CLIENT_SECRET" --base-url="$BASE_URL" \
  --scope="OR.Execution OR.Jobs OR.Administration OR.Folders"

echo "Initializing the project..."
uv run uipath init

echo "Packing agent..."
uv run uipath pack

echo "Running agent..."
uv run uipath run agent --file input.json

echo "Running agent again with empty UIPATH_JOB_KEY..."
export UIPATH_JOB_KEY=""
uv run uipath run agent --trace-file .uipath/traces.jsonl --file input.json >> local_run_output.log
