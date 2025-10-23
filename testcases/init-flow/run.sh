#!/bin/bash
set -e

echo "Syncing dependencies..."
uv sync

echo "Backing up pyproject.toml..."
cp pyproject.toml pyproject-overwrite.toml

echo "Creating new UiPath agent..."
uv run uipath new agent

# uipath new overwrites pyproject.toml, so we need to copy it back
echo "Restoring pyproject.toml..."
cp pyproject-overwrite.toml pyproject.toml
uv sync

echo "Authenticating with UiPath..."
uv run uipath auth --client-id="$CLIENT_ID" --client-secret="$CLIENT_SECRET" --base-url="$BASE_URL"

echo "Initializing UiPath..."
uv run uipath init

echo "Packing agent..."
uv run uipath pack

echo "Input from input.json file"
uv run uipath run agent --file input.json

echo "Running agent again with empty UIPATH_JOB_KEY..."
export UIPATH_JOB_KEY=""
uv run uipath run agent --trace-file .uipath/traces.jsonl --file input.json >> local_run_output.log