#!/bin/bash
set -e

TEMPLATE_DIR="../../template"

echo "Copying template files..."
cp "$TEMPLATE_DIR/main.py" main.py
cp "$TEMPLATE_DIR/langgraph.json" langgraph.json
cp "$TEMPLATE_DIR/input.json" input.json
cp "$TEMPLATE_DIR/uipath.json" uipath.json
cp -r "$TEMPLATE_DIR/evaluations" evaluations

echo "Syncing dependencies..."
uv sync

echo "Authenticating with UiPath..."
uv run uipath auth --client-id="$CLIENT_ID" --client-secret="$CLIENT_SECRET" --base-url="$BASE_URL"

echo "Initializing the project..."
uv run uipath init

echo "Running agent..."
uv run uipath run agent --file input.json

echo "Running agent again with empty UIPATH_JOB_KEY..."
export UIPATH_JOB_KEY=""
uv run uipath run agent --trace-file .uipath/traces.jsonl --file input.json >> local_run_output.log

echo "Running evaluation..."
uv run uipath eval --no-report --output-file eval_output.json
