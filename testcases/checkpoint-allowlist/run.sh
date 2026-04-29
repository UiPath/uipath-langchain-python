#!/bin/bash
set -e

SAMPLE_DIR="../../samples/checkpoint-allowlist"

echo "Copying sample files..."
cp "$SAMPLE_DIR/main.py" main.py
cp "$SAMPLE_DIR/langgraph.json" langgraph.json
cp "$SAMPLE_DIR/input.json" input.json

echo "Syncing dependencies..."
uv sync

echo "Authenticating with UiPath..."
uv run uipath auth --client-id="$CLIENT_ID" --client-secret="$CLIENT_SECRET" --base-url="$BASE_URL"

echo "Initializing the project..."
uv run uipath init

echo "Running agent (with serde block) — capturing log..."
uv run uipath run agent --file input.json 2>&1 | tee local_run_output.log
