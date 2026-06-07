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

# Clear any existing job key to use Console mode instead of SignalR mode
export UIPATH_JOB_KEY=""

echo "=== Running debug tests with pexpect ==="
uv run pytest src/test_debug.py -v -s
