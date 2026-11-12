#!/bin/bash
# Source before the traced run: the LLM Gateway adds its spans to the run's
# LLMOps trace only when the run has a trace id and a folder key.

rm -f .uipath/gateway_trace_id
GATEWAY_FOLDER_PATH="${GATEWAY_FOLDER_PATH:-Shared}"
# `|| true`: run.sh uses `set -e`, and a failed lookup should skip the check, not the test.
GATEWAY_FOLDER_KEY=$(uv run python ../common/resolve_folder_key.py "$GATEWAY_FOLDER_PATH" | tail -n 1) || true
GATEWAY_TRACE_ID=$(uv run python -c "import uuid; print(uuid.uuid4().hex)" | tail -n 1) || true

if [[ "$GATEWAY_FOLDER_KEY" =~ ^[0-9a-fA-F-]{36}$ && "$GATEWAY_TRACE_ID" =~ ^[0-9a-f]{32}$ ]]; then
    export UIPATH_FOLDER_KEY="$GATEWAY_FOLDER_KEY"
    export UIPATH_TRACE_ID="$GATEWAY_TRACE_ID"
    mkdir -p .uipath
    echo "$UIPATH_TRACE_ID" > .uipath/gateway_trace_id
    echo "LLM Gateway span check: trace $UIPATH_TRACE_ID, folder $GATEWAY_FOLDER_PATH"
else
    message="LLM Gateway span check skipped: could not resolve folder '$GATEWAY_FOLDER_PATH'"
    if [ "${GITHUB_ACTIONS:-}" = "true" ]; then echo "::warning::$message"; else echo "$message"; fi
fi
