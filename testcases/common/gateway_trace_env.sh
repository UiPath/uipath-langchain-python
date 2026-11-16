#!/bin/bash
# Source before the traced run: the LLM Gateway adds its spans to the run's
# LLMOps trace only when the run has a trace id and a folder key.

rm -f .uipath/gateway_trace_id
GATEWAY_FOLDER_PATH="${GATEWAY_FOLDER_PATH:-Shared}"
# The key is passed through a private file, never stdout. `|| true` keeps `set -e` from
# exiting silently; a failed lookup is reported below.
mkdir -p .uipath
GATEWAY_FOLDER_KEY_FILE=.uipath/gateway_folder_key
rm -f "$GATEWAY_FOLDER_KEY_FILE"
uv run python ../common/resolve_folder_key.py "$GATEWAY_FOLDER_PATH" "$GATEWAY_FOLDER_KEY_FILE" || true
GATEWAY_FOLDER_KEY=$(cat "$GATEWAY_FOLDER_KEY_FILE" 2>/dev/null) || true
rm -f "$GATEWAY_FOLDER_KEY_FILE"
GATEWAY_TRACE_ID=$(uv run python -c "import uuid; print(uuid.uuid4().hex)" | tail -n 1) || true

if [[ "$GATEWAY_FOLDER_KEY" =~ ^[0-9a-fA-F-]{36}$ && "$GATEWAY_TRACE_ID" =~ ^[0-9a-f]{32}$ ]]; then
    # Never print the folder key; on GitHub Actions, mask it in case anything does.
    if [ "${GITHUB_ACTIONS:-}" = "true" ]; then echo "::add-mask::$GATEWAY_FOLDER_KEY"; fi
    export UIPATH_FOLDER_KEY="$GATEWAY_FOLDER_KEY"
    export UIPATH_TRACE_ID="$GATEWAY_TRACE_ID"
    echo "$UIPATH_TRACE_ID" > .uipath/gateway_trace_id
    echo "LLM Gateway span check: trace $UIPATH_TRACE_ID, folder $GATEWAY_FOLDER_PATH"
else
    echo "LLM Gateway span check setup failed: could not resolve folder '$GATEWAY_FOLDER_PATH'" >&2
    return 1
fi
