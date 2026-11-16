#!/bin/bash
# Source before the traced run: the LLM Gateway adds its spans to the run's
# LLMOps trace only when the run has a trace id and a folder key.
#
# GATEWAY_FOLDER_KEY is the key of the folder the Gateway publishes its spans to. CI sets it
# from the <ENV>_GATEWAY_FOLDER_KEY secret; locally, export it yourself.

rm -f .uipath/gateway_trace_id
mkdir -p .uipath
GATEWAY_TRACE_ID=$(python3 -c "import uuid; print(uuid.uuid4().hex)") || true

if [[ "${GATEWAY_FOLDER_KEY:-}" =~ ^[0-9a-fA-F-]{36}$ && "$GATEWAY_TRACE_ID" =~ ^[0-9a-f]{32}$ ]]; then
    # Never print the folder key; on GitHub Actions, mask it in case anything does.
    if [ "${GITHUB_ACTIONS:-}" = "true" ]; then echo "::add-mask::$GATEWAY_FOLDER_KEY"; fi
    export UIPATH_FOLDER_KEY="$GATEWAY_FOLDER_KEY"
    export UIPATH_TRACE_ID="$GATEWAY_TRACE_ID"
    echo "$UIPATH_TRACE_ID" > .uipath/gateway_trace_id
    echo "LLM Gateway span check: trace $UIPATH_TRACE_ID"
else
    echo "LLM Gateway span check setup failed: set GATEWAY_FOLDER_KEY to the key of the folder the Gateway publishes to" >&2
    return 1
fi
