#!/bin/bash

# UiPath Robot Python - Local Development Startup Script
# Run this in a new terminal to start the local robot

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEBUG_FLAG=""
CLEAR_CACHE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG_FLAG="--debug"
            shift
            ;;
        --clear-cache)
            CLEAR_CACHE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--debug] [--clear-cache]"
            exit 1
            ;;
    esac
done

if [ "$CLEAR_CACHE" = true ]; then
    if [ -d "$SCRIPT_DIR/.uipath" ]; then
        echo "🗑️  Clearing local cache (.uipath)..."
        rm -rf "$SCRIPT_DIR/.uipath"
    else
        echo "ℹ️  No .uipath cache directory found, skipping"
    fi
fi

if [ ! -f "$SCRIPT_DIR/uipath.robot.toml" ]; then
    echo "📝 Generating uipath.robot.toml with path: $SCRIPT_DIR"
    cat > "$SCRIPT_DIR/uipath.robot.toml" <<EOF
[dependencies]
uipath-agents = { path = "$SCRIPT_DIR", editable = true }
EOF
else
    echo "📝 uipath.robot.toml already exists, skipping generation"
fi

echo "🔐 Authenticating with UiPath..."
if ! uv run uipath auth --alpha; then
    echo "⚠️  Auth failed, retrying with --force..."
    uv run uipath auth --alpha --force
fi

echo ""
echo "🤖 Starting UiPath Robot..."
echo "   The robot will connect to Orchestrator and listen for jobs."
echo "   Use runtime type 'Development' when starting jobs."
if [ -n "$DEBUG_FLAG" ]; then
    echo "   Debug mode enabled - debugger will listen on port 5678"
fi
echo ""

uv run uipath-robot $DEBUG_FLAG

