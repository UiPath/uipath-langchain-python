#!/bin/bash
set -e

CHAT_MODE=$([ "$USE_AZURE_CHAT" = "true" ] && echo "AZURE-CHAT" || echo "DEFAULT-CHAT")
echo "Running simple-local-mcp testcase with $CHAT_MODE mode..."
echo "USE_AZURE_CHAT=$USE_AZURE_CHAT"

# Authenticate with UiPath
echo "Authenticating with UiPath..."
uv run uipath auth --client-id="$CLIENT_ID" --client-secret="$CLIENT_SECRET" --base-url="$BASE_URL"

cd /app/testcases/simple-local-mcp

# Sync dependencies for this specific testcase
echo "Syncing dependencies..."
uv sync

# Pack the agent
echo "Packing agent..."
uv run uipath pack

# Run the agent
AGENT_INPUT=$(cat input.json)
echo "Running agent with $CHAT_MODE mode..."
echo "Input: $AGENT_INPUT"
uv run uipath run agent "$AGENT_INPUT"

# Validate output
echo "Validating output..."
python src/assert.py

echo "Simple-local-mcp testcase with $CHAT_MODE completed successfully."
