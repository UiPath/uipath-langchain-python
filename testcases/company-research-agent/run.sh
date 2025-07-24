#!/bin/bash
set -e

CHAT_MODE=$([ "$USE_AZURE_CHAT" = "true" ] && echo "AZURE-CHAT" || echo "DEFAULT-CHAT")
echo "Running company-research-agent testcase with $CHAT_MODE mode..."
echo "USE_AZURE_CHAT=$USE_AZURE_CHAT"

cd /app/testcases/company-research-agent

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

echo "Company-research-agent testcase with $CHAT_MODE completed successfully."
