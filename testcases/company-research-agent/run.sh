#!/bin/bash
set -e

CHAT_MODE=$([ "$USE_AZURE_CHAT" = "true" ] && echo "AZURE-CHAT" || echo "DEFAULT-CHAT")
echo "Running company-research-agent testcase with $CHAT_MODE mode..."
echo "USE_AZURE_CHAT=$USE_AZURE_CHAT"

# Authenticate with UiPath
echo "Authenticating with UiPath..."
uv run uipath auth --client-id="$CLIENT_ID" --client-secret="$CLIENT_SECRET" --base-url="$BASE_URL"

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

# Print the output file
echo "Printing output file..."
if [ -f "__uipath/output.json" ]; then
    echo "=== OUTPUT FILE CONTENT ==="
    cat __uipath/output.json
    echo "=== END OUTPUT FILE CONTENT ==="
else
    echo "ERROR: __uipath/output.json not found!"
    echo "Checking directory contents:"
    ls -la
    if [ -d "__uipath" ]; then
        echo "Contents of __uipath directory:"
        ls -la __uipath/
    else
        echo "__uipath directory does not exist!"
    fi
fi

# Validate output
echo "Validating output..."
python src/assert.py

echo "Company-research-agent testcase with $CHAT_MODE completed successfully."
