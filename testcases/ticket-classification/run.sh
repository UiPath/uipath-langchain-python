#!/bin/bash

# Authenticate with UiPath
echo "Authenticating with UiPath..."
uv run uipath auth --client-id="$CLIENT_ID" --client-secret="$CLIENT_SECRET" --base-url="$BASE_URL"

cd /app/testcases/ticket-classification

# Sync dependencies for this specific testcase
echo "Syncing dependencies..."
uv sync

# Pack the agent
echo "Packing agent..."
uv run uipath pack

# Run the agent
echo "Running agent with $CHAT_MODE mode..."
echo "Input from input.json file"
uv run uipath run agent --file input.json

echo "Resuming agent run by default with {'Answer': true}..."
uv run uipath run agent '{"Answer": true}' --resume;


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

echo "Testcase completed successfully."
