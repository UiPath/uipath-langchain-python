#!/bin/bash
cd /app/testcases/company-research-agent

# Sync dependencies for this specific testcase
echo "Syncing dependencies..."
uv sync

# Authenticate with UiPath
echo "Authenticating with UiPath..."
uv run uipath auth --client-id="$CLIENT_ID" --client-secret="$CLIENT_SECRET" --base-url="$BASE_URL" 

# Pack the agent
echo "Packing agent..."
uv run uipath pack

# Run the agent
echo "Input from input.json file"
uv run uipath run agent --file input.json

# Print the output file
source /app/testcases/common/print_output.sh
print_uipath_output

# Validate output
echo "Validating output..."
python src/assert.py || { echo "Validation failed!"; exit 1; }

echo "Testcase completed successfully."
