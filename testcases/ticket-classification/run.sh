#!/bin/bash
cd /app/testcases/ticket-classification

echo "Syncing dependencies..."
uv sync

echo "Authenticating with UiPath..."
uv run uipath auth --client-id="$CLIENT_ID" --client-secret="$CLIENT_SECRET" --base-url="$BASE_URL"

echo "Running uipath init..."
uv run uipath init

echo "Packing agent..."
uv run uipath pack

# run and them resume the agent to simulate user interaction
echo "Input from input.json file"
uv run uipath run agent --file input.json

echo "Resuming agent run by default with {'Answer': true}..."
uv run uipath run agent '{"Answer": true}' --resume;

source /app/testcases/common/print_output.sh
print_uipath_output

echo "Validating output..."
python src/assert.py || { echo "Validation failed!"; exit 1; }

echo "Testcase completed successfully."
