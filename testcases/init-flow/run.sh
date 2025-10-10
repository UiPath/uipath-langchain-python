#!/bin/bash
cd /app/testcases/init-flow

echo "Syncing dependencies..."
uv sync

echo "Backing up pyproject.toml..."
cp pyproject.toml pyproject-overwrite.toml

echo "Creating new UiPath agent..."
uv run uipath new agent

# uipath new overwrites pyproject.toml, so we need to copy it back
echo "Restoring pyproject.toml..."
cp pyproject-overwrite.toml pyproject.toml
uv sync

echo "Authenticating with UiPath..."
uv run uipath auth --client-id="$CLIENT_ID" --client-secret="$CLIENT_SECRET" --base-url="$BASE_URL"

echo "Initializing UiPath..."
uv run uipath init

echo "Packing agent..."
uv run uipath pack

echo "Input from input.json file"
uv run uipath run agent --file input.json

source /app/testcases/common/print_output.sh
print_uipath_output

echo "Validating output..."
python src/assert.py || { echo "Validation failed!"; exit 1; }

echo "Testcase completed successfully."
