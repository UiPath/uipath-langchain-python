#!/bin/bash

# Script to set up local development environment after git clean
# This script configures editable dependencies for the three linked repositories

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the workspace root directory (where this script is located)
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${GREEN}Setting up local development environment...${NC}"
echo "Workspace root: $WORKSPACE_ROOT"
echo ""

# Function to add or update [tool.uv.sources] section in pyproject.toml
add_uv_sources() {
    local pyproject_file="$1"
    local sources_content="$2"
    
    if [ ! -f "$pyproject_file" ]; then
        echo -e "${RED}Error: $pyproject_file not found!${NC}"
        return 1
    fi
    
    # Check if [tool.uv.sources] section already exists
    if grep -q "^\[tool\.uv\.sources\]" "$pyproject_file"; then
        echo -e "${YELLOW}[tool.uv.sources] section already exists in $(basename "$pyproject_file")${NC}"
        echo "  Verifying content..."
        
        # Check if the expected content is present
        if echo "$sources_content" | while IFS= read -r line; do
            if [ -n "$line" ]; then
                if ! grep -q "$line" "$pyproject_file"; then
                    echo -e "${YELLOW}  Warning: Expected line not found: $line${NC}"
                fi
            fi
        done; then
            echo -e "${GREEN}  Content looks correct${NC}"
        fi
    else
        echo -e "${GREEN}Adding [tool.uv.sources] section to $(basename "$pyproject_file")${NC}"
        echo "" >> "$pyproject_file"
        echo "$sources_content" >> "$pyproject_file"
    fi
}

# Step 1: Update uipath-agents-python/pyproject.toml
echo -e "${GREEN}Step 1: Updating uipath-agents-python/pyproject.toml${NC}"
AGENTS_PYPROJECT="$WORKSPACE_ROOT/uipath-agents-python/pyproject.toml"
AGENTS_SOURCES='[tool.uv.sources]
uipath = { path = "../uipath-python", editable = true }
uipath-langchain = { path = "../uipath-langchain-python", editable = true }'

add_uv_sources "$AGENTS_PYPROJECT" "$AGENTS_SOURCES"

# Step 2: Sync dependencies for uipath-agents-python
echo ""
echo -e "${GREEN}Step 2: Syncing dependencies for uipath-agents-python${NC}"
cd "$WORKSPACE_ROOT/uipath-agents-python"
if command -v uv &> /dev/null; then
    uv sync
    echo -e "${GREEN}✓ Dependencies synced for uipath-agents-python${NC}"
else
    echo -e "${RED}Error: uv command not found! Please install uv first.${NC}"
    exit 1
fi

# Step 3: Update uipath-langchain-python/pyproject.toml
echo ""
echo -e "${GREEN}Step 3: Updating uipath-langchain-python/pyproject.toml${NC}"
LANGCHAIN_PYPROJECT="$WORKSPACE_ROOT/uipath-langchain-python/pyproject.toml"
LANGCHAIN_SOURCES='[tool.uv.sources]
uipath = { path = "../uipath-python", editable = true }'

add_uv_sources "$LANGCHAIN_PYPROJECT" "$LANGCHAIN_SOURCES"

# Step 4: Sync dependencies for uipath-langchain-python
echo ""
echo -e "${GREEN}Step 4: Syncing dependencies for uipath-langchain-python${NC}"
cd "$WORKSPACE_ROOT/uipath-langchain-python"
if command -v uv &> /dev/null; then
    uv sync
    echo -e "${GREEN}✓ Dependencies synced for uipath-langchain-python${NC}"
else
    echo -e "${RED}Error: uv command not found! Please install uv first.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Local development environment setup complete!${NC}"
echo ""
echo "All repositories are now configured with editable dependencies:"
echo "  - uipath-agents-python → uipath-python, uipath-langchain-python"
echo "  - uipath-langchain-python → uipath-python"
