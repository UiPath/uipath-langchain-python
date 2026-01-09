#!/bin/bash

# Script to clean local development environment
# This script removes editable dependencies from pyproject.toml files and syncs dependencies

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the workspace root directory (where this script is located)
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${GREEN}Cleaning local development environment...${NC}"
echo "Workspace root: $WORKSPACE_ROOT"
echo ""

# Function to remove [tool.uv.sources] section from pyproject.toml
remove_uv_sources() {
    local pyproject_file="$1"
    
    if [ ! -f "$pyproject_file" ]; then
        echo -e "${RED}Error: $pyproject_file not found!${NC}"
        return 1
    fi
    
    # Check if [tool.uv.sources] section exists
    if grep -q "^\[tool\.uv\.sources\]" "$pyproject_file"; then
        echo -e "${GREEN}Removing [tool.uv.sources] section from $(basename "$pyproject_file")${NC}"
        
        # Use awk to remove the section and all its content until the next section or end of file
        # Remove from [tool.uv.sources] until next [ section or end of file
        awk '
            BEGIN { in_section = 0 }
            /^\[tool\.uv\.sources\]/ {
                in_section = 1
                next
            }
            in_section && /^\[/ {
                in_section = 0
                print
                next
            }
            in_section {
                next
            }
            !in_section {
                print
            }
        ' "$pyproject_file" > "$pyproject_file.tmp" && mv "$pyproject_file.tmp" "$pyproject_file"
        
        echo -e "${GREEN}✓ Section removed${NC}"
    else
        echo -e "${YELLOW}[tool.uv.sources] section not found in $(basename "$pyproject_file")${NC}"
    fi
}

# Step 1: Remove [tool.uv.sources] from uipath-agents-python/pyproject.toml
echo -e "${GREEN}Step 1: Removing [tool.uv.sources] from uipath-agents-python/pyproject.toml${NC}"
AGENTS_PYPROJECT="$WORKSPACE_ROOT/uipath-agents-python/pyproject.toml"
remove_uv_sources "$AGENTS_PYPROJECT"

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

# Step 3: Remove [tool.uv.sources] from uipath-langchain-python/pyproject.toml
echo ""
echo -e "${GREEN}Step 3: Removing [tool.uv.sources] from uipath-langchain-python/pyproject.toml${NC}"
LANGCHAIN_PYPROJECT="$WORKSPACE_ROOT/uipath-langchain-python/pyproject.toml"
remove_uv_sources "$LANGCHAIN_PYPROJECT"

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
echo -e "${GREEN}✓ Local development environment cleaned!${NC}"
echo ""
echo "All editable dependencies have been removed:"
echo "  - uipath-agents-python: [tool.uv.sources] section removed"
echo "  - uipath-langchain-python: [tool.uv.sources] section removed"
