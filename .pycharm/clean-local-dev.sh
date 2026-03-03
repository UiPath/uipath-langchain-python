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
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo -e "${GREEN}Cleaning local development environment...${NC}"
echo "Workspace root: $WORKSPACE_ROOT"
echo ""

# Function to remove a TOML section from pyproject.toml
remove_toml_section() {
    local pyproject_file="$1"
    local section_pattern="$2"
    local section_name="$3"

    if [ ! -f "$pyproject_file" ]; then
        echo -e "${RED}Error: $pyproject_file not found!${NC}"
        return 1
    fi

    if grep -q "$section_pattern" "$pyproject_file"; then
        echo -e "${GREEN}Removing $section_name from $(basename "$pyproject_file")${NC}"

        awk -v pattern="$section_pattern" '
            BEGIN { in_section = 0 }
            $0 ~ pattern {
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
        echo -e "${YELLOW}$section_name not found in $(basename "$pyproject_file")${NC}"
    fi
}

# Convenience wrapper for removing [tool.uv.sources]
remove_uv_sources() {
    remove_toml_section "$1" '^\[tool\.uv\.sources\]' '[tool.uv.sources]'
}

# Step 1: Remove local-only sections from uipath-agents-python/pyproject.toml
echo -e "${GREEN}Step 1: Cleaning uipath-agents-python/pyproject.toml${NC}"
AGENTS_PYPROJECT="$WORKSPACE_ROOT/uipath-agents-python/pyproject.toml"
remove_uv_sources "$AGENTS_PYPROJECT"
remove_toml_section "$AGENTS_PYPROJECT" \
    '^\[project\.entry-points\."uipath\.middlewares"\]' \
    '[project.entry-points."uipath.middlewares"]'

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
echo "All local-only configuration has been removed:"
echo "  - uipath-agents-python: [tool.uv.sources] section removed"
echo "  - uipath-agents-python: [project.entry-points.\"uipath.middlewares\"] section removed"
echo "  - uipath-langchain-python: [tool.uv.sources] section removed"
