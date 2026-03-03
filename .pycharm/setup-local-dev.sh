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
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

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

# Function to add a section to pyproject.toml if it doesn't already exist
add_pyproject_section() {
    local pyproject_file="$1"
    local section_header="$2"
    local section_content="$3"
    local grep_pattern="$4"

    if [ ! -f "$pyproject_file" ]; then
        echo -e "${RED}Error: $pyproject_file not found!${NC}"
        return 1
    fi

    if grep -q "$grep_pattern" "$pyproject_file"; then
        echo -e "${YELLOW}$section_header already exists in $(basename "$pyproject_file")${NC}"
    else
        echo -e "${GREEN}Adding $section_header to $(basename "$pyproject_file")${NC}"
        # Insert before [project.urls] to keep it with other entry-points
        sed -i '' "/^\[project\.urls\]/i\\
\\
$section_content
" "$pyproject_file"
    fi
}

# Step 1: Update uipath-agents-python/pyproject.toml (editable sources)
echo -e "${GREEN}Step 1: Updating uipath-agents-python/pyproject.toml${NC}"
AGENTS_PYPROJECT="$WORKSPACE_ROOT/uipath-agents-python/pyproject.toml"
AGENTS_SOURCES='[tool.uv.sources]
uipath = { path = "../uipath-python", editable = true }
uipath-langchain = { path = "../uipath-langchain-python", editable = true }'

add_uv_sources "$AGENTS_PYPROJECT" "$AGENTS_SOURCES"

# Step 1b: Add middlewares entry point for pull support
echo ""
echo -e "${GREEN}Step 1b: Adding middlewares entry point to uipath-agents-python/pyproject.toml${NC}"
MIDDLEWARE_SECTION='[project.entry-points."uipath.middlewares"]\
agents_middleware = "uipath_agents.middlewares:register_middleware"'
add_pyproject_section "$AGENTS_PYPROJECT" \
    '[project.entry-points."uipath.middlewares"]' \
    "$MIDDLEWARE_SECTION" \
    'uipath\.middlewares'

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

# Step 5: Add PyCharm Python Debug Server run configuration if not present
echo ""
DEBUG_PORT=5678
RUN_CONFIG_DIR="$WORKSPACE_ROOT/.idea/runConfigurations"
RUN_CONFIG_FILE="$RUN_CONFIG_DIR/Debug_Agent.xml"

if [ -f "$RUN_CONFIG_FILE" ]; then
    echo -e "${YELLOW}PyCharm debug run configuration already exists${NC}"
else
    echo -e "${GREEN}Step 5: Adding PyCharm Python Debug Server run configuration (port $DEBUG_PORT)${NC}"
    mkdir -p "$RUN_CONFIG_DIR"
    cat > "$RUN_CONFIG_FILE" << 'XMLEOF'
<component name="ProjectRunConfigurationManager">
  <configuration default="false" name="Debug Agent" type="PyRemoteDebugConfigurationType" factoryName="Python Remote Debug">
    <module name="uipath-agents-python" />
    <option name="PORT" value="5678" />
    <option name="HOST" value="localhost" />
    <PathMappingSettings>
      <option name="pathMappings">
        <list />
      </option>
    </PathMappingSettings>
    <option name="REDIRECT_OUTPUT" value="true" />
    <option name="SUSPEND_AFTER_CONNECT" value="true" />
    <method v="2" />
  </configuration>
</component>
XMLEOF
    echo -e "${GREEN}✓ Run configuration created at .idea/runConfigurations/Debug_Agent.xml${NC}"
fi

echo ""
echo -e "${GREEN}✓ Local development environment setup complete!${NC}"
echo ""
echo "All repositories are now configured with editable dependencies:"
echo "  - uipath-agents-python → uipath-python, uipath-langchain-python"
echo "  - uipath-agents-python → middlewares entry point (pull support)"
echo "  - uipath-langchain-python → uipath-python"
echo "  - PyCharm 'Debug Agent' run config (Python Remote Debug on port $DEBUG_PORT)"
