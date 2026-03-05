#!/usr/bin/env bash
# Upgrade uipath and uipath-langchain to their latest versions
set -euo pipefail

PYPROJECT="pyproject.toml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# State tracking for cleanup
ORIGINAL_BRANCH=""
STASHED_CHANGES=false
GIT_OPS_STARTED=false

# Cleanup function to restore original state on failure
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]] && $GIT_OPS_STARTED; then
        warn "Script failed, restoring original state..."

        # Return to original branch if we moved away
        if [[ -n "$ORIGINAL_BRANCH" ]]; then
            current=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
            if [[ "$current" != "$ORIGINAL_BRANCH" ]]; then
                warn "Returning to original branch: $ORIGINAL_BRANCH"
                git checkout "$ORIGINAL_BRANCH" 2>/dev/null || true
            fi
        fi

        # Restore stashed changes if we stashed them
        if $STASHED_CHANGES; then
            warn "Restoring stashed changes..."
            git stash pop 2>/dev/null || true
        fi
    fi
}

trap cleanup EXIT

# Check required tools
command -v curl >/dev/null 2>&1 || error "curl is required but not installed"
command -v python3 >/dev/null 2>&1 || error "python3 is required but not installed"
command -v uv >/dev/null 2>&1 || error "uv is required but not installed"
command -v git >/dev/null 2>&1 || error "git is required but not installed"

# Parse arguments
SKIP_GIT=false
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-git) SKIP_GIT=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-git    Skip git operations (branch creation, commit)"
            echo "  --dry-run     Show what would be done without making changes"
            echo "  -h, --help    Show this help message"
            exit 0
            ;;
        *) error "Unknown option: $1" ;;
    esac
done

# Get latest version from PyPI
get_pypi_version() {
    local package=$1
    curl -s "https://pypi.org/pypi/${package}/json" | python3 -c "
import sys, json
from packaging import version
data = json.load(sys.stdin)
releases = [r for r in data.get('releases', {}).keys() if data['releases'][r]]
releases.sort(key=lambda x: version.parse(x), reverse=True)
print(releases[0])
"
}

# Extract current version from pyproject.toml
get_current_version() {
    local package=$1
    grep -E "\"${package}(\\[.*\\])?==" "$PYPROJECT" | sed -E 's/.*==([0-9]+\.[0-9]+\.[0-9]+).*/\1/'
}

# Update version in pyproject.toml
update_version() {
    local package=$1
    local new_version=$2

    if $DRY_RUN; then
        info "[DRY-RUN] Would update $package to $new_version"
        return
    fi

    # Handle packages with extras like uipath-langchain[vertex,bedrock]
    sed -i.bak -E "s/(\"${package}(\\[.*\\])?)==[0-9]+\\.[0-9]+\\.[0-9]+\"/\\1==${new_version}\"/" "$PYPROJECT"
    rm -f "${PYPROJECT}.bak"
}

# Main execution
info "Fetching latest versions from PyPI..."

UIPATH_CURRENT=$(get_current_version "uipath")
LANGCHAIN_CURRENT=$(get_current_version "uipath-langchain")

UIPATH_LATEST=$(get_pypi_version "uipath")
LANGCHAIN_LATEST=$(get_pypi_version "uipath-langchain")

info "Current versions:"
echo "  uipath:          $UIPATH_CURRENT"
echo "  uipath-langchain: $LANGCHAIN_CURRENT"
echo ""
info "Latest versions:"
echo "  uipath:          $UIPATH_LATEST"
echo "  uipath-langchain: $LANGCHAIN_LATEST"
echo ""

# Check if updates are needed
if [[ "$UIPATH_CURRENT" == "$UIPATH_LATEST" ]] && [[ "$LANGCHAIN_CURRENT" == "$LANGCHAIN_LATEST" ]]; then
    info "All packages are already at their latest versions!"
    exit 0
fi

# Git operations
if ! $SKIP_GIT && ! $DRY_RUN; then
    info "Setting up git branch..."
    GIT_OPS_STARTED=true

    # Save original branch
    ORIGINAL_BRANCH=$(git rev-parse --abbrev-ref HEAD)

    # Stash uncommitted changes if any
    if ! git diff --quiet || ! git diff --cached --quiet; then
        info "Stashing uncommitted changes..."
        git stash push -m "upgrade-deps: auto-stash before dependency upgrade"
        STASHED_CHANGES=true
    fi

    # Fetch and checkout main
    git fetch origin
    git checkout main
    git pull origin main

    # Create new branch
    TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
    BRANCH_NAME="chore/bump-deps-$TIMESTAMP"
    git checkout -b "$BRANCH_NAME"
    info "Created branch: $BRANCH_NAME"
fi

# Update versions
info "Updating pyproject.toml..."

if [[ "$UIPATH_CURRENT" != "$UIPATH_LATEST" ]]; then
    update_version "uipath" "$UIPATH_LATEST"
    if ! $DRY_RUN; then
        info "Updated uipath: $UIPATH_CURRENT -> $UIPATH_LATEST"
    fi
fi

if [[ "$LANGCHAIN_CURRENT" != "$LANGCHAIN_LATEST" ]]; then
    update_version "uipath-langchain" "$LANGCHAIN_LATEST"
    if ! $DRY_RUN; then
        info "Updated uipath-langchain: $LANGCHAIN_CURRENT -> $LANGCHAIN_LATEST"
    fi
fi

if $DRY_RUN; then
    info "[DRY-RUN] Would regenerate lock file and sync dependencies"
    exit 0
fi

# Sync dependencies
info "Regenerating lock file..."
rm -f uv.lock
uv lock

info "Syncing dependencies..."
uv sync

# Git commit
if ! $SKIP_GIT; then
    info "Committing changes..."
    git add pyproject.toml uv.lock

    COMMIT_MSG="chore: bump uipath dependencies to latest versions

- uipath: $UIPATH_CURRENT -> $UIPATH_LATEST
- uipath-langchain: $LANGCHAIN_CURRENT -> $LANGCHAIN_LATEST"

    git commit -m "$COMMIT_MSG"

    info "Changes committed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Push the branch: git push -u origin $BRANCH_NAME"
    echo "  2. Create a PR to merge into main"

    if $STASHED_CHANGES; then
        echo ""
        echo "To return to your previous work:"
        echo "  3. git checkout $ORIGINAL_BRANCH"
        echo "  4. git stash pop"
    fi
fi

info "Done!"
