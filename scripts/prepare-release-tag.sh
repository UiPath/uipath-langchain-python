#!/usr/bin/env bash
# Prepare and push a new release tag
# SAFETY: This script ONLY pushes tags, NEVER the main branch
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
success() { echo -e "${GREEN}✓${NC} $1"; }

# Check required tools
command -v git >/dev/null 2>&1 || error "git is required but not installed"

# Parse arguments
FORCE=false
DRY_RUN=false
CUSTOM_TAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--force) FORCE=true; shift ;;
        -t|--tag) CUSTOM_TAG="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Prepare and push a new release tag (ONLY pushes tags, never main branch)"
            echo ""
            echo "Options:"
            echo "  -f, --force       Force overwrite an existing tag"
            echo "  -t, --tag <ver>   Specify exact tag version (e.g., v0.4.0)"
            echo "  --dry-run         Show what would be done without making changes"
            echo "  -h, --help        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Auto-increment minor version"
            echo "  $0 --force            # Force overwrite with auto-increment"
            echo "  $0 --tag v0.4.0       # Create specific tag"
            echo "  $0 -f -t v0.4.0       # Force create specific tag"
            exit 0
            ;;
        *) error "Unknown option: $1. Use -h for help." ;;
    esac
done

# Enforce main branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$CURRENT_BRANCH" != "main" ]]; then
    error "Release tags must be created from the main branch. Current branch: $CURRENT_BRANCH. Switch to main with: git checkout main"
fi

# Fetch latest main (read-only)
info "Fetching latest main branch (read-only)..."
if ! $DRY_RUN; then
    git fetch origin main
fi
success "Fetched latest main branch"

# Ensure local main is up to date with origin/main
LOCAL_SHA=$(git rev-parse HEAD)
REMOTE_SHA=$(git rev-parse origin/main)
if [[ "$LOCAL_SHA" != "$REMOTE_SHA" ]]; then
    error "Local main ($LOCAL_SHA) is out of sync with origin/main ($REMOTE_SHA). Run: git pull origin main"
fi

# Get latest tag
get_latest_tag() {
    git tag --sort=-v:refname 2>/dev/null | head -1
}

# Parse semantic version from tag (e.g., v0.3.0 -> "0 3 0")
parse_version() {
    local tag=$1
    echo "$tag" | sed -E 's/^v?([0-9]+)\.([0-9]+)\.([0-9]+)$/\1 \2 \3/'
}

# Increment minor version
increment_minor() {
    local tag=$1
    local parts
    read -r major minor patch <<< "$(parse_version "$tag")"

    if [[ -z "$major" ]]; then
        error "Failed to parse version from tag: $tag"
    fi

    # Increment minor, reset patch to 0
    local new_minor=$((minor + 1))
    echo "v${major}.${new_minor}.0"
}

# Determine target tag
if [[ -n "$CUSTOM_TAG" ]]; then
    # Ensure tag starts with 'v'
    if [[ ! "$CUSTOM_TAG" =~ ^v ]]; then
        CUSTOM_TAG="v$CUSTOM_TAG"
    fi
    NEW_TAG="$CUSTOM_TAG"
    info "Using specified tag: $NEW_TAG"
else
    LATEST_TAG=$(get_latest_tag)

    if [[ -z "$LATEST_TAG" ]]; then
        warn "No existing tags found, starting from v0.0.0"
        LATEST_TAG="v0.0.0"
    fi

    info "Latest tag: $LATEST_TAG"
    NEW_TAG=$(increment_minor "$LATEST_TAG")
    info "New tag (minor incremented): $NEW_TAG"
fi

# Check if tag already exists
TAG_EXISTS=false
if git rev-parse "$NEW_TAG" >/dev/null 2>&1; then
    TAG_EXISTS=true
fi

if $TAG_EXISTS && ! $FORCE; then
    error "Tag $NEW_TAG already exists. Use --force to overwrite."
fi

if $TAG_EXISTS && $FORCE; then
    warn "Tag $NEW_TAG exists and will be overwritten (--force)"
fi

# Dry run output
if $DRY_RUN; then
    echo ""
    info "[DRY-RUN] Would perform the following:"
    if $TAG_EXISTS && $FORCE; then
        echo "  - Delete local tag: git tag -d $NEW_TAG"
        echo "  - Create tag at origin/main: git tag $NEW_TAG origin/main"
        echo "  - Force push tag: git push -f origin $NEW_TAG"
    else
        echo "  - Create tag at origin/main: git tag $NEW_TAG origin/main"
        echo "  - Push tag: git push origin $NEW_TAG"
    fi
    echo ""
    info "[DRY-RUN] Main branch would NOT be modified or pushed"
    exit 0
fi

# Create and push tag
echo ""
if $FORCE && $TAG_EXISTS; then
    info "Force overwriting tag: $NEW_TAG"
    git tag -d "$NEW_TAG" 2>/dev/null || true
    git tag "$NEW_TAG" origin/main
    git push -f origin "$NEW_TAG"
    success "Tag $NEW_TAG force-created and force-pushed successfully"
    warn "Existing tag $NEW_TAG was overwritten"
else
    info "Creating new tag: $NEW_TAG"
    git tag "$NEW_TAG" origin/main
    git push origin "$NEW_TAG"
    success "Tag $NEW_TAG created and pushed successfully"
fi

success "Main branch was NOT modified or pushed"
echo ""
info "Release tag $NEW_TAG is ready!"
