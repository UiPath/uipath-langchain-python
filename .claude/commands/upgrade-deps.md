---
model: sonnet
---

# Upgrade Dependencies Command

Upgrade uipath and uipath-langchain to their latest versions.

## Arguments
$ARGUMENTS

Arguments can include:
- `--skip-git`: Skip git operations (branch creation, commit)
- `--dry-run`: Show what would be done without making changes
- No arguments: Full workflow (fetch main, create branch, update deps, commit)

## Instructions

Run the upgrade-deps script:

```bash
./scripts/upgrade-deps.sh $ARGUMENTS
```

## Error Handling

If the script fails, analyze the output and take appropriate action:

### Network/PyPI Errors
- **"curl" or "Failed to fetch"**: Check internet connection, retry the command
- **"packaging" import error**: Run `uv pip install packaging` first

### Git Errors
- **"uncommitted changes"**: The script auto-stashes, but if it fails, manually stash or commit changes first
- **"branch already exists"**: Delete the old branch with `git branch -D chore/bump-deps-<timestamp>` or use `--skip-git`
- **"checkout failed"**: Ensure working directory is clean, check `git status`

### Dependency Resolution Errors
- **"No solution found"**: Version conflict exists. Check the error message for conflicting packages. May need to manually adjust version constraints in pyproject.toml
- **"uv lock failed"**: Try `rm uv.lock && uv lock --refresh` to force regeneration

### Already Up-to-Date
- **"All packages are already at their latest versions!"**: No action needed, this is expected behavior

## Post-Execution

After successful execution, the script will output next steps:
1. Push the branch: `git push -u origin <branch-name>`
2. Create a PR to merge into main

If changes were stashed, restore them:
1. `git checkout <original-branch>`
2. `git stash pop`
