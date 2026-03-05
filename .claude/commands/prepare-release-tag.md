---
model: sonnet
---

# Prepare Release Tag Command

Create and push a new release tag. This command ONLY pushes tags, NEVER the main branch.

## Arguments
$ARGUMENTS

Arguments can include:
- `--force` or `-f`: Force overwrite an existing tag
- `--tag <version>` or `-t <version>`: Specify a specific tag version (e.g., `v0.4.0`)
- `--dry-run`: Show what would be done without making changes
- No arguments: Auto-increment minor version and create new tag

## Instructions

Run the prepare-release-tag script:

```bash
./scripts/prepare-release-tag.sh $ARGUMENTS
```

## Error Handling

If the script fails, analyze the output and take appropriate action:

### Git/Network Errors
- **"git fetch" failed**: Check network connection and repository access
- **"Permission denied"**: Verify you have push access for tags

### Tag Errors
- **"Tag already exists"**: Use `--force` flag to overwrite, or specify a different version with `--tag`
- **"Failed to parse version"**: Tag format is invalid. Tags must follow `vMAJOR.MINOR.PATCH` format (e.g., `v0.4.0`)

### No Tags Found
- **"No existing tags found"**: Script will start from `v0.0.0` and create `v0.1.0`. This is expected for new repositories.

### Force Push Warnings
- **"Existing tag was overwritten"**: This is expected when using `--force`. Be aware this may affect developers who already pulled the original tag.

## Version Increment Rules

- **Format:** `vMAJOR.MINOR.PATCH`
- **Auto-increment:** MINOR version +1, PATCH reset to 0
- **Examples:**
  - `v0.3.0` → `v0.4.0`
  - `v1.2.5` → `v1.3.0`
  - `v2.9.0` → `v2.10.0`

## Usage Examples

1. **Auto-increment minor version:**
   ```bash
   ./scripts/prepare-release-tag.sh
   ```

2. **Force overwrite with auto-increment:**
   ```bash
   ./scripts/prepare-release-tag.sh --force
   ```

3. **Specify exact tag version:**
   ```bash
   ./scripts/prepare-release-tag.sh --tag v0.4.0
   ```

4. **Force overwrite specific tag:**
   ```bash
   ./scripts/prepare-release-tag.sh -f -t v0.4.0
   ```

5. **Dry run to preview:**
   ```bash
   ./scripts/prepare-release-tag.sh --dry-run
   ```

## Safety Guarantee

This command ONLY pushes tags, NEVER the main branch:
- Uses `git fetch` (not `git pull`) to avoid local changes
- Uses `git push origin v<VERSION>` to explicitly push only the tag
