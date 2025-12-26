# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                          # Install dependencies
uv run uipath auth --alpha       # Authenticate
uv run uipath run agent.json '{}' # Run agent

uv run ruff format .             # Format
uv run ruff check . --fix        # Lint
uv run mypy .                    # Type check
uv run pytest tests/             # Test
```

## Code Style

- ALWAYS add typing annotations to each function or class, including return types
- Add Google-style docstrings to all public functions and classes
- SDK naming: `retrieve` (single by key), `retrieve_by_[field]` (by other field), `list` (multiple)
- Only use pytest (no unittest), all tests in `./tests` with typing annotations

## Code Comments Policy

Avoid redundant comments. Do NOT add comments that:
- Repeat what the method/variable name already says (e.g., `def get_agent_name` doesn't need `"""Get agent name."""`)
- Describe obvious single-line operations or simple delegations
- Duplicate information from class/module docstrings in method docstrings
- Add numbered step comments (# 1, # 2...) when a docstring already explains the sequence
- State what the code does when the code is self-explanatory from naming

DO add comments for:
- Non-obvious algorithms or tricks (e.g., XOR for key derivation)
- Ordering requirements that aren't apparent (e.g., "close inner span first, then outer")
- External schema references (e.g., "Matches C# Agents schema")
- Section dividers in long files (# --- Section Name ---)
- Constraints from external systems (e.g., "OTEL only supports primitives")
