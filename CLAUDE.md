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

## Architecture (`src/uipath_agents/`)

- `agent_graph_builder/` - Agent graph construction (config, graph, LLM utils, message utils, session debug state)
- `_bts/` - BTS runtime (attributes, callbacks, helpers, runtime, state, storage)
- `_cli/` - CLI integration (cli_pull, constants, runtime/)
- `_config/` - Configuration loading and feature flags
- `_errors/` - Error handling and exception mapping
- `_licensing/` - License consumption tracking and licensed runtime
- `_observability/` - Azure Monitor OpenTelemetry (event emitter, exporters/, llmops/instrumentors, llmops/spans, tracing)
- `_services/` - Flags and licensing services
- `voice/` - Voice agent runtime (graph, job_runtime)
- `middlewares.py` - Middleware registration
- `preload.py` - Preload module discovery
- `runtime.py` - Runtime factory registration

## Testing

Tests in `./tests` organized as:
- `unit/` - Unit tests (agent_graph_builder, bts, cli, config, errors, observability, services, voice)
- `integration/` - Trace parity tests with golden files
- `e2e/` - End-to-end tests requiring UiPath authentication
- `observability/` - LLMOps observability tests

Markers: `e2e` (requires auth), `slow` (long-running)

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
