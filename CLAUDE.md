# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`uipath-langchain` is a Python SDK that extends UiPath's Python SDK with LangChain/LangGraph integration. It implements the UiPath Runtime Protocol to deploy LangGraph agents to UiPath Cloud Platform. Requires Python 3.11+.

## Common Commands

```bash
# Install dependencies (uses uv)
uv sync --all-extras

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/path/to/test_file.py

# Run a single test
uv run pytest tests/path/to/test_file.py::test_name

# Lint
just lint              # ruff check + httpx client lint
just format            # ruff format check + fix

# Build
uv build
```

## Architecture

### Package: `src/uipath_langchain/`

- **`runtime/`** — `UiPathLangGraphRuntime` executes LangGraph graphs within the UiPath framework. Async execution with streaming, breakpoints, and message mapping. Registered as an entry point via `uipath_langchain.runtime:register_runtime_factory`.

- **`agent/`** — Agent implementation with sub-packages:
  - `react/` — ReAct agent pattern (agent, LLM node, router, tool node, guardrails)
  - `tools/` — Structured tools: context, escalation, extraction, integration, process, MCP adapters, durable interrupts. All inherit from `BaseUiPathStructuredTool`.
  - `guardrails/` — Input/output validation within agent execution
  - `multimodal/` — Multimodal invoke support
  - `wrappers/` — Agent decorators and wrappers

- **`chat/`** — LLM provider interfaces for OpenAI, Azure OpenAI, AWS Bedrock, Google Vertex AI. Uses **lazy imports** via `__getattr__` in `__init__.py` to keep CLI startup fast. Includes `hitl.py` with the `requires_approval` decorator for human-in-the-loop workflows. Factory pattern via `chat_model_factory.py`.

- **`retrievers/`** and **`vectorstores/`** — Context grounding retrieval and vector storage.

- **`guardrails/`** — Top-level guardrails with actions, enums, models, and middleware.

- **`_cli/`** — CLI commands (`uipath init`, `uipath new`) with project templates.

- **`_tracing/`** — OpenTelemetry instrumentation.

- **`_utils/`** — Shared utilities: HTTP request mixin, settings, sleep policy, environment helpers.

- **`middlewares.py`** — Entry point registered as `uipath_langchain.middlewares:register_middleware`.

### Entry Points (pyproject.toml)

The package registers two entry points consumed by the `uipath` CLI:
- `uipath.middleware` → `uipath_langchain.middlewares:register_middleware`
- `uipath.runtime_factory` → `uipath_langchain.runtime:register_runtime_factory`

## Key Conventions

- **httpx clients**: Always use `**get_httpx_client_kwargs()` when constructing `httpx.Client()` or `httpx.AsyncClient()`. A custom AST linter (`scripts/lint_httpx_client.py`) enforces this — it runs as part of `just lint`.

- **Lazy imports**: The `chat/` module defers heavy imports (langchain_openai, openai SDK) to optimize CLI startup. Use `__getattr__` pattern in `__init__.py` when adding new chat model providers.

- **Naming conventions for SDK methods**: `retrieve` (single by key), `retrieve_by_[field]` (single by other field), `list` (multiple resources).

- **Testing**: pytest only (no unittest). Tests in `./tests/` mirror source structure. Use pytest-asyncio for async tests (mode: auto). A circular import test (`test_no_circular_imports.py`) auto-discovers and validates all modules.

- **Type annotations**: All functions and classes require type annotations. Public APIs require Google-style docstrings.

- **Linting**: Ruff with rules E, F, B, I. Line length 88. mypy with pydantic plugin for type checking.

- **Bedrock/Vertex imports**: `bedrock.py` and `vertex.py` have per-file E402 ignores for conditional imports.
