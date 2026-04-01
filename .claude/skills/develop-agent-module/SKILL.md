---
name: develop-agent-module
description: Use when modifying, extending, or debugging anything inside the agent/ module — the ReAct agent graph, tools, guardrails, exceptions, multimodal, or wrappers. Guides clean architecture, module boundaries, and correct integration patterns. Triggers on "add a node to the agent graph", "modify routing", "add a guardrail action", "change agent state", "fix agent exceptions", "add multimodal support", or any work touching files under src/uipath_langchain/agent/.
allowed-tools: Read, Write, Edit, Glob, Grep, Bash, Agent
---

# Develop the Agent Module

Guidance for working inside `src/uipath_langchain/agent/` — the ReAct agent orchestration layer.

**Read first, code second.** Before changing anything, read the existing code in the subsystem you're modifying. Follow established patterns unless you have a concrete reason to diverge — and if you do, discuss it first.

## Architectural Constraints

The agent module is designed so that subsystems can be composed into different loop patterns without coupling:

- **`tools/` must not know about the loop.** Tools are standalone capabilities — they receive input, do work, return output. They must not import from `react/` (except shared types in `react/types.py` and helpers in `react/utils.py`). A tool should work regardless of whether it's orchestrated by a ReAct loop, a plan-and-execute loop, or something else entirely.
- **`guardrails/` must not know about the loop.** Guardrail evaluation and actions are generic validation — they inspect data, return pass/fail, and execute actions. The *wiring* of guardrails into a specific loop happens in `react/guardrails/`, not in `guardrails/` itself.
- **`react/` is one loop implementation, not the only possible one.** It owns graph construction, routing, and node lifecycle. It composes tools and guardrails but those subsystems don't depend back on it.
- **`exceptions/`, `multimodal/`, `messages/` are fully standalone.** They have no knowledge of the loop or each other.
- **`wrappers/` are loop-agnostic but impose state contracts.** They don't import the loop, but they do depend on specific state fields existing (e.g., `inner_state.job_attachments`). When adding a wrapper, ensure the state fields it expects are documented in `react/types.py`.

This means: if you're adding a tool and find yourself importing from `react/agent.py` or `react/router.py`, you're coupling the tool to the loop. Stop and rethink.

## Upstream Types (`uipath.agent` in uipath-python)

Agent definition models, resource configs, guardrail models, flow control tools (`END_EXECUTION_TOOL`, `RAISE_ERROR_TOOL`), and prompt templates all live in `uipath.agent.models.agent` in uipath-python. This package consumes them — it doesn't define them.

Key connection points:
- `tools/tool_factory.py` dispatches on `isinstance(resource, AgentXResourceConfig)` — each resource config type maps to a tool factory
- `guardrails/guardrails_factory.py` converts `AgentGuardrail` models into executable `(BaseGuardrail, GuardrailAction)` pairs

**Adding a new tool type requires the `AgentXResourceConfig` to exist in uipath-python first.** If it doesn't, that's a cross-package change.

## Where Does My Change Go?

| Change | Owner | Read first |
|--------|-------|------------|
| New tool type | `tools/<name>_tool.py` | `tools/process_tool.py` (simple), `tools/context_tool.py` (complex) |
| New guardrail action | `guardrails/actions/<name>_action.py` | `guardrails/actions/log_action.py` (simple), `guardrails/actions/escalate_action.py` (complex) |
| New graph node | `react/<name>_node.py` | `react/llm_node.py`, `react/types.py` |
| Routing changes | `react/router.py` or `react/router_conversational.py` | Both routers + `react/utils.py` |
| State fields | `react/types.py` + `react/reducers.py` | Existing state classes and their reducers |
| Error codes | `exceptions/exceptions.py` | Existing error code enums |
| Tool wrapper | `wrappers/` | `wrappers/job_attachment_wrapper.py` |
| Multimodal | `multimodal/` | `multimodal/invoke.py` |
| Guardrail evaluation | `guardrails/guardrail_nodes.py` | Existing scope-specific node creators |
| Guardrail subgraph wiring | `react/guardrails/guardrails_subgraph.py` | Existing subgraph creators |

After adding a new tool: **also** register it in `tools/tool_factory.py` → `_build_tool_for_resource()` and export from `tools/__init__.py`.

After adding a new guardrail action: **also** register it in `guardrails/guardrails_factory.py` → `build_guardrails_with_actions()` and export from `guardrails/actions/__init__.py`.

---

## Import Rules (Non-Obvious)

These are the constraints you can't discover by reading any single file:

**`react/agent.py` is the composition root.** It imports from everywhere in agent/. Nothing else may import from `react/agent.py` — only the runtime layer above does.

```
react/agent.py  ←  react/*, tools/, guardrails/, exceptions/
tools/*         ←  tools/*, react/types, react/utils, exceptions/, chat.hitl
guardrails/*    ←  guardrails/*, react/types, exceptions/, messages/
wrappers/*      ←  react/*, tools/tool_node
multimodal/*    ←  standalone (uipath._utils._ssl_context only)
exceptions/*    ←  standalone (uipath.runtime.errors, uipath.platform.errors)
messages/*      ←  standalone (langchain_core.messages)
```

**Hard rules:**
- tools/ and guardrails/ must **never** import from `react/agent.py` — import from `react/types.py` or `react/utils.py` instead
- agent/ must **never** import from downstream consumers — the dependency flows one way into this package, not out
- No direct `os.environ` reads — dependencies come through constructor parameters or SDK instances

---

## State Management Rules

The LangGraph state system has sharp edges. These rules prevent silent corruption:

- **Always return state updates as dicts from nodes** — LangGraph passes copies, so mutations on the state object are silently lost. The only way to update state is by returning dicts that the reducers merge.
- New fields on `InnerAgentGraphState` **must** have a reducer via `Annotated[T, reducer_func]` — use `merge_dicts` for dicts, `merge_objects` for nested BaseModel fields.
- To append messages: return `{"messages": [new_msg]}` — the `add_messages` reducer handles it.
- To **replace** messages (rare): use `Overwrite` from `langgraph.types`.
- `tools_storage` (in inner_state) is the shared key-value store for inter-tool communication — use it instead of inventing new state fields.

---

## Anti-Patterns

| Don't | Do Instead |
|-------|------------|
| Import from `react/agent.py` in tools/ or guardrails/ | Import from `react/types.py` or `react/utils.py` |
| Put routing logic in tool nodes | Return results; let the router decide |
| Mutate state object in a node (silently lost) | Return `{"messages": [...]}` dicts; let reducers merge |
| Read env vars inside tool/node functions | Accept config through factory parameters |
| Create tool classes | Use `create_<name>_tool()` factory functions |
| Add `AgentGraphState` fields without reducers | Use `Annotated[T, reducer_func]` |
| Catch and silently swallow exceptions | Re-raise or wrap with `AgentRuntimeError` |
| Use `asyncio.run()` for sync→async bridging | Use `asyncio.run_coroutine_threadsafe()` |
| Add guardrail eval logic outside `guardrail_nodes.py` | Create a scope-specific creator there |
| Put HITL logic in tool functions | Set `REQUIRE_CONVERSATIONAL_CONFIRMATION` metadata on the tool |
| Skip registering new tools in `tool_factory.py` | Always add dispatch entry in `_build_tool_for_resource()` |

---

## Exception Handling

Use the structured error types in `exceptions/` — never raise raw `Exception`, `ValueError`, or `RuntimeError`:

- **Runtime errors** (during execution): `AgentRuntimeError(code=AgentRuntimeErrorCode.X, title=..., detail=..., category=...)`
- **Startup errors** (during init): `AgentStartupError(code=AgentStartupErrorCode.X, title=..., detail=..., category=...)`
- **HTTP errors from platform calls**: catch `EnrichedException`, map via `raise_for_enriched()` in `exceptions/helpers.py`
- **LLM provider errors**: handled by `raise_for_provider_http_error()` in `exceptions/licensing.py`
- Always chain exceptions: `raise AgentRuntimeError(...) from e`

## Testing

Tests live in `tests/agent/` mirroring the source structure. Before writing new tests, read `tests/agent/tools/test_process_tool.py` for the standard fixture and mocking patterns.

Key conventions:
- Use `pytest-httpx` (`HTTPXMock`) for HTTP mocking — never make real network calls
- Use `monkeypatch.setenv()` / `monkeypatch.delenv()` for environment isolation
- Async tests need no decorator (`asyncio_mode = "auto"`)
- All test functions require type annotations
- Mock SDK dependencies via `AsyncMock` / `MagicMock` — tools and nodes receive them through constructor params

---

## Verification

After making changes:

- [ ] Code is in the correct subsystem (see table above)
- [ ] No imports from `react/agent.py` in tools/, guardrails/, or other leaf modules
- [ ] No imports from downstream consumers
- [ ] New state fields have reducers
- [ ] Node functions return dicts (not mutating state)
- [ ] New tools/actions registered in their respective factories
- [ ] New exports added to the subsystem's `__init__.py` and `__all__`
- [ ] `uv run ruff check src/uipath_langchain/agent/` passes
- [ ] `uv run mypy src/uipath_langchain/agent/` passes
- [ ] `uv run pytest tests/agent/` passes