# Escalation Tools Module Guide

> **CLAUDE: UPDATE THIS DOCUMENT**
>
> When you modify files in this module, you MUST update this document to reflect:
> - New or renamed shared primitives (update `common.py` section)
> - New escalation variants (add a column to the variant table)
> - Changes to the resource discriminator (update Escalation Type Discriminator section)
> - New or removed escalation-memory hooks (update Escalation Memory section)
>
> Keep the variant table and import map in sync with `__init__.py`.

## Overview

This module owns LangGraph tools for the three Action Center escalation
variants the Agent Builder runtime supports.  All three share one
resource concept (`AgentResourceType.ESCALATION`) and one channel model
(`AgentEscalationChannel`), but materialise the HITL task through
different platform endpoints.

The variants are discriminated by `escalation_type` on the resource
config:

| `escalation_type` | Resource config                              | Module          | Endpoint                                                                |
| ----------------- | -------------------------------------------- | --------------- | ----------------------------------------------------------------------- |
| `0`               | `AgentEscalationResourceConfig`              | `app_task.py`   | `tasks.create_async` (app-bound task; optional escalation memory)       |
| `1`               | `AgentIxpVsEscalationResourceConfig`         | `ixp_vs.py`     | `documents.create_validate_extraction_action_async` (DU validation)     |
| `2`               | `AgentQuickFormEscalationResourceConfig`     | `quick_form.py` | `tasks.create_quickform_async` (FormLib schema task)                    |

## Module Structure

```
src/uipath_langchain/agent/tools/escalation/
├── __init__.py    # Public exports: 3 factories + EscalationAction + recipient/asset resolvers
├── common.py      # Shared primitives (the seam)
├── app_task.py    # escalationType=0 — Action Center app task + escalation memory
├── quick_form.py  # escalationType=2 — FormLib schema task
├── ixp_vs.py      # escalationType=1 — DU validation action
└── memory.py      # Escalation memory cache lookup + ingest (only used by app_task today)
```

### Public Exports (`__init__.py`)

```python
from .app_task import create_escalation_tool
from .common import EscalationAction, resolve_asset, resolve_recipient_value
from .ixp_vs import create_ixp_escalation_tool
from .quick_form import create_quick_form_escalation_tool
```

`resolve_recipient_value` is re-exported because
`guardrails/actions/escalate_action.py` reaches into this package
(lazy import) to resolve recipients for escalation actions.

## Architecture

### common.py — The Seam

`common.py` is what makes `app_task.py` and `quick_form.py` thin.  It owns:

| Primitive                          | Purpose                                                                                                  |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `EscalationAction`                 | Outcome enum: `CONTINUE` / `END`.                                                                        |
| `resolve_recipient_value`          | Dispatches over `AgentEscalationRecipient` variants → `TaskRecipient`.                                   |
| `resolve_asset`                    | Asset-name → asset-value lookup via the SDK.                                                             |
| `_parse_task_data`                 | Strips/keeps fields based on input/output JSON schemas.                                                  |
| `_resolve_escalation_action`       | Looks up the channel's `outcome_mapping`; defaults to `CONTINUE`.                                        |
| `make_escalation_tool_output(M)`   | Builds the `EscalationToolOutput` pydantic model (`action`, `data: M`, `is_deleted`) for the mockable.   |
| `EscalationInvocationCtx`          | Dataclass: `agent_input`, `recipient`, `folder_path`, `task_title`, `serialized_data`.                   |
| `build_invocation_ctx`             | Assembles the preamble every variant runs before opening the durable interrupt.                          |
| `finalize_escalation_result`       | Post-processes the resolved task: handles `is_deleted`, parses outputs, resolves the action.             |
| `make_escalation_wrapper(channel)` | Returns the LangGraph tool wrapper: resolves task title, captures call metadata, maps `END` → exception. |

### Per-variant Factories

Each factory follows the same skeleton:

```python
def create_*_escalation_tool(resource, ...):
    channel = resource.channels[0]
    input_model = create_model(channel.input_schema)
    output_model = create_model(channel.output_schema)
    EscalationToolOutput = make_escalation_tool_output(output_model)

    async def tool_fn(**kwargs):
        ctx = await build_invocation_ctx(tool, channel, kwargs, input_model)

        @mockable(...)
        async def escalate(**_):
            @durable_interrupt
            async def create_task():
                # === The only meaningful difference per variant: ===
                # which platform call to make and what to pass it.
                ...
                return WaitEscalation(...)  # or WaitDocumentExtractionValidation(...)
            return await create_task()

        result = await escalate(**kwargs)
        return finalize_escalation_result(result, input_model=..., output_model=..., outcome_mapping=...)

    tool = StructuredToolWithArgumentProperties(...)
    tool.set_tool_wrappers(awrapper=make_escalation_wrapper(channel))
    return tool
```

`app_task.py` is the same skeleton plus the escalation-memory cache check
before the mockable, and the escalation-memory ingest after the result is
finalised (skipped when `result.is_deleted` to mirror the early-return
shape of the pre-refactor code).

`ixp_vs.py` does not use `build_invocation_ctx` /
`finalize_escalation_result` / `make_escalation_wrapper`: it suspends on
`WaitDocumentExtractionValidation` (not `WaitEscalation`), reads its
input from `tools_storage` rather than the tool's args, and detects
rejection through `documentRejectionDetails` instead of an outcome
mapping.  Its wrapper still uses `resolve_task_title` from
`tools/utils.py`.

## Escalation Memory

Memory lives in `memory.py` (moved wholesale from
`agent/tools/escalation_memory.py`).  It is owned by `app_task.py`
today; quick-form and ixp-vs do not call it.

**Lifecycle inside `create_escalation_tool`:**

1. Before the mockable: `_check_escalation_memory_cache(...)` returns a
   prior outcome if the input matches.  If hit, the tool short-circuits
   and returns the cached `{action, output, outcome}`.
2. After the mockable resolves (and only if `not result.is_deleted`):
   `_ingest_escalation_memory(...)` persists the outcome along with
   span/trace IDs so future agents can recall it.

The span/trace IDs come from `tool.metadata["_span_context"]` (set by
the LLMOps tool instrumentor in `uipath-agents`) and fall back to
`get_current_span_and_trace_ids()` / `UIPATH_TRACE_ID`.

To add memory to a new variant (e.g. quick-form), import the same
helpers from `.memory`, call them at the same two points, and skip
ingest when `result.is_deleted`.

## Cross-package Dependencies

```
agent/tools/escalation/        (this module)
├── common.py        → langchain_core, uipath.{agent,platform,runtime},
│                      uipath_langchain._utils.get_execution_folder_path,
│                      ...exceptions, ...react.types, ..tool_node, ..utils
├── memory.py        → uipath.platform.memory, uipath_langchain._utils,
│                      OTel
├── app_task.py      → .common, .memory, uipath.platform (UiPath, Task,
│                      WaitEscalation), uipath_langchain._utils
├── quick_form.py    → .common, uipath.platform (same)
└── ixp_vs.py        → uipath.platform.documents, ...exceptions,
                       ...react.types, ..structured_tool_with_output_type,
                       ..tool_node, ..utils (no .common — different shape)

Consumers
─────────
agent/tools/__init__.py        — re-exports the three factories
agent/tools/tool_factory.py    — dispatches on resource type → factory
agent/guardrails/actions/      — lazy-imports resolve_recipient_value
  escalate_action.py             from .escalation
```

## Tests

| Test file                                                  | Surface under test                          |
| ---------------------------------------------------------- | ------------------------------------------- |
| `tests/agent/tools/test_escalation_tool.py`                | App-task flow, common primitives, memory   |
| `tests/agent/tools/test_escalation_memory.py`              | Memory cache + ingest internals            |
| `tests/agent/tools/test_ixp_escalation_tool.py`            | IXP-VS extraction validation flow          |
| `tests/cli/test_agent_with_guardrails.py`                  | End-to-end escalation guardrails           |
| `tests/agent/guardrails/actions/test_escalate_action.py`   | Recipient resolution from guardrail action |

### Patch path conventions

Tests patch SDK calls at the module that performs the lookup:

- `escalation.common.UiPath` — when testing `resolve_asset`.
- `escalation.common.resolve_asset` — when testing `resolve_recipient_value`.
- `escalation.app_task.UiPath` — when testing the app-task creation flow.
- `escalation.app_task._check_escalation_memory_cache` / `._ingest_escalation_memory`
  / `._resolve_user_id` — when testing memory hooks (these are imported into
  `app_task.py` from `.memory`).
- `escalation.memory.UiPath` / `escalation.memory.UiPathConfig` — when testing
  memory internals (cache lookup, ingest request building).
- `escalation.ixp_vs.UiPath` — when testing the IXP validation flow.

## Guidelines for Changes

### Adding a new escalation variant

1. Add a `Literal[<N>]` discriminator on a new
   `Agent*EscalationResourceConfig` in `uipath-python`.
2. Add a new module under this package (e.g. `escalation/foo.py`)
   following the skeleton above.
3. If the variant maps to the standard `WaitEscalation` →
   `{action, output, outcome}` shape, reuse `build_invocation_ctx`,
   `finalize_escalation_result`, and `make_escalation_wrapper` from
   `common.py`.  If it diverges (like `ixp_vs.py`), write the
   minimum bespoke wrapper and keep `resolve_task_title` from
   `tools/utils.py`.
4. Re-export the factory from `__init__.py` and add an
   `isinstance(...)` branch in `tools/tool_factory.py`.
5. Add a row to the variant table at the top of this file.

### Adding a new shared primitive

1. Put it in `common.py`.
2. Re-export from `__init__.py` only if it has consumers outside the
   subpackage (today: `EscalationAction`, `resolve_asset`,
   `resolve_recipient_value`).
3. Update the "common.py — The Seam" table above.

### Touching escalation memory

1. Edit `memory.py` directly.
2. If you change a public name imported by `app_task.py`, update both
   the import and the `Patch path conventions` table above.
3. Memory writes must remain idempotent w.r.t. `result.is_deleted` —
   never ingest for a deleted task.
