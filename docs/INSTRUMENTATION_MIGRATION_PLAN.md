# Instrumentation Migration Plan: uipath-langchain → uipath-agents-python

## Executive Summary

**Goal**: Remove manual instrumentation from `uipath-langchain-python` and create an OpenInference-style custom instrumentor in `uipath-agents-python` (low-code repo).

**Why**: Keep uipath-langchain library "clean and native" so same tools/loop work with:
- Low-code agents (custom UiPath instrumentation)
- Coded agents (native LangGraph/LangSmith tracing)

---

## Current State Analysis

### Files with Manual Instrumentation in uipath-langchain-python

| File | What it does | Lines of instrumentation |
|------|--------------|-------------------------|
| `_tracing/tracer.py` | UiPathTracer class with span creation methods | ~240 lines |
| `_tracing/schema.py` | SpanType enum, SpanName class | ~50 lines |
| `_tracing/_instrument_traceable.py` | LangSmith @traceable adapter | ~100 lines |
| `runtime/runtime.py` | Agent run span wrapping execute/stream | ~50 lines added |
| `agent/react/llm_node.py` | LLM call + Model run spans | ~30 lines added |
| `agent/tools/tool_node.py` | Tool call spans | ~15 lines added |

### Key Integration Points (polluted code)

```python
# runtime/runtime.py - lines 73-75, 161-166
if is_custom_instrumentation_enabled():
    return await self._execute_traced(input, options)

# llm_node.py - lines 44-45
if is_custom_instrumentation_enabled():
    return await _llm_node_instrumented(llm, messages, model)

# tool_node.py - lines 37-40
if is_custom_instrumentation_enabled():
    with tracer.start_tool_call(tool.name):
        return await base_node.ainvoke(state)
```

---

## Target Architecture

### OpenInference Pattern (from Arize)

```python
class LangChainInstrumentor(BaseInstrumentor):
    def _instrument(self):
        # Wrap BaseCallbackManager.__init__ to inject tracer
        wrapt.wrap_function_wrapper(
            "langchain_core.callbacks",
            "BaseCallbackManager.__init__",
            self._wrap_callback_init
        )

    def _wrap_callback_init(self, wrapped, instance, args, kwargs):
        result = wrapped(*args, **kwargs)
        instance.add_handler(self._tracer, True)  # Inject tracer as callback
        return result
```

### Proposed UiPath Instrumentor (in uipath-agents-python)

```python
class UiPathLangGraphInstrumentor(BaseInstrumentor):
    """Custom instrumentor for UiPath span schema without modifying langchain code."""

    def _instrument(self):
        # Option 1: Callback injection (like OpenInference)
        # Option 2: LangGraph-specific hooks
        # Option 3: Monkey-patch runtime only (minimal)
```

---

## Migration Options

### Option A: Pure Callback-Based (OpenInference-style)

**Approach**: Inject `UiPathTracerCallback` into LangGraph's callback system

**Pros**:
- No modification to uipath-langchain
- Standard LangChain pattern
- Works with any LangGraph agent

**Cons**:
- Spans emit on END not START (Scott's concern)
- Guardrail sub-graphs generate many spans
- May change with LangGraph SDK updates

**Gaps identified by Scott**:
> "spans are only emitted when nodes complete - but if we start a long-running tool call, we want the span to show up when the tool call starts"

### Option B: Hybrid - Callback + Runtime Wrapper

**Approach**:
- Callback for LLM/tool spans (most things)
- Thin wrapper in low-code for agent run span + start-time emission

**Implementation in uipath-agents-python**:
```python
# In low-code repo, wrap the runtime
class InstrumentedRuntime:
    def __init__(self, base_runtime: UiPathLangGraphRuntime):
        self._runtime = base_runtime
        self._tracer = UiPathTracer()

    async def execute(self, ...):
        with self._tracer.start_agent_run(...):
            # Add callback to runtime's graph
            self._runtime.graph.callbacks.append(UiPathTracerCallback(self._tracer))
            return await self._runtime.execute(...)
```

**Pros**:
- Spans start immediately (not on completion)
- Agent run span controlled by low-code
- LLM/tool spans via callback (cleaner)

**Cons**:
- Still some runtime awareness needed
- Two patterns (wrapper + callback)

### Option C: Full Wrapper in Low-Code (Recommended)

**Approach**: Move ALL instrumentation to uipath-agents-python via wrapper/middleware

**In uipath-agents-python**:
```python
class UiPathInstrumentedAgent:
    """Wraps any LangGraph agent with UiPath instrumentation."""

    def __init__(self, graph: CompiledStateGraph, tracer: UiPathTracer):
        self._graph = graph
        self._tracer = tracer

    async def invoke(self, input, config):
        # Inject tracing callback for this invocation
        tracing_callback = UiPathTracingCallback(self._tracer)
        config = merge_callbacks(config, [tracing_callback])

        with self._tracer.start_agent_run(agent_name=...):
            return await self._graph.ainvoke(input, config)
```

**UiPathTracingCallback handles**:
- on_llm_start → start_llm_call() (immediate span)
- on_llm_end → complete span
- on_tool_start → start_tool_call() (immediate span)
- on_tool_end → complete span

**Pros**:
- uipath-langchain stays 100% clean
- Full control over span timing (START not just END)
- Can customize guardrail span aggregation
- Works for low-code only; coded agents use native tracing

**Cons**:
- Need to understand LangGraph callback system deeply
- Some edge cases with async/streaming

---

## Recommended Approach: Option C

### Phase 1: Port Infrastructure to uipath-agents-python

**Move these files**:
```
uipath-langchain-python/           uipath-agents-python/
_tracing/tracer.py          →      tracing/tracer.py
_tracing/schema.py          →      tracing/schema.py
```

**Create new callback-based instrumentor**:
```
uipath-agents-python/
  tracing/
    __init__.py
    tracer.py              # UiPathTracer (moved)
    schema.py              # SpanType, SpanName (moved)
    callback.py            # NEW: UiPathTracingCallback
    instrumentor.py        # NEW: UiPathLangGraphInstrumentor
```

### Phase 2: Implement UiPathTracingCallback

```python
from langchain_core.callbacks import BaseCallbackHandler
from opentelemetry.trace import Span
from typing import Any, Dict, List, Optional
from uuid import UUID

class UiPathTracingCallback(BaseCallbackHandler):
    """LangGraph callback that creates UiPath-schema spans."""

    def __init__(self, tracer: UiPathTracer):
        self._tracer = tracer
        self._run_spans: Dict[UUID, Span] = {}  # Track active spans by run_id

    # --- LLM Events ---
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str],
        *, run_id: UUID, **kwargs
    ) -> None:
        """Start LLM span immediately (not on completion)."""
        model_name = serialized.get("kwargs", {}).get("model_name", "unknown")

        # Create outer LLM call span
        llm_span = self._tracer._tracer.start_span("LLM call")
        llm_span.set_attribute("type", "completion")
        self._run_spans[run_id] = llm_span

        # Create inner model run span
        model_span = self._tracer._tracer.start_span("Model run")
        model_span.set_attribute("type", "llmCall")
        model_span.set_attribute("model", model_name)
        self._run_spans[f"{run_id}_model"] = model_span

    def on_llm_end(self, response, *, run_id: UUID, **kwargs) -> None:
        """Complete LLM spans."""
        # Close model span
        if f"{run_id}_model" in self._run_spans:
            self._run_spans[f"{run_id}_model"].end()
        # Close LLM call span
        if run_id in self._run_spans:
            self._run_spans[run_id].end()

    # --- Tool Events ---
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str,
        *, run_id: UUID, **kwargs
    ) -> None:
        """Start tool span immediately."""
        tool_name = serialized.get("name", "unknown")
        span = self._tracer._tracer.start_span(f"Tool call - {tool_name}")
        span.set_attribute("type", "toolCall")
        span.set_attribute("toolName", tool_name)
        self._run_spans[run_id] = span

    def on_tool_end(self, output: str, *, run_id: UUID, **kwargs) -> None:
        """Complete tool span."""
        if run_id in self._run_spans:
            self._run_spans[run_id].end()
```

### Phase 3: Clean Up uipath-langchain-python

**Remove**:
```
src/uipath_langchain/_tracing/           # Entire directory
```

**Modify** (remove instrumentation branches):

`runtime/runtime.py`:
```python
# REMOVE: is_custom_instrumentation_enabled() checks
# REMOVE: _execute_traced, _stream_traced methods
# KEEP: pure _execute_core, _stream_core logic only
```

`agent/react/llm_node.py`:
```python
# REMOVE: is_custom_instrumentation_enabled() check
# REMOVE: _llm_node_instrumented function
# KEEP: simple llm.ainvoke() call
```

`agent/tools/tool_node.py`:
```python
# REMOVE: tracer imports
# REMOVE: is_custom_instrumentation_enabled() check
# KEEP: simple base_node.ainvoke() call
```

### Phase 4: Integration in uipath-agents-python

**Usage in low-code agent**:
```python
from uipath_agents.tracing import UiPathTracer, UiPathTracingCallback

# When creating agent runtime
tracer = UiPathTracer()
callback = UiPathTracingCallback(tracer)

# Inject into graph execution
async def run_agent(graph, input):
    with tracer.start_agent_run(agent_name="MyAgent"):
        config = {"callbacks": [callback]}
        return await graph.ainvoke(input, config)
```

---

## Handling Scott's Concerns

### 1. "Spans emit when tool calls START not END"

**Solution**: `on_tool_start` creates span immediately:
```python
def on_tool_start(self, ...):
    span = tracer.start_span(...)  # Span visible NOW
    self._run_spans[run_id] = span
```

### 2. "Guardrails = sub-graphs but want 1 span"

**Solution**: Filter/aggregate in callback:
```python
def on_chain_start(self, serialized, ...):
    chain_name = serialized.get("name", "")
    if "guardrail" in chain_name.lower():
        # Create single guardrail span, ignore sub-spans
        if not self._active_guardrail_span:
            self._active_guardrail_span = tracer.start_span("Guardrail")
```

### 3. "Don't want spans to change with SDK updates"

**Solution**: Own the span schema completely in callback:
```python
# Our callback defines exact span names/attributes
# SDK changes don't affect our output as long as callbacks fire
span.set_attribute("type", SpanType.TOOL_CALL.value)  # Our schema
span.set_attribute("toolName", tool_name)              # Our attribute
```

### 4. "Support OpenInference standards for evaluators"

**Solution**: Emit both UiPath and OpenInference attributes:
```python
def on_llm_end(self, response, ...):
    span = self._run_spans[run_id]
    # UiPath attributes
    span.set_attribute("type", "llmCall")
    # OpenInference attributes (for evaluators)
    span.set_attribute("llm.model_name", model_name)
    span.set_attribute("llm.token_count.completion", token_count)
```

---

## Testing Strategy

### In uipath-agents-python

1. **Unit tests**: Mock graph, verify callback methods create correct spans
2. **Integration tests**: Run actual agent, capture spans with InMemoryExporter
3. **Schema validation**: JSON compare against expected trace structure

### In uipath-langchain-python

1. **Remove**: All tracing tests (`tests/tracing/`)
2. **Verify**: Existing tests pass without instrumentation code
3. **Add**: Tests that confirm NO tracing happens by default

---

## Migration Checklist

### uipath-agents-python (add)

- [ ] Create `tracing/` module
- [ ] Port `UiPathTracer` class
- [ ] Port `SpanType`, `SpanName` enums
- [ ] Implement `UiPathTracingCallback`
- [ ] Create integration wrapper
- [ ] Add unit tests
- [ ] Add integration tests
- [ ] Update low-code agent to use new tracing

### uipath-langchain-python (remove)

- [ ] Delete `src/uipath_langchain/_tracing/` directory
- [ ] Remove tracing imports from `runtime/runtime.py`
- [ ] Remove `_execute_traced`, `_stream_traced` methods
- [ ] Remove tracing from `llm_node.py`
- [ ] Remove tracing from `tool_node.py`
- [ ] Remove `is_custom_instrumentation_enabled()` references
- [ ] Delete `tests/tracing/` directory
- [ ] Update `runtime/factory.py` if needed
- [ ] Remove env var `UIPATH_CUSTOM_INSTRUMENTATION` documentation

---

## Timeline & Dependencies

```
Week 1: Port tracer + schema to uipath-agents-python
        ↓
Week 2: Implement UiPathTracingCallback + tests
        ↓
Week 3: Integrate into low-code agent, E2E testing
        ↓
Week 4: Remove instrumentation from uipath-langchain-python
        ↓
Week 5: Final testing + release coordination
```

**Dependency**: Both repos need coordinated release

---

## Open Questions

1. **Where exactly in uipath-agents-python should tracing live?**
   - `uipath_agents/tracing/`?
   - `uipath_agents/langchain/tracing/`?

2. **How to handle coded agents?**
   - They should NOT use this callback (use native LangSmith)
   - Need flag or separate entry point

3. **Streaming spans?**
   - Current impl creates span on stream start
   - Need to verify callback timing with astream()

4. **Cross-repo testing?**
   - How to test integration before both releases?
   - Maybe: branch dependency in pyproject.toml for testing

---

## Appendix: Key Files Reference

### OpenInference Instrumentor (reference)
https://github.com/Arize-ai/openinference/blob/main/python/instrumentation/openinference-instrumentation-langchain/src/openinference/instrumentation/langchain/__init__.py

### LangChain Callback Protocol
https://python.langchain.com/docs/modules/callbacks/

### Current UiPath Tracing (to be moved)
- `src/uipath_langchain/_tracing/tracer.py`
- `src/uipath_langchain/_tracing/schema.py`
