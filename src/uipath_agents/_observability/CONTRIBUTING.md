# Contributing to LLMOps Traces

Internal guide for adding/modifying trace spans in the LLMOps instrumentation system.

## Architecture Overview

```
┌─────────────────────────────────────────┐
│   LlmOpsInstrumentationCallback         │  ← LangChain callback handler
│   (delegates to instrumentors)          │
└─────────────┬───────────────────────────┘
              │
              ├──────────────┬──────────────┬───────────────┐
              ▼              ▼              ▼               ▼
      ┌─────────────┐ ┌─────────────┐ ┌──────────────┐ ┌──────────┐
      │LlmInstrument│ │ToolInstrume │ │ Guardrail    │ │ Span     │
      │   or        │ │    ntor     │ │ Instrumentor │ │Hierarchy │
      └──────┬──────┘ └──────┬──────┘ └──────┬───────┘ └────┬─────┘
             │               │                │               │
             └───────────────┴────────────────┴───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │   LlmOpsSpanFactory          │  ← Span creation
              │   (typed schemas)            │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │   OpenTelemetry SDK          │  ← trace.get_tracer()
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │   LLMOps / AppInsights       │  ← Exporters
              └──────────────────────────────┘
```

**Key directories:**
- `llmops/callback.py` - Main callback handler (delegates to instrumentors)
- `llmops/instrumentors/` - Specialized instrumentors (LLM, Tool, Guardrail)
- `llmops/spans/` - Span factory and typed span schemas
- `llmops/spans/span_attributes/` - Typed Pydantic attribute classes
- `llmops/spans/span_name.py` - Span type enums and display names
- `llmops/span_hierarchy.py` - Manages parent-child span relationships
- `instrumented_runtime.py` - Runtime wrapper that manages agent span lifecycle

**Dual instrumentation:** We use manual instrumentation (not auto-instrumentation). OpenInference spans are filtered out; we emit custom spans tailored to the LLMOps platform. See [Dual Instrumentation](https://uipath.atlassian.net/wiki/spaces/~7120201d2c956b7d1c4065a7ba3947a7b34ebd/pages/90030669947/Dual+Instrumentation+-+Manual+OpenInference).

---

## Adding Attributes to Existing Spans

### Example: Add `execution_type` and `agent_version` to AgentRun span

**PR Reference:** [#145 - span attributes parity](https://github.com/UiPath/uipath-agents-python/pull/145)

**Step 1: Define attribute in span_attributes**

```python
# llmops/spans/span_attributes/core.py
class AgentRunSpanAttributes(BaseSpanAttributes):
    # ... existing fields ...

    # New fields - use camelCase alias for JSON
    execution_type: Optional[int] = Field(None, alias="executionType")
    agent_version: Optional[str] = Field(None, alias="agentVersion")
```

**Step 2: Add helper to read from environment (if external)**

```python
# llmops/instrumentors/attribute_helpers.py
ENV_UIPATH_IS_DEBUG = "UIPATH_IS_DEBUG"
ENV_UIPATH_PROCESS_VERSION = "UIPATH_PROCESS_VERSION"

def get_execution_type() -> int:
    """Debug=0 if UIPATH_IS_DEBUG=true, else Runtime=1."""
    if os.getenv(ENV_UIPATH_IS_DEBUG, "").lower() == "true":
        return 0  # Debug
    return 1  # Runtime

def get_agent_version() -> Optional[str]:
    return os.getenv(ENV_UIPATH_PROCESS_VERSION) or None
```

**Step 3: Pass to span creation in span schema**

```python
# llmops/spans/spans_schema/agent.py - AgentSpanSchema.start_agent_run()
attrs = AgentRunSpanAttributes(
    agent_name=agent_name,
    # ... existing ...
    execution_type=get_execution_type(),
    agent_version=get_agent_version(),
)
```

**Step 4: (Optional) Propagate from instrumented runtime**

```python
# instrumented_runtime.py - if value comes from agent definition
# Properties can be passed when starting the span via the span factory
```

---

## Adding a New Span Type

### Example: Add `agentTool` span for agent-as-tool invocations

**PR Reference:** [#162 - agentTool spans](https://github.com/UiPath/uipath-agents-python/pull/162)

**Step 1: Add span type to SpanKeys**

```python
# llmops/spans/span_keys.py
class SpanType:
    # ... existing ...
    AGENT_TOOL = "agentTool"  # New type
```

**Step 2: Create attribute class in span_attributes**

```python
# llmops/spans/span_attributes/tools.py
class AgentToolSpanAttributes(ToolCallSpanAttributes):
    """Attributes for agent-as-tool spans."""

    job_id: Optional[str] = Field(None, alias="jobId")
    job_details_uri: Optional[str] = Field(None, alias="jobDetailsUri")

    @property
    def type(self) -> str:
        return SpanType.AGENT_TOOL
```

**Step 3: Add span schema class**

```python
# llmops/spans/spans_schema/tool.py
class ToolSpanSchema:
    def start_agent_tool(
        self,
        agent_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        parent_span: Optional[Span] = None,
    ) -> Span:
        """Start an agent tool span (agent-as-tool invocation)."""
        parent = parent_span or trace.get_current_span()
        context = trace.set_span_in_context(parent) if parent else None

        span = self._tracer.start_span(
            SpanName.tool_call(agent_name),
            kind=SpanKind.INTERNAL,
            context=context,
        )

        attrs = AgentToolSpanAttributes(
            tool_name=agent_name,
            arguments=arguments,
        )
        self._apply_attributes(span, attrs)
        self._upsert_started_fn(span)
        return span
```

**Step 4: Add method to LlmOpsSpanFactory**

```python
# llmops/spans/span_factory.py
class LlmOpsSpanFactory:
    def start_agent_tool(
        self,
        agent_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        parent_span: Optional[Span] = None,
    ) -> Span:
        """Start an agent tool span."""
        return self._tool_schema.start_agent_tool(
            agent_name=agent_name,
            arguments=arguments,
            parent_span=parent_span,
        )
```

**Step 5: Wire up in instrumentor (for LangChain events)**

```python
# llmops/instrumentors/tool_instrumentor.py - on_tool_start()
def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
    tool_type = kwargs.get("metadata", {}).get("tool_type")

    if tool_type == "agent":
        child_span = self._state.span_factory.start_agent_tool(
            agent_name=tool_name,
            arguments=tool_input,
            parent_span=tool_span,
        )
    # ... handle other tool types
```

---

## Span Lifecycle Patterns

### Context Manager (auto-end)

```python
with span_factory.start_agent_run(agent_name="MyAgent") as span:
    # span auto-ends on exit
    pass
```

### Manual Start/End (via span schemas)

```python
# Start span (via schema)
span = agent_schema.start_llm_call(parent_span=parent)
try:
    # ... work ...
    llm_schema.end_llm_call(span, result=llm_result)
except Exception as e:
    llm_schema.end_llm_call_error(span, e)
```

### Suspend/Resume (interruptible tools)

```python
# On suspend
span_factory.upsert_span_suspended(span)  # Status=UNSET, no end_time

# On resume
span_factory.upsert_span_complete(span)   # Status=OK/ERROR with end_time
```

### Instrumentor Pattern (callback-driven)

```python
# Instrumentors handle LangChain events and delegate to span schemas
class ToolSpanInstrumentor:
    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        # Extract metadata and create span via factory
        span = self._state.span_factory.start_tool_call(...)
        self._state.tool_spans[run_id] = span

    def on_tool_end(self, output, *, run_id, **kwargs):
        # Retrieve and complete span
        span = self._state.tool_spans.pop(run_id)
        self._state.span_factory.end_tool_call(span, output)
```

---

## Attribute Serialization

OTEL only accepts primitives. Complex objects are JSON serialized:

```python
# llmops/spans/spans_schema/base.py - _apply_attributes()
if isinstance(value, (dict, list)):
    span.set_attribute(key, json.dumps(value))
else:
    span.set_attribute(key, value)
```

Use Pydantic aliases for camelCase JSON output:

```python
# llmops/spans/span_attributes/base.py
class MySpanAttributes(BaseSpanAttributes):
    my_field: str = Field(..., alias="myField")  # → "myField" in JSON
```

---

## Testing

```python
# tests/unit/observability/test_span_factory.py
def test_agent_tool_span_attributes(span_factory, span_exporter):
    span = span_factory.start_agent_tool("SubAgent", {"x": 1})
    span_factory._tool_schema.end_tool_call(span, output="result")

    spans = span_exporter.get_finished_spans()
    attrs = dict(spans[0].attributes)

    assert attrs["type"] == "agentTool"
    assert attrs["toolName"] == "SubAgent"
    assert json.loads(attrs["arguments"]) == {"x": 1}
```

Test instrumentors separately:

```python
# tests/unit/observability/test_instrumentors.py
def test_tool_instrumentor_creates_span(tool_instrumentor, state):
    tool_instrumentor.on_tool_start(
        serialized={}, input_str="test", run_id=uuid4()
    )
    assert len(state.tool_spans) == 1
```

---

## Checklist for New Spans/Attributes

- [ ] Define complete attributes with proper field names and aliases
- [ ] Use camelCase aliases for JSON serialization in attribute classes
- [ ] Add span type to `llmops/spans/span_keys.py` if new type
- [ ] Create or update span schema in `llmops/spans/spans_schema/`
- [ ] Add method to `LlmOpsSpanFactory` if needed
- [ ] Wire up in appropriate instrumentor (`llmops/instrumentors/`)
- [ ] Add unit tests for span factory and instrumentor
- [ ] Test with LLMOps trace viewer
- [ ] Consider suspend/resume if span can be interrupted
- [ ] Update span hierarchy management if parent-child relationships change

## Reference PRs

| Change | PR |
|--------|-----|
| Traces refactoring (callback → instrumentors, tracer → span factory) | feat/traces branch |
| Propagate top-level span properties | [#189](https://github.com/UiPath/uipath-agents-python/pull/189) |
| Parent LLM spans to tool span when called from within tools | [#181](https://github.com/UiPath/uipath-agents-python/pull/181) |
| Add model attribute to LLM call span | [#193](https://github.com/UiPath/uipath-agents-python/pull/193) |
| Span attributes parity (execution_type, agent_version, etc.) | [#145](https://github.com/UiPath/uipath-agents-python/pull/145) |
| AgentTool spans + tool call fixes | [#162](https://github.com/UiPath/uipath-agents-python/pull/162) |

## Key Refactoring Changes

**Renamed Components:**
- `UiPathTracingCallback` → `LlmOpsInstrumentationCallback`
- `UiPathTracer` → `LlmOpsSpanFactory`
- `TelemetryRuntimeWrapper` → `InstrumentedRuntime`

**File Reorganization:**
- `callback.py` → `llmops/callback.py` (refactored to delegate to instrumentors)
- `tracer.py` → `llmops/spans/span_factory.py`
- `schema.py` → `llmops/spans/span_name.py` + `llmops/spans/span_keys.py`
- `span_attributes.py` → `llmops/spans/span_attributes/*.py` (split by domain)
- `runtime_wrapper.py` → `instrumented_runtime.py`

**New Architecture:**
- Introduced **Instrumentors** pattern - specialized handlers for LLM, Tool, and Guardrail events
- Introduced **Span Schemas** - typed span creation classes (AgentSpanSchema, LlmSpanSchema, ToolSpanSchema, GuardrailSpanSchema)
- Introduced **SpanHierarchyManager** - manages parent-child span relationships across run IDs
- Introduced **InstrumentationState** - shared state between callback and instrumentors
