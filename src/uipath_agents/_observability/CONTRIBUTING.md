# Contributing to URT Traces

Internal guide for adding/modifying trace spans in the Unified Runtime Traces system.

## Architecture Overview

```
┌─────────────────────────────────────────┐
│   UiPathTracingCallback                 │  ← LangChain callback handler
│   (intercepts LLM/tool events)          │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│   UiPathTracer                          │  ← Manual span creation
│   (typed attributes, span lifecycle)   │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│   OpenTelemetry SDK                     │  ← trace.get_tracer()
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│   LLMOps / AppInsights                  │  ← Exporters
└─────────────────────────────────────────┘
```

**Key files:**
- `tracer.py` - Core tracer with span creation methods
- `callback.py` - LangChain integration
- `span_attributes.py` - Typed Pydantic attribute classes
- `schema.py` - Span type enums and names

**Dual instrumentation:** We use manual instrumentation (not auto-instrumentation). OpenInference spans are filtered out; we emit our own matching C# Temporal schema. See [Dual Instrumentation](https://uipath.atlassian.net/wiki/spaces/~7120201d2c956b7d1c4065a7ba3947a7b34ebd/pages/90030669947/Dual+Instrumentation+-+Manual+OpenInference).

---

## Adding Attributes to Existing Spans

### Example: Add `execution_type` and `agent_version` to AgentRun span

**PR Reference:** [#145 - span attributes parity](https://github.com/UiPath/uipath-agents-python/pull/145)

**Step 1: Define attribute in span_attributes.py**

```python
# span_attributes.py
class AgentRunSpanAttributes(BaseSpanAttributes):
    # ... existing fields ...

    # New fields - use camelCase alias for JSON
    execution_type: Optional[int] = Field(None, alias="executionType")
    agent_version: Optional[str] = Field(None, alias="agentVersion")
```

**Step 2: Add helper to read from environment (if external)**

```python
# span_attributes.py
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

**Step 3: Pass to span creation in tracer.py**

```python
# tracer.py - start_agent_run()
attrs = AgentRunSpanAttributes(
    agent_name=agent_name,
    # ... existing ...
    execution_type=get_execution_type(),
    agent_version=get_agent_version(),
)
```

**Step 4: (Optional) Propagate from runtime wrapper**

```python
# runtime_wrapper.py - if value comes from agent definition
attrs.is_conversational = agent_definition.is_conversational
```

---

## Adding a New Span Type

### Example: Add `agentTool` span for agent-as-tool invocations

**PR Reference:** [#162 - agentTool spans](https://github.com/UiPath/uipath-agents-python/pull/162)

**Step 1: Add span type to schema.py**

```python
# schema.py
class SpanType(str, Enum):
    # ... existing ...
    AGENT_TOOL = "agentTool"  # New type
```

**Step 2: Create attribute class in span_attributes.py**

```python
# span_attributes.py
class AgentToolSpanAttributes(ToolCallSpanAttributes):
    """Attributes for agent-as-tool spans."""

    job_id: Optional[str] = Field(None, alias="jobId")
    job_details_uri: Optional[str] = Field(None, alias="jobDetailsUri")

    @property
    def type(self) -> str:
        return SpanType.AGENT_TOOL
```

**Step 3: Add creation method to tracer.py**

```python
# tracer.py
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
    self.upsert_span_started(span)
    return span
```

**Step 4: Wire up in callback.py (for LangChain events)**

```python
# callback.py - on_tool_start()
def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
    tool_type = kwargs.get("metadata", {}).get("tool_type")

    if tool_type == "agent":
        child_span = self._tracer.start_agent_tool(
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
with tracer.start_agent_run(agent_name="MyAgent") as span:
    # span auto-ends on exit
    pass
```

### Manual Start/End (for callbacks)

```python
span = tracer.start_llm_call(parent_span=parent)
try:
    # ... work ...
    tracer.end_span_ok(span)
except Exception as e:
    tracer.end_span_error(span, e)
```

### Suspend/Resume (interruptible tools)

```python
# On suspend
tracer.upsert_span_suspended(span)  # Status=UNSET, no end_time

# On resume
tracer.upsert_span_complete(span)   # Status=OK/ERROR with end_time
```

---

## Attribute Serialization

OTEL only accepts primitives. Complex objects are JSON serialized:

```python
# tracer.py - _apply_attributes()
if isinstance(value, (dict, list)):
    span.set_attribute(key, json.dumps(value))
else:
    span.set_attribute(key, value)
```

Use Pydantic aliases for camelCase JSON output:

```python
class MySpanAttributes(BaseSpanAttributes):
    my_field: str = Field(..., alias="myField")  # → "myField" in JSON
```

---

## Testing

```python
# tests/unit/observability/test_tracer.py
def test_agent_tool_span_attributes(self, tracer, span_exporter):
    span = tracer.start_agent_tool("SubAgent", {"x": 1})
    tracer.end_span_ok(span)

    spans = span_exporter.get_finished_spans()
    attrs = dict(spans[0].attributes)

    assert attrs["type"] == "agentTool"
    assert attrs["toolName"] == "SubAgent"
    assert json.loads(attrs["arguments"]) == {"x": 1}
```

---

## Checklist for New Spans/Attributes

- [ ] Match C# Temporal schema (check `Execution.Shared/Traces/`)
- [ ] Use camelCase aliases for JSON serialization
- [ ] Add unit tests for attribute serialization
- [ ] Update `SpanAttributes` type union if new span type
- [ ] Test with LLMOps trace viewer
- [ ] Consider suspend/resume if span can be interrupted

## Reference PRs

| Change | PR |
|--------|-----|
| Span attributes parity (execution_type, agent_version, etc.) | [#145](https://github.com/UiPath/uipath-agents-python/pull/145) |
| AgentTool spans + tool call fixes | [#162](https://github.com/UiPath/uipath-agents-python/pull/162) |
