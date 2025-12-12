# Tracing Approach Decision: Manual Instrumentation vs Span Processor

> **Decision:** Manual Instrumentation
> **Date:** December 2024
> **Status:** Approved

---

## Executive Summary

We evaluated two approaches for transforming verbose OpenInference/LangGraph traces into the UiPath Agents schema:

1. **Span Processor** - Intercept OpenInference spans, transform/filter/reparent
2. **Manual Instrumentation** - Instrument agent code directly, matching C# Agents pattern

After analysis, **Manual Instrumentation** was selected due to exact schema match, lower overhead, and elimination of perceived complexity through new discoveries.

---

## Approach Comparison

### Span Processor Approach

```
LangGraph Agent Code (unchanged)
        ↓
OpenInference Auto-Instrumentation
        ↓
[All verbose spans emitted: LangGraph, init, agent, route_agent, UiPathChat, terminate...]
        ↓
LangGraphCollapsingSpanProcessor
  - Buffer node spans
  - Drop noise (init, route_agent, terminate)
  - Reparent LLM spans to synthetic "Agent run"
  - Synthesize guardrail spans
  - Transform attributes (OpenInference → UiPath)
        ↓
Exporter
```

### Manual Instrumentation Approach

```
LangGraph Agent Code (instrumented)
  - @traced decorators / context managers
  - Direct span creation with UiPath schema
        ↓
[Only meaningful spans emitted: Agent run, LLM call, Tool call, Guardrails...]
        ↓
Exporter
```

---

## Pros and Cons Comparison

### Span Processor

| Pros | Cons |
|------|------|
| Zero agent code changes | Memory overhead (buffering until LangGraph ends) |
| Works with any LangGraph agent | Latency overhead (transform pipeline) |
| Auto token counting preserved | Fake guardrail timing (synthesized from LLM span) |
| Auto I/O serialization preserved | Lost tool type info (process vs integration vs context_grounding) |
| Single point of transformation | Complex nested LangGraph detection |
| | Attribute mapping approximations (OpenInference → UiPath) |
| | Maintenance tied to OpenInference output changes |
| | Cannot emit progressive state naturally (on_start hack needed) |

### Manual Instrumentation

| Pros | Cons |
|------|------|
| Exact UiPath schema match | Must extract tokens from UsageMetadata |
| Lower memory overhead (no buffering) | Must serialize messages manually |
| Lower latency (direct emit) | Must maintain instrumentation code |
| Real guardrail timing (actual execution) | Third-party libs not auto-traced |
| Tool type information preserved | Code changes required in agent files |
| Native progressive state (upsert pattern) | |
| Matches C# Agents exactly | |
| Co-located code + instrumentation | |

---

## Cons Eliminated / Workarounds

| Original Con | Discovery | Resolution |
|--------------|-----------|------------|
| **"Must extract tokens manually"** | `UsageMetadata` already populated by `UiPathChat` classes (`models.py:184-187`) | 3 lines to extract: `response.usage_metadata.input_tokens`, etc. |
| **"Must serialize messages manually"** | C# Agents does this manually too (~30 lines in `CoreEntityMapper.cs`) | Write once, reuse. Gives us exact format control matching C# |
| **"Third-party libs not auto-traced"** | C# Agents doesn't trace them either - by design. Traces business ops, not internals | Same approach: trace "Context Grounding Tool", not internal embedding/vector calls |
| **"Code changes required"** | We own `agent.py`, `llm_node.py`, `tool_node.py`, `terminate_node.py` | Changes co-located with code they trace. Natural maintenance pattern |
| **"Must maintain instrumentation"** | Instrumentation is 5-10 lines per node, not complex | Feature flag allows fallback to OpenInference if issues arise |

---

## Key Discoveries That Changed The Decision

### 1. Token Counting Already Available

```python
# UiPathChat already populates UsageMetadata
# src/uipath_langchain/chat/models.py:184-187
ai_message = AIMessage(
    content=message.get("content", ""),
    usage_metadata=UsageMetadata(
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
    ),
)
```

**Extraction is trivial:**
```python
if response.usage_metadata:
    prompt_tokens = response.usage_metadata.input_tokens
    completion_tokens = response.usage_metadata.output_tokens
```

### 2. C# Agents Uses 100% Manual Instrumentation

No auto-instrumentation for:
- LLM calls → Manual `TraceSpan<CompletionSpanAttributes>`
- Tool calls → Manual `TraceSpan<ToolCallSpanAttributes>`
- Context grounding → Manual `TraceSpan<ContextGroundingToolSpanAttributes>`
- Embeddings → Not traced (internal to ECS service)

### 3. `route_agent` Not Traced in C#

C# has no "routing decision" span. The routing decision is:
- Implicit in `ToolCalls` attribute of LLM span
- Visible from which child spans execute next

**Conclusion:** Don't transform it, don't create it.

### 4. Guardrail Timing Must Be Real

Span Processor synthesizes guardrail spans with borrowed timestamps:
```python
guardrail_span["start_time"] = llm_span["start_time"]  # Fake!
```

Manual instrumentation traces actual guardrail execution:
```python
with tracer.start_guardrail(...) as span:  # Real timing
    result = await guardrail.evaluate(state)
```

---

## Performance Comparison

| Metric | Span Processor | Manual Instrumentation |
|--------|---------------|----------------------|
| Memory | Higher (buffer all spans) | Lower (no buffering) |
| Latency | Higher (transform pipeline) | Lower (direct emit) |
| Span count | Same output, more internal work | Fewer spans created overall |
| CPU | Higher (regex matching, reparenting) | Lower (direct attribute set) |

---

## Schema Fidelity Comparison

| Aspect | Span Processor | Manual Instrumentation |
|--------|---------------|----------------------|
| Span names | Exact match possible | Exact match |
| Span types | Mapped from OpenInference | Native UiPath types |
| Attribute names | Must transform (camelCase) | Native camelCase |
| Tool types | Lost (OpenInference doesn't know) | Preserved (process, integration, etc.) |
| Guardrail details | Synthesized | Real evaluation data |
| Token counts | Preserved from OpenInference | Extracted from UsageMetadata |

---

## Risk Mitigation

### Feature Flag

```bash
UIPATH_TRACING_MODE=manual|openinference|hybrid
```

- **manual** - Full manual instrumentation (target state)
- **openinference** - Current behavior (fallback)
- **hybrid** - Manual for agent-level, OpenInference for debugging

### Rollback Path

If issues arise with manual instrumentation:
1. Set `UIPATH_TRACING_MODE=openinference`
2. Behavior reverts to current state
3. No code deployment needed

---

## Decision Rationale

### Why NOT Span Processor

1. **Complexity for solved problems** - Token counting and I/O serialization are already available
2. **Lossy transformation** - Tool types, real guardrail timing lost
3. **Performance overhead** - Buffering and transformation add latency
4. **Maintenance coupling** - Tied to OpenInference output format changes
5. **Synthesized data** - Fake guardrail spans don't reflect reality

### Why Manual Instrumentation

1. **Exact C# parity** - Same schema, same patterns, same trace output
2. **Lower overhead** - No buffering, no transformation pipeline
3. **Full fidelity** - Tool types, real timing, all attributes preserved
4. **Natural maintenance** - Instrumentation co-located with code
5. **Feature flag safety** - Can revert to OpenInference instantly

---

## Implementation Path

See [manual-instrumentation-plan.md](./manual-instrumentation-plan.md) for detailed implementation plan.

### Summary

| PR | Scope | Focus |
|----|-------|-------|
| PR 1 | Foundation - Structure Only | Get span hierarchy right (Agent run → LLM call). Minimal attributes. |
| PR 2 | Full Attributes + Tool Spans | Complete attributes, message serialization, token extraction, tool instrumentation |
| PR 3 | Guardrails + Context Grounding | Complete coverage, JSON comparison tests for C# parity |

---

## Appendix: Trace Output Comparison

### Current (OpenInference)

```
LangGraph                           4.36s
├── init                            1ms    ← noise
├── agent                           4.32s  ← noise
│   ├── UiPathChat [gpt-4]          4.22s
│   └── route_agent                 1ms    ← noise
├── A_Plus_B                        6ms
│   └── A_Plus_B                    4ms
├── agent                           1.33s  ← noise
│   ├── UiPathChat [gpt-4]          1.23s
│   └── route_agent                 1ms    ← noise
└── terminate                       0ms    ← noise
```

### Target (Manual / C# Agents)

```
Agent run - Agent                   13.85s
├── Agent input guardrail check     1.04s
│   └── Pre-execution governance    89ms
├── LLM call                        8.19s
│   ├── LLM input guardrail check   1.59s
│   │   └── Pre-execution governance 380ms
│   ├── Model run [gpt-4]           2.02s
│   └── LLM output guardrail check  928ms
│       └── Post-execution governance 74ms
├── Tool call - A_Plus_B            6ms
├── LLM call                        ...
├── Agent output guardrail check    929ms
│   └── Post-execution governance   71ms
└── Agent output                    0ms
```

---

## Conclusion

Manual instrumentation is the correct choice because:

1. **The perceived complexity was illusory** - Token counting and serialization are trivial
2. **C# Agents validates the approach** - They use 100% manual instrumentation
3. **Span processor adds overhead for problems that don't exist**
4. **Feature flag provides safe rollback** - Zero risk to adopt

**Final decision: Implement manual instrumentation with feature flag for gradual rollout.**
