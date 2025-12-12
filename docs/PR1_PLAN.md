# PR 1: Manual Instrumentation - Foundation (Structure Only)

> **Goal**: Get the basic span hierarchy right. Agent run → LLM call nesting with minimal attributes.

## Target Span Structure (PR 1 Scope)

```
Agent run - {AgentName}            <- type: agentRun
└── LLM call                       <- type: completion
```

### What We're NOT Doing in PR 1:
- Tool call spans (PR 2)
- Guardrail spans (PR 3)
- Full attribute schemas (PR 2)
- Message serialization (PR 2)
- Token extraction (PR 2)

---

## Implementation Files

### 1. `src/uipath_langchain/_tracing/schema.py`

Minimal schema with only what's needed for PR 1:

```python
"""Span types and names matching C# Agents schema."""
from enum import Enum


class SpanType(str, Enum):
    """Span types matching C# Agents SpanType.cs"""
    AGENT_RUN = "agentRun"
    COMPLETION = "completion"
    AGENT_OUTPUT = "agentOutput"


class SpanName:
    """Span names matching C# Agents SpanName.cs"""

    @staticmethod
    def agent_run(agent_name: str) -> str:
        return f"Agent run - {agent_name}"

    LLM_CALL = "LLM call"
    AGENT_OUTPUT = "Agent output"
```

### 2. `src/uipath_langchain/_tracing/config.py`

Simple boolean feature flag:

```python
"""Tracing configuration - feature flag for custom instrumentation."""
import os


# Simple boolean flag
# True = custom manual instrumentation (target state)
# False = OpenInference auto-instrumentation (for debugging)
CUSTOM_INSTRUMENTATION_ENABLED = os.getenv(
    "UIPATH_CUSTOM_INSTRUMENTATION", "false"
).lower() == "true"


def is_custom_instrumentation_enabled() -> bool:
    """Check if custom instrumentation is enabled."""
    return CUSTOM_INSTRUMENTATION_ENABLED
```

### 3. `src/uipath_langchain/_tracing/tracer.py`

Core tracer with only `start_agent_run` and `start_llm_call`:

```python
"""Manual instrumentation tracer matching C# Agents pattern."""
import json
from contextlib import contextmanager
from typing import Any, Generator, Optional

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from .schema import SpanName, SpanType


class UiPathTracer:
    """Manual tracer matching C# Agents TraceSpan pattern.

    PR 1 Scope: Agent run and LLM call spans only.
    """

    def __init__(self):
        self._tracer = trace.get_tracer("uipath-langchain", "1.0.0")

    @contextmanager
    def start_agent_run(
        self,
        agent_name: str,
        *,
        agent_id: Optional[str] = None,
    ) -> Generator[Span, None, None]:
        """Start an agent run span - matches C# AgentRunSpan.

        PR 1: Minimal attributes - just type and agentName.
        """
        span_name = SpanName.agent_run(agent_name)

        with self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.INTERNAL,
        ) as span:
            # Minimal attributes for PR 1
            span.set_attribute("type", SpanType.AGENT_RUN.value)
            span.set_attribute("agentName", agent_name)
            if agent_id:
                span.set_attribute("agentId", agent_id)
            span.set_attribute("source", "langchain")

            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_attribute("error", json.dumps({
                    "message": str(e),
                    "type": type(e).__name__
                }))
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    @contextmanager
    def start_llm_call(
        self,
        model_name: str,
    ) -> Generator[Span, None, None]:
        """Start an LLM call span - matches C# CompletionSpan.

        PR 1: Minimal attributes - just type and model.
        """
        with self._tracer.start_as_current_span(
            SpanName.LLM_CALL,
            kind=SpanKind.INTERNAL,
        ) as span:
            # Minimal attributes for PR 1
            span.set_attribute("type", SpanType.COMPLETION.value)
            span.set_attribute("model", model_name)

            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_attribute("error", json.dumps({
                    "message": str(e),
                    "type": type(e).__name__
                }))
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    def emit_agent_output(self, output: Any) -> None:
        """Emit agent output span."""
        with self._tracer.start_as_current_span(
            SpanName.AGENT_OUTPUT,
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("type", SpanType.AGENT_OUTPUT.value)
            if isinstance(output, (dict, list)):
                span.set_attribute("output", json.dumps(output))
            else:
                span.set_attribute("output", str(output))
            span.set_status(Status(StatusCode.OK))


# Global tracer instance
_tracer: Optional[UiPathTracer] = None


def get_tracer() -> UiPathTracer:
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = UiPathTracer()
    return _tracer
```

### 4. Update `src/uipath_langchain/_tracing/__init__.py`

```python
from ._instrument_traceable import _instrument_traceable_attributes
from .config import (
    CUSTOM_INSTRUMENTATION_ENABLED,
    is_custom_instrumentation_enabled,
)
from .schema import SpanName, SpanType
from .tracer import UiPathTracer, get_tracer

__all__ = [
    # Existing
    "_instrument_traceable_attributes",
    # Config
    "CUSTOM_INSTRUMENTATION_ENABLED",
    "is_custom_instrumentation_enabled",
    # Schema
    "SpanType",
    "SpanName",
    # Tracer
    "UiPathTracer",
    "get_tracer",
]
```

### 5. Modify `src/uipath_langchain/agent/react/llm_node.py`

Add conditional manual instrumentation:

```python
"""LLM node implementation for LangGraph."""
from typing import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.tools import BaseTool

from uipath_langchain._tracing import get_tracer, is_custom_instrumentation_enabled

from .constants import MAX_SUCCESSIVE_COMPLETIONS
from .types import AgentGraphState
from .utils import count_successive_completions


def create_llm_node(
    model: BaseChatModel,
    tools: Sequence[BaseTool] | None = None,
):
    """Invoke LLM with tools and dynamically control tool_choice."""
    bindable_tools = list(tools) if tools else []
    base_llm = model.bind_tools(bindable_tools) if bindable_tools else model

    async def llm_node(state: AgentGraphState):
        messages: list[AnyMessage] = state.messages

        successive_completions = count_successive_completions(messages)
        if successive_completions >= MAX_SUCCESSIVE_COMPLETIONS:
            llm = base_llm.bind(tool_choice="required")
        else:
            llm = base_llm

        # Check if custom instrumentation is enabled
        if is_custom_instrumentation_enabled():
            return await _llm_node_instrumented(llm, messages, model)
        else:
            # Original behavior - OpenInference handles tracing
            response = await llm.ainvoke(messages)
            if not isinstance(response, AIMessage):
                raise TypeError(
                    f"LLM returned {type(response).__name__} instead of AIMessage"
                )
            return {"messages": [response]}

    return llm_node


async def _llm_node_instrumented(llm, messages: list[AnyMessage], model: BaseChatModel):
    """LLM node with manual instrumentation (PR 1: minimal attributes)."""
    tracer = get_tracer()

    # Extract model name
    model_name = getattr(model, "model_name", getattr(model, "model", "unknown"))

    with tracer.start_llm_call(model_name=model_name) as span:
        response = await llm.ainvoke(messages)

        if not isinstance(response, AIMessage):
            raise TypeError(
                f"LLM returned {type(response).__name__} instead of AIMessage"
            )

        # PR 2 will add: content, toolCalls, usage attributes

        return {"messages": [response]}
```

### 6. Modify `src/uipath_langchain/agent/react/agent.py`

Wrap agent execution with agent run span:

```python
# Add to imports:
from uipath_langchain._tracing import get_tracer, is_custom_instrumentation_enabled

# Modify create_agent function to accept agent_name parameter:
def create_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    messages: Sequence[SystemMessage | HumanMessage]
    | Callable[[InputT], Sequence[SystemMessage | HumanMessage]],
    *,
    input_schema: Type[InputT] | None = None,
    output_schema: Type[OutputT] | None = None,
    config: AgentGraphConfig | None = None,
    guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] | None = None,
    agent_name: str = "Agent",  # NEW parameter
) -> StateGraph[AgentGraphState, None, InputT, OutputT]:
    # ... existing code ...
```

> Note: The agent run span wrapping is complex because LangGraph manages the execution.
> For PR 1, we'll instrument at the runtime level instead (see section 7).

### 7. Modify `src/uipath_langchain/runtime/runtime.py`

Add agent run span at runtime invocation:

```python
# In UiPathLangGraphRuntime.invoke() method:

async def invoke(self, input_data: dict[str, Any], ...) -> dict[str, Any]:
    if is_custom_instrumentation_enabled():
        tracer = get_tracer()
        with tracer.start_agent_run(
            agent_name=self._entrypoint,
            agent_id=str(self._runtime_id),
        ):
            return await self._invoke_internal(input_data, ...)
    else:
        return await self._invoke_internal(input_data, ...)
```

---

## Unit Tests

### `tests/tracing/test_config.py`

```python
"""Test feature flag configuration."""
import os
from unittest import mock

import pytest


def test_custom_instrumentation_disabled_by_default():
    """Default should be disabled."""
    with mock.patch.dict(os.environ, {}, clear=True):
        # Re-import to pick up new env
        from uipath_langchain._tracing.config import is_custom_instrumentation_enabled
        # Note: module caching means we need importlib.reload in real test
        # For now, just document expected behavior
        pass


def test_custom_instrumentation_enabled():
    """Env var should enable custom instrumentation."""
    with mock.patch.dict(os.environ, {"UIPATH_CUSTOM_INSTRUMENTATION": "true"}):
        pass  # Test implementation


def test_custom_instrumentation_case_insensitive():
    """TRUE, True, true should all work."""
    for value in ["TRUE", "True", "true", "TrUe"]:
        with mock.patch.dict(os.environ, {"UIPATH_CUSTOM_INSTRUMENTATION": value}):
            pass  # Test implementation
```

### `tests/tracing/test_schema.py`

```python
"""Test span schema matches C# Agents."""
import pytest

from uipath_langchain._tracing.schema import SpanName, SpanType


def test_span_types_match_csharp():
    """Verify span types match C# Agents SpanType.cs exactly."""
    assert SpanType.AGENT_RUN.value == "agentRun"
    assert SpanType.COMPLETION.value == "completion"
    assert SpanType.AGENT_OUTPUT.value == "agentOutput"


def test_span_names_match_csharp():
    """Verify span names match C# Agents SpanName.cs exactly."""
    assert SpanName.agent_run("TestAgent") == "Agent run - TestAgent"
    assert SpanName.LLM_CALL == "LLM call"
    assert SpanName.AGENT_OUTPUT == "Agent output"
```

### `tests/tracing/test_tracer.py`

```python
"""Test UiPathTracer produces correct span structure."""
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry import trace

from uipath_langchain._tracing.tracer import UiPathTracer


@pytest.fixture
def tracer_with_exporter():
    """Setup tracer with in-memory exporter for testing."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    tracer = UiPathTracer()
    return tracer, exporter


def test_agent_run_span_structure(tracer_with_exporter):
    """Test agent run span has correct name and type."""
    tracer, exporter = tracer_with_exporter

    with tracer.start_agent_run("TestAgent", agent_id="123"):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == "Agent run - TestAgent"
    assert span.attributes["type"] == "agentRun"
    assert span.attributes["agentName"] == "TestAgent"
    assert span.attributes["agentId"] == "123"
    assert span.attributes["source"] == "langchain"


def test_llm_call_span_structure(tracer_with_exporter):
    """Test LLM call span has correct name and type."""
    tracer, exporter = tracer_with_exporter

    with tracer.start_llm_call("gpt-4"):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == "LLM call"
    assert span.attributes["type"] == "completion"
    assert span.attributes["model"] == "gpt-4"


def test_span_hierarchy(tracer_with_exporter):
    """Test LLM call is child of agent run."""
    tracer, exporter = tracer_with_exporter

    with tracer.start_agent_run("TestAgent") as agent_span:
        with tracer.start_llm_call("gpt-4") as llm_span:
            pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 2

    # Find spans by name
    agent_span = next(s for s in spans if "Agent run" in s.name)
    llm_span = next(s for s in spans if s.name == "LLM call")

    # Verify hierarchy
    assert llm_span.parent.span_id == agent_span.context.span_id


def test_agent_output_span(tracer_with_exporter):
    """Test agent output span."""
    tracer, exporter = tracer_with_exporter

    tracer.emit_agent_output({"result": "test"})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == "Agent output"
    assert span.attributes["type"] == "agentOutput"
```

---

## E2E Test Plan

### Setup Local Development

1. In `uipath-langchain-python/pyproject.toml`, add:
```toml
[tool.uv.sources]
uipath = { path = "../uipath-python", editable = true }
```

2. In `uipath-agents-python/pyproject.toml`, add:
```toml
[tool.uv.sources]
uipath = { path = "../uipath-python", editable = true }
uipath-langchain = { path = "../uipath-langchain-python", editable = true }
```

3. Sync both projects:
```bash
cd ~/repos/uipath-langchain-python && uv sync
cd ~/repos/uipath-agents-python && uv sync
```

### Run E2E Test

```bash
cd ~/repos/uipath-agents-python/examples/basic

# Enable custom instrumentation
export UIPATH_CUSTOM_INSTRUMENTATION=true

# Run the agent
uv run uipath run agent.json '{}'
```

### Verify Span Structure

The trace output should show:
```
Agent run - Agent                  <- type: agentRun
└── LLM call                       <- type: completion
    (may have multiple LLM calls)
```

NOT the verbose OpenInference structure:
```
LangGraph
├── init
├── agent
│   ├── UiPathChat
│   └── route_agent
└── terminate
```

---

## Success Criteria for PR 1

1. **Feature flag works**: `UIPATH_CUSTOM_INSTRUMENTATION=true` enables custom tracing
2. **Span hierarchy correct**: Agent run → LLM call parent-child relationship
3. **Span types match C#**: `agentRun`, `completion`, `agentOutput`
4. **Span names match C#**: `Agent run - {name}`, `LLM call`, `Agent output`
5. **No OpenInference noise**: No `init`, `route_agent`, `terminate` spans when enabled
6. **Unit tests pass**: All schema and tracer tests pass
7. **E2E works**: Sample agent runs with correct trace structure

---

## Files Changed Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `_tracing/schema.py` | NEW | SpanType, SpanName |
| `_tracing/config.py` | NEW | Feature flag |
| `_tracing/tracer.py` | NEW | UiPathTracer class |
| `_tracing/__init__.py` | MODIFY | Add exports |
| `agent/react/llm_node.py` | MODIFY | Add conditional instrumentation |
| `runtime/runtime.py` | MODIFY | Add agent run span wrapper |
| `tests/tracing/test_*.py` | NEW | Unit tests |
