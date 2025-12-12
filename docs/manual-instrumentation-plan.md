# Manual Instrumentation Implementation Plan

> Goal: Replace OpenInference auto-instrumentation with manual instrumentation matching C# Agents schema exactly.

## Overview

### Current State (OpenInference)
```
LangGraph                          <- OpenInference auto-span
├── init                           <- Noise (drop)
├── agent                          <- Noise (drop)
│   ├── UiPathChat                 <- LLM span (transform)
│   └── route_agent                <- Noise (drop)
├── A_Plus_B                       <- Tool span (transform)
├── agent                          <- Noise (drop)
│   ├── UiPathChat                 <- LLM span (transform)
│   └── route_agent                <- Noise (drop)
└── terminate                      <- Noise (drop)
```

### Target State (C# Agents Schema)
```
Agent run - {AgentName}            <- type: agentRun
├── Agent input guardrail check    <- type: agentPreGuardrails
│   └── Pre-execution governance   <- type: preGovernance
├── LLM call                       <- type: completion
│   ├── LLM input guardrail check  <- type: llmPreGuardrails
│   │   └── Pre-execution governance
│   ├── Model run                  <- type: completion (actual LLM data)
│   └── LLM output guardrail check <- type: llmPostGuardrails
│       └── Post-execution governance
├── Tool call - {ToolName}         <- type: toolCall
│   ├── Tool input guardrail check <- type: toolPreGuardrails
│   └── Tool output guardrail check<- type: toolPostGuardrails
├── LLM call                       <- (repeat for each LLM iteration)
├── Agent output guardrail check   <- type: agentPostGuardrails
│   └── Post-execution governance
└── Agent output                   <- type: agentOutput
```

---

## Feature Flag Design

### Environment Variable
```bash
# Simple boolean toggle - enabled = custom instrumentation, disabled = OpenInference (for debugging)
UIPATH_CUSTOM_INSTRUMENTATION=true|false
```

### Settings Integration
```python
# src/uipath_langchain/_utils/_settings.py
import os

# Simple boolean flag - no hybrid mode needed
# True = custom manual instrumentation (target state)
# False = OpenInference auto-instrumentation (for debugging)
CUSTOM_INSTRUMENTATION_ENABLED = os.getenv("UIPATH_CUSTOM_INSTRUMENTATION", "false").lower() == "true"

def is_custom_instrumentation_enabled() -> bool:
    """Check if custom instrumentation is enabled."""
    return CUSTOM_INSTRUMENTATION_ENABLED
```

### CLI Flag
```bash
uipath-langchain run --custom-instrumentation
```

---

## PR 1: Foundation - Structure Only

> **Scope**: Get the basic span structure right. Agent run + LLM call spans only. No tool spans, no guardrails. Minimal attributes - just enough to verify hierarchy is correct.

### 1.1 Span Types & Attributes Schema

Create schema definitions matching C# Agents exactly.

**File:** `src/uipath_langchain/_tracing/schema.py`

```python
"""Span types and attributes matching C# Agents schema."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import json

class SpanType(str, Enum):
    """Span types matching C# Agents SpanType.cs"""
    AGENT_RUN = "agentRun"
    COMPLETION = "completion"
    LLM_CALL = "llmCall"
    TOOL_CALL = "toolCall"

    # Guardrails
    AGENT_PRE_GUARDRAILS = "agentPreGuardrails"
    AGENT_POST_GUARDRAILS = "agentPostGuardrails"
    LLM_PRE_GUARDRAILS = "llmPreGuardrails"
    LLM_POST_GUARDRAILS = "llmPostGuardrails"
    TOOL_PRE_GUARDRAILS = "toolPreGuardrails"
    TOOL_POST_GUARDRAILS = "toolPostGuardrails"

    # Governance
    PRE_GOVERNANCE = "preGovernance"
    POST_GOVERNANCE = "postGovernance"
    GUARDRAIL_EVALUATION = "guardrailEvaluation"
    GUARDRAIL_ESCALATION = "guardrailEscalation"

    # Tools
    PROCESS_TOOL = "processTool"
    INTEGRATION_TOOL = "integrationTool"
    CONTEXT_GROUNDING_TOOL = "contextGroundingTool"
    ESCALATION_TOOL = "escalationTool"
    MCP_TOOL = "mcpTool"

    # Output
    AGENT_OUTPUT = "agentOutput"


class SpanName:
    """Span names matching C# Agents SpanName.cs"""

    @staticmethod
    def agent_run(agent_name: str, is_conversational: bool = False) -> str:
        prefix = "Conversational " if is_conversational else ""
        return f"{prefix}Agent run - {agent_name}"

    LLM_CALL = "LLM call"
    MODEL_RUN = "Model run"

    @staticmethod
    def tool_call(tool_name: str) -> str:
        return f"Tool call - {tool_name}"

    # Guardrails
    AGENT_INPUT_GUARDRAIL = "Agent input guardrail check"
    AGENT_OUTPUT_GUARDRAIL = "Agent output guardrail check"
    LLM_INPUT_GUARDRAIL = "LLM input guardrail check"
    LLM_OUTPUT_GUARDRAIL = "LLM output guardrail check"
    TOOL_INPUT_GUARDRAIL = "Tool input guardrail check"
    TOOL_OUTPUT_GUARDRAIL = "Tool output guardrail check"

    # Governance
    PRE_GOVERNANCE = "Pre-execution governance"
    POST_GOVERNANCE = "Post-execution governance"

    # Output
    AGENT_OUTPUT = "Agent output"


class ToolType(str, Enum):
    """Tool types for toolCall spans."""
    PROCESS = "process"
    INTEGRATION = "integration"
    CONTEXT_GROUNDING = "contextGrounding"
    ESCALATION = "escalation"
    MCP = "mcp"
    CUSTOM = "custom"


@dataclass
class AgentRunAttributes:
    """Attributes for agentRun span - matches AgentRunSpanAttributes.cs"""
    type: str = field(default=SpanType.AGENT_RUN)
    agent_id: str = ""
    agent_name: str = ""
    agent_version: str = ""
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    input_schema: Optional[dict] = None
    output_schema: Optional[dict] = None
    input: dict = field(default_factory=dict)
    output: Optional[dict] = None
    source: str = "langchain"
    is_conversational: Optional[bool] = None
    error: Optional[dict] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in {
            "type": self.type,
            "agentId": self.agent_id,
            "agentName": self.agent_name,
            "agentVersion": self.agent_version,
            "systemPrompt": self.system_prompt,
            "userPrompt": self.user_prompt,
            "inputSchema": self.input_schema,
            "outputSchema": self.output_schema,
            "input": self.input,
            "output": self.output,
            "source": self.source,
            "isConversational": self.is_conversational,
            "error": self.error,
        }.items() if v is not None}


@dataclass
class CompletionAttributes:
    """Attributes for completion span - matches CompletionSpanAttributes.cs"""
    type: str = field(default=SpanType.COMPLETION)
    model: str = ""
    settings: dict = field(default_factory=dict)  # {maxTokens, temperature}
    content: Optional[str] = None
    tool_calls: Optional[list] = None
    usage: Optional[dict] = None  # {promptTokens, completionTokens, totalTokens}
    error: Optional[dict] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in {
            "type": self.type,
            "model": self.model,
            "settings": self.settings,
            "content": self.content,
            "toolCalls": self.tool_calls,
            "usage": self.usage,
            "error": self.error,
        }.items() if v is not None}


@dataclass
class ToolCallAttributes:
    """Attributes for toolCall span - matches ToolCallSpanAttributes.cs"""
    type: str = field(default=SpanType.TOOL_CALL)
    call_id: str = ""
    tool_name: str = ""
    tool_type: str = ToolType.CUSTOM
    arguments: dict = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[dict] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in {
            "type": self.type,
            "callId": self.call_id,
            "toolName": self.tool_name,
            "toolType": self.tool_type,
            "arguments": self.arguments,
            "result": self.result,
            "error": self.error,
        }.items() if v is not None}


@dataclass
class ContextGroundingAttributes:
    """Attributes for contextGroundingTool span."""
    type: str = field(default=SpanType.CONTEXT_GROUNDING_TOOL)
    retrieval_mode: str = "semantic"  # semantic, summarization, deepRag
    query: str = ""
    threshold: Optional[float] = None
    number_of_results: Optional[int] = None
    filter: Optional[str] = None
    index_id: Optional[str] = None
    results: Optional[list] = None
    error: Optional[dict] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in {
            "type": self.type,
            "retrievalMode": self.retrieval_mode,
            "query": self.query,
            "threshold": self.threshold,
            "numberOfResults": self.number_of_results,
            "filter": self.filter,
            "indexId": self.index_id,
            "results": self.results,
            "error": self.error,
        }.items() if v is not None}


@dataclass
class GuardrailAttributes:
    """Attributes for guardrail spans."""
    type: str = ""  # Set based on guardrail type
    error: Optional[dict] = None

    def to_dict(self) -> dict:
        return {"type": self.type, "error": self.error} if self.error else {"type": self.type}


@dataclass
class AgentOutputAttributes:
    """Attributes for agentOutput span."""
    type: str = field(default=SpanType.AGENT_OUTPUT)
    output: Any = None

    def to_dict(self) -> dict:
        return {"type": self.type, "output": self.output}
```

### 1.2 Core Tracer Infrastructure

**File:** `src/uipath_langchain/_tracing/tracer.py`

```python
"""Manual instrumentation tracer matching C# Agents pattern."""
import json
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Generator, Optional

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from .schema import (
    AgentOutputAttributes,
    AgentRunAttributes,
    CompletionAttributes,
    GuardrailAttributes,
    SpanName,
    SpanType,
    ToolCallAttributes,
)


@dataclass
class SpanContext:
    """Context for tracking span hierarchy."""
    trace_id: str
    parent_span_id: Optional[str]
    agent_id: Optional[str] = None


class UiPathTracer:
    """Manual tracer matching C# Agents TraceSpan pattern.

    Key differences from OpenInference:
    - Creates exact UiPath schema spans
    - Supports upsert pattern (emit at start, update at end)
    - No noise spans (init, route_agent, terminate)
    """

    def __init__(self):
        self._tracer = trace.get_tracer("uipath-langchain", "1.0.0")

    @contextmanager
    def start_agent_run(
        self,
        agent_name: str,
        input_data: dict,
        *,
        agent_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        is_conversational: bool = False,
    ) -> Generator[tuple[Span, AgentRunAttributes], None, None]:
        """Start an agent run span - matches C# AgentRunSpan."""
        attrs = AgentRunAttributes(
            agent_id=agent_id or str(uuid.uuid4()),
            agent_name=agent_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            input=input_data,
            is_conversational=is_conversational if is_conversational else None,
        )

        span_name = SpanName.agent_run(agent_name, is_conversational)

        with self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.INTERNAL,
        ) as span:
            # Set initial attributes
            for key, value in attrs.to_dict().items():
                if isinstance(value, (dict, list)):
                    span.set_attribute(key, json.dumps(value))
                else:
                    span.set_attribute(key, value)

            try:
                yield span, attrs
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                attrs.error = {"message": str(e), "type": type(e).__name__}
                span.set_attribute("error", json.dumps(attrs.error))
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    @contextmanager
    def start_llm_call(
        self,
        model_name: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Generator[tuple[Span, CompletionAttributes], None, None]:
        """Start an LLM call span - matches C# CompletionSpan."""
        settings = {}
        if max_tokens is not None:
            settings["maxTokens"] = max_tokens
        if temperature is not None:
            settings["temperature"] = temperature

        attrs = CompletionAttributes(
            model=model_name,
            settings=settings,
        )

        with self._tracer.start_as_current_span(
            SpanName.LLM_CALL,
            kind=SpanKind.INTERNAL,
        ) as span:
            for key, value in attrs.to_dict().items():
                if isinstance(value, (dict, list)):
                    span.set_attribute(key, json.dumps(value))
                else:
                    span.set_attribute(key, value)

            try:
                yield span, attrs
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                attrs.error = {"message": str(e), "type": type(e).__name__}
                span.set_attribute("error", json.dumps(attrs.error))
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    def complete_llm_call(
        self,
        span: Span,
        attrs: CompletionAttributes,
        *,
        content: Optional[str] = None,
        tool_calls: Optional[list] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        """Complete an LLM call span with response data."""
        attrs.content = content
        attrs.tool_calls = tool_calls
        attrs.usage = {
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "totalTokens": total_tokens,
        }

        # Update span attributes
        if content:
            span.set_attribute("content", content)
        if tool_calls:
            span.set_attribute("toolCalls", json.dumps(tool_calls))
        span.set_attribute("usage", json.dumps(attrs.usage))

    @contextmanager
    def start_tool_call(
        self,
        tool_name: str,
        call_id: str,
        arguments: dict,
        tool_type: str = "custom",
    ) -> Generator[tuple[Span, ToolCallAttributes], None, None]:
        """Start a tool call span - matches C# ToolCallSpan."""
        attrs = ToolCallAttributes(
            call_id=call_id,
            tool_name=tool_name,
            tool_type=tool_type,
            arguments=arguments,
        )

        with self._tracer.start_as_current_span(
            SpanName.tool_call(tool_name),
            kind=SpanKind.INTERNAL,
        ) as span:
            for key, value in attrs.to_dict().items():
                if isinstance(value, (dict, list)):
                    span.set_attribute(key, json.dumps(value))
                else:
                    span.set_attribute(key, value)

            try:
                yield span, attrs
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                attrs.error = {"message": str(e), "type": type(e).__name__}
                span.set_attribute("error", json.dumps(attrs.error))
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    def complete_tool_call(
        self,
        span: Span,
        attrs: ToolCallAttributes,
        result: Any,
    ) -> None:
        """Complete a tool call span with result."""
        attrs.result = result
        if isinstance(result, (dict, list)):
            span.set_attribute("result", json.dumps(result))
        else:
            span.set_attribute("result", str(result))

    @contextmanager
    def start_guardrail(
        self,
        span_type: SpanType,
        span_name: str,
    ) -> Generator[tuple[Span, GuardrailAttributes], None, None]:
        """Start a guardrail span."""
        attrs = GuardrailAttributes(type=span_type)

        with self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("type", span_type)

            try:
                yield span, attrs
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                attrs.error = {"message": str(e), "type": type(e).__name__}
                span.set_attribute("error", json.dumps(attrs.error))
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    def emit_agent_output(
        self,
        output: Any,
    ) -> None:
        """Emit agent output span."""
        attrs = AgentOutputAttributes(output=output)

        with self._tracer.start_as_current_span(
            SpanName.AGENT_OUTPUT,
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("type", SpanType.AGENT_OUTPUT)
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

### 1.3 Feature Flag Implementation

**File:** `src/uipath_langchain/_tracing/config.py`

```python
"""Tracing configuration and feature flags."""
import os
from enum import Enum
from typing import Optional


class TracingMode(str, Enum):
    """Tracing mode selection."""
    OPENINFERENCE = "openinference"  # Current behavior (default)
    MANUAL = "manual"                 # Full manual instrumentation
    HYBRID = "hybrid"                 # Manual agent-level, OpenInference for internals


_tracing_mode: Optional[TracingMode] = None


def get_tracing_mode() -> TracingMode:
    """Get the current tracing mode."""
    global _tracing_mode
    if _tracing_mode is None:
        mode = os.getenv("UIPATH_TRACING_MODE", TracingMode.OPENINFERENCE)
        _tracing_mode = TracingMode(mode)
    return _tracing_mode


def set_tracing_mode(mode: TracingMode) -> None:
    """Set the tracing mode (for CLI/programmatic use)."""
    global _tracing_mode
    _tracing_mode = mode


def is_manual_tracing() -> bool:
    """Check if manual tracing is enabled."""
    return get_tracing_mode() in (TracingMode.MANUAL, TracingMode.HYBRID)


def is_openinference_enabled() -> bool:
    """Check if OpenInference should be enabled."""
    return get_tracing_mode() in (TracingMode.OPENINFERENCE, TracingMode.HYBRID)
```

### 1.4 Message Serialization Utilities

**File:** `src/uipath_langchain/_tracing/serialize.py`

```python
"""Message serialization matching C# Agents format."""
import json
from typing import Any, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


def serialize_message(message: BaseMessage) -> dict:
    """Serialize a LangChain message to C# Agents format."""
    result = {
        "role": _get_role(message),
        "content": message.content if isinstance(message.content, str) else "",
    }

    if isinstance(message, AIMessage) and message.tool_calls:
        result["toolCalls"] = [
            {
                "id": tc.get("id", ""),
                "name": tc.get("name", ""),
                "arguments": tc.get("args", {}),
            }
            for tc in message.tool_calls
        ]

    if isinstance(message, ToolMessage):
        result["toolCallId"] = message.tool_call_id

    return result


def serialize_messages(messages: Sequence[BaseMessage]) -> list[dict]:
    """Serialize a list of messages."""
    return [serialize_message(m) for m in messages]


def _get_role(message: BaseMessage) -> str:
    """Get the role string for a message."""
    if isinstance(message, SystemMessage):
        return "system"
    elif isinstance(message, HumanMessage):
        return "user"
    elif isinstance(message, AIMessage):
        return "assistant"
    elif isinstance(message, ToolMessage):
        return "tool"
    return message.type


def serialize_tool_calls(tool_calls: list) -> list[dict]:
    """Serialize tool calls to C# Agents format."""
    return [
        {
            "id": tc.get("id", ""),
            "name": tc.get("name", ""),
            "arguments": tc.get("args", {}),
        }
        for tc in tool_calls
    ]


def extract_usage_from_response(response: AIMessage) -> dict:
    """Extract token usage from AIMessage response."""
    usage = {"promptTokens": 0, "completionTokens": 0, "totalTokens": 0}

    if response.usage_metadata:
        usage["promptTokens"] = response.usage_metadata.input_tokens or 0
        usage["completionTokens"] = response.usage_metadata.output_tokens or 0
        usage["totalTokens"] = response.usage_metadata.total_tokens or 0

    return usage
```

### 1.5 Update Module Exports

**File:** `src/uipath_langchain/_tracing/__init__.py`

```python
from ._instrument_traceable import _instrument_traceable_attributes
from .config import (
    TracingMode,
    get_tracing_mode,
    is_manual_tracing,
    is_openinference_enabled,
    set_tracing_mode,
)
from .schema import (
    AgentOutputAttributes,
    AgentRunAttributes,
    CompletionAttributes,
    ContextGroundingAttributes,
    GuardrailAttributes,
    SpanName,
    SpanType,
    ToolCallAttributes,
    ToolType,
)
from .serialize import (
    extract_usage_from_response,
    serialize_message,
    serialize_messages,
    serialize_tool_calls,
)
from .tracer import UiPathTracer, get_tracer

__all__ = [
    # Existing
    "_instrument_traceable_attributes",
    # Config
    "TracingMode",
    "get_tracing_mode",
    "set_tracing_mode",
    "is_manual_tracing",
    "is_openinference_enabled",
    # Schema
    "SpanType",
    "SpanName",
    "ToolType",
    "AgentRunAttributes",
    "CompletionAttributes",
    "ToolCallAttributes",
    "ContextGroundingAttributes",
    "GuardrailAttributes",
    "AgentOutputAttributes",
    # Serialization
    "serialize_message",
    "serialize_messages",
    "serialize_tool_calls",
    "extract_usage_from_response",
    # Tracer
    "UiPathTracer",
    "get_tracer",
]
```

### PR 1 Deliverables

| File | Purpose |
|------|---------|
| `_tracing/schema.py` | Span types only (SpanType, SpanName enums) |
| `_tracing/tracer.py` | Core tracer with `start_agent_run` and `start_llm_call` only |
| `_tracing/config.py` | Simple boolean feature flag |
| `_tracing/__init__.py` | Updated exports |

> **Note**: Message serialization (`serialize.py`) and full attribute schemas deferred to PR 2. Focus here is just span hierarchy.

### PR 1 Testing

```python
# tests/tracing/test_schema.py
def test_span_types_match_csharp():
    """Verify span types match C# Agents SpanType.cs"""
    assert SpanType.AGENT_RUN == "agentRun"
    assert SpanType.COMPLETION == "completion"
    assert SpanType.TOOL_CALL == "toolCall"
    # ... all types

def test_agent_run_attributes_serialization():
    """Verify AgentRunAttributes matches C# schema."""
    attrs = AgentRunAttributes(
        agent_id="123",
        agent_name="TestAgent",
        input={"query": "test"},
    )
    d = attrs.to_dict()
    assert d["type"] == "agentRun"
    assert d["agentId"] == "123"
    assert "agentName" in d  # camelCase
```

---

## PR 2: Core Agent Instrumentation + Full Attributes

> **Scope**: Add tool call spans, full attribute schemas, message serialization, token extraction.

### 2.1 Instrumented LLM Node

**File:** `src/uipath_langchain/agent/react/llm_node.py` (modified)

```python
"""LLM node implementation with manual instrumentation."""
from typing import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.tools import BaseTool

from uipath_langchain._tracing import (
    extract_usage_from_response,
    get_tracer,
    is_manual_tracing,
    serialize_messages,
    serialize_tool_calls,
)

from .constants import MAX_SUCCESSIVE_COMPLETIONS
from .types import AgentGraphState
from .utils import count_successive_completions


def create_llm_node(
    model: BaseChatModel,
    tools: Sequence[BaseTool] | None = None,
):
    """Create LLM node with optional manual instrumentation."""
    bindable_tools = list(tools) if tools else []
    base_llm = model.bind_tools(bindable_tools) if bindable_tools else model

    async def llm_node(state: AgentGraphState):
        messages: list[AnyMessage] = state.messages

        successive_completions = count_successive_completions(messages)
        if successive_completions >= MAX_SUCCESSIVE_COMPLETIONS:
            llm = base_llm.bind(tool_choice="required")
        else:
            llm = base_llm

        if is_manual_tracing():
            return await _llm_node_instrumented(llm, messages, model)
        else:
            # Original behavior - OpenInference handles tracing
            response = await llm.ainvoke(messages)
            if not isinstance(response, AIMessage):
                raise TypeError(f"LLM returned {type(response).__name__} instead of AIMessage")
            return {"messages": [response]}

    return llm_node


async def _llm_node_instrumented(llm, messages: list[AnyMessage], model: BaseChatModel):
    """LLM node with manual instrumentation."""
    tracer = get_tracer()

    # Extract model settings
    model_name = getattr(model, "model_name", getattr(model, "model", "unknown"))
    max_tokens = getattr(model, "max_tokens", None)
    temperature = getattr(model, "temperature", None)

    with tracer.start_llm_call(
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
    ) as (span, attrs):
        # Set input messages
        span.set_attribute("input.messages", serialize_messages(messages))

        # Invoke LLM
        response = await llm.ainvoke(messages)

        if not isinstance(response, AIMessage):
            raise TypeError(f"LLM returned {type(response).__name__} instead of AIMessage")

        # Extract and set response data
        usage = extract_usage_from_response(response)
        tool_calls = serialize_tool_calls(response.tool_calls) if response.tool_calls else None

        tracer.complete_llm_call(
            span,
            attrs,
            content=response.content if isinstance(response.content, str) else None,
            tool_calls=tool_calls,
            prompt_tokens=usage["promptTokens"],
            completion_tokens=usage["completionTokens"],
            total_tokens=usage["totalTokens"],
        )

        return {"messages": [response]}
```

### 2.2 Instrumented Tool Node

**File:** `src/uipath_langchain/agent/tools/tool_node.py` (modified)

```python
"""Tool node with manual instrumentation."""
import json
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

from uipath_langchain._tracing import (
    ToolType,
    get_tracer,
    is_manual_tracing,
)

from .context_tool import UiPathContextGroundingTool
from .escalation_tool import UiPathEscalationTool
from .integration_tool import UiPathIntegrationTool
from .process_tool import UiPathProcessTool


def _get_tool_type(tool: BaseTool) -> str:
    """Determine tool type for tracing."""
    if isinstance(tool, UiPathProcessTool):
        return ToolType.PROCESS
    elif isinstance(tool, UiPathIntegrationTool):
        return ToolType.INTEGRATION
    elif isinstance(tool, UiPathContextGroundingTool):
        return ToolType.CONTEXT_GROUNDING
    elif isinstance(tool, UiPathEscalationTool):
        return ToolType.ESCALATION
    # Check for MCP tools
    elif hasattr(tool, "_mcp_server"):
        return ToolType.MCP
    return ToolType.CUSTOM


def create_tool_node(tools: list[BaseTool]) -> dict[str, Any]:
    """Create tool nodes with optional manual instrumentation."""
    tool_map = {tool.name: tool for tool in tools}
    nodes = {}

    for tool_name, tool in tool_map.items():
        nodes[f"action:{tool_name}"] = _create_single_tool_node(tool)

    return nodes


def _create_single_tool_node(tool: BaseTool):
    """Create a single tool node."""
    tool_type = _get_tool_type(tool)

    async def tool_node(state):
        messages = state.messages
        last_message = messages[-1]

        if not hasattr(last_message, "tool_calls"):
            return {"messages": []}

        tool_calls = [tc for tc in last_message.tool_calls if tc["name"] == tool.name]
        results = []

        for tool_call in tool_calls:
            if is_manual_tracing():
                result = await _invoke_tool_instrumented(
                    tool, tool_call, tool_type
                )
            else:
                # Original behavior
                result = await tool.ainvoke(tool_call["args"])

            results.append(
                ToolMessage(
                    content=json.dumps(result) if not isinstance(result, str) else result,
                    tool_call_id=tool_call["id"],
                    name=tool.name,
                )
            )

        return {"messages": results}

    return tool_node


async def _invoke_tool_instrumented(
    tool: BaseTool,
    tool_call: dict,
    tool_type: str,
) -> Any:
    """Invoke tool with manual instrumentation."""
    tracer = get_tracer()

    with tracer.start_tool_call(
        tool_name=tool.name,
        call_id=tool_call["id"],
        arguments=tool_call["args"],
        tool_type=tool_type,
    ) as (span, attrs):
        result = await tool.ainvoke(tool_call["args"])
        tracer.complete_tool_call(span, attrs, result)
        return result
```

### 2.3 Instrumented Agent Wrapper

**File:** `src/uipath_langchain/agent/react/agent.py` (modified)

Add agent-level span wrapper:

```python
"""Agent creation with manual instrumentation support."""
import os
from typing import Callable, Sequence, Type, TypeVar, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pydantic import BaseModel
from uipath.platform.guardrails import BaseGuardrail

from uipath_langchain._tracing import (
    SpanName,
    SpanType,
    get_tracer,
    is_manual_tracing,
)

from ..guardrails import create_llm_guardrails_subgraph
from ..guardrails.actions import GuardrailAction
from ..tools import create_tool_node
from .init_node import create_init_node
from .llm_node import create_llm_node
from .router import route_agent
from .terminate_node import create_terminate_node
from .tools import create_flow_control_tools
from .types import AgentGraphConfig, AgentGraphNode, AgentGraphState

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


def create_state_with_input(input_schema: Type[InputT]):
    InnerAgentGraphState = type(
        "InnerAgentGraphState",
        (AgentGraphState, input_schema),
        {},
    )
    cast(type[BaseModel], InnerAgentGraphState).model_rebuild()
    return InnerAgentGraphState


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
    agent_name: str = "Agent",  # NEW: For span naming
) -> StateGraph[AgentGraphState, None, InputT, OutputT]:
    """Build agent graph with optional manual instrumentation."""
    if config is None:
        config = AgentGraphConfig()

    os.environ["LANGCHAIN_RECURSION_LIMIT"] = str(config.recursion_limit)

    agent_tools = list(tools)
    flow_control_tools: list[BaseTool] = create_flow_control_tools(output_schema)
    llm_tools: list[BaseTool] = [*agent_tools, *flow_control_tools]

    init_node = create_init_node(messages)
    tool_nodes = create_tool_node(agent_tools)
    terminate_node = create_terminate_node(output_schema, agent_name)  # Pass agent_name

    InnerAgentGraphState = create_state_with_input(
        input_schema if input_schema is not None else BaseModel
    )

    builder: StateGraph[AgentGraphState, None, InputT, OutputT] = StateGraph(
        InnerAgentGraphState, input_schema=input_schema, output_schema=output_schema
    )

    # Wrap init node with agent run span if manual tracing
    if is_manual_tracing():
        init_node = _wrap_init_with_agent_span(init_node, agent_name, messages)

    builder.add_node(AgentGraphNode.INIT, init_node)

    for tool_name, tool_node in tool_nodes.items():
        builder.add_node(tool_name, tool_node)

    builder.add_node(AgentGraphNode.TERMINATE, terminate_node)
    builder.add_edge(START, AgentGraphNode.INIT)

    llm_node = create_llm_node(model, llm_tools)
    llm_with_guardrails_subgraph = create_llm_guardrails_subgraph(
        (AgentGraphNode.LLM, llm_node), guardrails
    )
    builder.add_node(AgentGraphNode.AGENT, llm_with_guardrails_subgraph)
    builder.add_edge(AgentGraphNode.INIT, AgentGraphNode.AGENT)

    tool_node_names = list(tool_nodes.keys())
    builder.add_conditional_edges(
        AgentGraphNode.AGENT,
        route_agent,
        [AgentGraphNode.AGENT, *tool_node_names, AgentGraphNode.TERMINATE],
    )

    for tool_name in tool_node_names:
        builder.add_edge(tool_name, AgentGraphNode.AGENT)

    builder.add_edge(AgentGraphNode.TERMINATE, END)

    return builder


def _wrap_init_with_agent_span(init_node, agent_name: str, messages):
    """Wrap init node to start agent run span."""
    async def wrapped_init(state):
        tracer = get_tracer()

        # Extract system/user prompts from messages
        system_prompt = None
        user_prompt = None

        if callable(messages):
            resolved_messages = messages(state)
        else:
            resolved_messages = messages

        for msg in resolved_messages:
            if isinstance(msg, SystemMessage):
                system_prompt = msg.content
            elif isinstance(msg, HumanMessage):
                user_prompt = msg.content

        # Start agent run span (will be closed in terminate_node)
        # Store span context in state for later access
        with tracer.start_agent_run(
            agent_name=agent_name,
            input_data=state.model_dump() if hasattr(state, "model_dump") else {},
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        ) as (span, attrs):
            # Store span reference for terminate node
            state._agent_span = span
            state._agent_attrs = attrs

            result = await init_node(state)
            return result

    return wrapped_init
```

### 2.4 Updated Terminate Node

**File:** `src/uipath_langchain/agent/react/terminate_node.py` (modified)

```python
"""Terminate node with agent output span emission."""
from typing import Type, TypeVar

from pydantic import BaseModel

from uipath_langchain._tracing import get_tracer, is_manual_tracing

from .types import AgentGraphState

OutputT = TypeVar("OutputT", bound=BaseModel)


def create_terminate_node(
    output_schema: Type[OutputT] | None = None,
    agent_name: str = "Agent",
):
    """Create terminate node that emits agent output span."""

    async def terminate_node(state: AgentGraphState):
        # Extract output from state
        output = _extract_output(state, output_schema)

        if is_manual_tracing():
            tracer = get_tracer()

            # Emit agent output span
            tracer.emit_agent_output(output)

            # Update agent run span with output (if we have reference)
            if hasattr(state, "_agent_span") and hasattr(state, "_agent_attrs"):
                state._agent_attrs.output = output
                state._agent_span.set_attribute("output", str(output))

        return output

    return terminate_node


def _extract_output(state: AgentGraphState, output_schema: Type[OutputT] | None):
    """Extract output from state based on schema."""
    # Implementation based on existing terminate_node logic
    # ... (keep existing logic)
    pass
```

### PR 2 Deliverables

| File | Changes |
|------|---------|
| `_tracing/serialize.py` | Message serialization + token extraction |
| `_tracing/schema.py` | Full attribute dataclasses (AgentRunAttributes, CompletionAttributes, ToolCallAttributes) |
| `agent/react/llm_node.py` | Full attributes on LLM span |
| `agent/tools/tool_node.py` | Add tool type detection + instrumentation |
| `agent/react/agent.py` | Agent run span wrapper with full attributes |
| `agent/react/terminate_node.py` | Agent output span emission |

### PR 2 Testing

```python
# tests/tracing/test_llm_instrumentation.py
@pytest.mark.parametrize("tracing_mode", [TracingMode.MANUAL, TracingMode.OPENINFERENCE])
async def test_llm_node_creates_correct_spans(tracing_mode):
    """Verify LLM node creates correct span structure."""
    set_tracing_mode(tracing_mode)

    # Run LLM node
    result = await llm_node(state)

    # Get exported spans
    spans = get_exported_spans()

    if tracing_mode == TracingMode.MANUAL:
        assert any(s.name == "LLM call" for s in spans)
        llm_span = next(s for s in spans if s.name == "LLM call")
        assert llm_span.attributes["type"] == "completion"
        assert "usage" in llm_span.attributes
```

---

## PR 3: Guardrails, Context Grounding & JSON Comparison Tests

> **Scope**: Complete instrumentation for guardrails, context grounding. Add JSON comparison test infrastructure to verify C# parity.

### 3.1 Guardrail Instrumentation

**File:** `src/uipath_langchain/agent/guardrails/guardrail_nodes.py` (modified)

```python
"""Guardrail nodes with manual instrumentation."""
from uipath_langchain._tracing import (
    SpanName,
    SpanType,
    get_tracer,
    is_manual_tracing,
)


async def create_guardrail_node(
    guardrail,
    action,
    stage: str,  # "agent_pre", "agent_post", "llm_pre", "llm_post", "tool_pre", "tool_post"
):
    """Create guardrail node with instrumentation."""

    # Map stage to span type and name
    stage_mapping = {
        "agent_pre": (SpanType.AGENT_PRE_GUARDRAILS, SpanName.AGENT_INPUT_GUARDRAIL),
        "agent_post": (SpanType.AGENT_POST_GUARDRAILS, SpanName.AGENT_OUTPUT_GUARDRAIL),
        "llm_pre": (SpanType.LLM_PRE_GUARDRAILS, SpanName.LLM_INPUT_GUARDRAIL),
        "llm_post": (SpanType.LLM_POST_GUARDRAILS, SpanName.LLM_OUTPUT_GUARDRAIL),
        "tool_pre": (SpanType.TOOL_PRE_GUARDRAILS, SpanName.TOOL_INPUT_GUARDRAIL),
        "tool_post": (SpanType.TOOL_POST_GUARDRAILS, SpanName.TOOL_OUTPUT_GUARDRAIL),
    }

    span_type, span_name = stage_mapping[stage]
    governance_type = SpanType.PRE_GOVERNANCE if "pre" in stage else SpanType.POST_GOVERNANCE
    governance_name = SpanName.PRE_GOVERNANCE if "pre" in stage else SpanName.POST_GOVERNANCE

    async def guardrail_node(state):
        if is_manual_tracing():
            tracer = get_tracer()

            # Guardrail wrapper span
            with tracer.start_guardrail(span_type, span_name) as (guard_span, guard_attrs):
                # Nested governance span
                with tracer.start_guardrail(governance_type, governance_name) as (gov_span, gov_attrs):
                    # Execute guardrail
                    result = await guardrail.evaluate(state)

                    if result.triggered:
                        await action.execute(state, result)

                    return state
        else:
            # Original behavior
            result = await guardrail.evaluate(state)
            if result.triggered:
                await action.execute(state, result)
            return state

    return guardrail_node
```

### 3.2 Context Grounding Instrumentation

**File:** `src/uipath_langchain/retrievers/context_grounding_retriever.py` (modified)

```python
"""Context grounding retriever with manual instrumentation."""
from uipath_langchain._tracing import (
    ContextGroundingAttributes,
    SpanName,
    SpanType,
    get_tracer,
    is_manual_tracing,
)


class UiPathContextGroundingRetriever:
    """Retriever with manual instrumentation matching C# ContextGroundingToolWorkflow."""

    async def _aretrieve(self, query: str, **kwargs) -> list:
        """Retrieve with optional instrumentation."""
        if is_manual_tracing():
            return await self._aretrieve_instrumented(query, **kwargs)
        return await self._aretrieve_original(query, **kwargs)

    async def _aretrieve_instrumented(self, query: str, **kwargs) -> list:
        """Retrieve with manual instrumentation."""
        tracer = get_tracer()

        attrs = ContextGroundingAttributes(
            retrieval_mode=kwargs.get("retrieval_mode", "semantic"),
            query=query,
            threshold=kwargs.get("threshold"),
            number_of_results=kwargs.get("top_k"),
            index_id=self.index_id,
        )

        with tracer._tracer.start_as_current_span(
            f"Context Grounding - {self.index_name}",
            kind=SpanKind.INTERNAL,
        ) as span:
            # Set input attributes
            for key, value in attrs.to_dict().items():
                span.set_attribute(key, value if not isinstance(value, (dict, list)) else json.dumps(value))

            # Execute retrieval
            results = await self._aretrieve_original(query, **kwargs)

            # Set output
            attrs.results = [doc.page_content for doc in results]
            span.set_attribute("results", json.dumps(attrs.results))

            return results
```

### 3.3 JSON Comparison Test Infrastructure

**File:** `tests/tracing/conftest.py`

```python
"""Test fixtures for tracing tests."""
import json
from dataclasses import dataclass
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture
def span_exporter():
    """Create in-memory span exporter for testing."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


@pytest.fixture
def exported_spans(span_exporter):
    """Get exported spans as list."""
    def _get_spans():
        return span_exporter.get_finished_spans()
    return _get_spans


def spans_to_json(spans) -> list[dict]:
    """Convert spans to JSON-comparable format."""
    result = []
    for span in spans:
        span_dict = {
            "name": span.name,
            "parent_id": format(span.parent.span_id, "016x") if span.parent else None,
            "status": span.status.status_code.name,
            "attributes": dict(span.attributes) if span.attributes else {},
        }
        result.append(span_dict)
    return result
```

**File:** `tests/tracing/test_json_comparison.py`

```python
"""JSON comparison tests - spans should match C# Agents output."""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from tests.tracing.conftest import spans_to_json


def normalize_span_for_comparison(span: dict) -> dict:
    """Normalize span for comparison (remove IDs, timestamps)."""
    normalized = deepcopy(span)

    # Remove volatile fields
    volatile_fields = ["id", "trace_id", "parent_id", "start_time", "end_time"]
    for field in volatile_fields:
        normalized.pop(field, None)

    # Sort attributes for consistent comparison
    if "attributes" in normalized:
        normalized["attributes"] = dict(sorted(normalized["attributes"].items()))

    return normalized


def deep_compare_spans(actual: list[dict], expected: list[dict]) -> tuple[bool, str]:
    """Deep compare span lists, ignoring IDs.

    Returns:
        (match: bool, diff_message: str)
    """
    if len(actual) != len(expected):
        return False, f"Span count mismatch: {len(actual)} vs {len(expected)}"

    # Sort by name for consistent comparison
    actual_sorted = sorted(actual, key=lambda s: s.get("name", ""))
    expected_sorted = sorted(expected, key=lambda s: s.get("name", ""))

    for i, (a, e) in enumerate(zip(actual_sorted, expected_sorted)):
        a_norm = normalize_span_for_comparison(a)
        e_norm = normalize_span_for_comparison(e)

        if a_norm != e_norm:
            return False, f"Span {i} mismatch:\nActual: {json.dumps(a_norm, indent=2)}\nExpected: {json.dumps(e_norm, indent=2)}"

    return True, ""


class TestSpanJsonComparison:
    """Test that manual instrumentation produces identical JSON to C# Agents."""

    @pytest.fixture
    def expected_agent_run_spans(self):
        """Expected spans for a simple agent run."""
        return [
            {
                "name": "Agent run - TestAgent",
                "attributes": {
                    "type": "agentRun",
                    "agentName": "TestAgent",
                    "source": "langchain",
                },
            },
            {
                "name": "Agent input guardrail check",
                "attributes": {"type": "agentPreGuardrails"},
            },
            {
                "name": "Pre-execution governance",
                "attributes": {"type": "preGovernance"},
            },
            {
                "name": "LLM call",
                "attributes": {
                    "type": "completion",
                    "model": "gpt-4",
                },
            },
            {
                "name": "LLM input guardrail check",
                "attributes": {"type": "llmPreGuardrails"},
            },
            {
                "name": "Pre-execution governance",
                "attributes": {"type": "preGovernance"},
            },
            {
                "name": "Model run",
                "attributes": {"type": "completion"},
            },
            {
                "name": "LLM output guardrail check",
                "attributes": {"type": "llmPostGuardrails"},
            },
            {
                "name": "Post-execution governance",
                "attributes": {"type": "postGovernance"},
            },
            {
                "name": "Agent output guardrail check",
                "attributes": {"type": "agentPostGuardrails"},
            },
            {
                "name": "Post-execution governance",
                "attributes": {"type": "postGovernance"},
            },
            {
                "name": "Agent output",
                "attributes": {"type": "agentOutput"},
            },
        ]

    async def test_agent_run_span_structure(self, exported_spans, expected_agent_run_spans):
        """Test that agent run produces correct span structure."""
        # Run agent
        # ... (test setup)

        actual = spans_to_json(exported_spans())
        match, diff = deep_compare_spans(actual, expected_agent_run_spans)

        assert match, diff

    async def test_llm_call_attributes(self, exported_spans):
        """Test LLM call span has correct attributes."""
        # Run LLM call
        # ...

        spans = exported_spans()
        llm_span = next(s for s in spans if s.name == "LLM call")

        # Must have these attributes (matching C# CompletionSpanAttributes)
        assert llm_span.attributes["type"] == "completion"
        assert "model" in llm_span.attributes
        assert "usage" in llm_span.attributes

        usage = json.loads(llm_span.attributes["usage"])
        assert "promptTokens" in usage  # camelCase like C#
        assert "completionTokens" in usage
        assert "totalTokens" in usage

    async def test_tool_call_attributes(self, exported_spans):
        """Test tool call span has correct attributes."""
        # Run tool
        # ...

        spans = exported_spans()
        tool_span = next(s for s in spans if "Tool call" in s.name)

        # Must have these attributes (matching C# ToolCallSpanAttributes)
        assert tool_span.attributes["type"] == "toolCall"
        assert "callId" in tool_span.attributes
        assert "toolName" in tool_span.attributes
        assert "toolType" in tool_span.attributes
        assert "arguments" in tool_span.attributes


class TestCSharpCompatibility:
    """Test exact compatibility with C# Agents trace output."""

    @pytest.fixture
    def csharp_reference_traces(self):
        """Load reference traces exported from C# Agents."""
        path = Path(__file__).parent / "fixtures" / "csharp_agent_traces.json"
        with open(path) as f:
            return json.load(f)

    async def test_matches_csharp_output(self, exported_spans, csharp_reference_traces):
        """Verify Python traces match C# traces exactly (except IDs)."""
        # Run equivalent agent
        # ...

        actual = spans_to_json(exported_spans())
        expected = csharp_reference_traces["spans"]

        match, diff = deep_compare_spans(actual, expected)
        assert match, f"Python traces don't match C# reference:\n{diff}"
```

**File:** `tests/tracing/fixtures/csharp_agent_traces.json`

```json
{
  "description": "Reference traces from C# Agents for compatibility testing",
  "agent_name": "TestAgent",
  "spans": [
    {
      "name": "Agent run - TestAgent",
      "attributes": {
        "type": "agentRun",
        "agentId": "...",
        "agentName": "TestAgent",
        "agentVersion": "1.0.0",
        "systemPrompt": "You are a helpful assistant.",
        "userPrompt": "Hello",
        "input": {},
        "output": null,
        "source": "langchain"
      }
    },
    {
      "name": "Agent input guardrail check",
      "attributes": {
        "type": "agentPreGuardrails"
      }
    },
    {
      "name": "Pre-execution governance",
      "attributes": {
        "type": "preGovernance"
      }
    },
    {
      "name": "LLM call",
      "attributes": {
        "type": "completion",
        "model": "gpt-4o-2024-11-20",
        "settings": {
          "maxTokens": 4096,
          "temperature": 0.7
        }
      }
    },
    {
      "name": "Model run",
      "attributes": {
        "type": "completion",
        "model": "gpt-4o-2024-11-20",
        "content": "Hello! How can I help you?",
        "toolCalls": null,
        "usage": {
          "promptTokens": 15,
          "completionTokens": 8,
          "totalTokens": 23
        }
      }
    },
    {
      "name": "Agent output guardrail check",
      "attributes": {
        "type": "agentPostGuardrails"
      }
    },
    {
      "name": "Post-execution governance",
      "attributes": {
        "type": "postGovernance"
      }
    },
    {
      "name": "Agent output",
      "attributes": {
        "type": "agentOutput",
        "output": "Hello! How can I help you?"
      }
    }
  ]
}
```

### 3.4 Integration Test

**File:** `tests/tracing/test_integration.py`

```python
"""Integration tests comparing OpenInference vs Manual instrumentation."""
import pytest

from uipath_langchain._tracing import TracingMode, set_tracing_mode


class TestTracingModeComparison:
    """Test that both modes produce functionally equivalent results."""

    @pytest.fixture(params=[TracingMode.OPENINFERENCE, TracingMode.MANUAL])
    def tracing_mode(self, request):
        """Parameterize tests to run with both modes."""
        set_tracing_mode(request.param)
        yield request.param

    async def test_agent_executes_correctly(self, tracing_mode, agent, input_data):
        """Agent produces same output regardless of tracing mode."""
        result = await agent.ainvoke(input_data)

        # Output should be identical
        assert result["output"] == "expected_output"

    async def test_manual_mode_fewer_spans(self, exported_spans):
        """Manual mode should produce fewer, more focused spans."""
        set_tracing_mode(TracingMode.OPENINFERENCE)
        # Run agent
        openinference_spans = exported_spans()

        set_tracing_mode(TracingMode.MANUAL)
        # Run agent
        manual_spans = exported_spans()

        # Manual should have fewer spans (no init, route_agent, terminate noise)
        assert len(manual_spans) < len(openinference_spans)

        # Manual should have all meaningful spans
        manual_names = {s.name for s in manual_spans}
        assert "Agent run - TestAgent" in manual_names
        assert "LLM call" in manual_names
        assert "Agent output" in manual_names

        # Manual should NOT have noise spans
        assert not any("init" in s.name for s in manual_spans)
        assert not any("route_agent" in s.name for s in manual_spans)
```

### PR 3 Deliverables

| File | Purpose |
|------|---------|
| `agent/guardrails/guardrail_nodes.py` | Guardrail instrumentation |
| `retrievers/context_grounding_retriever.py` | Context grounding instrumentation |
| `tests/tracing/conftest.py` | Test fixtures |
| `tests/tracing/test_json_comparison.py` | JSON comparison tests |
| `tests/tracing/fixtures/csharp_agent_traces.json` | C# reference traces |
| `tests/tracing/test_integration.py` | Integration tests |

---

## Implementation Checklist

### PR 1: Foundation - Structure Only
- [ ] Create `_tracing/schema.py` with SpanType and SpanName enums
- [ ] Create `_tracing/tracer.py` with `start_agent_run` and `start_llm_call` only
- [ ] Create `_tracing/config.py` with boolean feature flag
- [ ] Update `_tracing/__init__.py` exports
- [ ] Test: Verify span hierarchy is correct (Agent run → LLM call nesting)

### PR 2: Full Attributes + Tool Spans
- [ ] Add `_tracing/serialize.py` with message serialization and token extraction
- [ ] Add full attribute dataclasses to schema.py
- [ ] Modify `llm_node.py` with full attributes (usage, content, toolCalls)
- [ ] Modify `tool_node.py` with tool type detection + instrumentation
- [ ] Modify `agent.py` with agent run span wrapper + full attributes
- [ ] Modify `terminate_node.py` with agent output emission
- [ ] Add tests for LLM span attributes (camelCase, token counts)
- [ ] Add tests for tool span attributes (toolType classification)

### PR 3: Guardrails + Context Grounding + Testing
- [ ] Add guardrail span instrumentation
- [ ] Add context grounding instrumentation
- [ ] Create JSON comparison test infrastructure
- [ ] Create C# reference trace fixtures
- [ ] Add integration tests comparing modes
- [ ] Performance benchmark both modes

---

## Rollout Plan

### Stage 1: Internal Testing
```bash
UIPATH_TRACING_MODE=manual pytest tests/tracing/
```

### Stage 2: Opt-in Beta
- Document feature flag
- Enable for internal agents
- Collect feedback

### Stage 3: Default Migration
- Change default from `openinference` to `manual`
- Keep `openinference` available for debugging
- Monitor for issues

### Stage 4: Deprecation
- Mark `openinference` mode as deprecated
- Remove in future major version

---

## Success Criteria

1. **Schema Match**: All span types and attributes match C# Agents exactly
2. **JSON Comparison**: Deep compare of exported JSON passes (ignoring IDs)
3. **Performance**: Manual mode has lower overhead than OpenInference
4. **Functionality**: Agent behavior identical in both modes
5. **Token Counting**: Usage metadata captured correctly
6. **Tool Types**: All tool types (process, integration, context_grounding) identified correctly
