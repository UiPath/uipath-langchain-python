# Observability

This directory contains the observability infrastructure for UiPath Agents, including telemetry, tracing, and span exporters.

## LlmOpsFileExporter

The `LlmOpsFileExporter` exports trace spans to a local file in the internal UiPath format - the same format used by `LlmOpsHttpExporter` when sending traces to the UiPath platform.

**Note:** This exporter is only available for local development/debugging.

### Usage

Use the `LLMOPS_TRACE_FILE` environment variable to enable file export:

```bash
# Export traces in UiPath internal format
LLMOPS_TRACE_FILE=traces.jsonl uv run uipath run agent.json '{}'
```

### Using Both Exporters

You can export traces in both OpenTelemetry format (via `--trace-file`) and UiPath internal format (via `LLMOPS_TRACE_FILE`) simultaneously:

```bash
LLMOPS_TRACE_FILE=internal_traces.jsonl uv run uipath run agent.json '{}' --trace-file otel_traces.jsonl
```

This produces:
- `otel_traces.jsonl` - OpenTelemetry/LangGraph schema format
- `internal_traces.jsonl` - UiPath internal schema format

### Output Format

The exporter writes spans in JSON Lines format (one JSON object per line). Each span follows the internal UiPath schema:

```json
{
  "Id": "41850f68-19b5-429d-807a-4bc55b891ed9",
  "TraceId": "6d85e05c-ec91-41e4-8f68-b4b0d3c5d4b4",
  "ParentId": "c6d388a8-198d-40af-b8d0-bfd6b15cdc38",
  "Name": "Model run",
  "StartTime": "2026-01-14T20:23:09.973680",
  "EndTime": "2026-01-14T20:23:11.464381",
  "SpanType": "completion",
  "Status": 1,
  "Attributes": "{\"type\": \"completion\", \"model\": \"gpt-4.1-2025-04-14\", ...}",
  "OrganizationId": "...",
  "TenantId": "...",
  "FolderKey": "...",
  "ExecutionType": 1
}
```

### Span Types

The exporter produces spans with types defined in `schema.py`. See the `SpanType` enum in [schema.py](./schema.py) for the complete and authoritative list of span types.

Common span types include:
- `agentRun` - Root span for agent execution
- `llmCall` / `completion` - LLM calls and API execution
- `toolCall` - Tool execution (including processTool, integrationTool, contextGroundingTool, mcpTool)
- `agentOutput` - Final agent output
- Guardrail spans - Pre/post guardrails for agent, LLM, and tool scopes

### Format Comparison

#### OpenTelemetry Format (`--trace-file`)

Standard OpenTelemetry JSON format with OpenInference attributes:

```json
{
  "name": "Model run",
  "context": {
    "trace_id": "0x8a0836fc0b39756f0ffb0437a01e7d78",
    "span_id": "0xec61b0a660066f67"
  },
  "kind": "SpanKind.INTERNAL",
  "attributes": {
    "telemetry.filter": "drop",
    "input.value": "...",
    "openinference.span.kind": "CHAIN"
  }
}
```

#### UiPath Internal Format (`LLMOPS_TRACE_FILE`)

UiPath platform schema with processed attributes:

```json
{
  "Id": "41850f68-19b5-429d-807a-4bc55b891ed9",
  "TraceId": "6d85e05c-ec91-41e4-8f68-b4b0d3c5d4b4",
  "Name": "Model run",
  "SpanType": "completion",
  "Attributes": "{\"type\": \"completion\", \"model\": \"gpt-4.1-2025-04-14\"}",
  "Status": 1,
  "ExecutionType": 1
}
```

### Implementation Details

The `LlmOpsFileExporter`:
- Uses the same conversion logic as `LlmOpsHttpExporter`
- Filters spans marked with `telemetry.filter="drop"`
- Processes attributes according to span type (completion, toolCall, etc.)
- Writes to a JSON Lines file for easy streaming and parsing
- Automatically creates the output directory if it doesn't exist
- Clears the file on initialization to avoid appending to old traces
- Only initializes when running locally (no `job_key` detected)

### Architecture

The exporter is automatically registered in `AgentsRuntimeFactory._setup_instrumentation()` when the `LLMOPS_TRACE_FILE` environment variable is set. It integrates with the OpenTelemetry trace manager and processes spans through the same pipeline as other exporters.

```
┌─────────────────┐
│  UiPath Tracer  │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  Trace Manager      │
│  ┌───────────────┐  │
│  │SourceMarker   │  │  (filters spans)
│  │Processor      │  │
│  └───────────────┘  │
│                     │
│  Exporters:         │
│  ┌───────────────┐  │
│  │LlmOpsHttp     │──┼──► UiPath Platform
│  │Exporter       │  │
│  └───────────────┘  │
│  ┌───────────────┐  │
│  │LlmOpsFile     │──┼──► traces.jsonl
│  │Exporter       │  │    (if LLMOPS_TRACE_FILE set)
│  └───────────────┘  │
└─────────────────────┘
```

### Reading the Output

Since the output is JSON Lines format, you can easily process it with standard tools:

```bash
# Pretty print all spans
cat traces.jsonl | jq '.'

# Filter by span type
cat traces.jsonl | jq 'select(.SpanType == "completion")'

# Extract specific fields
cat traces.jsonl | jq '{Name, SpanType, Status}'

# Count spans by type
cat traces.jsonl | jq -r '.SpanType' | sort | uniq -c
```
