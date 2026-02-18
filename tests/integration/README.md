# Agent Trace Integration Tests

Validates UiPath agent execution traces match expected structure and attributes.

## Structure

```
integration/
├── calculator/          # Calculator agent tests
│   ├── expected_traces/
│   │   ├── golden.json  # Expected trace structure
│   │   └── config.json  # Parity comparison config
│   └── test_traces.py
└── basic/              # Basic agent tests
    └── ...
```

## Running Tests

```bash
# Set required environment variables
export UIPATH_CLIENT_ID="..."
export UIPATH_CLIENT_SECRET="..."
export UIPATH_URL="https://alpha.uipath.com/org/tenant"

# Run all integration tests
uv run pytest tests/integration/ -m e2e

# Run specific agent tests
uv run pytest tests/integration/calculator/ -m e2e
```

## What's Validated

Each test suite validates:
1. **No extra spans** - Actual trace has no unexpected spans
2. **Span hierarchy** - Parent-child relationships match expected structure
3. **Span attributes** - Attributes match golden trace (with configured ignores/formats)

## Creating Tests for a New Agent

### 1. Create the Agent
```bash
# Add agent definition to examples/
examples/{agent_name}/
└── agent.json
```

### 2. Run and Capture Traces
```bash
# Run the agent with trace capture
export LLMOPS_TRACE_FILE=/tmp/trace.json
uv run uipath run examples/{agent_name}/agent.json '{}'

# Copy the trace as golden
mkdir -p tests/integration/{agent_name}/expected_traces/
cp /tmp/trace.json tests/integration/{agent_name}/expected_traces/golden.json
```

### 3. Create Parity Config
Create `tests/integration/{agent_name}/expected_traces/config.json` to configure comparison rules (see Configuration section below).

### 4. Create Test Class
```python
# tests/integration/{agent_name}/test_traces.py
from pathlib import Path
from tests.integration.conftest import AgentTraceTest

EXPECTED = Path(__file__).parent / "expected_traces"

class TestMyAgent(AgentTraceTest):
    GOLDEN = EXPECTED / "golden.json"
    CONFIG = EXPECTED / "config.json"
    AGENT_DIR = "{agent_name}"
    AGENT_INPUT = '{}'  # JSON string for agent input
```

### 5. Run Tests
```bash
uv run pytest tests/integration/{agent_name}/ -m e2e
```

## Configuration (config.json)

The `config.json` file controls how traces are compared:

```json
{
  "description": "Human-readable description",
  "ignore_spans": [],  // Span names to skip entirely

  "fields": {
    "format": {
      "Id": "hex_id",           // Validate format without exact match
      "*Time*": "iso_datetime",  // Wildcards match multiple fields
      "*Key": "uuid"
    },
    "ignore": [
      "OrganizationId",          // Skip these top-level fields
      "TenantId"
    ]
  },

  "attributes": {
    "format": {
      "*Id": "uuid"              // Format validation in Attributes JSON
    },
    "ignore": [
      "output",                  // Skip these in Attributes JSON
      "input.value",             // Supports nested paths
      "usage"
    ]
  }
}
```

### Format Types
- `hex_id` - Hexadecimal ID (e.g., span IDs)
- `uuid` - UUID format
- `iso_datetime` - ISO 8601 timestamp

### Wildcards
- `*Time*` matches `StartTime`, `EndTime`, `UpdatedAt`
- `*Key` matches `ProcessKey`, `JobKey`, `FolderKey`

## Span Type Reference

- `"LLM call"` spans → SpanType: `"llmCall"`
- `"Model run"` spans → SpanType: `"completion"`
