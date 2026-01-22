## CLI Commands Reference

The UiPath Python SDK provides a comprehensive CLI for managing coded agents and automation projects. All commands should be executed with `uv run uipath <command>`.

### Command Overview

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `init` | Initialize agent project | Creating a new agent or updating schema |
| `run` | Execute agent | Running agent locally or testing |
| `eval` | Evaluate agent | Testing agent performance with evaluation sets |

---

### `uipath init`

**Description:** Initialize the project.

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--no-agents-md-override` | flag | false | Won't override existing .agent files and AGENTS.md file. |

**Usage Examples:**

```bash
# Initialize a new agent project
uv run uipath init

# Initialize with specific entrypoint
uv run uipath init main.py

# Initialize and infer bindings from code
uv run uipath init --infer-bindings
```

**When to use:** Run this command when you've modified the Input/Output models and need to regenerate the `uipath.json` schema file.

---

### `uipath run`

**Description:** Execute the project.

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `entrypoint` | No | N/A |
| `input` | No | N/A |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--resume` | flag | false | Resume execution from a previous state |
| `-f`, `--file` | value | `Sentinel.UNSET` | File path for the .json input |
| `--input-file` | value | `Sentinel.UNSET` | Alias for '-f/--file' arguments |
| `--output-file` | value | `Sentinel.UNSET` | File path where the output will be written |
| `--trace-file` | value | `Sentinel.UNSET` | File path where the trace spans will be written (JSON Lines format) |
| `--state-file` | value | `Sentinel.UNSET` | File path where the state file is stored for persisting execution state. If not provided, a temporary file will be used. |
| `--debug` | flag | false | Enable debugging with debugpy. The process will wait for a debugger to attach. |
| `--debug-port` | value | `5678` | Port for the debug server (default: 5678) |
| `--keep-state-file` | flag | false | Keep the temporary state file even when not resuming and no job id is provided |

**Usage Examples:**

```bash
# Run agent with inline JSON input
uv run uipath run main.py '{"query": "What is the weather?"}'

# Run agent with input from file
uv run uipath run main.py --file input.json

# Run agent and save output to file
uv run uipath run agent '{"task": "Process data"}' --output-file result.json

# Run agent with debugging enabled
uv run uipath run main.py '{"input": "test"}' --debug --debug-port 5678

# Resume agent execution from previous state
uv run uipath run --resume
```

**When to use:** Run this command to execute your agent locally for development, testing, or debugging. Use `--debug` flag to attach a debugger for step-by-step debugging.

---

### `uipath eval`

**Description:** Run an evaluation set against the agent.

    Args:
        entrypoint: Path to the agent script to evaluate (optional, will auto-discover if not specified)
        eval_set: Path to the evaluation set JSON file (optional, will auto-discover if not specified)
        eval_ids: Optional list of evaluation IDs
        eval_set_run_id: Custom evaluation set run ID (optional, will generate UUID if not specified)
        workers: Number of parallel workers for running evaluations
        no_report: Do not report the evaluation results
        enable_mocker_cache: Enable caching for LLM mocker responses
        report_coverage: Report evaluation coverage
        model_settings_id: Model settings ID to override agent settings
        trace_file: File path where traces will be written in JSONL format
        max_llm_concurrency: Maximum concurrent LLM requests
        input_overrides: Input field overrides mapping (direct field override with deep merge)
        resume: Resume execution from a previous suspended state
    

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `entrypoint` | No | N/A |
| `eval_set` | No | N/A |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--eval-set-run-id` | value | `Sentinel.UNSET` | Custom evaluation set run ID (if not provided, a UUID will be generated) |
| `--no-report` | flag | false | Do not report the evaluation results |
| `--workers` | value | `1` | Number of parallel workers for running evaluations (default: 1) |
| `--output-file` | value | `Sentinel.UNSET` | File path where the output will be written |
| `--enable-mocker-cache` | flag | false | Enable caching for LLM mocker responses |
| `--report-coverage` | flag | false | Report evaluation coverage |
| `--model-settings-id` | value | `"default"` | Model settings ID from evaluation set to override agent settings (default: 'default') |
| `--trace-file` | value | `Sentinel.UNSET` | File path where traces will be written in JSONL format |
| `--max-llm-concurrency` | value | `20` | Maximum concurrent LLM requests (default: 20) |
| `--resume` | flag | false | Resume execution from a previous suspended state |

**Usage Examples:**

```bash
# Run evaluation with auto-discovered files
uv run uipath eval

# Run evaluation with specific entrypoint and eval set
uv run uipath eval main.py eval_set.json

# Run evaluation without reporting results
uv run uipath eval --no-report

# Run evaluation with custom number of workers
uv run uipath eval --workers 4

# Save evaluation output to file
uv run uipath eval --output-file eval_results.json
```

**When to use:** Run this command to test your agent's performance against a predefined evaluation set. This helps validate agent behavior and measure quality metrics.

---

### Common Workflows

**1. Creating a New Agent:**
```bash
# Step 1: Initialize project
uv run uipath init

# Step 2: Run agent to test
uv run uipath run main.py '{"input": "test"}'

# Step 3: Evaluate agent performance
uv run uipath eval
```

**2. Development & Testing:**
```bash
# Run with debugging
uv run uipath run main.py '{"input": "test"}' --debug

# Test with input file
uv run uipath run main.py --file test_input.json --output-file test_output.json
```

**3. Schema Updates:**
```bash
# After modifying Input/Output models, regenerate schema
uv run uipath init --infer-bindings
```

### Configuration File (uipath.json)

The `uipath.json` file is automatically generated by `uipath init` and defines your agent's schema and bindings.

**Structure:**

```json
{
  "entryPoints": [
    {
      "filePath": "agent",
      "uniqueId": "uuid-here",
      "type": "agent",
      "input": {
        "type": "object",
        "properties": { ... },
        "description": "Input schema",
        "required": [ ... ]
      },
      "output": {
        "type": "object",
        "properties": { ... },
        "description": "Output schema",
        "required": [ ... ]
      }
    }
  ],
  "bindings": {
    "version": "2.0",
    "resources": []
  }
}
```

**When to Update:**

1. **After Modifying Input/Output Models**: Run `uv run uipath init --infer-bindings` to regenerate schemas
2. **Changing Entry Point**: Update `filePath` if you rename or move your main file
3. **Manual Schema Adjustments**: Edit `input.jsonSchema` or `output.jsonSchema` directly if needed
4. **Bindings Updates**: The `bindings` section maps the exported graph variable - update if you rename your graph

**Important Notes:**

- The `uniqueId` should remain constant for the same agent
- Always use `type: "agent"` for LangGraph agents
- The `jsonSchema` must match your Pydantic models exactly
- Re-run `uipath init --infer-bindings` instead of manual edits when possible

