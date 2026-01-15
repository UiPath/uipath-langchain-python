# Tool-Calling Suspend/Resume Agent

A simple agent demonstrating the suspend/resume pattern for RPA process invocations using LangGraph's `interrupt()` function.

## Overview

This agent calls `interrupt(InvokeProcess(...))` to suspend execution while waiting for an external RPA process to complete. The evaluation runtime detects the suspension and extracts trigger information for resumption.

## Features

- Single-node graph demonstrating tool-calling suspend pattern
- Uses LangGraph's `interrupt()` to suspend execution
- **Two graph variants**:
  - `graph.py`: Uses `InvokeProcess` for real RPA integration
  - `graph_simple.py`: Uses simple dict payload (no authentication required)
- Comprehensive evaluation sets with trajectory validation

## Running Evaluations

```bash
# Navigate to the sample directory
cd samples/tool-calling-suspend-resume

# Run evaluation
uv run uipath eval graph evaluations/eval-sets/test_suspend_resume.json
```

## Evaluation Sets

### `test_simple_no_auth.json`
Quick test using the simple graph variant (no authentication required):
- Tests basic suspend/resume pattern with dict payload
- Validates suspension detection in logs
- Best for initial testing and development

### `test_suspend_resume.json`
Tests with actual RPA invocation:
- Validates agent calls `interrupt()` with proper `InvokeProcess` structure
- Checks for suspension indicators in logs
- Uses both LLM-based trajectory evaluator and contains-based evaluator
- Requires UiPath authentication

### `test_with_evaluators.json`
Tests evaluator execution after completion:
- Modifies the agent to complete without suspending
- Validates that evaluators run and produce scores
- Useful for verifying evaluator configuration

## Architecture

```
graph.py (LangGraph Agent)
    ↓
invoke_process_node → interrupt(InvokeProcess(...))
    ↓
SUSPENDS execution
    ↓
Runtime detects suspension
    ↓
Extracts triggers
    ↓
Skips evaluators (run after resume)
```

## Key Components

- **graph.py**: Main agent with `InvokeProcess` for RPA invocation
- **graph_simple.py**: Simplified agent using dict payload (no auth required)
- **evaluations/**: Evaluation sets for testing suspend/resume behavior
  - **eval-sets/test_simple_no_auth.json**: Quick test without authentication
  - **eval-sets/test_suspend_resume.json**: Full RPA invocation test
- **pyproject.toml**: Package metadata
- **uipath.json**: Agent configuration
- **langgraph.json**: Graph entrypoint definitions

## How It Works

1. **Agent Execution**: Agent runs and reaches `interrupt(InvokeProcess(...))`
2. **Suspension**: LangGraph raises interrupt, runtime detects `SUSPENDED` status
3. **Trigger Extraction**: Runtime extracts trigger with process details
4. **Evaluator Skip**: Evaluators are skipped during suspension
5. **Resume** (when implemented): Process completes, agent resumes
6. **Evaluator Execution**: Evaluators run on final output

## Expected Output

When running the evaluation, you should see:
```
🔴 DETECTED SUSPENSION → Runtime detects status change
📋 Extracted N trigger(s) → Shows trigger details
⏭️ Skipping evaluators → Explains why no evaluation
✅ Passing through triggers → Shows trigger propagation
```

The evaluation result will show `status: SUSPENDED` with trigger information in the output JSON.
