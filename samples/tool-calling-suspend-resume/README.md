# Tool-Calling Suspend/Resume Agent

A simple agent demonstrating the suspend/resume pattern for RPA process invocations using LangGraph's `interrupt()` function.

## Overview

This agent calls `interrupt(InvokeProcess(...))` to suspend execution while waiting for an external RPA process to complete. The evaluation runtime detects the suspension and extracts trigger information for resumption.

## Features

- Single-node graph demonstrating tool-calling suspend pattern
- Uses LangGraph's `interrupt()` to suspend execution
- Proper `InvokeProcess` structure for RPA invocation
- Includes checkpointer (required for interrupts)
- Comprehensive evaluation sets with trajectory validation

## Running Evaluations

```bash
# Navigate to the sample directory
cd samples/tool-calling-suspend-resume

# Run evaluation
uv run uipath eval graph evaluations/eval-sets/test_suspend_resume.json
```

## Evaluation Sets

### `test_suspend_resume.json`
Tests the actual suspend/resume behavior:
- Validates agent calls `interrupt()` with proper `InvokeProcess` structure
- Checks for suspension indicators in logs
- Uses both LLM-based trajectory evaluator and contains-based evaluator

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

- **graph.py**: Main agent with single node that calls `interrupt()`
- **evaluations/**: Evaluation sets and evaluator configurations
  - **eval-sets/**: Test cases for suspend and evaluator testing
  - **evaluators/**: LLM trajectory and contains evaluator configs
- **pyproject.toml**: Package metadata
- **uipath.json**: Agent configuration

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
