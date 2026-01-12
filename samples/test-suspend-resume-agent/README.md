# Test Suspend/Resume Agent

A simple test agent that demonstrates the suspend/resume pattern for RPA process invocations with comprehensive logging.

## Overview

This agent has a single node that calls `interrupt(InvokeProcess(...))` to suspend execution while waiting for an external RPA process to complete.

## Features

- Single-node graph for simple testing
- Uses LangGraph's `interrupt()` to suspend execution
- Demonstrates proper `InvokeProcess` structure
- Includes checkpointer for interrupt support
- **Comprehensive logging** to visualize suspend/resume flow

## Running the Agent

```bash
# Run evaluation with detailed logging
cd /home/chibionos/r2/uipath-langchain-python/samples/test-suspend-resume-agent
uv run --with ../../. uipath eval agent evaluations/test_suspend_resume.json
```

## Logging Output

The enhanced logging clearly shows the suspend/resume lifecycle:

### 1. **Execution Start**
```
================================================================================
EVAL RUNTIME: Starting evaluation execution
EVAL RUNTIME: Execution ID: <uuid>
EVAL RUNTIME: Job ID: None
EVAL RUNTIME: Resume mode: False
================================================================================
```

### 2. **Agent Suspension**
```
==============================================================================
AGENT NODE: Starting invoke_process_node
AGENT NODE: Received query: Test suspend and resume with RPA process
AGENT NODE: Created InvokeProcess request: {...}
🔴 AGENT NODE: About to call interrupt() - SUSPENDING EXECUTION
==============================================================================
```

### 3. **Runtime Detects Suspension**
```
================================================================================
🔴 EVAL RUNTIME: DETECTED SUSPENSION for eval 'Basic suspend test'
EVAL RUNTIME: Agent returned SUSPENDED status
EVAL RUNTIME: Extracted 2 trigger(s) from suspended execution
EVAL RUNTIME: Trigger 1: {interruptId, triggerType, apiResume...}
EVAL RUNTIME: Trigger 2: {interruptId, triggerType, apiResume...}
================================================================================
```

### 4. **Trigger Pass-Through**
```
================================================================================
EVAL RUNTIME: Collecting triggers from all evaluation runs
EVAL RUNTIME: ✅ Passing through 4 trigger(s) to top-level result
EVAL RUNTIME: Pass-through trigger 1: {...}
EVAL RUNTIME: Pass-through trigger 2: {...}
...
================================================================================
```

### 5. **Resume (when --resume flag is used)**
```
================================================================================
🟢 EVAL RUNTIME: RESUME MODE ENABLED - Will resume from suspended state
================================================================================
...
🟢 AGENT NODE: Execution RESUMED after interrupt()
AGENT NODE: RPA process has completed
AGENT NODE: Returning result for query: ...
```

## Testing Suspend/Resume

The agent will:
1. Accept a query input
2. Create an `InvokeProcess` request
3. Call `interrupt()` to suspend execution (🔴 logged)
4. Return SUSPENDED status with trigger information

The serverless executor will then:
1. Detect the SUSPENDED status (🔴 logged)
2. Extract the trigger information (logged with details)
3. Start the RPA job
4. Wait for completion
5. Resume execution with `--resume` flag (🟢 logged)

## Key Logging Symbols

- 🔴 **Red dot**: Suspension point
- 🟢 **Green dot**: Resume point
- ✅ **Checkmark**: Successful trigger pass-through
- `====`: Section separators for clarity
