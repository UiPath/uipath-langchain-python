# Test Suspend/Resume Agent

A simple test agent that demonstrates the suspend/resume pattern for RPA process invocations.

## Overview

This agent has a single node that calls `interrupt(InvokeProcess(...))` to suspend execution while waiting for an external RPA process to complete.

## Features

- Single-node graph for simple testing
- Uses LangGraph's `interrupt()` to suspend execution
- Demonstrates proper `InvokeProcess` structure
- Includes checkpointer for interrupt support

## Running the Agent

```bash
# Run directly
uipath run graph.py '{"query": "Test suspend and resume"}'

# Run evaluation
uipath eval graph evaluations/test_suspend_resume.json
```

## Testing Suspend/Resume

The agent will:
1. Accept a query input
2. Create an `InvokeProcess` request
3. Call `interrupt()` to suspend execution
4. Return SUSPENDED status with trigger information

The serverless executor will then:
1. Detect the SUSPENDED status
2. Extract the trigger information
3. Start the RPA job
4. Wait for completion
5. Resume execution with the job results
