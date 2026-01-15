# Tool-Calling Suspend/Resume Sample

This sample demonstrates how agents can suspend execution when invoking external RPA processes, then resume seamlessly when those processes complete. This is useful for long-running automations where you don't want to block the agent execution.

## Quick Start

**Run the interactive demo** to see suspend/resume in action:

```bash
cd samples/tool-calling-suspend-resume
uv run python demo_suspend_resume.py
```

This runs a complete demo showing:
1. Agent executes and calls `interrupt()` → **SUSPENDS**
2. State saves to SQLite (`__uipath/state.db`)
3. Simulates process restart
4. Agent resumes from checkpoint → **COMPLETES**

## What is Suspend/Resume?

The suspend/resume pattern allows agents to:
- **Suspend** execution at specific points (e.g., when calling an RPA process)
- **Persist** their state to disk
- **Resume** execution later when external work completes
- **Continue** seamlessly from where they left off

This is critical for:
- Long-running RPA automations
- Human-in-the-loop workflows
- External API calls with async callbacks
- Multi-step processes across systems

## How It Works

### The `interrupt()` Function

The key is LangGraph's `interrupt()` function:

```python
async def invoke_process_node(state: State) -> State:
    logger.info("About to invoke RPA process...")

    # 🔴 Execution SUSPENDS here!
    # State is saved to SQLite checkpoint
    resume_data = interrupt({
        "message": "Waiting for RPA process",
        "process": "MyProcess"
    })

    # 🟢 This code runs AFTER resume
    logger.info(f"Process completed: {resume_data}")
    return State(result=resume_data)
```

### The Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. SUSPEND PHASE                                                │
├─────────────────────────────────────────────────────────────────┤
│  Agent executes → reaches interrupt()                           │
│         ↓                                                       │
│  LangGraph suspends execution                                   │
│         ↓                                                       │
│  State saved to __uipath/state.db                              │
│         ↓                                                       │
│  Returns SUSPENDED status with triggers                         │
│         ↓                                                       │
│  Python process can safely exit                                 │
└─────────────────────────────────────────────────────────────────┘

        ... time passes, external work completes ...

┌─────────────────────────────────────────────────────────────────┐
│ 2. RESUME PHASE                                                 │
├─────────────────────────────────────────────────────────────────┤
│  New Python process starts                                      │
│         ↓                                                       │
│  Loads state from __uipath/state.db                            │
│         ↓                                                       │
│  Invokes with Command(resume=result_data)                      │
│         ↓                                                       │
│  Execution continues from interrupt()                           │
│         ↓                                                       │
│  Agent completes and returns final result                       │
└─────────────────────────────────────────────────────────────────┘
```

## Files in This Sample

### Core Agent Files
- **`graph.py`** - Agent using real `InvokeProcess` for RPA integration
- **`graph_simple.py`** - Simplified agent using dict payload (no auth needed)
- **`langgraph.json`** - Defines graph entrypoints: `graph` and `agent-simple`

### Demo & Testing
- **`demo_suspend_resume.py`** - 🎬 **START HERE!** Interactive demo of suspend/resume
- **`test_suspend_resume_with_validation.py`** - Comprehensive test with checkpoint assertions
- **`test_suspend_step1.py`** - Suspend step (can be run separately)
- **`test_suspend_step2.py`** - Resume step (can be run separately)
- **`inspect_state.py`** - Utility to decode and inspect checkpoint database

### Configuration
- **`pyproject.toml`** - Python dependencies
- **`uipath.json`** - Agent configuration
- **`evaluations/`** - Evaluation sets for testing suspend/resume behavior

## Running the Demo

### Full Demo (Recommended)
```bash
uv run python demo_suspend_resume.py
```

Runs both suspend and resume steps automatically. Watch the logs to see:
- `🔴 About to call interrupt()` - Execution about to suspend
- `✅ Agent suspended` - State saved to disk
- `🟢 Execution RESUMED` - Agent continues after resume

### Step-by-Step Demo
Run suspend and resume separately to simulate real-world process separation:

```bash
# Terminal 1: Run suspend step
uv run python demo_suspend_resume.py suspend
# Process exits, state saved to __uipath/state.db

# Terminal 2: Run resume step (simulates separate process)
uv run python demo_suspend_resume.py resume
# Loads state and completes execution
```

## Using with UiPath Evaluation Runtime

Test with the evaluation runtime to see how triggers are extracted:

```bash
# Simple variant (no authentication required)
uv run uipath eval agent-simple evaluations/eval-sets/test_simple_no_auth.json

# Full RPA variant (requires authentication)
uv run uipath eval graph evaluations/eval-sets/test_suspend_resume.json
```

Expected output:
```
🔴 DETECTED SUSPENSION → Runtime detects SUSPENDED status
📋 Extracted 1 trigger(s) → Shows InvokeProcess trigger details
⏭️ Skipping evaluators → Evaluators run after resume
✅ Result: SUSPENDED with triggers
```

## Inspecting the State Database

Want to see what's stored in the checkpoint?

```bash
uv run python inspect_state.py
```

This decodes the msgpack data in `__uipath/state.db` and shows:
- Checkpoint chain (parent-child relationships)
- Interrupt data (what was passed to `interrupt()`)
- Resume data (what was passed to `Command(resume=...)`)
- Channel values (state at each checkpoint)

## Key Components

### AsyncSqliteSaver
Persists checkpoints to SQLite:
```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async with AsyncSqliteSaver.from_conn_string("__uipath/state.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
```

### Thread ID
Critical for resume - must match the suspend invocation:
```python
config = {"configurable": {"thread_id": "my-unique-thread-id"}}

# Suspend
await graph.ainvoke(input_data, config=config)

# Resume (same thread_id!)
await graph.ainvoke(Command(resume=resume_data), config=config)
```

### Command API
Provides resume data to the waiting `interrupt()`:
```python
from langgraph.types import Command

# Resume with specific data
result = await graph.ainvoke(
    Command(resume={"status": "completed", "output": "success"}),
    config=config
)
```

## Integration with Evaluations

When using `uipath eval`, the evaluation runtime:
1. Detects when `interrupt()` is called
2. Checks for SUSPENDED status in the result
3. Extracts `InvokeProcess` triggers
4. **Skips evaluators** (they run after resume)
5. Returns SUSPENDED status with triggers to orchestrator

Later, when the RPA process completes:
1. Orchestrator invokes with `--resume` flag
2. Runtime loads checkpoint and resumes
3. Agent completes execution
4. **Evaluators run** on final output

## Two Graph Variants

### 1. graph.py (Production)
Uses real `InvokeProcess` for RPA integration:
```python
from uipath.platform.common import InvokeProcess

invoke_request = InvokeProcess(
    name="TestProcess",
    input_arguments={"query": state.query},
    process_folder_path="Shared",
)
process_output = interrupt(invoke_request)
```

Requires UiPath authentication and proper process setup.

### 2. graph_simple.py (Development)
Uses simple dict for testing without authentication:
```python
resume_data = interrupt({
    "message": "Waiting for completion",
    "query": state.query
})
```

Perfect for local development and testing suspend/resume logic.

## Common Patterns

### Pattern 1: Simple Suspend/Resume
```python
# Suspend with data
result_data = interrupt({"action": "call_api", "url": "..."})

# Resume provides the data
await graph.ainvoke(Command(resume={"status": "ok", "data": "..."}))
```

### Pattern 2: RPA Process Invocation
```python
# Suspend with InvokeProcess
from uipath.platform.common import InvokeProcess

process_output = interrupt(InvokeProcess(
    name="MyProcess",
    input_arguments={"param": "value"}
))

# Resume with process output
await graph.ainvoke(Command(resume={"outputArg": "result"}))
```

### Pattern 3: Multiple Suspensions
```python
# First suspension
data1 = interrupt({"step": 1})
# ... do work with data1 ...

# Second suspension
data2 = interrupt({"step": 2})
# ... do work with data2 ...
```

Each `interrupt()` creates a new checkpoint!

## Troubleshooting

### "No checkpoint found"
- Make sure you ran the suspend step first
- Check that `__uipath/state.db` exists
- Verify you're using the same `thread_id`

### "Agent doesn't suspend"
- Ensure you're using a checkpointer: `graph.compile(checkpointer=...)`
- Check that `interrupt()` is actually called in your code
- Look for `__interrupt__` field in the result

### "Resume starts from beginning"
- Use `Command(resume=data)` not just passing data
- Verify thread_id matches between suspend and resume
- Check that checkpointer points to the same database file

## Next Steps

1. **Run the demo**: `uv run python demo_suspend_resume.py`
2. **Inspect the state**: `uv run python inspect_state.py`
3. **Test with eval runtime**: `uv run uipath eval agent-simple evaluations/eval-sets/test_simple_no_auth.json`
4. **Build your own**: Use `graph_simple.py` as a template

## Resources

- [LangGraph Interrupts Documentation](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/)
- [UiPath Evaluation Runtime](https://github.com/UiPath/uipath-python)
- [AsyncSqliteSaver Reference](https://langchain-ai.github.io/langgraph/reference/checkpoints/)
