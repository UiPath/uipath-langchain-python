# Tool-Calling Suspend/Resume Sample

This sample demonstrates how agents can suspend execution at specific points, then resume seamlessly when external work completes. This is useful for long-running automations where you don't want to block the agent execution.

## Quick Start

Test the suspend/resume flow using the evaluation command:

```bash
cd samples/tool-calling-suspend-resume

# Step 1: Run to suspend
uv run uipath eval agent-simple evaluations/eval-sets/test_simple_no_auth.json

# Step 2: Resume with input override
uv run uipath eval agent-simple evaluations/eval-sets/test_simple_no_auth.json --resume --input-overrides '{"query": "Test suspend with simple payload"}'
```

## What is Suspend/Resume?

The suspend/resume pattern allows agents to:
- **Suspend** execution at specific points (e.g., when waiting for external work)
- **Persist** their state to disk
- **Resume** execution later when the work completes
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
async def suspend_node(state: State) -> State:
    logger.info("About to suspend execution...")

    # 🔴 Execution SUSPENDS here!
    # State is saved to SQLite checkpoint
    resume_data = interrupt({
        "message": "Waiting for external completion",
        "query": state.query
    })

    # 🟢 This code runs AFTER resume
    logger.info(f"Received resume data: {resume_data}")
    result = f"Completed with resume data: {resume_data}"
    return {"query": state.query, "result": result}
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
│  Returns SUSPENDED status                                       │
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

### Core Files
- **`graph_simple.py`** - Simple agent demonstrating suspend/resume with dict payload
- **`agent-simple.py`** - Symlink to graph_simple.py (referenced by uipath.json)
- **`uipath.json`** - Agent configuration
- **`langgraph.json`** - Graph definition for agent-simple
- **`pyproject.toml`** - Python dependencies

### Evaluation Files
- **`evaluations/eval-sets/test_simple_no_auth.json`** - Test cases for suspend/resume
- **`evaluations/evaluators/contains_evaluator.json`** - Evaluator checking completion

## Running the Sample

### Step 1: Suspend

Run the evaluation without `--resume`:

```bash
uv run uipath eval agent-simple evaluations/eval-sets/test_simple_no_auth.json
```

**What happens**:
- Agent executes and calls `interrupt()` → suspends
- State is saved to `__uipath/state.db`
- Evaluation runtime detects SUSPENDED status
- Process exits

**Expected output**:
```
EVAL RUNTIME: Resume mode: False
🔴 EVAL RUNTIME: DETECTED SUSPENSION
EVAL RUNTIME: Agent returned SUSPENDED status
EVAL RUNTIME: Extracted trigger(s) from suspended execution
✓ Basic suspend/resume with query - No evaluators
```

### Step 2: Resume

Resume execution with the `--resume` flag and provide input override:

```bash
uv run uipath eval agent-simple evaluations/eval-sets/test_simple_no_auth.json --resume --input-overrides '{"query": "Test suspend with simple payload"}'
```

**What happens**:
- Loads state from `__uipath/state.db`
- Continues execution from `interrupt()`
- Agent completes and returns result
- Evaluators run on final output

**Expected output**:
```
EVAL RUNTIME: Resume mode: True
🟢 AGENT NODE: Execution RESUMED after interrupt()
AGENT NODE: Received resume data: ...
✓ Basic suspend/resume with query - Completed with resume data
```

## Key Components

### AsyncSqliteSaver

Persists checkpoints to SQLite:

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async def _create_graph():
    checkpointer = AsyncSqliteSaver.from_conn_string("__uipath/state.db")
    return builder.compile(checkpointer=checkpointer)
```

### Thread ID

Critical for resume - must match the suspend invocation. The evaluation runtime handles this automatically.

### Command API

The runtime uses the Command API to provide resume data:

```python
from langgraph.types import Command

# Resume with specific data
result = await graph.ainvoke(
    Command(resume={"status": "completed", "output": "success"}),
    config=config
)
```

## Agent Implementation

The core agent (`graph_simple.py`) is simple:

```python
from langgraph.graph import StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel

class Input(BaseModel):
    query: str

class Output(BaseModel):
    result: str

class State(BaseModel):
    query: str
    result: str = ""

async def suspend_node(state: State) -> State:
    """Node that suspends execution."""
    # Interrupt with simple dict (no RPA invocation needed)
    resume_data = interrupt({
        "message": "Waiting for external completion",
        "query": state.query
    })

    # This code executes after resume
    result = f"Completed with resume data: {resume_data}"
    return {"query": state.query, "result": result}

# Build the graph
builder = StateGraph(state_schema=State)
builder.add_node("suspend_node", suspend_node)
builder.add_edge(START, "suspend_node")
builder.add_edge("suspend_node", END)

# Compile with AsyncSqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async def _create_graph():
    checkpointer = AsyncSqliteSaver.from_conn_string("__uipath/state.db")
    return builder.compile(checkpointer=checkpointer)
```

## Troubleshooting

### "No checkpoint found"
- Make sure you ran the suspend step first
- Check that `__uipath/state.db` exists
- Clean state between runs if needed: `rm -rf __uipath/state.db`

### "Agent doesn't suspend"
- Ensure you're using a checkpointer: `builder.compile(checkpointer=...)`
- Check that `interrupt()` is actually called in your code
- Look for SUSPENDED status in the output

### "Resume starts from beginning"
- Use `--resume` flag when resuming
- Verify the state file exists at `__uipath/state.db`

## Next Steps

1. **Run the suspend step**: `uv run uipath eval agent-simple evaluations/eval-sets/test_simple_no_auth.json`
2. **Run the resume step**: `uv run uipath eval agent-simple evaluations/eval-sets/test_simple_no_auth.json --resume --input-overrides '{"query": "Test suspend with simple payload"}'`
3. **Build your own**: Use `graph_simple.py` as a template for your suspend/resume workflows

## Resources

- [LangGraph Interrupts Documentation](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/)
- [UiPath Python SDK](https://github.com/UiPath/uipath-python)
- [AsyncSqliteSaver Reference](https://langchain-ai.github.io/langgraph/reference/checkpoints/)
