# Manual Testing Guide for Suspend/Resume

This guide shows how to manually test the suspend/resume functionality using CLI commands.

## Step 1: Initial Execution (Suspend Phase)

Run the agent - it will suspend at the `interrupt()` call:

```bash
uv run uipath run agent-simple --input '{"query": "test manual suspend"}'
```

Expected output:
```
Status: SUSPENDED
Output: {
  'abc123...': {
    'message': 'Waiting for external completion',
    'query': 'test manual suspend'
  }
}
```

The key here is the **interrupt_id** (the long hash like `abc123...`). This is needed for resume.

## Step 2: Inspect What Was Saved

### Check the checkpoint:
```bash
uv run python -c "
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from graph_simple import builder
import asyncio

async def check():
    async with AsyncSqliteSaver.from_conn_string('__uipath/state.db') as saver:
        graph = builder.compile(checkpointer=saver)
        state = await graph.aget_state({'configurable': {'thread_id': 'default'}})
        print('State values:', state.values)
        print('Next tasks:', state.next)

asyncio.run(check())
"
```

### Check triggers in database:
```bash
sqlite3 __uipath/state.db "SELECT runtime_id, interrupt_id FROM __uipath_resume_triggers"
```

## Step 3: Resume Execution

### Option A: Using CLI Resume (If Available)

```bash
# If the uipath CLI supports resume with data:
uv run uipath resume agent-simple \
  --thread-id default \
  --resume-data '{"<interrupt_id>": "MY RESUME DATA"}'
```

Replace `<interrupt_id>` with the actual interrupt ID from Step 1.

### Option B: Using Python Script (Recommended)

Create a resume script:

```bash
cat > test_manual_resume.py << 'EOF'
import asyncio
from uipath.runtime import UiPathRuntimeContext, UiPathExecuteOptions
from uipath_langchain.runtime.factory import UiPathLangGraphRuntimeFactory

async def main():
    # Prompt user for interrupt_id
    print("Enter the interrupt_id from the suspend output:")
    interrupt_id = input("> ").strip()

    print("\nEnter the data you want to provide for resume:")
    resume_data_value = input("> ").strip()

    # Create runtime
    ctx = UiPathRuntimeContext()
    factory = UiPathLangGraphRuntimeFactory(ctx)
    runtime = await factory.new_runtime(entrypoint="agent-simple", runtime_id="default")

    # Resume with provided data
    resume_input = {interrupt_id: resume_data_value}
    options = UiPathExecuteOptions(resume=True)

    print(f"\nResuming with data: {resume_input}")
    result = await runtime.execute(input=resume_input, options=options)

    print(f"\n✅ Status: {result.status}")
    print(f"Output: {result.output}")

    await factory.dispose()

if __name__ == "__main__":
    asyncio.run(main())
EOF

uv run python test_manual_resume.py
```

Example interaction:
```
Enter the interrupt_id from the suspend output:
> abc123def456...

Enter the data you want to provide for resume:
> Completed by manual testing

✅ Status: SUCCESSFUL
Output: {'query': 'test manual suspend', 'result': 'Completed with resume data: Completed by manual testing'}
```

## Step 4: Verify Final State

Check that the execution completed:

```bash
uv run python -c "
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from graph_simple import builder
import asyncio

async def check():
    async with AsyncSqliteSaver.from_conn_string('__uipath/state.db') as saver:
        graph = builder.compile(checkpointer=saver)
        state = await graph.aget_state({'configurable': {'thread_id': 'default'}})
        print('Final state:', state.values)
        print('Next tasks:', state.next)  # Should be empty

asyncio.run(check())
"
```

Expected output:
```
Final state: {'query': 'test manual suspend', 'result': 'Completed with resume data: Completed by manual testing'}
Next tasks: ()
```

## Full End-to-End Test Script

For convenience, here's a complete script that does both phases:

```bash
cat > test_full_cycle.py << 'EOF'
import asyncio
from uipath.runtime import UiPathRuntimeContext, UiPathExecuteOptions
from uipath_langchain.runtime.factory import UiPathLangGraphRuntimeFactory

async def main():
    ctx = UiPathRuntimeContext()
    factory = UiPathLangGraphRuntimeFactory(ctx)
    runtime = await factory.new_runtime(entrypoint="agent-simple", runtime_id="manual_test")

    print("=" * 80)
    print("PHASE 1: Execute and Suspend")
    print("=" * 80)

    result1 = await runtime.execute(input={"query": "test full cycle"})
    print(f"Status: {result1.status}")
    print(f"Interrupts: {result1.output}")

    if result1.status.name != "SUSPENDED":
        print("ERROR: Expected SUSPENDED status")
        return

    interrupt_id = list(result1.output.keys())[0]
    print(f"\n✓ Got interrupt_id: {interrupt_id[:16]}...")

    print("\n" + "=" * 80)
    print("PHASE 2: Resume")
    print("=" * 80)

    user_data = input("Enter data to provide for resume (or press Enter for default): ").strip()
    if not user_data:
        user_data = "Manual test completed"

    resume_input = {interrupt_id: user_data}
    options = UiPathExecuteOptions(resume=True)
    result2 = await runtime.execute(input=resume_input, options=options)

    print(f"\n✅ Status: {result2.status}")
    print(f"Final output: {result2.output}")

    await factory.dispose()

if __name__ == "__main__":
    asyncio.run(main())
EOF

uv run python test_full_cycle.py
```

## Common Issues

### Issue: "No checkpoint found"
- Make sure you're using the same `thread_id` / `runtime_id` for both suspend and resume
- Default is `"default"` for `uipath run`

### Issue: "Field required" validation error
- This was the bug we just fixed - make sure `graph_simple.py` returns a dict, not a State object

### Issue: "No triggers found in database"
- Triggers might have been deleted by a previous failed resume attempt
- Re-run the suspend phase (Step 1)

### Issue: Empty resume data
- Make sure you're providing the correct interrupt_id from the suspend output
- The interrupt_id is the key in the output dict from the suspend phase
