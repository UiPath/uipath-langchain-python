#!/usr/bin/env python3
"""
Test script to demonstrate suspend/resume flow.

This script:
1. Runs the agent which suspends with interrupt(InvokeProcess)
2. Captures the trigger/inbox information
3. Simulates resume by calling the agent again with the trigger payload
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from graph import Input, graph


async def test_suspend_resume():
    """Test the suspend and resume flow."""

    print("\n" + "=" * 80)
    print("STEP 1: Initial execution - agent will SUSPEND")
    print("=" * 80 + "\n")

    # Create a config with thread_id for checkpointing
    config = {
        "configurable": {
            "thread_id": "test-thread-123",
        }
    }

    # Run the agent - it will suspend at interrupt()
    input_data = Input(query="Test suspend and resume with RPA process")

    try:
        result = await graph.ainvoke(input_data.model_dump(), config)  # type: ignore[arg-type]
        print(f"\n✅ First execution result: {result}")
    except Exception as e:
        # LangGraph raises an exception when interrupted
        print(f"\n🔴 Agent SUSPENDED (as expected): {type(e).__name__}")
        print("   This is normal behavior - the agent called interrupt()")

    # Get the current state to see what was captured
    print("\n" + "=" * 80)
    print("STEP 2: Checking agent state after suspension")
    print("=" * 80 + "\n")

    state_snapshot = await graph.aget_state(config)  # type: ignore[arg-type]
    print(f"State values: {state_snapshot.values}")
    print(f"Next node to execute: {state_snapshot.next}")

    # Check if there are any tasks (interrupts) pending
    if state_snapshot.tasks:
        print(f"\n🔴 Found {len(state_snapshot.tasks)} pending task(s):")
        for i, task in enumerate(state_snapshot.tasks, 1):
            print(f"   Task {i}: {task}")
            if hasattr(task, "interrupts") and task.interrupts:
                for interrupt in task.interrupts:
                    print(f"   Interrupt value: {interrupt.value}")

    print("\n" + "=" * 80)
    print("STEP 3: RESUMING execution")
    print("=" * 80 + "\n")

    # Resume by providing a response to the interrupt
    # In a real scenario, this would be the result from the RPA process
    resume_payload = {
        "status": "completed",
        "result": "RPA process completed successfully",
        "data": {"processed": True},
    }

    print(f"Resuming with payload: {json.dumps(resume_payload, indent=2)}")

    # Resume execution - LangGraph will continue from where it left off
    try:
        # Use Command to resume
        result = await graph.ainvoke(Command(resume=resume_payload), config)  # type: ignore[arg-type]
        print(f"\n🟢 Resume execution result: {result}")
    except Exception as e:
        print(f"\n⚠️  Resume attempt: {type(e).__name__}: {e}")

        # Alternative: Just invoke again without any input
        # LangGraph should continue from the checkpoint
        print("\nTrying alternative resume method (invoke without input)...")
        result = await graph.ainvoke(None, config)  # type: ignore[arg-type]
        print(f"\n🟢 Alternative resume result: {result}")

    print("\n" + "=" * 80)
    print("STEP 4: Verification - checking final state")
    print("=" * 80 + "\n")

    final_state = await graph.aget_state(config)  # type: ignore[arg-type]
    print(f"Final state values: {final_state.values}")
    print(f"Next node: {final_state.next}")
    print(f"Pending tasks: {len(final_state.tasks) if final_state.tasks else 0}")

    print("\n✅ Test completed!")


if __name__ == "__main__":
    # Import Command for resume
    from langgraph.types import Command

    asyncio.run(test_suspend_resume())
