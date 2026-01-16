"""Test resume by passing resume data directly (without API triggers)."""
import asyncio
import os
from pathlib import Path

from uipath.runtime import UiPathRuntimeContext
from uipath_langchain.runtime.factory import UiPathLangGraphRuntimeFactory
from uipath.runtime import UiPathExecuteOptions


async def main():
    # Create context and factory
    ctx = UiPathRuntimeContext()
    factory = UiPathLangGraphRuntimeFactory(ctx)

    runtime_id = "test_direct_resume"

    print("=" * 80)
    print("PHASE 1: Initial execution (will suspend)")
    print("=" * 80)

    runtime = await factory.new_runtime(
        entrypoint="agent-simple",
        runtime_id=runtime_id,
    )

    # Execute - will suspend at interrupt()
    result = await runtime.execute(input={"query": "test direct resume"})

    print(f"Status: {result.status}")
    print(f"Output (interrupts): {result.output}")

    # Get the interrupt_id from the output
    interrupt_ids = list(result.output.keys())
    if not interrupt_ids:
        print("ERROR: No interrupts found!")
        return

    interrupt_id = interrupt_ids[0]
    print(f"Interrupt ID: {interrupt_id}")

    print("\n" + "=" * 80)
    print("PHASE 2: Resume with direct input")
    print("=" * 80)

    # Prepare resume data - map interrupt_id to the data we want to provide
    resume_data = {interrupt_id: "MANUAL RESUME DATA"}
    print(f"Providing resume data: {resume_data}")

    # Resume by passing resume data directly as input
    options = UiPathExecuteOptions(resume=True)
    result2 = await runtime.execute(input=resume_data, options=options)

    print(f"\nStatus: {result2.status}")
    print(f"Output: {result2.output}")

    # Cleanup
    await factory.dispose()
    print("\n✅ Test completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
