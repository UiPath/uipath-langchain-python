"""Interactive test script for suspend/resume functionality.

This script guides you through both phases:
1. Execute and suspend
2. Resume with your own data
"""
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
    print(f"\n✓ Status: {result1.status}")
    print(f"✓ Interrupts: {result1.output}")

    if result1.status.name != "SUSPENDED":
        print("\n❌ ERROR: Expected SUSPENDED status")
        await factory.dispose()
        return

    interrupt_id = list(result1.output.keys())[0]
    print(f"\n✓ Got interrupt_id: {interrupt_id}")

    print("\n" + "=" * 80)
    print("PHASE 2: Resume")
    print("=" * 80)

    print("\nThe agent is now suspended and waiting for resume data.")
    print("Enter the data you want to provide (this will be passed to the interrupt() return):")
    user_data = input("> ").strip()

    if not user_data:
        user_data = "Manual test completed"
        print(f"(Using default: '{user_data}')")

    resume_input = {interrupt_id: user_data}
    print(f"\n→ Resuming with: {resume_input}")

    options = UiPathExecuteOptions(resume=True)
    result2 = await runtime.execute(input=resume_input, options=options)

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    print(f"✅ Status: {result2.status}")
    print(f"✅ Final output: {result2.output}")

    if result2.output and "result" in result2.output:
        print(f"\n✓ Agent result: {result2.output['result']}")

    await factory.dispose()
    print("\n🎉 Test completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
