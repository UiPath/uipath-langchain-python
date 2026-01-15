#!/usr/bin/env python3
"""Interactive demo of suspend/resume flow across separate Python processes.

This script demonstrates the complete suspend/resume lifecycle:
1. Agent executes and suspends at interrupt()
2. State persists to SQLite database (__uipath/state.db)
3. Separate invocation resumes from checkpoint
4. Execution completes with resume data

Run this demo:
    python demo_suspend_resume.py
"""

import asyncio
import logging
import sys
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class State(BaseModel):
    """Agent state containing query and result."""

    query: str
    result: str = ""


async def suspend_node(state: State) -> State:
    """Node that suspends execution using interrupt().

    This demonstrates the suspend/resume pattern:
    1. Logs that it's about to suspend
    2. Calls interrupt() which suspends execution and saves state
    3. When resumed, receives data passed to Command(resume=...)
    4. Returns final result with resume data
    """
    logger.info("=" * 80)
    logger.info("🔴 AGENT: About to call interrupt() - execution will SUSPEND here")
    logger.info(f"AGENT: Current query: {state.query}")
    logger.info("=" * 80)

    # This is where execution suspends!
    # The interrupt() call saves state and returns control
    # Execution only continues past this line when resumed
    resume_data = interrupt(
        {"message": "Waiting for external process", "query": state.query}
    )

    # Everything below this line runs AFTER resume
    logger.info("=" * 80)
    logger.info("🟢 AGENT: Execution RESUMED!")
    logger.info(f"AGENT: Received resume data: {resume_data}")
    logger.info("=" * 80)

    result = f"Query: {state.query} | Resume data: {resume_data}"
    return State(query=state.query, result=result)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


async def run_suspend():
    """Step 1: Run agent until it suspends."""
    print_section("STEP 1: Execute agent until suspension")

    db_path = Path("__uipath/state.db")
    db_path.parent.mkdir(exist_ok=True)

    # Clean up any previous state
    if db_path.exists():
        db_path.unlink()
        logger.info("🧹 Cleaned previous state database\n")

    # Build the graph
    builder = StateGraph(state_schema=State)
    builder.add_node("suspend_node", suspend_node)
    builder.add_edge(START, "suspend_node")
    builder.add_edge("suspend_node", END)

    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        # Use a consistent thread_id - this is crucial for resume!
        config = {"configurable": {"thread_id": "demo-suspend-resume"}}

        input_data = {"query": "What is the meaning of life?"}
        logger.info(f"Invoking agent with input: {input_data}")

        result = await graph.ainvoke(input_data, config=config)

        print("\n" + "-" * 80)
        if "__interrupt__" in result:
            logger.info("✅ SUCCESS: Agent suspended as expected!")
            logger.info(f"   Interrupt data: {result['__interrupt__']}")
            logger.info(f"   State saved to: {db_path} ({db_path.stat().st_size} bytes)")
            print("-" * 80)
            print("\n💡 The agent is now suspended. State has been saved to disk.")
            print("   In a real scenario, this process would exit here.")
            print("   Run this script again to see the resume step.")
            return True
        else:
            logger.error("❌ FAILED: Agent did not suspend!")
            logger.error(f"   Result: {result}")
            return False


async def run_resume():
    """Step 2: Resume from suspended state."""
    print_section("STEP 2: Resume from checkpoint")

    db_path = Path("__uipath/state.db")

    if not db_path.exists():
        logger.error(f"❌ No checkpoint found at {db_path}")
        logger.error("   Please run the suspend step first!")
        return False

    logger.info(f"Loading state from: {db_path} ({db_path.stat().st_size} bytes)")

    # Build the SAME graph (must match!)
    builder = StateGraph(state_schema=State)
    builder.add_node("suspend_node", suspend_node)
    builder.add_edge(START, "suspend_node")
    builder.add_edge("suspend_node", END)

    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        # Use the SAME thread_id as suspend!
        config = {"configurable": {"thread_id": "demo-suspend-resume"}}

        # Provide resume data - this gets passed to the waiting interrupt()
        resume_data = {"status": "completed", "answer": 42}
        logger.info(f"Resuming with data: {resume_data}\n")

        # The magic: Command(resume=...) tells LangGraph to resume
        result = await graph.ainvoke(Command(resume=resume_data), config=config)

        print("\n" + "-" * 80)
        if "result" in result:
            logger.info("✅ SUCCESS: Execution completed!")
            logger.info(f"   Final result: {result['result']}")
            print("-" * 80)
            print(
                "\n💡 The agent has completed execution after resume. "
                "State has been updated."
            )
            return True
        else:
            logger.error("❌ FAILED: No result in output!")
            logger.error(f"   Result: {result}")
            return False


async def run_full_demo():
    """Run complete suspend/resume demo."""
    print("\n" + "🎬" * 40)
    print("  SUSPEND/RESUME DEMO")
    print("  Demonstrating state persistence across Python processes")
    print("🎬" * 40)

    # Step 1: Suspend
    suspend_success = await run_suspend()
    if not suspend_success:
        logger.error("\n❌ Demo failed at suspend step")
        return False

    print("\n⏸️  SIMULATING PROCESS EXIT/RESTART...")
    await asyncio.sleep(1)  # Dramatic pause

    # Step 2: Resume
    resume_success = await run_resume()
    if not resume_success:
        logger.error("\n❌ Demo failed at resume step")
        return False

    print_section("🎉 DEMO COMPLETE!")
    print("Key takeaways:")
    print("  1. interrupt() suspends execution and saves state to SQLite")
    print("  2. State persists across Python process restarts")
    print("  3. Command(resume=data) provides data to waiting interrupt()")
    print("  4. The agent completes execution seamlessly after resume")
    print("\n✨ Check __uipath/state.db to see the persisted checkpoints!")

    return True


def main():
    """Entry point for the demo."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "suspend":
            print("Running suspend step only...")
            success = asyncio.run(run_suspend())
        elif command == "resume":
            print("Running resume step only...")
            success = asyncio.run(run_resume())
        elif command == "help":
            print("Usage:")
            print("  python demo_suspend_resume.py           # Run full demo")
            print("  python demo_suspend_resume.py suspend   # Run suspend step only")
            print("  python demo_suspend_resume.py resume    # Run resume step only")
            print("  python demo_suspend_resume.py help      # Show this help")
            return
        else:
            print(f"Unknown command: {command}")
            print("Run 'python demo_suspend_resume.py help' for usage")
            sys.exit(1)
    else:
        # Run full demo
        success = asyncio.run(run_full_demo())

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
