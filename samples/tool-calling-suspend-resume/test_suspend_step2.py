"""Step 2: Resume from suspended state."""

import asyncio
import logging
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class State(BaseModel):
    query: str
    result: str = ""

async def suspend_node(state: State) -> State:
    logger.info("=" * 80)
    logger.info("AGENT NODE: Starting suspend_node")
    logger.info(f"AGENT NODE: Received query: {state.query}")
    logger.info("🔴 AGENT NODE: About to call interrupt() - SUSPENDING EXECUTION")
    logger.info("=" * 80)

    # Interrupt with simple dict
    resume_data = interrupt({"message": "Waiting for external completion", "query": state.query})

    # This WILL execute after resume
    logger.info("=" * 80)
    logger.info("🟢 AGENT NODE: Execution RESUMED after interrupt()")
    logger.info(f"AGENT NODE: Received resume data: {resume_data}")
    logger.info("=" * 80)

    result = f"Completed with resume data: {resume_data}"
    return State(query=state.query, result=result)

async def main():
    # Setup database
    db_path = Path("__uipath/state.db")

    if not db_path.exists():
        logger.error("❌ No state.db found! Run test_suspend_step1.py first")
        return

    logger.info(f"Loading state from: {db_path} ({db_path.stat().st_size} bytes)")

    # Build graph (same as step 1)
    builder = StateGraph(state_schema=State)
    builder.add_node("suspend_node", suspend_node)
    builder.add_edge(START, "suspend_node")
    builder.add_edge("suspend_node", END)

    # Compile with same checkpointer
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        # SAME config as step 1 (thread_id must match!)
        config = {"configurable": {"thread_id": "test-123"}}

        logger.info("=" * 80)
        logger.info("STEP 2: Resume execution")
        logger.info("=" * 80)

        try:
            # To resume, we need to pass None as input and use Command to provide resume data
            # The resume data should be provided via Command API
            resume_data = {"status": "completed", "message": "Process finished successfully"}
            logger.info(f"Resuming with data: {resume_data}")

            # Option 1: Resume with None (will continue from checkpoint)
            # result = await graph.ainvoke(None, config=config)

            # Option 2: Update state via update_state and then resume
            # First, let's try just resuming with Command
            from langgraph.types import Command

            result = await graph.ainvoke(Command(resume=resume_data), config=config)
            logger.info(f"Result: {result}")

            if "result" in result:
                logger.info(f"✅ COMPLETED! Final result: {result['result']}")
            else:
                logger.warning("⚠️ Result doesn't contain 'result' field")

        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
