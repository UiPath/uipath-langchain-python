"""Step 1: Run agent until it suspends."""

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

    # This won't execute until resume
    logger.info("=" * 80)
    logger.info("🟢 AGENT NODE: Execution RESUMED after interrupt()")
    logger.info(f"AGENT NODE: Received resume data: {resume_data}")
    logger.info("=" * 80)

    result = f"Completed with resume data: {resume_data}"
    return State(query=state.query, result=result)

async def main():
    # Setup database
    db_path = Path("__uipath/state.db")
    db_path.parent.mkdir(exist_ok=True)

    if db_path.exists():
        db_path.unlink()
        logger.info("🧹 Cleaned up old state")

    # Build graph
    builder = StateGraph(state_schema=State)
    builder.add_node("suspend_node", suspend_node)
    builder.add_edge(START, "suspend_node")
    builder.add_edge("suspend_node", END)

    # Compile with AsyncSqliteSaver
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "test-123"}}
        input_data = {"query": "Test suspend and resume"}

        logger.info("=" * 80)
        logger.info("STEP 1: First execution (will suspend)")
        logger.info("=" * 80)

        try:
            result = await graph.ainvoke(input_data, config=config)
            logger.info(f"Result: {result}")

            if "__interrupt__" in result:
                logger.info("✅ SUSPENDED! Interrupt detected")
                logger.info(f"State saved to: {db_path}")
                logger.info(f"State file size: {db_path.stat().st_size} bytes")
                logger.info("")
                logger.info("To resume, run: python test_suspend_step2.py")
            else:
                logger.error("❌ Did not suspend!")

        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
