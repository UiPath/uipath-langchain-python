"""Test suspend/resume flow with checkpoint validation.

This script demonstrates and validates the complete suspend/resume lifecycle:
1. Agent executes and suspends at interrupt()
2. State persists to SQLite database
3. Separate process resumes from checkpoint
4. Execution completes with resume data

Run this test:
    uv run python test_suspend_resume_with_validation.py
"""

import asyncio
import logging
import sqlite3
import msgpack
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class State(BaseModel):
    query: str
    result: str = ""

async def suspend_node(state: State) -> State:
    """Node that suspends execution."""
    logger.info("AGENT: Starting suspend_node")
    logger.info(f"AGENT: Query: {state.query}")
    logger.info("AGENT: 🔴 Calling interrupt() - SUSPENDING")

    resume_data = interrupt({"message": "Waiting for completion", "query": state.query})

    logger.info("AGENT: 🟢 RESUMED after interrupt()")
    logger.info(f"AGENT: Resume data: {resume_data}")

    result = f"Completed with: {resume_data}"
    return State(query=state.query, result=result)

def validate_checkpoints(db_path: Path):
    """Validate checkpoint structure and data."""
    logger.info("=" * 80)
    logger.info("VALIDATING CHECKPOINTS")
    logger.info("=" * 80)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Check checkpoint count
    cursor.execute("SELECT COUNT(*) FROM checkpoints")
    count = cursor.fetchone()[0]
    logger.info(f"Checkpoint count: {count}")
    assert count >= 2, f"Expected at least 2 checkpoints, got {count}"

    # Check for interrupt data
    cursor.execute("""
        SELECT value FROM writes
        WHERE channel = '__interrupt__'
        LIMIT 1
    """)
    interrupt_row = cursor.fetchone()
    assert interrupt_row is not None, "No __interrupt__ write found"

    interrupt_data = msgpack.unpackb(interrupt_row[0], raw=False, strict_map_key=False)
    logger.info(f"✅ Interrupt data found: {len(str(interrupt_data))} bytes")

    # Check for resume data
    cursor.execute("""
        SELECT value FROM writes
        WHERE channel = '__resume__'
        LIMIT 1
    """)
    resume_row = cursor.fetchone()

    if resume_row:
        resume_data = msgpack.unpackb(resume_row[0], raw=False, strict_map_key=False)
        logger.info(f"✅ Resume data found: {resume_data}")
        assert isinstance(resume_data, (dict, list)), "Resume data should be dict or list"

    # Check for final result
    cursor.execute("""
        SELECT value FROM writes
        WHERE channel = 'result'
        LIMIT 1
    """)
    result_row = cursor.fetchone()

    if result_row:
        result_data = msgpack.unpackb(result_row[0], raw=False, strict_map_key=False)
        logger.info(f"✅ Final result found: {result_data}")
        assert "Completed with" in result_data, "Result should contain completion message"

    # Validate checkpoint chain
    cursor.execute("""
        SELECT checkpoint_id, parent_checkpoint_id
        FROM checkpoints
        ORDER BY rowid
    """)
    checkpoints = cursor.fetchall()

    logger.info(f"\nCheckpoint chain validation:")
    for idx, (cp_id, parent_id) in enumerate(checkpoints, 1):
        logger.info(f"  {idx}. {cp_id[:20]}... (parent: {parent_id[:20] + '...' if parent_id else 'root'})")

        if idx == 1:
            assert parent_id is None, "First checkpoint should have no parent"
        else:
            assert parent_id is not None, f"Checkpoint {idx} should have parent"

    conn.close()
    logger.info("✅ All checkpoint validations passed!")

async def main():
    """Run complete suspend/resume test with validation."""
    db_path = Path("__uipath/state.db")
    db_path.parent.mkdir(exist_ok=True)

    # Clean up
    if db_path.exists():
        db_path.unlink()
        logger.info("🧹 Cleaned old state\n")

    # Build graph
    builder = StateGraph(state_schema=State)
    builder.add_node("suspend_node", suspend_node)
    builder.add_edge(START, "suspend_node")
    builder.add_edge("suspend_node", END)

    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-123"}}

        # ===== STEP 1: SUSPEND =====
        logger.info("=" * 80)
        logger.info("STEP 1: First execution (will suspend)")
        logger.info("=" * 80)

        input_data = {"query": "Test suspend and resume"}
        result1 = await graph.ainvoke(input_data, config=config)

        logger.info(f"\nResult: {result1}")
        assert "__interrupt__" in result1, "Should have __interrupt__ field"
        assert db_path.exists(), "state.db should exist"

        initial_size = db_path.stat().st_size
        logger.info(f"✅ SUSPENDED! State saved ({initial_size} bytes)")

        # Validate after suspend
        validate_checkpoints(db_path)

        # ===== STEP 2: RESUME =====
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Resume execution (simulating separate process)")
        logger.info("=" * 80)

        resume_data = {"status": "completed", "message": "Process finished"}
        logger.info(f"Resuming with: {resume_data}\n")

        result2 = await graph.ainvoke(Command(resume=resume_data), config=config)

        logger.info(f"\nResult: {result2}")
        assert "result" in result2, "Should have result field"
        assert "Completed with" in result2["result"], "Result should contain completion message"

        final_size = db_path.stat().st_size
        logger.info(f"✅ COMPLETED! Final state size: {final_size} bytes")

        # Validate after resume
        validate_checkpoints(db_path)

        logger.info("\n" + "=" * 80)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
