"""
Pexpect-based tests for uipath debug command.

Tests the interactive debugger functionality including:
- Single breakpoint
- Multiple breakpoints
- List breakpoints (l command)
- Remove breakpoint (r command)
- Quit debugger (q command)
- Step mode (s command)

Interactions are recorded and printed at the end of each test.
"""

import sys
from pathlib import Path

import pexpect
import pytest

# Add testcases to path for common imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from testcases.common import PromptTest


# The command to run for all tests
COMMAND = "uv run uipath debug agent --file input.json"
# The debugger prompt
PROMPT = r"> "
# Timeout for expect operations
TIMEOUT = 30

# Expected final value: 10 * 2 + 100 * 3 - 50 + 10 = 320
# (10*2=20, 20+100=120, 120*3=360, 360-50=310, 310+10=320)
EXPECTED_FINAL_VALUE = "320"


def test_single_breakpoint():
    """Test setting and hitting a single breakpoint."""
    test = PromptTest(
        command=COMMAND,
        test_name="debug_single_breakpoint",
        prompt=PROMPT,
        timeout=TIMEOUT,
    )
    try:
        test.start()

        test.send_command("b process_step_2", expect=r"Breakpoint set at: process_step_2")
        test.send_command("c", expect=r"BREAKPOINT.*process_step_2.*before")
        test.send_command("c", expect=r"Debug session completed")

        test.expect_eof()

        # Additional assertions on log file
        output = test.get_output()
        assert "processed_value" in output and EXPECTED_FINAL_VALUE in output, \
            f"Final processed_value of {EXPECTED_FINAL_VALUE} not found"

    except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF) as e:
        print("\nERROR: Pexpect failed.", file=sys.stderr)
        print(f"Failure: {type(e).__name__}", file=sys.stderr)
        print(f"\n--- Output before failure ---\n{test.before}", file=sys.stderr)
        pytest.fail(f"Test failed: {e}")
    finally:
        test.close()


def test_multiple_breakpoints():
    """Test setting and hitting multiple breakpoints."""
    test = PromptTest(
        command=COMMAND,
        test_name="debug_multiple_breakpoints",
        prompt=PROMPT,
        timeout=TIMEOUT,
    )
    try:
        test.start()

        test.send_command("b process_step_2", expect=r"Breakpoint set at: process_step_2")
        test.send_command("b process_step_4", expect=r"Breakpoint set at: process_step_4")
        test.send_command("c", expect=r"BREAKPOINT.*process_step_2.*before")
        test.send_command("c", expect=r"BREAKPOINT.*process_step_4.*before")
        test.send_command("c", expect=r"Debug session completed")

        test.expect_eof()

        # Additional assertions on log file
        output = test.get_output()
        breakpoint_count = output.count("BREAKPOINT")
        assert breakpoint_count >= 2, \
            f"Expected at least 2 breakpoints hit, got {breakpoint_count}"
        assert "processed_value" in output and EXPECTED_FINAL_VALUE in output, \
            f"Final processed_value of {EXPECTED_FINAL_VALUE} not found"

    except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF) as e:
        print("\nERROR: Pexpect failed.", file=sys.stderr)
        print(f"Failure: {type(e).__name__}", file=sys.stderr)
        print(f"\n--- Output before failure ---\n{test.before}", file=sys.stderr)
        pytest.fail(f"Test failed: {e}")
    finally:
        test.close()


def test_list_breakpoints():
    """Test listing active breakpoints with 'l' command."""
    test = PromptTest(
        command=COMMAND,
        test_name="debug_list_breakpoints",
        prompt=PROMPT,
        timeout=TIMEOUT,
    )
    try:
        test.start()

        test.send_command("b process_step_2", expect=r"Breakpoint set at: process_step_2")
        test.send_command("b process_step_3", expect=r"Breakpoint set at: process_step_3")
        test.send_command("l", expect=r"Active breakpoints:")
        test.send_command("c", expect=r"BREAKPOINT.*process_step_2.*before")
        test.send_command("c", expect=r"BREAKPOINT.*process_step_3.*before")
        test.send_command("c", expect=r"Debug session completed")

        test.expect_eof()

        # Additional assertions on log file
        output = test.get_output()
        assert "process_step_2" in output and "process_step_3" in output, \
            "Not all breakpoints shown in list"

    except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF) as e:
        print("\nERROR: Pexpect failed.", file=sys.stderr)
        print(f"Failure: {type(e).__name__}", file=sys.stderr)
        print(f"\n--- Output before failure ---\n{test.before}", file=sys.stderr)
        pytest.fail(f"Test failed: {e}")
    finally:
        test.close()


def test_remove_breakpoint():
    """Test removing a breakpoint with 'r' command."""
    test = PromptTest(
        command=COMMAND,
        test_name="debug_remove_breakpoint",
        prompt=PROMPT,
        timeout=TIMEOUT,
    )
    try:
        test.start()

        test.send_command("b process_step_2", expect=r"Breakpoint set at: process_step_2")
        test.send_command("b process_step_4", expect=r"Breakpoint set at: process_step_4")
        test.send_command("l", expect=r"Active breakpoints:")
        test.send_command("r process_step_2", expect=r"Breakpoint removed: process_step_2")
        test.send_command("l", expect=r"process_step_4")
        # Now, continue and ensure we ONLY stop at step_4 (not step_2)
        test.send_command("c", expect=r"BREAKPOINT.*process_step_4.*before")
        test.send_command("c", expect=r"Debug session completed")

        test.expect_eof()

    except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF) as e:
        print("\nERROR: Pexpect failed.", file=sys.stderr)
        print(f"Failure: {type(e).__name__}", file=sys.stderr)
        print(f"\n--- Output before failure ---\n{test.before}", file=sys.stderr)
        pytest.fail(f"Test failed: {e}")
    finally:
        test.close()


def test_quit_debugger():
    """Test quitting the debugger early with 'q' command."""
    test = PromptTest(
        command=COMMAND,
        test_name="debug_quit",
        prompt=PROMPT,
        timeout=TIMEOUT,
    )
    try:
        test.start()

        test.send_command("b process_step_3", expect=r"Breakpoint set at: process_step_3")
        test.send_command("c", expect=r"BREAKPOINT.*process_step_3.*before")
        # Quit - no specific output expected, just EOF
        test.send_command("q")

        test.expect_eof()

        # Additional assertions on log file
        output = test.get_output()

        # Steps 1 and 2 should have executed before the breakpoint
        assert "step_1_double" in output, "step_1 did not execute before quit"
        assert "step_2_add_100" in output, "step_2 did not execute before quit"

        # Step 3 should NOT have executed (we quit at the breakpoint BEFORE step_3)
        assert "step_3_multiply_3" not in output, \
            "step_3 should not have executed - quit was before step_3"

    except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF) as e:
        print("\nERROR: Pexpect failed.", file=sys.stderr)
        print(f"Failure: {type(e).__name__}", file=sys.stderr)
        print(f"\n--- Output before failure ---\n{test.before}", file=sys.stderr)
        pytest.fail(f"Test failed: {e}")
    finally:
        test.close()


def test_step_mode():
    """Test step mode - breaks on every node."""
    test = PromptTest(
        command=COMMAND,
        test_name="debug_step_mode",
        prompt=PROMPT,
        timeout=TIMEOUT,
    )
    try:
        test.start()

        test.send_command("s", expect=r"BREAKPOINT.*prepare_input.*before")
        test.send_command("s", expect=r"BREAKPOINT.*process_step_1.*before")
        test.send_command("s", expect=r"BREAKPOINT.*process_step_2.*before")
        test.send_command("s", expect=r"BREAKPOINT.*process_step_3.*before")
        test.send_command("s", expect=r"BREAKPOINT.*process_step_4.*before")
        test.send_command("s", expect=r"BREAKPOINT.*process_step_5.*before")
        test.send_command("s", expect=r"BREAKPOINT.*finalize.*before")
        test.send_command("s", expect=r"Debug session completed")

        test.expect_eof()

        # Additional assertions on log file
        output = test.get_output()

        # Count breakpoints - should have 7 (one per node)
        breakpoint_count = output.count("BREAKPOINT")
        assert breakpoint_count >= 7, \
            f"Expected at least 7 breakpoints in step mode, got {breakpoint_count}"

        # Check all steps executed
        assert "step_1_double" in output, "step_1 not found in step mode output"
        assert "step_2_add_100" in output, "step_2 not found in step mode output"
        assert "step_3_multiply_3" in output, "step_3 not found in step mode output"
        assert "step_4_subtract_50" in output, "step_4 not found in step mode output"
        assert "step_5_add_10" in output, "step_5 not found in step mode output"

        # Check final value
        assert "processed_value" in output and EXPECTED_FINAL_VALUE in output, \
            f"Final processed_value of {EXPECTED_FINAL_VALUE} not found"

    except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF) as e:
        print("\nERROR: Pexpect failed.", file=sys.stderr)
        print(f"Failure: {type(e).__name__}", file=sys.stderr)
        print(f"\n--- Output before failure ---\n{test.before}", file=sys.stderr)
        pytest.fail(f"Test failed: {e}")
    finally:
        test.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
