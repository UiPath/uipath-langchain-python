"""
Pexpect-based tests for uipath debug command.

Tests the interactive debugger functionality including:
- Single breakpoint
- Multiple breakpoints
- List breakpoints (l command)
- Remove breakpoint (r command)
- Quit debugger (q command)
- Step mode (s command)
"""

import re
import pexpect
import sys
import pytest

# The command to run for all tests
COMMAND = "uv run uipath debug agent --file input.json"
# The debugger prompt
PROMPT = r"> "
# Timeout for expect operations
TIMEOUT = 30

# Expected final value: 10 * 2 + 100 * 3 - 50 + 10 = 320
# (10*2=20, 20+100=120, 120*3=360, 360-50=310, 310+10=320)
EXPECTED_FINAL_VALUE = "320"


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def read_log(filename: str) -> str:
    """Read and strip ANSI from log file."""
    with open(filename, 'r', encoding='utf-8') as f:
        return strip_ansi(f.read())


def run_test(interactions, log_file):
    """
    A generic test runner for a sequence of debugger interactions.

    Args:
        interactions (list): A list of (command, expected_response) tuples.
        log_file (str): File to log the complete session output.
    """
    print(f"\n--- Running test, logging to {log_file} ---")

    child = pexpect.spawn(COMMAND, encoding='utf-8', timeout=TIMEOUT)
    try:
        # Log everything to the specified file
        child.logfile_read = open(log_file, "w")

        for i, (command, expected_response) in enumerate(interactions):
            # Wait for the prompt before sending a command
            child.expect(PROMPT)
            print(f"Interaction {i+1}: Sending command '{command}'")
            child.sendline(command)

            # Check for the expected response
            if expected_response:
                print(f"Interaction {i+1}: Expecting '{expected_response}'")
                child.expect(expected_response)

        # After all interactions, wait for the process to end
        print("Waiting for process to complete...")
        child.expect(pexpect.EOF)
        print("--- Test completed successfully ---")

    except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF) as e:
        print("\nERROR: Pexpect failed.", file=sys.stderr)
        print(f"Failure during: pexpect.{type(e).__name__}", file=sys.stderr)
        print("\n--- Child Output (Before Failure) ---", file=sys.stderr)
        print(child.before, file=sys.stderr)
        pytest.fail(f"Test failed in {log_file}: {e}")
    finally:
        child.close()


# === Debug Command Tests ===

def test_single_breakpoint():
    """Test setting and hitting a single breakpoint."""
    interactions = [
        ("b process_step_2", r"Breakpoint set at: process_step_2"),
        ("c", r"BREAKPOINT.*process_step_2.*before"),
        ("c", r"Debug session completed")
    ]
    run_test(interactions, "debug_single_breakpoint.log")

    # Additional assertions on log file
    output = read_log("debug_single_breakpoint.log")
    assert "processed_value" in output and EXPECTED_FINAL_VALUE in output, \
        f"Final processed_value of {EXPECTED_FINAL_VALUE} not found"


def test_multiple_breakpoints():
    """Test setting and hitting multiple breakpoints."""
    interactions = [
        ("b process_step_2", r"Breakpoint set at: process_step_2"),
        ("b process_step_4", r"Breakpoint set at: process_step_4"),
        ("c", r"BREAKPOINT.*process_step_2.*before"),
        ("c", r"BREAKPOINT.*process_step_4.*before"),
        ("c", r"Debug session completed")
    ]
    run_test(interactions, "debug_multiple_breakpoints.log")

    # Additional assertions on log file
    output = read_log("debug_multiple_breakpoints.log")
    breakpoint_count = output.count("BREAKPOINT")
    assert breakpoint_count >= 2, \
        f"Expected at least 2 breakpoints hit, got {breakpoint_count}"
    assert "processed_value" in output and EXPECTED_FINAL_VALUE in output, \
        f"Final processed_value of {EXPECTED_FINAL_VALUE} not found"


def test_list_breakpoints():
    """Test listing active breakpoints with 'l' command."""
    interactions = [
        ("b process_step_2", r"Breakpoint set at: process_step_2"),
        ("b process_step_3", r"Breakpoint set at: process_step_3"),
        ("l", r"Active breakpoints:"),  # Check that list shows breakpoints
        ("c", r"BREAKPOINT.*process_step_2.*before"),
        ("c", r"BREAKPOINT.*process_step_3.*before"),
        ("c", r"Debug session completed")
    ]
    run_test(interactions, "debug_list_breakpoints.log")

    # Additional assertions on log file
    output = read_log("debug_list_breakpoints.log")
    assert "process_step_2" in output and "process_step_3" in output, \
        "Not all breakpoints shown in list"


def test_remove_breakpoint():
    """Test removing a breakpoint with 'r' command."""
    interactions = [
        ("b process_step_2", r"Breakpoint set at: process_step_2"),
        ("b process_step_4", r"Breakpoint set at: process_step_4"),
        ("l", r"Active breakpoints:"),  # Verify both are set
        ("r process_step_2", r"Breakpoint removed: process_step_2"),
        ("l", r"process_step_4"),  # Verify only step_4 is left
        # Now, continue and ensure we ONLY stop at step_4 (not step_2)
        ("c", r"BREAKPOINT.*process_step_4.*before"),
        ("c", r"Debug session completed")
    ]
    run_test(interactions, "debug_remove_breakpoint.log")


def test_quit_debugger():
    """Test quitting the debugger early with 'q' command."""
    interactions = [
        ("b process_step_3", r"Breakpoint set at: process_step_3"),
        ("c", r"BREAKPOINT.*process_step_3.*before"),
        ("q", None)  # No specific output expected, just EOF
    ]
    run_test(interactions, "debug_quit.log")

    # Additional assertions on log file
    output = read_log("debug_quit.log")

    # Steps 1 and 2 should have executed before the breakpoint
    assert "step_1_double" in output, "step_1 did not execute before quit"
    assert "step_2_add_100" in output, "step_2 did not execute before quit"

    # Step 3 should NOT have executed (we quit at the breakpoint BEFORE step_3)
    assert "step_3_multiply_3" not in output, \
        "step_3 should not have executed - quit was before step_3"


def test_step_mode():
    """Test step mode - breaks on every node."""
    interactions = [
        ("s", r"BREAKPOINT.*prepare_input.*before"),
        ("s", r"BREAKPOINT.*process_step_1.*before"),
        ("s", r"BREAKPOINT.*process_step_2.*before"),
        ("s", r"BREAKPOINT.*process_step_3.*before"),
        ("s", r"BREAKPOINT.*process_step_4.*before"),
        ("s", r"BREAKPOINT.*process_step_5.*before"),
        ("s", r"BREAKPOINT.*finalize.*before"),
        ("s", r"Debug session completed")
    ]
    run_test(interactions, "debug_step_mode.log")

    # Additional assertions on log file
    output = read_log("debug_step_mode.log")

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
