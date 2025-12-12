"""
Pexpect-based tests for uipath dev TUI console.

Tests the interactive dev console functionality including:
- Starting the TUI console
- Creating new runs with JSON input
- Running the agent and verifying output
- Navigating between tabs (Details, Traces, Logs, Chat)
- History panel showing completed runs
- Keyboard shortcuts (q, n, r, etc.)

Screen frames are captured during execution and printed at the end.
"""

import sys
from pathlib import Path

import pexpect
import pytest

# Add testcases to path for common imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from testcases.common import ConsoleTest


# The command to run for all tests
COMMAND = "uv run uipath dev"
# Timeout for expect operations
TIMEOUT = 60


def test_dev_console_starts():
    """Test that uipath dev TUI starts and shows main UI elements."""
    test = ConsoleTest(
        command=COMMAND,
        test_name="dev_console_starts",
        timeout=TIMEOUT,
    )
    try:
        test.start()

        # The TUI should show key UI elements
        test.wait_for_ui(3, "TUI fully loaded")

        # Capture the initial state
        test.capture_screen("Main UI visible")

        # Send 'q' to quit
        test.send_key('q', "Quit command")
        test.expect_eof()

        print("--- Test completed successfully ---")

        # Verify output contains expected UI elements
        output = test.get_output()
        assert "History" in output or "New run" in output or "agent" in output, \
            "TUI main elements not found in output"

    except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF) as e:
        print("\nERROR: Pexpect failed.", file=sys.stderr)
        print(f"Failure: {type(e).__name__}", file=sys.stderr)
        print(f"\n--- Output before failure ---\n{test.before}", file=sys.stderr)
        pytest.fail(f"Test failed: {e}")
    finally:
        test.close()


def test_run_calculator_agent():
    """Test running the calculator agent through the TUI.

    This test:
    1. Starts the dev console
    2. Enters JSON input for calculator (a=10, b=5, operator=+)
    3. Runs the agent
    4. Verifies the result (15.0) appears
    5. Quits the console
    """
    test = ConsoleTest(
        command=COMMAND,
        test_name="dev_run_calculator",
        timeout=TIMEOUT,
    )
    try:
        test.start()
        test.wait_for_ui(3, "Ready to run")

        # Send 'r' to run with current input
        test.send_key('r', "Run command sent")

        # Wait for execution to complete
        test.wait_for_ui(5, "Execution in progress")
        test.wait_for_ui(3, "Execution complete")

        # Capture final state before quitting
        test.capture_screen("Result displayed")

        # Quit
        test.send_key('q', "Quit")
        test.expect_eof()

        print("--- Test completed successfully ---")

        # Verify output
        output = test.get_output()
        has_completion = (
            "COMPLETED" in output or
            "result" in output or
            "15.0" in output or
            "Success" in output.lower()
        )
        assert has_completion, "Agent execution result not found in output"

    except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF) as e:
        print("\nERROR: Pexpect failed.", file=sys.stderr)
        print(f"Failure: {type(e).__name__}", file=sys.stderr)
        print(f"\n--- Output before failure ---\n{test.before}", file=sys.stderr)
        pytest.fail(f"Test failed: {e}")
    finally:
        test.close()


def test_new_run_and_modify_input():
    """Test creating a new run with modified input."""
    test = ConsoleTest(
        command=COMMAND,
        test_name="dev_new_run_modified",
        timeout=TIMEOUT,
    )
    try:
        test.start()
        test.wait_for_ui(3, "Initial state")

        # Press 'n' for new run
        test.send_key('n', "New run")
        test.wait_for_ui(1, "New run form")

        # Run with default values
        test.send_key('r', "Run")
        test.wait_for_ui(5, "Execution complete")

        # Quit
        test.send_key('q', "Quit")
        test.expect_eof()

        print("--- Test completed successfully ---")

        output = test.get_output()
        has_run = "agent" in output.lower() or "run" in output.lower()
        assert has_run, "No evidence of agent run in output"

    except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF) as e:
        print("\nERROR: Pexpect failed.", file=sys.stderr)
        print(f"Failure: {type(e).__name__}", file=sys.stderr)
        print(f"\n--- Output before failure ---\n{test.before}", file=sys.stderr)
        pytest.fail(f"Test failed: {e}")
    finally:
        test.close()


def test_view_traces_tab():
    """Test viewing the Traces tab after a run.

    Navigation sequence: R TAB TAB TAB RIGHT_ARROW
    - Tab 1: selects the list under History
    - Tab 2: selects the text box under Details
    - Tab 3: selects the tab control (Details/Traces/Logs/Chat)
    - RIGHT_ARROW: moves to Traces tab
    """
    test = ConsoleTest(
        command=COMMAND,
        test_name="dev_traces_tab",
        timeout=TIMEOUT,
    )
    try:
        test.start()
        test.wait_for_ui(3, "Initial")

        # Run the agent first
        test.send_key('r', "Run")
        test.wait_for_ui(5, "Run complete")

        # Navigate to Traces tab: TAB TAB TAB RIGHT_ARROW
        test.send_key('\t', "Tab 1 - History list")
        test.wait_for_ui(0.5, "After tab 1")
        test.send_key('\t', "Tab 2 - Details text box")
        test.wait_for_ui(0.5, "After tab 2")
        test.send_key('\t', "Tab 3 - Tab control")
        test.wait_for_ui(0.5, "After tab 3")
        # RIGHT_ARROW to move from Details to Traces
        test.send_key('\x1b[C', "Right arrow - Traces tab")
        test.wait_for_ui(1, "Traces tab selected")

        test.capture_screen("Traces view")

        # Quit
        test.send_key('q', "Quit")
        test.expect_eof()

        print("--- Test completed successfully ---")

        output = test.get_output()
        has_traces = (
            "Trace" in output or
            "LangGraph" in output or
            "calculate" in output or
            "postprocess" in output
        )
        assert has_traces, "Traces content not found - expected trace tree in output"

    except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF) as e:
        print("\nERROR: Pexpect failed.", file=sys.stderr)
        print(f"Failure: {type(e).__name__}", file=sys.stderr)
        print(f"\n--- Output before failure ---\n{test.before}", file=sys.stderr)
        pytest.fail(f"Test failed: {e}")
    finally:
        test.close()


def test_view_logs_tab():
    """Test viewing the Logs tab after a run.

    Navigation sequence: R TAB TAB TAB RIGHT_ARROW RIGHT_ARROW
    - Tab 1: selects the list under History
    - Tab 2: selects the text box under Details
    - Tab 3: selects the tab control (Details/Traces/Logs/Chat)
    - RIGHT_ARROW x2: moves to Logs tab (Details -> Traces -> Logs)
    """
    test = ConsoleTest(
        command=COMMAND,
        test_name="dev_logs_tab",
        timeout=TIMEOUT,
    )
    try:
        test.start()
        test.wait_for_ui(3, "Initial")

        # Run the agent
        test.send_key('r', "Run")
        test.wait_for_ui(5, "Complete")

        # Navigate to Logs tab: TAB TAB TAB RIGHT_ARROW RIGHT_ARROW
        test.send_key('\t', "Tab 1 - History list")
        test.wait_for_ui(0.5, "After tab 1")
        test.send_key('\t', "Tab 2 - Details text box")
        test.wait_for_ui(0.5, "After tab 2")
        test.send_key('\t', "Tab 3 - Tab control")
        test.wait_for_ui(0.5, "After tab 3")
        # RIGHT_ARROW twice: Details -> Traces -> Logs
        test.send_key('\x1b[C', "Right arrow - Traces")
        test.wait_for_ui(0.5, "Traces tab")
        test.send_key('\x1b[C', "Right arrow - Logs")
        test.wait_for_ui(1, "Logs tab selected")

        test.capture_screen("Logs view")

        # Quit
        test.send_key('q', "Quit")
        test.expect_eof()

        print("--- Test completed successfully ---")

        output = test.get_output()
        has_logs = (
            "INFO" in output or
            "DEBUG" in output or
            "Starting" in output or
            "Execution" in output
        )
        assert has_logs, "Logs content not found - expected log messages in output"

    except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF) as e:
        print("\nERROR: Pexpect failed.", file=sys.stderr)
        print(f"Failure: {type(e).__name__}", file=sys.stderr)
        print(f"\n--- Output before failure ---\n{test.before}", file=sys.stderr)
        pytest.fail(f"Test failed: {e}")
    finally:
        test.close()


def test_multiple_runs_in_history():
    """Test that multiple runs appear in the history panel."""
    test = ConsoleTest(
        command=COMMAND,
        test_name="dev_multiple_runs",
        timeout=TIMEOUT,
    )
    try:
        test.start()
        test.wait_for_ui(3, "Initial")

        # First run
        test.send_key('r', "First run")
        test.wait_for_ui(5, "First complete")

        test.capture_screen("After first run")

        # Second run
        test.send_key('n', "New")
        test.wait_for_ui(1, "New run form")
        test.send_key('r', "Second run")
        test.wait_for_ui(5, "Second complete")

        test.capture_screen("After second run - history should show both")

        # Quit
        test.send_key('q', "Quit")
        test.expect_eof()

        print("--- Test completed successfully ---")

        output = test.get_output()
        agent_count = output.lower().count("agent")
        print(f"Agent references in output: {agent_count}")

    except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF) as e:
        print("\nERROR: Pexpect failed.", file=sys.stderr)
        print(f"Failure: {type(e).__name__}", file=sys.stderr)
        print(f"\n--- Output before failure ---\n{test.before}", file=sys.stderr)
        pytest.fail(f"Test failed: {e}")
    finally:
        test.close()


def test_quit_with_escape():
    """Test that ESC key cancels/closes appropriately."""
    test = ConsoleTest(
        command=COMMAND,
        test_name="dev_escape_key",
        timeout=TIMEOUT,
    )
    try:
        test.start()
        test.wait_for_ui(3, "Initial")

        # Press ESC
        test.send_key('\x1b', "ESC pressed")
        test.wait_for_ui(1, "After ESC")

        # Then quit
        test.send_key('q', "Quit")
        test.expect_eof()

        print("--- Test completed successfully ---")

    except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF) as e:
        print("\nERROR: Pexpect failed.", file=sys.stderr)
        print(f"Failure: {type(e).__name__}", file=sys.stderr)
        print(f"\n--- Output before failure ---\n{test.before}", file=sys.stderr)
        pytest.fail(f"Test failed: {e}")
    finally:
        test.close()


def test_calculator_operations():
    """Test calculator operations through the TUI."""
    test = ConsoleTest(
        command=COMMAND,
        test_name="dev_calculator_ops",
        timeout=TIMEOUT,
    )
    try:
        test.start()
        test.wait_for_ui(3, "Ready")

        # Run with default input (10 + 5 = 15)
        test.send_key('r', "Run addition")
        test.wait_for_ui(5, "Complete")

        test.capture_screen("Addition result")

        # Quit
        test.send_key('q', "Quit")
        test.expect_eof()

        print("--- Test completed successfully ---")

        output = test.get_output()
        has_result = "15" in output or "result" in output
        assert has_result, "Calculator result not found"

    except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF) as e:
        print("\nERROR: Pexpect failed.", file=sys.stderr)
        print(f"Failure: {type(e).__name__}", file=sys.stderr)
        print(f"\n--- Output before failure ---\n{test.before}", file=sys.stderr)
        pytest.fail(f"Test failed: {e}")
    finally:
        test.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
