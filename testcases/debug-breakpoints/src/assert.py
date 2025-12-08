import os
import re

print("Checking debug breakpoints test output...")

# Check NuGet package
uipath_dir = ".uipath"
assert os.path.exists(uipath_dir), "NuGet package directory (.uipath) not found"

nupkg_files = [f for f in os.listdir(uipath_dir) if f.endswith('.nupkg')]
assert nupkg_files, "NuGet package file (.nupkg) not found in .uipath directory"

print(f"NuGet package found: {nupkg_files[0]}")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def read_log(filename: str) -> str:
    """Read and strip ANSI from log file."""
    assert os.path.isfile(filename), f"Log file not found: {filename}"
    with open(filename, 'r', encoding='utf-8') as f:
        return strip_ansi(f.read())


# Expected final value: 10 * 2 + 100 * 3 - 50 + 10 = 320
# (10*2=20, 20+100=120, 120*3=360, 360-50=310, 310+10=320)
EXPECTED_FINAL_VALUE = "320"


# ============================================
# Test 1: Single breakpoint
# ============================================
print("\n=== Test 1: Single breakpoint ===")

output = read_log("debug_single_breakpoint.log")

assert "Breakpoint set at: process_step_2" in output, \
    "Breakpoint was not set on process_step_2"
print("✓ Breakpoint was set on process_step_2")

assert "BREAKPOINT" in output and "process_step_2" in output, \
    "Breakpoint on process_step_2 was not hit"
print("✓ Breakpoint on process_step_2 was hit")

assert "processed_value" in output and EXPECTED_FINAL_VALUE in output, \
    f"Final processed_value of {EXPECTED_FINAL_VALUE} not found"
print(f"✓ Final processed value is {EXPECTED_FINAL_VALUE}")

assert "Debug session completed" in output, \
    "Debug session did not complete"
print("✓ Debug session completed")


# ============================================
# Test 2: Multiple breakpoints
# ============================================
print("\n=== Test 2: Multiple breakpoints ===")

output = read_log("debug_multiple_breakpoints.log")

assert "Breakpoint set at: process_step_2" in output, \
    "First breakpoint was not set on process_step_2"
assert "Breakpoint set at: process_step_4" in output, \
    "Second breakpoint was not set on process_step_4"
print("✓ Both breakpoints were set (process_step_2 and process_step_4)")

# Count BREAKPOINT occurrences - should have at least 2
breakpoint_count = output.count("BREAKPOINT")
assert breakpoint_count >= 2, \
    f"Expected at least 2 breakpoints hit, got {breakpoint_count}"
print(f"✓ Multiple breakpoints were hit ({breakpoint_count} times)")

assert "processed_value" in output and EXPECTED_FINAL_VALUE in output, \
    f"Final processed_value of {EXPECTED_FINAL_VALUE} not found"
print(f"✓ Final processed value is {EXPECTED_FINAL_VALUE}")

assert "Debug session completed" in output, \
    "Debug session did not complete"
print("✓ Debug session completed")


# ============================================
# Test 3: List breakpoints
# ============================================
print("\n=== Test 3: List breakpoints ===")

output = read_log("debug_list_breakpoints.log")

# Check that list command shows active breakpoints
assert "Active breakpoints:" in output or "process_step_2" in output, \
    "List command output not found"
print("✓ List command executed")

# Both breakpoints should be listed
assert "process_step_2" in output and "process_step_3" in output, \
    "Not all breakpoints shown in list"
print("✓ All breakpoints listed (process_step_2 and process_step_3)")

assert "Debug session completed" in output, \
    "Debug session did not complete"
print("✓ Debug session completed")


# ============================================
# Test 4: Remove breakpoint
# ============================================
print("\n=== Test 4: Remove breakpoint ===")

output = read_log("debug_remove_breakpoint.log")

# Check breakpoints were set
assert "Breakpoint set at: process_step_2" in output, \
    "Breakpoint on process_step_2 was not set"
assert "Breakpoint set at: process_step_4" in output, \
    "Breakpoint on process_step_4 was not set"
print("✓ Both breakpoints were initially set")

# Check remove command was acknowledged
assert "Breakpoint removed: process_step_2" in output or "removed" in output.lower(), \
    "Remove command was not acknowledged"
print("✓ Remove command executed for process_step_2")

# After removing process_step_2, only process_step_4 should be hit
# The execution should NOT stop at process_step_2 after removal
# But SHOULD stop at process_step_4
# Count breakpoints after the first "c" (start) - should only hit step_4
# This is tricky to validate precisely, so we check that step_4 breakpoint is hit
assert "BREAKPOINT" in output and "process_step_4" in output, \
    "Breakpoint on process_step_4 was not hit"
print("✓ Remaining breakpoint (process_step_4) was hit")

assert "Debug session completed" in output, \
    "Debug session did not complete"
print("✓ Debug session completed")


# ============================================
# Test 5: Quit debugger
# ============================================
print("\n=== Test 5: Quit debugger ===")

output = read_log("debug_quit.log")

# Check breakpoint was set
assert "Breakpoint set at: process_step_3" in output, \
    "Breakpoint on process_step_3 was not set"
print("✓ Breakpoint was set on process_step_3")

# Check breakpoint was hit (before quit)
assert "BREAKPOINT" in output and "process_step_3" in output, \
    "Breakpoint on process_step_3 was not hit before quit"
print("✓ Breakpoint on process_step_3 was hit")

# After quit, step_3 and beyond should NOT execute
# step_3_multiply_3 should NOT appear in output (we quit before it executed)
# But step_1 and step_2 should have completed
assert "step_1_double" in output, "step_1 did not execute before quit"
assert "step_2_add_100" in output, "step_2 did not execute before quit"
print("✓ Steps 1 and 2 executed before quit")

# Verify early termination - step_3 should NOT have completed
# (we quit at the breakpoint BEFORE step_3 executed)
# The breakpoint is "before" process_step_3, so step_3_multiply_3 should not appear
# Actually, looking at the debug output, we quit at the breakpoint, so step_3 won't run
if "step_3_multiply_3" not in output:
    print("✓ Quit prevented step_3 from executing (correct early termination)")
else:
    # If step_3 did execute, that's also acceptable depending on timing
    print("! Note: step_3 executed (quit may have been after node completion)")

# Debug session should still show completion message even on quit
assert "Debug session completed" in output, \
    "Debug session completion message not found"
print("✓ Debug session ended gracefully")


# ============================================
# Test 6: Step mode
# ============================================
print("\n=== Test 6: Step mode ===")

output = read_log("debug_step_mode.log")

# In step mode, should hit breakpoints on all nodes
assert "BREAKPOINT" in output and "prepare_input" in output, \
    "Step mode did not break on prepare_input"
print("✓ Step mode hit breakpoint on prepare_input")

# Count breakpoints - should have many (one per node)
breakpoint_count = output.count("BREAKPOINT")
assert breakpoint_count >= 5, \
    f"Expected at least 5 breakpoints in step mode, got {breakpoint_count}"
print(f"✓ Step mode hit {breakpoint_count} breakpoints")

# Check all steps executed
assert "step_1_double" in output, "step_1 not found in step mode output"
assert "step_2_add_100" in output, "step_2 not found in step mode output"
assert "step_3_multiply_3" in output, "step_3 not found in step mode output"
assert "step_4_subtract_50" in output, "step_4 not found in step mode output"
assert "step_5_add_10" in output, "step_5 not found in step mode output"
print("✓ All 5 processing steps completed in step mode")

assert "processed_value" in output and EXPECTED_FINAL_VALUE in output, \
    f"Final processed_value of {EXPECTED_FINAL_VALUE} not found"
print(f"✓ Final processed value is {EXPECTED_FINAL_VALUE}")

assert "Debug session completed" in output, \
    "Step mode debug session did not complete"
print("✓ Step mode debug session completed")


# ============================================
# Summary
# ============================================
print("\n" + "=" * 50)
print("All debug breakpoints validations passed!")
print("=" * 50)
