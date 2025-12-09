#!/bin/bash
set -e

echo "Syncing dependencies..."
uv sync

echo "Authenticating with UiPath..."
uv run uipath auth --client-id="$CLIENT_ID" --client-secret="$CLIENT_SECRET" --base-url="$BASE_URL"

echo "Initializing the project..."
uv run uipath init

echo "Packing agent..."
uv run uipath pack

# Clear any existing job key to use Console mode instead of SignalR mode
export UIPATH_JOB_KEY=""

# ============================================
# Test 1: Single breakpoint
# ============================================
echo "=== Test 1: Single breakpoint ==="
echo "Setting breakpoint on 'process_step_2' and running debug session..."

# Debug session:
# 1. "b process_step_2" - set breakpoint
# 2. "c" - start execution
# 3. "c" - continue from breakpoint
printf "b process_step_2\nc\nc\n" | uv run uipath debug agent --file input.json 2>&1 | tee debug_single_breakpoint.log

# ============================================
# Test 2: Multiple breakpoints
# ============================================
echo "=== Test 2: Multiple breakpoints ==="
echo "Setting breakpoints on 'process_step_2' and 'process_step_4'..."

# Debug session:
# 1. "b process_step_2" - set first breakpoint
# 2. "b process_step_4" - set second breakpoint
# 3. "c" - start execution
# 4. "c" - continue from first breakpoint (step_2)
# 5. "c" - continue from second breakpoint (step_4)
printf "b process_step_2\nb process_step_4\nc\nc\nc\n" | uv run uipath debug agent --file input.json 2>&1 | tee debug_multiple_breakpoints.log

# ============================================
# Test 3: List breakpoints
# ============================================
echo "=== Test 3: List breakpoints ==="
echo "Testing 'l' command to list breakpoints..."

# Debug session:
# 1. "b process_step_2" - set breakpoint
# 2. "b process_step_3" - set another breakpoint
# 3. "l" - list all breakpoints
# 4. "c" - start and continue to completion
# 5. "c" - continue from step_2
# 6. "c" - continue from step_3
printf "b process_step_2\nb process_step_3\nl\nc\nc\nc\n" | uv run uipath debug agent --file input.json 2>&1 | tee debug_list_breakpoints.log

# ============================================
# Test 4: Remove breakpoint
# ============================================
echo "=== Test 4: Remove breakpoint ==="
echo "Testing 'r' command to remove a breakpoint..."

# Debug session:
# 1. "b process_step_2" - set breakpoint on step_2
# 2. "b process_step_4" - set breakpoint on step_4
# 3. "l" - list breakpoints (should show both)
# 4. "r process_step_2" - remove breakpoint from step_2
# 5. "l" - list breakpoints (should show only step_4)
# 6. "c" - start execution (should NOT stop at step_2, only at step_4)
# 7. "c" - continue from step_4
printf "b process_step_2\nb process_step_4\nl\nr process_step_2\nl\nc\nc\n" | uv run uipath debug agent --file input.json 2>&1 | tee debug_remove_breakpoint.log

# ============================================
# Test 5: Quit debugger early
# ============================================
echo "=== Test 5: Quit debugger ==="
echo "Testing 'q' command to quit debugger early..."

# Debug session:
# 1. "b process_step_3" - set breakpoint
# 2. "c" - start execution
# 3. "q" - quit at the breakpoint (before step_3 executes)
# Note: This will exit early, so we use || true to not fail the script
printf "b process_step_3\nc\nq\n" | uv run uipath debug agent --file input.json 2>&1 | tee debug_quit.log || true

# ============================================
# Test 6: Step mode (full graph)
# ============================================
echo "=== Test 6: Step mode ==="
echo "Running debug session with step mode (breaks on every node)..."

# Debug session with step mode - need to use 's' for each step to stay in step mode
# ('c' disables step mode, 's' keeps it enabled)
# 7 nodes: prepare_input, process_step_1, process_step_2, process_step_3, process_step_4, process_step_5, finalize
printf "s\ns\ns\ns\ns\ns\ns\ns\n" | uv run uipath debug agent --file input.json 2>&1 | tee debug_step_mode.log
