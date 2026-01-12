# Suspend/Resume Testing Guide

This guide shows you how to test the complete suspend/resume cycle with the test agent.

## Overview

The suspend/resume pattern allows an agent to:
1. **Suspend** execution when it needs external work (e.g., RPA process)
2. Wait for that work to complete
3. **Resume** execution with the results

## Testing Methods

### Method 1: Automated Test Script (Python)

Run the Python test script that automates the entire cycle:

```bash
cd /home/chibionos/r2/uipath-langchain-python/samples/test-suspend-resume-agent
python3 test_suspend_resume.py
```

**What it does:**
- ✅ Runs agent until it suspends
- ✅ Inspects the checkpoint state
- ✅ Simulates resume with mock RPA results
- ✅ Verifies final state

**Output shows:**
```
STEP 1: Initial execution - agent will SUSPEND
🔴 Agent SUSPENDED (as expected)

STEP 2: Checking agent state after suspension
State values: {'query': 'Test suspend...', 'result': ''}
Found 1 pending task(s)

STEP 3: RESUMING execution
Resuming with payload: {...}
🟢 Resume execution result: {...}

STEP 4: Verification - checking final state
✅ Test completed!
```

---

### Method 2: CLI Testing (Shell Script)

Run the shell script for CLI-based testing:

```bash
cd /home/chibionos/r2/uipath-langchain-python/samples/test-suspend-resume-agent
./test_cli_suspend_resume.sh
```

**What it does:**
- ✅ Runs agent via `uipath run`
- ✅ Captures output with triggers
- ✅ Extracts inbox_id from triggers
- ✅ Shows how to resume

**Key outputs:**
- `output_suspended.json` - Contains trigger information
- `trace_suspended.jsonl` - Execution trace
- `run_output.log` - Console output with logs

---

### Method 3: Manual Step-by-Step

#### Step 1: Run agent and capture suspension

```bash
cd /home/chibionos/r2/uipath-langchain-python/samples/test-suspend-resume-agent

uv run --with ../../. uipath run agent \
  '{"query": "Test suspend"}' \
  --output-file output.json \
  --trace-file trace.jsonl
```

**Expected output:**
```
================================================================================
AGENT NODE: Starting invoke_process_node
AGENT NODE: Received query: Test suspend
🔴 AGENT NODE: About to call interrupt() - SUSPENDING EXECUTION
================================================================================
```

#### Step 2: Examine the output file

```bash
cat output.json | python3 -m json.tool
```

**Look for:**
```json
{
  "triggers": [
    {
      "interruptId": "...",
      "triggerType": "Api",
      "apiResume": {
        "inboxId": "e6e17acd-786c-46df-852c-76aeb9ffb29d",
        "request": {
          "name": "TestProcess",
          "input_arguments": {...}
        }
      }
    }
  ]
}
```

#### Step 3: Resume execution

**Option A: Using UiPath SDK (for real Orchestrator jobs)**

```python
from uipath.platform import UiPath

sdk = UiPath()

# Resume with RPA process results
await sdk.jobs.resume_async(
    inbox_id="e6e17acd-786c-46df-852c-76aeb9ffb29d",
    payload={
        "status": "completed",
        "result": "RPA process completed successfully",
        "output_data": {"processed_items": 42}
    }
)
```

**Option B: Using uipath run --resume (for checkpointed agents)**

```bash
uv run --with ../../. uipath run agent --resume
```

This resumes from the last checkpoint (requires persistent checkpointer, not MemorySaver).

---

## Understanding the Flow

### 1. Agent Suspends

When the agent calls `interrupt(InvokeProcess(...))`:

```python
# In graph.py
invoke_request = InvokeProcess(
    name="TestProcess",
    input_arguments={"query": state.query, "data": "test_data"},
    process_folder_path="Shared",
)

interrupt(invoke_request)  # ← Execution suspends here
```

**Logs show:**
```
🔴 AGENT NODE: About to call interrupt() - SUSPENDING EXECUTION
```

### 2. Runtime Detects Suspension

The eval runtime detects `UiPathRuntimeStatus.SUSPENDED`:

**Logs show:**
```
🔴 EVAL RUNTIME: DETECTED SUSPENSION for eval 'Basic suspend test'
EVAL RUNTIME: Agent returned SUSPENDED status
EVAL RUNTIME: Extracted 2 trigger(s) from suspended execution
EVAL RUNTIME: Trigger 1: {interruptId, triggerType, apiResume...}
```

### 3. Trigger Information Captured

The trigger contains everything needed to resume:
- `interruptId` - Unique ID for this suspension point
- `inboxId` - Used to send resume payload
- `request` - The original InvokeProcess request
- `triggerType` - Type of trigger (Api, QueueItem, etc.)

### 4. Resume Execution

When resumed, the agent continues **after** the `interrupt()` call:

```python
interrupt(invoke_request)  # ← Was suspended here

# Execution resumes here ↓
logger.info("🟢 AGENT NODE: Execution RESUMED after interrupt()")
return State(query=state.query, result="Process completed")
```

**Logs show:**
```
🟢 AGENT NODE: Execution RESUMED after interrupt()
AGENT NODE: RPA process has completed
```

---

## Integration with Orchestrator

In a real deployment:

### 1. Agent Suspends → Orchestrator Job Suspends

```python
# Agent code
interrupt(InvokeProcess(name="ProcessInvoice", ...))
```

↓

```
Orchestrator Job Status: SUSPENDED
Trigger saved with inbox_id
```

### 2. Webhook Notification

Orchestrator sends webhook:
```json
{
  "Type": "job.suspended",
  "Job": {
    "Id": "abc-123",
    "Status": "Suspended"
  }
}
```

### 3. External System Processes

Your system:
- Receives webhook
- Extracts inbox_id from job triggers
- Starts RPA process
- Waits for completion

### 4. Resume Call

When RPA process completes:
```python
await sdk.jobs.resume_async(
    inbox_id="<from_trigger>",
    payload={"invoice_processed": True, "total": 1500.50}
)
```

### 5. Agent Continues

Agent receives the payload and continues:
```python
# The payload is available in the resumed state
result = payload["total"]  # 1500.50
```

---

## Troubleshooting

### Agent doesn't suspend

**Check:**
- ✅ Graph has checkpointer configured
- ✅ Using `interrupt()` correctly
- ✅ Not catching the interrupt exception

### Can't find triggers in output

**Check:**
- ✅ Using `--output-file` flag
- ✅ Eval runtime is passing through triggers
- ✅ Look for `triggers` or `trigger` field in JSON

### Resume doesn't work

**Check:**
- ✅ Correct inbox_id from trigger
- ✅ UiPath credentials configured
- ✅ Job still in SUSPENDED state
- ✅ Checkpointer persists state (MemorySaver is in-memory only)

---

## Key Files

- `graph.py` - Agent implementation with interrupt()
- `test_suspend_resume.py` - Automated Python test
- `test_cli_suspend_resume.sh` - CLI-based test script
- `evaluations/test_suspend_resume.json` - Eval set for testing
- `README.md` - Quick start guide
- `SUSPEND_RESUME_GUIDE.md` - This comprehensive guide

---

## Next Steps

1. **Test locally**: Run `python3 test_suspend_resume.py`
2. **Test with CLI**: Run `./test_cli_suspend_resume.sh`
3. **Deploy to Orchestrator**: Package and deploy agent
4. **Set up webhooks**: Configure webhook endpoint for job.suspended
5. **Implement resume logic**: Handle webhook and call `sdk.jobs.resume_async()`

For production use, consider:
- Using persistent checkpointer (not MemorySaver)
- Error handling for failed RPA processes
- Timeout handling for stuck jobs
- Retry logic for resume calls
