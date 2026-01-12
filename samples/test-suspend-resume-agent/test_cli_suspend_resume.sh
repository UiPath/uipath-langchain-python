#!/bin/bash

# Test script for suspend/resume using CLI commands
# This demonstrates the complete suspend/resume cycle using uipath CLI

set -e

echo "================================================================================"
echo "TEST: Suspend/Resume using UiPath CLI"
echo "================================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

AGENT_DIR="/home/chibionos/r2/uipath-langchain-python/samples/test-suspend-resume-agent"
OUTPUT_FILE="output_suspended.json"
TRACE_FILE="trace_suspended.jsonl"

cd "$AGENT_DIR"

echo -e "${YELLOW}STEP 1: Run agent - it will SUSPEND at interrupt()${NC}"
echo "Command: uv run --with ../../. uipath run agent '{\"query\": \"Test suspend\"}' --output-file $OUTPUT_FILE --trace-file $TRACE_FILE"
echo ""

# Run the agent - it will suspend
set +e  # Don't exit on error
uv run --with ../../. uipath run agent '{"query": "Test suspend and resume"}' \
    --output-file "$OUTPUT_FILE" \
    --trace-file "$TRACE_FILE" 2>&1 | tee run_output.log
EXIT_CODE=$?
set -e

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Agent execution completed${NC}"
else
    echo -e "${RED}✗ Agent execution returned error code $EXIT_CODE (expected for suspend)${NC}"
fi

echo ""
echo "================================================================================"
echo -e "${YELLOW}STEP 2: Check output file for trigger information${NC}"
echo "================================================================================"
echo ""

if [ -f "$OUTPUT_FILE" ]; then
    echo "Output file content:"
    cat "$OUTPUT_FILE" | python3 -m json.tool
    echo ""

    # Extract trigger information
    echo -e "${YELLOW}Extracting trigger information...${NC}"
    TRIGGER_INFO=$(cat "$OUTPUT_FILE" | python3 -c "
import json, sys
data = json.load(sys.stdin)
if 'trigger' in data:
    print(json.dumps(data['trigger'], indent=2))
elif 'triggers' in data and data['triggers']:
    print(json.dumps(data['triggers'][0], indent=2))
else:
    print('No triggers found')
" 2>/dev/null || echo "Could not parse triggers")

    echo "First trigger:"
    echo "$TRIGGER_INFO"
else
    echo -e "${RED}⚠️  Output file not found${NC}"
fi

echo ""
echo "================================================================================"
echo -e "${YELLOW}STEP 3: Extract inbox_id for resume${NC}"
echo "================================================================================"
echo ""

# In a real scenario, you would:
# 1. Extract the inbox_id from the trigger
# 2. Wait for the RPA process to complete
# 3. Call resume with the job result

INBOX_ID=$(cat "$OUTPUT_FILE" | python3 -c "
import json, sys
data = json.load(sys.stdin)
triggers = data.get('triggers', [data.get('trigger')]) if 'trigger' in data or 'triggers' in data else []
for trigger in triggers:
    if trigger and 'apiResume' in trigger:
        print(trigger['apiResume']['inboxId'])
        break
" 2>/dev/null || echo "")

if [ -n "$INBOX_ID" ]; then
    echo -e "${GREEN}✓ Found inbox_id: $INBOX_ID${NC}"
else
    echo -e "${RED}⚠️  Could not extract inbox_id from triggers${NC}"
fi

echo ""
echo "================================================================================"
echo -e "${YELLOW}STEP 4: How to RESUME execution${NC}"
echo "================================================================================"
echo ""

cat << 'EOF'
To resume execution, you would use:

METHOD 1: Using UiPath SDK (in Python)
--------------------------------------
from uipath.platform import UiPath

sdk = UiPath()

# Resume with the result from your RPA process
await sdk.jobs.resume_async(
    inbox_id="<inbox_id>",
    payload={
        "status": "completed",
        "result": "Process completed successfully",
        "data": {...}
    }
)

METHOD 2: Using uipath run --resume (for LangGraph agents)
----------------------------------------------------------
# This resumes from the saved checkpoint
uv run --with ../../. uipath run agent --resume

METHOD 3: Using REST API directly
----------------------------------
# POST to /odata/Jobs/UiPath.Server.Configuration.OData.Resume
curl -X POST "https://<your-orchestrator>/odata/Jobs/UiPath.Server.Configuration.OData.Resume" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "inboxId": "<inbox_id>",
    "payload": {...}
  }'

EOF

echo ""
echo "================================================================================"
echo -e "${YELLOW}STEP 5: Simulating resume with LangGraph checkpoint${NC}"
echo "================================================================================"
echo ""

echo "For this test agent, resume would continue from the interrupt() point."
echo "The agent would then execute the code after interrupt() and return results."
echo ""
echo -e "${GREEN}✅ Test script completed!${NC}"
echo ""
echo "Key files created:"
echo "  - $OUTPUT_FILE (contains trigger information)"
echo "  - $TRACE_FILE (execution trace)"
echo "  - run_output.log (console output)"
