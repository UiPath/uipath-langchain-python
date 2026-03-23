import os
import json
from trace_assert import assert_traces

print("Checking template agent output...")

# Check agent output file
output_file = "__uipath/output.json"
assert os.path.isfile(output_file), "Agent output file not found"

with open(output_file, 'r', encoding='utf-8') as f:
    output_data = json.load(f)

status = output_data.get("status")
assert status == "successful", f"Agent execution failed with status: {status}"

print("Agent execution status: successful")

assert "output" in output_data, "Missing 'output' field in agent response"

output_content = output_data["output"]
assert "response" in output_content, "Missing 'response' field in output"

response = output_content["response"]
assert response and isinstance(response, str), "Response field is empty or not a string"

print(f"Agent response: {response[:200]}...")

# Check local run output
with open("local_run_output.log", 'r', encoding='utf-8') as f:
    local_run_output = f.read()

assert "Successful execution." in local_run_output, \
    f"Response does not contain 'Successful execution.'. Actual response: {local_run_output}"

# Check traces
assert_traces(".uipath/traces.jsonl", "expected_traces.json")

# Check evaluation output
eval_output_file = "eval_output.json"
assert os.path.isfile(eval_output_file), "Evaluation output file not found"

with open(eval_output_file, 'r', encoding='utf-8') as f:
    eval_data = json.load(f)

print(f"Evaluation output: {json.dumps(eval_data, indent=2)[:500]}")

eval_results = eval_data.get("evaluationSetResults", [])
eval_by_name = {e["evaluationName"]: e for e in eval_results}

assert "Current time tool should be called before web search" in eval_by_name, \
    "Missing evaluation result for: Current time tool should be called before web search"
assert "UiPath details" in eval_by_name, \
    "Missing evaluation result for: UiPath details"

print(f"Found {len(eval_by_name)} evaluation results")

# Tool call order evaluator should have perfect score (1.0)
tool_order = eval_by_name["Current time tool should be called before web search"]
for run_result in tool_order["evaluationRunResults"]:
    score = run_result["result"]["score"]
    assert score == 1.0, f"Tool call order score should be 1.0, got: {score}"
print("  ✓ tool call order: perfect score (1.0)")

# LLM judge evaluator should score above 0.7
llm_judge = eval_by_name["UiPath details"]
for run_result in llm_judge["evaluationRunResults"]:
    score = run_result["result"]["score"]
    assert score > 0.7, f"LLM judge score should be > 0.7, got: {score}"
    print(f"  ✓ llm judge output: score {score}")

print("All validations passed successfully!")
