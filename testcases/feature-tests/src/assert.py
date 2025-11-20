import os
import json
from trace_assert import assert_traces

print("Checking feature tests agent output...")

# Check NuGet package
uipath_dir = ".uipath"
assert os.path.exists(uipath_dir), "NuGet package directory (.uipath) not found"

nupkg_files = [f for f in os.listdir(uipath_dir) if f.endswith('.nupkg')]
assert nupkg_files, "NuGet package file (.nupkg) not found in .uipath directory"

print(f"NuGet package found: {nupkg_files[0]}")

# Check agent output file
output_file = "__uipath/output.json"
assert os.path.isfile(output_file), "Agent output file not found"

print("Agent output file found")

# Check status and required fields
with open(output_file, 'r', encoding='utf-8') as f:
    output_data = json.load(f)

# Check status
status = output_data.get("status")
assert status == "successful", f"Agent execution failed with status: {status}"

print("Agent execution status: successful")

# Check required fields for feature tests agent
assert "output" in output_data, "Missing 'output' field in agent response"

output_content = output_data["output"]

# Validate test_type field
assert "test_type" in output_content, "Missing 'test_type' field in output"
test_type = output_content["test_type"]
assert test_type in ["streaming", "invoke", "streaming_with_tools", "both_apis"], \
    f"Invalid test_type: {test_type}"

print(f"Test type: {test_type}")

# Validate success field
assert "success" in output_content, "Missing 'success' field in output"
success = output_content["success"]
assert success is True, f"Test did not succeed: {success}"

print("Test success: True")

# Validate result_summary field
assert "result_summary" in output_content, "Missing 'result_summary' field in output"
result_summary = output_content["result_summary"]
assert result_summary and result_summary.strip() != "", "Result summary is empty"

print(f"Result summary: {result_summary}")

# Validate optional fields based on test type
if test_type in ["streaming", "streaming_with_tools", "both_apis"]:
    assert "chunks_received" in output_content, f"Missing 'chunks_received' field for {test_type} test"
    chunks_received = output_content["chunks_received"]
    assert chunks_received is not None and chunks_received > 0, \
        f"Expected positive chunks_received, got: {chunks_received}"
    print(f"Chunks received: {chunks_received}")

if test_type in ["streaming", "invoke"]:
    assert "content_length" in output_content, f"Missing 'content_length' field for {test_type} test"
    content_length = output_content["content_length"]
    assert content_length is not None and content_length > 0, \
        f"Expected positive content_length, got: {content_length}"
    print(f"Content length: {content_length}")

if test_type == "streaming_with_tools":
    assert "tool_calls_count" in output_content, "Missing 'tool_calls_count' field for streaming_with_tools test"
    tool_calls_count = output_content["tool_calls_count"]
    # Tool calls count can be 0 or more, depending on whether the LLM decided to use tools
    assert tool_calls_count is not None, "tool_calls_count should not be None"
    print(f"Tool calls count: {tool_calls_count}")

# Check local run output
with open("local_run_output.log", 'r', encoding='utf-8') as f:
    local_run_output = f.read()

# Check if response contains 'Successful execution.'
assert "Successful execution." in local_run_output, \
    f"Response does not contain 'Successful execution.'. Actual response: {local_run_output}"

# Validate traces
assert_traces(".uipath/traces.jsonl", "expected_traces.json")

print("All validations passed successfully!")
