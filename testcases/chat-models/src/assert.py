import os
import json
from trace_assert import assert_traces

print("Checking feature tests agent output...")

uipath_dir = ".uipath"
assert os.path.exists(uipath_dir), "NuGet package directory (.uipath) not found"

nupkg_files = [f for f in os.listdir(uipath_dir) if f.endswith('.nupkg')]
assert nupkg_files, "NuGet package file (.nupkg) not found in .uipath directory"

print(f"NuGet package found: {nupkg_files[0]}")

output_file = "__uipath/output.json"
assert os.path.isfile(output_file), "Agent output file not found"

print("Agent output file found")

with open(output_file, 'r', encoding='utf-8') as f:
    output_data = json.load(f)

status = output_data.get("status")
assert status == "successful", f"Agent execution failed with status: {status}"

print("Agent execution status: successful")

assert "output" in output_data, "Missing 'output' field in agent response"

output_content = output_data["output"]

assert "success" in output_content, "Missing 'success' field in output"
success = output_content["success"]

assert "result_summary" in output_content, "Missing 'result_summary' field in output"
result_summary = output_content["result_summary"]
assert result_summary and result_summary.strip() != "", "Result summary is empty"

print(f"\nTest Results:")
print(f"  Success: {success}")
print(f"  Summary:\n{result_summary}")


assert success is True, f"Test did not succeed. See detailed results above."

assert "chunks_received" in output_content, "Missing 'chunks_received' field"
chunks_received = output_content["chunks_received"]
assert chunks_received is not None and chunks_received > 0, \
    f"Expected positive chunks_received, got: {chunks_received}"
print(f"Chunks received: {chunks_received}")

assert "content_length" in output_content, "Missing 'content_length' field"
content_length = output_content["content_length"]
assert content_length is not None and content_length > 0, \
    f"Expected positive content_length, got: {content_length}"
print(f"Content length: {content_length}")

assert "tool_calls_count" in output_content, "Missing 'tool_calls_count' field"
tool_calls_count = output_content["tool_calls_count"]
assert tool_calls_count is not None and tool_calls_count > 0, \
    f"Expected positive tool_calls_count, got: {tool_calls_count}"
print(f"Tool calls count: {tool_calls_count}")

with open("local_run_output.log", 'r', encoding='utf-8') as f:
    local_run_output = f.read()

assert "Successful execution." in local_run_output, \
    f"Response does not contain 'Successful execution.'. Actual response: {local_run_output}"

assert_traces(".uipath/traces.jsonl", "expected_traces.json")

print("All validations passed successfully!")
