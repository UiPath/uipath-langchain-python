import json
import os

from trace_assert import assert_traces

print("Checking model-onboarding agent output...")

uipath_dir = ".uipath"
assert os.path.exists(uipath_dir), "NuGet package directory (.uipath) not found"

nupkg_files = [f for f in os.listdir(uipath_dir) if f.endswith(".nupkg")]
assert nupkg_files, "NuGet package file (.nupkg) not found in .uipath directory"
print(f"NuGet package found: {nupkg_files[0]}")

output_file = "__uipath/output.json"
assert os.path.isfile(output_file), "Agent output file not found"
print("Agent output file found")

with open(output_file, "r", encoding="utf-8") as f:
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

print("\nTest Results:")
print(f"  Success: {success}")
print(f"  Summary:\n{result_summary}")

assert success is True, "Test did not succeed. See detailed results above."

# The second (empty UIPATH_JOB_KEY) local run appends to this log.
with open("local_run_output.log", "r", encoding="utf-8") as f:
    local_run_output = f.read()

assert "Successful execution." in local_run_output, (
    f"Response does not contain 'Successful execution.'. "
    f"Actual response: {local_run_output}"
)

assert_traces(".uipath/traces.jsonl", "expected_traces.json")

print("All validations passed successfully!")
