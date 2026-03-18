import json

with open("__uipath/output.json", "r", encoding="utf-8") as f:
    output_data = json.load(f)

output_content = output_data["output"]
result_summary = output_content["result_summary"]

print(f"Success: {output_content['success']}")
print(f"Summary:\n{result_summary}")

assert output_content["success"] is True, "Test did not succeed. See summary above."
