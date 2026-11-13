import json

from gateway_trace_assert import assert_gateway_traces
from trace_assert import assert_traces, load_traces

with open("__uipath/output.json", encoding="utf-8") as f:
    output = json.load(f)
assert output.get("response"), f"Missing 'response' in output: {output}"
assert output.get("direct_model_reply", "").startswith("[fake direct provider]"), output

with open("local_run_output.log", encoding="utf-8") as f:
    assert "Successful execution." in f.read(), "Run did not succeed"

assert_traces(".uipath/traces.jsonl", "expected_traces.json")

# The Gateway model's client-side LLM span is skipped.
with open("expected_traces.json", encoding="utf-8") as f:
    forbidden = set(json.load(f)["forbidden_span_names"])
found = forbidden & {span["name"] for span in load_traces(".uipath/traces.jsonl")}
assert not found, f"Unexpected client-side LLM spans: {found}"
print(f"✓ No client-side LLM span for: {', '.join(sorted(forbidden))}")

assert_gateway_traces(".uipath/traces.jsonl", "expected_traces.json")

print("All validations passed successfully!")
