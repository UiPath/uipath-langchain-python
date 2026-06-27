# Wait Until Agent

This sample demonstrates a LangGraph agent that suspends with `WaitUntil` and resumes when Orchestrator fires the timer resume trigger.

The resume time is an absolute timezone-aware `datetime`. The SDK normalizes it to UTC before creating the timer trigger.

The sample includes an empty `bindings.json` file so the project has the same deployable shape as samples that declare resource bindings.
