# Invoke Process Timeout Agent

This sample demonstrates timeout handling for typed interrupts.

The parent graph invokes `CHILD_PROCESS_NAME = "timeout-child-agent"` with `InvokeProcess(..., timeout=600)`. If that process does not complete before the timeout, Orchestrator resumes the parent through the timer trigger first. The parent uses `assert_no_timeout`, which raises `UiPathTimeoutError` on timeout.

The child process is declared in `bindings.json`. The sample code passes `process_folder_path="Shared"` when invoking the child process so the process lookup is explicit and can be changed with the binding to any process in the target organization.
