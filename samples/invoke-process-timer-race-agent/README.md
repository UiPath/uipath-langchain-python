# Invoke Process Timer Race Agent

This sample demonstrates racing a child process interrupt with a timer interrupt.

The parent graph suspends with `interrupt([InvokeProcess(...), WaitUntil(...)])`. If `CHILD_PROCESS_NAME = "timeout-child-agent"` does not complete before the timer fires, Orchestrator resumes the parent through the timer trigger first and the sample raises `TimeoutError`.

The child process is declared in `bindings.json`. The sample code passes `process_folder_path="Shared"` when invoking the child process so the process lookup is explicit and can be changed with the binding to any process in the target organization.
