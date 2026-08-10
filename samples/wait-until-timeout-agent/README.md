# Wait Until Timeout Agent

This sample shows how a LangGraph agent can wait for whichever happens first:
a child UiPath process completes, or a timer expires.

The parent agent uses:

```python
interrupt([
    InvokeProcess(...),
    WaitUntil(...),
])
```

If the child process completes first, the interrupt returns the child process
output. If the timer completes first, `assert_no_timeout(...)` raises
`UiPathTimeoutError`.

## Files

- `graph.py`: parent agent graph.
- `bindings.json`: process binding for the child process.
- `uipath.json`: UiPath agent configuration.
- `langgraph.json`: LangGraph graph entrypoint.

## Child Process

The sample references a process named `timeout-child-agent` in the `Shared`
folder:

```python
CHILD_PROCESS_NAME = "timeout-child-agent"
CHILD_PROCESS_FOLDER_PATH = "Shared"
```

For local runs, change `CHILD_PROCESS_NAME` and `CHILD_PROCESS_FOLDER_PATH` to
match a process in your tenant.

For cloud runs, deploy the package and choose an override from your tenant for
the `timeout-child-agent.Shared` process binding. The binding is already
configured in `bindings.json`.

For timeout testing, use a child process that runs longer than 10 minutes, or
reduce the `WaitUntil` duration in `graph.py`.

## Run

```bash
cd samples/wait-until-timeout-agent
uv sync
uipath run agent '{"message": "start child work"}'
```

When publishing the sample, use the existing `timeout-child-agent.Shared`
binding and select the process you want the parent agent to invoke.
