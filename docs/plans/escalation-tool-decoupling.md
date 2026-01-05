# Escalation Tool Decoupling Plan

## Problem Statement

The escalation tool is currently coupled to LangGraph's loop implementation:
- Wrapper returns `Command(goto=AgentGraphNode.TERMINATE)` - LangGraph-specific
- Cannot be reused in coded agents, deterministic flows, or other loop implementations
- Tools in the public repo should be drop-in replaceable across different agent architectures

## Current Implementation

```
escalation_tool_fn()
    → returns {action: END/CONTINUE, output: ...}

escalation_wrapper()
    → if action == END: return Command(goto=TERMINATE)
    → else: return output

UiPathToolNode._process_result()
    → if Command: return it (routes to TERMINATE)
    → else: wrap in ToolMessage
```

**Flow for termination:**
```
Tool → Wrapper → Command → Router → TERMINATE node → raises AgentTerminationException
```

The termination already ends in an exception - just delayed through graph machinery.

## Proposed Solution

**Move termination signal earlier using exception pattern:**

```
Tool → raises AgentToolTerminationRequest → Loop catches → handles appropriately
```

Each loop implementation handles the exception its own way:
- **LangGraph loop**: catch in `UiPathToolNode` → convert to `Command(goto=TERMINATE)`
- **Coded agent**: catch → terminate gracefully with output
- **Deterministic loop**: catch → return final result

## Why Exceptions for Flow Control are Acceptable

### Python's Design Philosophy

1. **Built-in precedent**: `StopIteration`, `SystemExit`, `GeneratorExit` are all exceptions used for signaling, not errors
   - Source: [Python docs](https://docs.python.org/3/library/exceptions.html#SystemExit)

2. **EAFP idiom**: "Easier to Ask Forgiveness than Permission" is idiomatic Python
   - Source: [Microsoft Python Blog](https://devblogs.microsoft.com/python/idiomatic-python-eafp-versus-lbyl/)
   - > "Since exceptions are used for control flow like in EAFP, Python implementations work hard to make exceptions a cheap operation"

3. **SystemExit pattern**: `sys.exit()` raises an exception specifically so cleanup handlers can run
   - > "A call to sys.exit() is translated into an exception so that clean-up handlers (finally clauses) can be executed"

4. **BaseException hierarchy**: Signals inherit from `BaseException` (not `Exception`) to avoid accidental catching:
   ```
   BaseException
   ├── GeneratorExit      # signal
   ├── KeyboardInterrupt  # signal
   ├── SystemExit         # signal
   └── Exception          # actual errors
   ```

### References
- [Real Python - LBYL vs EAFP](https://realpython.com/python-lbyl-vs-eafp/)
- [Hacker News Discussion](https://news.ycombinator.com/item?id=20672619)
- [Python Glossary - EAFP](https://realpython.com/ref/glossary/eafp/)

## Implementation Steps

### Step 1: Create Exception Class

```python
# src/uipath_langchain/agent/exceptions/exceptions.py

class AgentToolTerminationRequest(BaseException):
    """Tool requested agent termination (signal, not error).

    Inherits from BaseException (like SystemExit) to avoid
    accidental catching by `except Exception` blocks.
    """
    def __init__(
        self,
        source: str,
        title: str,
        detail: str,
        output: Any = None
    ):
        self.source = source
        self.title = title
        self.detail = detail
        self.output = output
        super().__init__(title)
```

### Step 2: Modify Escalation Tool

```python
# src/uipath_langchain/agent/tools/escalation_tool.py

async def escalation_tool_fn(**kwargs: Any) -> dict[str, Any]:
    result = interrupt(CreateEscalation(...))

    escalation_action = getattr(result, "action", None)
    escalation_output = getattr(result, "data", {})

    outcome_str = channel.outcome_mapping.get(escalation_action)
    outcome = EscalationAction(outcome_str) if outcome_str else EscalationAction.CONTINUE

    if outcome == EscalationAction.END:
        raise AgentToolTerminationRequest(
            source=AgentTerminationSource.ESCALATION,
            title=f"Agent run ended based on escalation outcome with directive {escalation_action}",
            detail=f"Escalation output: {escalation_output}",
            output=escalation_output
        )

    # CONTINUE case - return data for agent to use
    return escalation_output

# Remove escalation_wrapper - no longer needed
# Remove StructuredToolWithWrapper - use regular StructuredTool
```

### Step 3: Update UiPathToolNode

```python
# src/uipath_langchain/agent/tools/tool_node.py

async def _afunc(self, state: Any, config: RunnableConfig | None = None) -> OutputType:
    call = self._extract_tool_call(state)
    if call is None:
        return None

    try:
        if self.awrapper:
            filtered_state = self._filter_state(state, self.awrapper)
            result = await self.awrapper(self.tool, call, filtered_state)
        else:
            result = await self.tool.ainvoke(call["args"])
        return self._process_result(call, result)

    except AgentToolTerminationRequest as e:
        # Convert tool termination request to graph Command
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"{e.title}. {e.detail}",
                        tool_call_id=call["id"],
                    )
                ],
                "termination": {
                    "source": e.source,
                    "title": e.title,
                    "detail": e.detail,
                },
            },
            goto=AgentGraphNode.TERMINATE,
        )
```

### Step 4: Cleanup

- Remove `StructuredToolWithWrapper` class
- Remove `ToolWrapperMixin` if no other tools use wrappers
- Update `__init__.py` exports
- Update tests

## Benefits

| Aspect | Before | After |
|--------|--------|-------|
| Tool portability | Coupled to LangGraph | Framework-agnostic |
| Wrapper needed | Yes | No |
| Code in tool | Returns dict | Raises signal |
| Loop handling | Implicit via Command | Explicit try/catch |
| Reuse in coded agents | Not possible | Just catch exception |
| Follows Python idiom | Partially | Yes (SystemExit pattern) |

## Testing Plan

1. Unit test: `AgentToolTerminationRequest` exception creation
2. Unit test: Escalation tool raises on END outcome
3. Unit test: Escalation tool returns data on CONTINUE
4. Integration test: LangGraph loop handles termination correctly
5. Integration test: Escalation with reject → terminate flow
6. Integration test: Escalation with approve → continue flow

## Migration Notes

- This is a breaking change for anyone using `escalation_wrapper` directly
- Public API (`create_escalation_tool`) remains the same
- Behavior is identical - just internal implementation change
