# Gym Sample Usage Guide

This sample demonstrates how to build agents that support **both CLI execution** with properly typed inputs **and batch evaluation** across multiple datapoints using two separate graph builders.

## Summary

**The Solution:** Use **two separate graph building methods**:
1. `build_cli_graph()` - Accepts input at runtime (for CLI)
2. `build_evaluation_graph(input)` - Pre-binds input at build time (for evaluation)

This approach follows the pattern from `ticket-classification` example with the updated `input_schema=` and `output_schema=` parameters.

## Two Modes of Operation

### 1. CLI Mode (Single Execution with Runtime Input)

Use `uipath run` for single agent executions:

```bash
# Run with properly typed CalculatorInput
uv run uipath run calculator '{"expression": "2 + 2"}'
```

**How it works:**
- Graph built with `build_cli_graph()`
- Uses `StateGraph(State, input_schema=CalculatorInput, output_schema=CalculatorOutput)`
- Has a `prepare_input` node that receives `CalculatorInput` and returns `StateBaseClass`
- Runtime: CLI passes `{"expression": "2 + 2"}` → validated → passed to `prepare_input` → converted to State with messages

### 2. Evaluation Mode (Batch Execution with Pre-bound Input)

Run all datapoints for comprehensive evaluation:

```bash
# Run evaluation across all datapoints
uv run python -m gym_sample.run

# With verbose output
uv run python -m gym_sample.run --verbose
```

**How it works:**
- Graphs built with `build_evaluation_graph(datapoint.input)`
- Each graph captures its datapoint input at build time via a closure
- Runtime: Just call `graph.ainvoke({})` - input already bound
- No runtime input needed

## Architecture

### Key Classes & Methods

**`BasicLoop` class has two graph builders:**

```python
def build_cli_graph(self) -> StateGraph:
    """Build graph for CLI mode - accepts input at runtime."""
    graph = StateGraph(
        StateBaseClass,
        input_schema=self.scenario.input_schema,
        output_schema=self.scenario.output_schema
    )
    # prepare_input node receives input_schema type at runtime
    graph.add_node("prepare_input", self.prepare_input_from_runtime)
    ...
    return graph

def build_evaluation_graph(self, agent_input: Dict[str, Any]) -> StateGraph:
    """Build graph for evaluation mode - pre-binds input at build time."""
    graph = StateGraph(
        StateBaseClass,
        input_schema=self.scenario.input_schema,
        output_schema=self.scenario.output_schema
    )
    final_agent_input = self.scenario.input_schema.model_validate(agent_input)

    # Closure captures the input at build time
    def prepare_with_bound_input(state: StateBaseClass | None = None) -> StateBaseClass:
        return self.prepare_input_node(agent_input=final_agent_input)

    graph.add_node("prepare_input", prepare_with_bound_input)
    ...
    return graph
```

### Entry Points

**`calculator_agent()` - CLI entry point:**
```python
@asynccontextmanager
async def calculator_agent(
    agent_input: CalculatorInput | None = None
) -> AsyncGenerator[StateGraph, None]:
    """Pre-configured calculator agent entry point for CLI usage."""
    agent_scenario = get_agents()["calculator"]
    loop = BasicLoop(scenario=agent_scenario, llm=get_model(), ...)

    # Build CLI graph that accepts input at runtime
    graph = loop.build_cli_graph()
    yield graph
```

**`agents_with_datapoints()` - Evaluation entry point:**
```python
@asynccontextmanager
async def agents_with_datapoints(
    agent_name: str = "calculator"
) -> AsyncGenerator[List[tuple[StateGraph, Datapoint]], None]:
    """Create all LangGraph agents for evaluation mode."""
    agent_scenario = get_agents()[agent_name]
    loop = BasicLoop(scenario=agent_scenario, llm=get_model(), ...)

    graphs = []
    for datapoint in agent_scenario.datapoints:
        # Build evaluation graph with pre-bound input
        graph = loop.build_evaluation_graph(datapoint.input)
        graphs.append((graph, datapoint))

    yield graphs
```

### Input Handling Flow

**CLI Mode:**
1. User runs: `uipath run calculator '{"expression": "2 + 2"}'`
2. CLI validates JSON against `CalculatorInput` schema
3. Passes `CalculatorInput(expression="2 + 2")` to graph at runtime
4. `prepare_input_from_runtime` receives the `CalculatorInput`
5. Converts to `StateBaseClass` with system/human messages
6. Rest of graph executes normally

**Evaluation Mode:**
1. `build_evaluation_graph({"expression": "15.0 + 7.0 * 3.0"})` called
2. Input validated and captured in closure
3. Graph returned with input pre-bound
4. Later: `graph.ainvoke({})` - empty dict because input already bound
5. `prepare_with_bound_input` closure executes, using captured input
6. Returns `StateBaseClass` with messages
7. Rest of graph executes normally

## Key Implementation Details

### Why Two Separate Builders?

**Problem:** Cannot use the same graph for both modes because:
- CLI mode needs to accept input at `graph.ainvoke(input)` time
- Evaluation mode needs input pre-bound to avoid passing it multiple times

**Solution:** Separate builders optimized for each use case:
- CLI: `prepare_input` node signature: `(graph_input: CalculatorInput) -> StateBaseClass`
- Evaluation: `prepare_input` node signature: `(state: StateBaseClass | None) -> StateBaseClass` with input captured in closure

### Why Closure Instead of `partial`?

Using `partial(self.prepare_input_node, agent_input=input)` causes conflicts because:
- LangGraph passes state as first positional argument
- `partial` binds `agent_input` as keyword argument
- Result: `TypeError: got multiple values for argument 'agent_input'`

Closure avoids this by accepting the positional arg and ignoring it:
```python
def prepare_with_bound_input(state: StateBaseClass | None = None) -> StateBaseClass:
    # Ignore incoming state, use captured input from closure
    return self.prepare_input_node(agent_input=final_agent_input)
```

### How Output Schema Works

The `output_schema` is returned by wrapping the `end_execution` tool in a node function:

```python
def end_execution_node(state: StateBaseClass) -> BaseModel:
    """Wrapper node that calls end_execution tool and returns output_schema."""
    self.scenario.end_execution_tool.run(state)
    # Return the output_schema populated from state.result
    return self.scenario.output_schema.model_validate(state.result)

graph.add_node("end_execution", end_execution_node)
```

**Why this approach:**
- The `end_execution` tool validates the LLM's args against the schema
- Stores the validated data in `state.result`
- The wrapper node converts `state.result` to the proper `output_schema` type
- LangGraph recognizes this as the final output since it matches `output_schema=`

**Result:** When the graph completes, you get properly typed output:
```json
{
  "answer": 36.0
}
```

Instead of just an empty dict `{}`.

## Adding New Agents

To add a new agent with proper typing:

1. **Define input/output schemas in `graph.py`:**
```python
class MyAgentInput(BaseModel):
    query: str
    max_results: int = 5

class MyAgentOutput(BaseModel):
    results: List[str]
```

2. **Add to `get_agents()` in `graph.py`:**
```python
def get_agents() -> Dict[str, AgentBaseClass]:
    return {
        "my_agent": AgentBaseClass(
            system_prompt="...",
            user_prompt="Process: {query}",
            input_schema=MyAgentInput,
            output_schema=MyAgentOutput,
            tools=[...],
            end_execution_tool=EndExecutionTool(
                args_schema={
                    "type": "object",
                    "properties": {
                        "results": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["results"]
                },
                output_schema=MyAgentOutput,  # IMPORTANT: Pass output_schema here!
            ),
            datapoints=get_my_datapoints(),
        ),
    }
```

**Note:** The `end_execution_tool` needs both:
- `args_schema`: JSON schema the LLM uses when calling the tool
- `output_schema`: The Pydantic model for the final graph output

3. **Create typed entry point in `graph.py`:**
```python
@asynccontextmanager
async def my_agent(
    agent_input: MyAgentInput | None = None
) -> AsyncGenerator[StateGraph, None]:
    """Pre-configured entry point for my_agent."""
    agent_scenario = get_agents()["my_agent"]
    loop = BasicLoop(scenario=agent_scenario, llm=get_model(), ...)
    graph = loop.build_cli_graph()
    yield graph
```

4. **Register in `langgraph.json`:**
```json
{
  "graphs": {
    "my_agent": "./src/gym_sample/graph.py:my_agent"
  }
}
```

5. **Use from CLI:**
```bash
uv run uipath run my_agent '{"query": "test", "max_results": 10}'
```

## Benefits of This Approach

✅ **Type Safety**: Proper Pydantic models for CLI inputs
✅ **Flexibility**: Support both single execution and batch evaluation
✅ **Testability**: Easy to run comprehensive evaluations
✅ **Clarity**: Clear separation between execution modes
✅ **No Conflicts**: Each mode has its own optimized graph builder
✅ **Follows Patterns**: Based on ticket-classification example

## Common Patterns

### Running specific datapoints
```bash
# Via environment variable
AGENT_INPUT=3 uv run uipath run calculator

# Or just run evaluation mode
uv run python -m gym_sample.run --verbose
```

### Programmatic usage
```python
# CLI mode - pass input at runtime
async with calculator_agent() as graph:
    compiled = graph.compile()
    result = await compiled.ainvoke(CalculatorInput(expression="2 + 2"))
    print(result)

# Evaluation mode - input pre-bound
async with agents_with_datapoints("calculator") as graphs:
    for graph, datapoint in graphs:
        compiled = graph.compile()
        result = await compiled.ainvoke({})  # Empty dict!
        print(f"Result for {datapoint.input}: {result}")
```

## Troubleshooting

**"got multiple values for argument"** → Using `partial` instead of closure in evaluation mode
**"Field required"** in CLI mode → Check `input_schema=` is set correctly
**401 Unauthorized** → Run `uv run uipath auth` first
**Input not being used** in evaluation → Check `ainvoke({})` not `ainvoke(input)`
