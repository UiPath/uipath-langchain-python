# Joke Agent (Decorator-based Guardrails)

A simple LangGraph agent that generates family-friendly jokes based on a given topic using UiPath's LLM. This sample demonstrates all three guardrail decorator types — PII, Prompt Injection, and Deterministic — applied directly to the LLM, agent, and tool without a middleware stack.

## Requirements

- Python 3.11+

## Installation

```bash
uv venv -p 3.11 .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

## Usage

Run the joke agent:

```bash
uv run uipath run agent '{"topic": "banana"}'
```

### Input Format

```json
{
    "topic": "banana"
}
```

### Output Format

```json
{
    "joke": "Why did the banana go to the doctor? Because it wasn't peeling well!"
}
```

## Guardrails Overview

This sample achieves full parity with the middleware-based `joke-agent` sample using only decorators. The table below shows which scope each guardrail covers:

| Decorator | Target | Scope | Action |
|---|---|---|---|
| `@prompt_injection_guardrail` | `create_llm` factory | LLM | `BlockAction` — blocks on detection |
| `@pii_detection_guardrail` | `create_llm` factory | LLM | `LogAction(WARNING)` — logs and continues |
| `@pii_detection_guardrail` | `analyze_joke_syntax` tool | TOOL | `LogAction(WARNING)` — logs email/phone |
| `@deterministic_guardrail` | `analyze_joke_syntax` tool | TOOL (PRE) | `CustomFilterAction` — replaces "donkey" with "[censored]" |
| `@deterministic_guardrail` | `analyze_joke_syntax` tool | TOOL (PRE) | `BlockAction` — blocks jokes > 1000 chars |
| `@deterministic_guardrail` | `analyze_joke_syntax` tool | TOOL (POST) | `CustomFilterAction` — always-on output transform |
| `@pii_detection_guardrail` | `create_joke_agent` factory | AGENT | `LogAction(WARNING)` — logs agent-level PII |

## Guardrail Decorators

### LLM-level guardrails

Stacked decorators on a factory function. The outermost decorator runs first:

```python
@prompt_injection_guardrail(
    threshold=0.5,
    action=BlockAction(),
    name="LLM Prompt Injection Detection",
    enabled_for_evals=False,  # default is True
)
@pii_detection_guardrail(
    entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL, 0.5)],
    action=LogAction(severity_level=LoggingSeverityLevel.WARNING),
    name="LLM PII Detection",
)
def create_llm():
    return UiPathChat(model="gpt-4o-2024-08-06", temperature=0.7)

llm = create_llm()
```

### Tool-level guardrails

`@deterministic_guardrail` applies local rule functions — no UiPath API call. Rules receive the tool input dict and return `True` to signal a violation. `@pii_detection_guardrail` at TOOL scope evaluates via the UiPath guardrails API.

```python
@deterministic_guardrail(
    rules=[lambda args: "donkey" in args.get("joke", "").lower()],
    action=CustomFilterAction(word_to_filter="donkey", replacement="[censored]"),
    stage=GuardrailExecutionStage.PRE,
    name="Joke Content Word Filter",
    enabled_for_evals=False,  # default is True
)
@deterministic_guardrail(
    rules=[lambda args: len(args.get("joke", "")) > 1000],
    action=BlockAction(),
    stage=GuardrailExecutionStage.PRE,
    name="Joke Content Length Limiter",
)
@deterministic_guardrail(
    rules=[],           # empty rules = always apply (unconditional transform)
    action=CustomFilterAction(word_to_filter="words", replacement="words++"),
    stage=GuardrailExecutionStage.POST,
    name="Joke Content Always Filter",
)
@pii_detection_guardrail(
    entities=[
        PIIDetectionEntity(PIIDetectionEntityType.EMAIL, 0.5),
        PIIDetectionEntity(PIIDetectionEntityType.PHONE_NUMBER, 0.5),
    ],
    action=LogAction(severity_level=LoggingSeverityLevel.WARNING),
    name="Tool PII Detection",
)
@tool
def analyze_joke_syntax(joke: str) -> str:
    ...
```

### Agent-level guardrail

```python
@pii_detection_guardrail(
    entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL, 0.5)],
    action=LogAction(
        severity_level=LoggingSeverityLevel.WARNING,
        message="PII detected from agent guardrails decorator",
    ),
    name="Agent PII Detection",
    enabled_for_evals=False,  # default is True
)
def create_joke_agent():
    return create_agent(model=llm, tools=[analyze_joke_syntax], ...)

agent = create_joke_agent()
```

### Custom action

`CustomFilterAction` (defined locally in `graph.py`) demonstrates how to implement a custom `GuardrailAction`. When a violation is detected it replaces the offending word in the tool input dict or string, logs the change, then returns the modified data so execution continues with the sanitised input:

```python
@dataclass
class CustomFilterAction(GuardrailAction):
    word_to_filter: str
    replacement: str = "***"

    def handle_validation_result(self, result, data, guardrail_name):
        # filter word from dict/str and return modified data
        ...
```

## Rule semantics (`@deterministic_guardrail`)

- A rule with **1 parameter** receives the tool input dict (`PRE` stage).
- A rule with **2 parameters** receives `(input_dict, output_dict)` (`POST` stage).
- A rule returns `True` to signal a **violation**, `False` to **pass**.
- **All** rules must detect a violation for the guardrail to trigger. If any rule passes, the guardrail passes.
- **Empty `rules=[]`** always triggers the action (useful for unconditional transforms).

## `enabled_for_evals` override

All decorator guardrails accept `enabled_for_evals` (default `True`). Set it to `False`
when you want runtime guardrail behavior but do not want that guardrail enabled for eval scenarios.

## Verification

To manually verify each guardrail fires, run from this directory:

```bash
uv run uipath run agent '{"topic": "donkey"}'
```

**Scenario 1 — word filter (PRE):** the LLM includes "donkey" in the joke passed to `analyze_joke_syntax`. `CustomFilterAction` replaces it with `[censored]` before the tool executes. Look for `[FILTER][Joke Content Word Filter]` in stdout.

**Scenario 2 — length limiter (PRE):** if the generated joke exceeds 1000 characters, `BlockAction` raises `AgentRuntimeError(TERMINATION_GUARDRAIL_VIOLATION)` before the tool is called.

**Scenario 3 — PII at tool and agent scope:** supply a topic containing an email address:

```bash
uv run uipath run agent '{"topic": "donkey, test@example.com"}'
```

Both the agent-scope and LLM-scope `@pii_detection_guardrail` decorators log a `WARNING` when the email is detected. The tool-scope `@pii_detection_guardrail` logs when the email reaches the tool input.

## Differences from the Middleware Approach (`joke-agent`)

| Aspect | Middleware (`joke-agent`) | Decorator (`joke-agent-decorator`) |
|---|---|---|
| Configuration | Middleware class instances passed to `create_agent(middleware=[...])` | `@decorator` stacked on the target object |
| Scope | Explicit `scopes=[...]` list | Inferred automatically from the decorated object |
| Tool guardrails | `UiPathDeterministicGuardrailMiddleware(tools=[...])` | `@deterministic_guardrail` directly on the `@tool` |
| Custom loops | Not supported (requires `create_agent`) | Works in any custom LangChain loop |
| API calls | Via middleware stack | Direct `uipath.guardrails.evaluate_guardrail()` |

## Example Topics

- `"banana"` — normal run, all guardrails pass
- `"donkey"` — triggers the word filter on `analyze_joke_syntax`
- `"donkey, test@example.com"` — triggers word filter + PII guardrails at all scopes
- `"computer"`, `"coffee"`, `"pizza"`, `"weather"`
