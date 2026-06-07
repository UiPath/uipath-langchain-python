# Joke Agent (Decorator-based Guardrails)

A simple LangGraph agent that generates family-friendly jokes based on a given topic using UiPath's LLM. This sample demonstrates the unified `@guardrail` decorator applied to LLM factories, tools, agent factories, graph nodes, and plain Python functions — without a middleware stack.

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

This sample uses a single unified `@guardrail` decorator with three components:

- **`validator`** — what to check (`CustomValidator`, `PIIValidator`, `PromptInjectionValidator`)
- **`action`** — what to do on violation (`BlockAction`, `LogAction`, or a custom `GuardrailAction`)
- **`stage`** — when to check (`GuardrailExecutionStage.PRE` or `POST`)

| Decorator | Target | Validator | Action |
|---|---|---|---|
| `@guardrail(validator=PromptInjectionValidator(...))` | `create_llm` factory | Prompt Injection | `BlockAction` — blocks on detection |
| `@guardrail(validator=PIIValidator(...EMAIL...))` | `create_llm` factory | PII (Email) | `LogAction(WARNING)` — logs and continues |
| `@guardrail(validator=CustomValidator(...))` | `analyze_joke_syntax` tool | Custom (word check) | `CustomFilterAction` — replaces "donkey" with "[censored]" |
| `@guardrail(validator=CustomValidator(...))` | `analyze_joke_syntax` tool | Custom (length check) | `BlockAction` — blocks jokes > 1000 chars |
| `@guardrail(validator=CustomValidator(...))` | `analyze_joke_syntax` tool | Custom (always true) | `CustomFilterAction` — always-on output transform (POST) |
| `@guardrail(validator=PIIValidator(...EMAIL, PHONE...))` | `analyze_joke_syntax` tool | PII (Email, Phone) | `LogAction(WARNING)` — logs email/phone |
| `@guardrail(validator=PIIValidator(...PERSON...))` | `create_joke_agent` factory | PII (Person) | `BlockAction` — blocks person names |
| `@guardrail(validator=PIIValidator(...PERSON...))` | `joke_node` graph node | PII (Person) | `BlockAction` — blocks person names in node input |
| `@guardrail(validator=CustomValidator(...))` | `format_joke_for_display` function | Custom (word check) | `CustomFilterAction` — replaces "donkey" in display output |

## The `@guardrail` Decorator

### LLM-level guardrails

Stacked decorators on a factory function. The outermost decorator runs first:

```python
@guardrail(
    validator=PromptInjectionValidator(threshold=0.5),
    action=BlockAction(),
    name="LLM Prompt Injection Detection",
    stage=GuardrailExecutionStage.PRE,
)
@guardrail(
    validator=pii_email,
    action=LogAction(severity_level=LoggingSeverityLevel.WARNING),
    name="LLM PII Detection",
    stage=GuardrailExecutionStage.PRE,
)
def create_llm():
    return UiPathChat(model="gpt-4o-2024-08-06", temperature=0.7)

llm = create_llm()
```

### Tool-level guardrails

`CustomValidator` applies local rule functions — no UiPath API call. The validator receives the tool input dict and returns `True` to signal a violation. `PIIValidator` evaluates via the UiPath guardrails API.

```python
@guardrail(
    validator=CustomValidator(lambda args: "donkey" in args.get("joke", "").lower()),
    action=CustomFilterAction(word_to_filter="donkey", replacement="[censored]"),
    stage=GuardrailExecutionStage.PRE,
    name="Joke Content Word Filter",
)
@guardrail(
    validator=CustomValidator(lambda args: len(args.get("joke", "")) > 1000),
    action=BlockAction(title="Joke is too long", detail="The generated joke is too long"),
    stage=GuardrailExecutionStage.PRE,
    name="Joke Content Length Limiter",
)
@guardrail(
    validator=CustomValidator(lambda args: True),
    action=CustomFilterAction(word_to_filter="words", replacement="words++"),
    stage=GuardrailExecutionStage.POST,
    name="Joke Content Always Filter",
)
@guardrail(
    validator=pii_email_phone,
    action=LogAction(
        severity_level=LoggingSeverityLevel.WARNING,
        message="Email or phone number detected",
    ),
    name="Tool PII Detection",
    stage=GuardrailExecutionStage.PRE,
)
@tool
def analyze_joke_syntax(joke: str) -> str:
    ...
```

### Agent-level guardrail

```python
@guardrail(
    validator=PIIValidator(
        entities=[PIIDetectionEntity(PIIDetectionEntityType.PERSON, threshold=0.5)],
    ),
    action=BlockAction(
        title="Person name detection",
        detail="Person name detected and is not allowed",
    ),
    name="Agent PII Detection",
    stage=GuardrailExecutionStage.PRE,
)
def create_joke_agent():
    return create_agent(model=llm, tools=[analyze_joke_syntax], ...)

agent = create_joke_agent()
```

### Graph node guardrail

```python
@guardrail(
    validator=PIIValidator(
        entities=[PIIDetectionEntity(PIIDetectionEntityType.PERSON, threshold=0.5)],
    ),
    action=BlockAction(
        title="Person name detection in topic",
        detail="Person name detected in the node input and is not allowed",
    ),
    name="Node Input PII Detection",
    stage=GuardrailExecutionStage.PRE,
)
async def joke_node(state: Input) -> Output:
    ...
```

### Plain Python function guardrail with `GuardrailExclude`

`@guardrail` also works on plain Python functions. Use `Annotated[..., GuardrailExclude()]` to exclude specific parameters from guardrail evaluation:

```python
@guardrail(
    validator=CustomValidator(lambda args: "donkey" in args.get("topic", "").lower()),
    action=CustomFilterAction(word_to_filter="donkey", replacement="[topic redacted]"),
    stage=GuardrailExecutionStage.PRE,
    name="Topic Word Filter",
)
def format_joke_for_display(
    topic: str,
    joke: str,
    config: Annotated[dict[str, Any], GuardrailExclude()],
) -> str:
    ...
```

The PRE guardrail receives `{"topic": ..., "joke": ...}` — the `config` parameter is excluded.

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

## Validator Types

| Validator | What it does | API call? |
|---|---|---|
| `CustomValidator(fn)` | Calls `fn(args_dict)` — returns `True` for violation | No (local) |
| `PIIValidator(entities=[...])` | Detects PII entities (email, phone, person, etc.) | Yes (UiPath API) |
| `PromptInjectionValidator(threshold=...)` | Detects prompt injection attempts | Yes (UiPath API) |

## Reusable Validators

Validators can be declared once and reused across multiple `@guardrail` decorators:

```python
pii_email = PIIValidator(
    entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL, threshold=0.5)],
)

pii_email_phone = PIIValidator(
    entities=[
        PIIDetectionEntity(PIIDetectionEntityType.EMAIL, threshold=0.5),
        PIIDetectionEntity(PIIDetectionEntityType.PHONE_NUMBER, threshold=0.5),
    ],
)
```

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

Both the agent-scope and LLM-scope PII guardrails log a `WARNING` when the email is detected. The tool-scope PII guardrail logs when the email reaches the tool input.

## Differences from the Middleware Approach (`joke-agent`)

| Aspect | Middleware (`joke-agent`) | Decorator (`joke-agent-decorator`) |
|---|---|---|
| Configuration | Middleware class instances passed to `create_agent(middleware=[...])` | `@guardrail(validator=..., action=..., stage=...)` stacked on the target |
| Scope | Explicit `scopes=[...]` list | Inferred automatically from the decorated object |
| Validator + Action | Bundled inside middleware class | Separate, composable objects |
| Tool guardrails | `UiPathDeterministicGuardrailMiddleware(tools=[...])` | `@guardrail` directly on the `@tool` |
| Plain functions | Not supported | `@guardrail` works on any callable, with `GuardrailExclude` for parameter exclusion |
| Custom loops | Not supported (requires `create_agent`) | Works in any custom LangChain/LangGraph graph |
| API calls | Via middleware stack | Direct `uipath.guardrails.evaluate_guardrail()` for built-in validators; local for `CustomValidator` |

## Example Topics

- `"banana"` — normal run, all guardrails pass
- `"donkey"` — triggers the word filter on `analyze_joke_syntax`
- `"donkey, test@example.com"` — triggers word filter + PII guardrails at all scopes
- `"computer"`, `"coffee"`, `"pizza"`, `"weather"`