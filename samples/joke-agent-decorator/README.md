# Joke Agent (Decorator-based Guardrails)

A simple LangGraph agent that generates family-friendly jokes based on a given topic using UiPath's LLM. This version demonstrates guardrail decorators instead of middleware-based guardrails.

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

The agent accepts a simple topic-based input:

```json
{
    "topic": "banana"
}
```

The `topic` field should be a string representing the subject for the joke. The agent will automatically convert this to the appropriate message format internally.

### Output Format

```json
{
    "joke": "Why did the banana go to the doctor? Because it wasn't peeling well!"
}
```

## Features

- Generates family-friendly jokes appropriate for all ages
- Uses UiPath's LLM (UiPathChat) for joke generation
- LangChain compatible implementation using `create_agent`
- Custom LoggingMiddleware that logs input and output
- **Decorator-based guardrails** - Uses `@pii_guardrail` decorator instead of middleware

## Guardrails

This agent uses decorator-based guardrails instead of middleware. Guardrails are applied using the `@pii_guardrail` decorator directly on the LLM and agent, which evaluates guardrails using `uipath.guardrails.evaluate_guardrail()` directly.

### Decorator Approach

The decorator approach allows you to apply guardrails directly to objects without needing to configure middleware:

```python
# Apply to LLM
llm = pii_guardrail(
    entities=[Entity(PIIDetectionEntity.EMAIL, 0.5)],
    action=LogAction(severity_level=AgentGuardrailSeverityLevel.WARNING),
    name="LLM PII Detection",
)(UiPathChat(model="gpt-4o"))

# Apply to agent function
@pii_guardrail(
    entities=[Entity(PIIDetectionEntity.EMAIL, 0.5)],
    action=LogAction(severity_level=AgentGuardrailSeverityLevel.WARNING),
    name="Agent PII Detection",
)
def create_agent():
    return create_agent(...)
```

### Differences from Middleware Approach

- **Direct evaluation**: Decorators call `uipath.guardrails.evaluate_guardrail()` directly, not through middleware
- **Scope inference**: Scope is automatically detected (LLM for LLM instances, AGENT for agent functions)
- **Simpler API**: No need to specify scopes or create middleware instances
- **Works in custom loops**: Decorators work even when not using LangChain's middleware system

## Example Topics

Try different topics like:
- "banana"
- "computer"
- "coffee"
- "pizza"
- "weather"
