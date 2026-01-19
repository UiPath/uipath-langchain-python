# Joke Agent

A simple LangGraph agent that generates family-friendly jokes based on a given topic using UiPath's LLM.

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
- Simple, clean architecture following UiPath agent patterns

## Middleware

The agent includes multiple types of middleware:

### LoggingMiddleware (Custom)

A custom middleware that logs:
- **Input**: The topic/query when the agent starts execution
- **Output**: The generated joke when the agent completes

The middleware is implemented using `AgentMiddleware` from LangChain and demonstrates how to create custom middleware for agents. You can find the implementation in `middleware.py`.

### PIIMiddleware (LangChain Built-in - Example)

The agent uses LangChain's built-in `PIIMiddleware` as an example:
- **Email addresses**: Blocked in user input
- **Credit card numbers**: Blocked in user input
- **URLs**: Blocked in user input

When any of these PII types are detected in the input, the agent will raise a `PIIDetectionError` and stop execution. This demonstrates the blocking strategy.

### UiPathPIIDetection (UiPath Guardrails)

The agent also uses UiPath's guardrails system for PII detection:
- **Email addresses**: Detected with configurable threshold (0.5)
- **Credit card numbers**: Detected with configurable threshold (0.5)

The guardrail is configured to log warnings when PII is detected at Agent and LLM scopes. This uses UiPath's built-in guardrails service which provides enterprise-grade PII detection.

Example configuration:
```python
UiPathPIIDetection(
    scopes=[GuardrailScope.AGENT, GuardrailScope.LLM],
    action=LogAction(severity_level=AgentGuardrailSeverityLevel.WARNING),
    entities=[
        Entity(PIIDetectionEntity.EMAIL, 0.5),
        Entity(PIIDetectionEntity.CREDIT_CARD_NUMBER, 0.5),
    ],
)
```

When PII is detected, the guardrail logs a warning message but allows execution to continue (LogAction behavior). This demonstrates the logging strategy and shows how to use UiPath's guardrails service.

## Example Topics

Try different topics like:
- "banana"
- "computer"
- "coffee"
- "pizza"
- "weather"
