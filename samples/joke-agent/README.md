# Joke Agent

A LangGraph agent that generates family-friendly jokes based on a given topic using UiPath's LLM. The agent includes comprehensive guardrails for PII detection, prompt injection prevention, and content validation.

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
- Uses UiPath's LLM (UiPathChat) with GPT-4o model
- LangChain compatible implementation using `create_agent`
- Includes a tool for analyzing joke syntax (word count and letter count)
- Comprehensive guardrails system with multiple middleware layers
- Custom logging middleware that logs input and output
- Simple, clean architecture following UiPath agent patterns

## Agent Architecture

The agent is built using LangGraph's `StateGraph` with custom input/output schemas:

- **Input Schema**: `Input` with a `topic` field
- **Output Schema**: `Output` with a `joke` field
- **LLM**: UiPathChat with model `gpt-4o-2024-08-06` and temperature `0.7`

### Tools

The agent includes one tool:

#### `analyze_joke_syntax`

Analyzes the syntax of a joke by counting words and letters. The tool:
- Counts the number of words (split by whitespace)
- Counts the number of letters (alphabetic characters only, excluding spaces and punctuation)
- Returns a formatted string with the analysis results

The agent uses this tool automatically as part of its joke generation process.

## Middleware

The agent includes multiple layers of middleware for logging, PII detection, prompt injection prevention, and content validation:

### LoggingMiddleware (Custom)

A custom middleware that logs:
- **Input**: The topic/query when the agent starts execution
- **Output**: The generated joke when the agent completes

The middleware is implemented using `AgentMiddleware` from LangChain and demonstrates how to create custom middleware for agents. You can find the implementation in `middleware.py`.

### UiPathPIIDetectionMiddleware

The agent uses two instances of UiPath's PII detection middleware:

#### 1. Agent/LLM Scope PII Detector

- **Name**: "My personal PII detector"
- **Scopes**: `AGENT`, `LLM`
- **Action**: `LogAction` with `WARNING` severity level
- **Entities**:
  - Email addresses (threshold: 0.5)
  - Credit card numbers (threshold: 0.5)

This guardrail monitors PII in agent input and LLM interactions, logging warnings when detected but allowing execution to continue.

#### 2. Tool Scope PII Detector

- **Name**: "Tool PII detector"
- **Scopes**: `TOOL`
- **Action**: `LogAction` with `WARNING` severity level
- **Entities**:
  - Email addresses (threshold: 0.5)
  - Credit card numbers (threshold: 0.5)
  - Phone numbers (threshold: 0.5)
- **Tool Names**: `analyze_joke_syntax`

This guardrail specifically monitors the `analyze_joke_syntax` tool for PII, logging warnings when detected.

### UiPathPromptInjectionMiddleware

- **Name**: "Prompt Injection Detection"
- **Scopes**: `LLM`
- **Action**: `BlockAction` (blocks execution when prompt injection is detected)
- **Threshold**: 0.5

This guardrail prevents prompt injection attacks by blocking requests that appear to contain prompt injection attempts at the LLM scope.

### UiPathDeterministicGuardrailMiddleware

The agent includes two deterministic guardrails that use custom rules:

#### 1. Joke Content Validator

- **Name**: "Joke Content Validator"
- **Tool Names**: `analyze_joke_syntax`
- **Stage**: `PRE` (executes before tool invocation)
- **Rules**: Detects if the word "donkey" appears in the joke (case-insensitive)
- **Action**: `CustomFilterAction` that replaces "donkey" with "*"

This guardrail demonstrates how to implement custom filtering actions. When the word "donkey" is detected in tool input, it is automatically filtered and replaced with "*", and a log message is generated.

#### 2. Joke Content Length Limiter

- **Name**: "Joke Content Length Limiter"
- **Tool Names**: `analyze_joke_syntax`
- **Stage**: `PRE` (executes before tool invocation)
- **Rules**: Detects if the joke length exceeds 1000 characters
- **Action**: `BlockAction` (blocks execution when limit is exceeded)

This guardrail prevents excessively long jokes by blocking tool execution when the joke exceeds 1000 characters.

## Custom Actions

The agent demonstrates how to create custom guardrail actions through the `CustomFilterAction` class in `middleware.py`. This action:

- Filters/replaces specific words in tool input
- Logs violations with details showing original and filtered text
- Returns the filtered input data to be used instead
- Works with both string and dictionary input data

This serves as an example for developers who want to implement their own custom guardrail actions beyond the built-in `LogAction` and `BlockAction`.

## Example Topics

Try different topics like:
- "banana"
- "computer"
- "coffee"
- "pizza"
- "weather"

## System Prompt

The agent uses a system prompt that instructs it to:
1. Generate a family-friendly joke based on the given topic
2. Use the `analyze_joke_syntax` tool to analyze the joke's syntax
3. Ensure the output includes the joke
4. Create jokes that are appropriate for children, free from offensive language, clever, and not based on stereotypes

The agent automatically follows this process when generating jokes.
