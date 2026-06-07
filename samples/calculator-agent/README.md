# Calculator Agent

A simple LangGraph agent that performs basic arithmetic operations and demonstrates nested traced invocations.

## Requirements

- Python 3.11+

## Installation

```bash
uv venv -p 3.11 .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

## Usage

Run the calculator agent:

```bash
uipath run agent '{"a": 10, "b": 5, "operator": "+"}'
```

### Input Format

```json
{
    "a": 10,
    "b": 5,
    "operator": "+"
}
```

Supported operators: `+`, `-`, `*`, `/`

### Output Format

```json
{
    "result": 15.0
}
```
