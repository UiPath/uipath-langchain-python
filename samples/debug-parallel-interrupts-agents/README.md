# Debug Agent

A LangGraph agent with 2 parallel branches with multiple interrupts for testing breakpoints and step-by-step debugging scenarios.

## Installation

```bash
uv venv -p 3.11 .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

## Usage

```bash
uipath debug agent '{}'
```
