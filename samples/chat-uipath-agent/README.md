# Movie Chat Agent

An AI assistant using LangGraph's built-in react agent with UiPath LLM and DuckDuckGo search for movie research and recommendations.

## Requirements

- Python 3.11+

## Installation

```bash
uv venv -p 3.11 .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

## Usage

```bash
uipath run agent '{"messages": [{"type": "human", "content": "Tell me about the movie Inception"}]}'
```
