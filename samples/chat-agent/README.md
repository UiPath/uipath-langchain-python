# Movie Chat Agent

An AI assistant using LangGraph and Tavily search for movie research and recommendations.

## Requirements

- Python 3.11+
- Anthropic API key
- Tavily API key

## Installation

```bash
uv venv -p 3.11 .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

Set your API keys as environment variables in .env

```bash
ANTHROPIC_API_KEY=your_anthropic_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Usage

```bash
uipath run agent '{"messages": [{"type": "human", "content": "Tell me about the movie Inception"}]}'
```
