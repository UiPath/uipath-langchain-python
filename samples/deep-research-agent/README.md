# Deep Research Agent

Research agent using deep agents with web search and subagents.

> **Note:** This sample uses source files from `uipath-langchain-python` using a local reference. Changes to `src/uipath_langchain/` are reflected immediately without reinstalling.

## Setup

```bash
uv sync
uv run uipath auth --alpha
```

## Search Provider

The agent supports two search providers, controlled by the `SEARCH_PROVIDER` environment variable:

### UiPath Integration

Uses UiPath's built-in web search integration.

```bash
# No extra setup needed, uses web_search_config.json
SEARCH_PROVIDER=uipath
```

Configure `web_search_config.json` with your UiPath Integration Tool settings.

### Tavily (default)

Uses Tavily search API.

```bash
# Install tavily dependency
uv sync --extra tavily
```

Set in `.env`:
```
SEARCH_PROVIDER=tavily
TAVILY_API_KEY=your_tavily_api_key
```

## Configuration

Set in `.env`:
```
LANGCHAIN_RECURSION_LIMIT=1000
```

## Usage

```bash
uv run uipath run agent '{"messages": [{"role": "user", "content": "Research the impact of AI in the field of gene sequencing "}]}' --output-file result.txt
```

## Architecture

```
Main Agent (Research Lead)
├── researcher (Web Search)
└── reviewer (Quality check)
```
