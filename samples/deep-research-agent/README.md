# Deep Research Agent

Research agent using deep agents with web search and subagents.

> **Note:** This sample uses source files from `uipath-langchain-python` using a local reference. Changes to `src/uipath_langchain/` are reflected immediately without reinstalling.

## Setup

```bash
uv sync
uv run uipath auth --alpha
```

Configure `web_search_config.json` with your UiPath Integration Tool settings.

Set in `.env`:
```
LANGCHAIN_RECURSION_LIMIT=100
```

## Usage

```bash
uv run uipath run agent '{"messages": [{"role": "user", "content": "Research the impact of AI in the field of gene sequencing "}]}' --output-file result.txt
```

## Architecture

```
Main Agent (Research Lead)
├── researcher (UiPath Web Search)
└── reviewer (Quality check)
```
