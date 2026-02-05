# Deep Research Agent

Research agent using deep agents with web search and subagents.

> **Note:** This sample uses source files from `uipath-langchain-python` using a local reference. Changes to `src/uipath_langchain/` are reflected immediately without reinstalling.

## Setup

```bash
uv sync
uv run uipath auth --alpha
```

Set in `.env`:
```
TAVILY_API_KEY=your_key
LANGCHAIN_RECURSION_LIMIT=100
```

## Usage

```bash
uv run uipath run agent '{"messages": [{"role": "user", "content": "Research the impact of AI in the field of gene sequencing "}]}' --output-file result.txt
```

## Architecture

```
Main Agent (Research Lead)
├── researcher (Tavily search)
└── reviewer (Quality check)
```
