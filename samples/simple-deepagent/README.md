# Simple DeepAgent

A research agent using the DeepAgents library with planning, sub-agent delegation, and web search capabilities.

## Overview

This sample demonstrates the DeepAgents framework, which provides advanced agentic patterns beyond simple tool-calling loops. DeepAgents combines:
- **Planning** through task decomposition
- **Subagent spawning** for specialized tasks (researcher and critic)
- **File system access** for memory management
- **Tool use** with Tavily search

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
uipath run agent '{"messages": [{"type": "human", "content": "Research the history of artificial intelligence"}]}'
```

The agent will:
1. Break down complex research questions into sub-tasks
2. Delegate research to the specialized **researcher** subagent
3. Use the **critic** subagent to review and provide feedback on findings
4. Use web search to gather information
5. Organize findings into a structured response

## How It Works

Unlike simple ReAct agents, DeepAgents can:
- Plan multiple steps ahead using a built-in task decomposition tool
- Spawn specialized subagents with isolated context:
  - **Researcher**: Gathers information using web search
  - **Critic**: Reviews outputs for quality and completeness
- Maintain state through filesystem access
- Handle complex, multi-step research workflows with iterative refinement
