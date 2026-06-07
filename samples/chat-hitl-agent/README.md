# Culinary Research & Recipe Assistant

An AI-powered culinary assistant using LangGraph with Human-in-the-Loop (HITL) capabilities. The agent uses Claude 3.7 Sonnet and Tavily search to research ingredients, provide recipes, and answer cooking questions with approval gates for tool execution.

## Features

- **Culinary Expertise**: Research ingredients, cooking methods, and food science
- **Recipe Assistance**: Provide accurate recipes and ingredient substitutions
- **Human-in-the-Loop**: Request approval before executing web searches
- **Powered by Claude 3.7 Sonnet**: Advanced reasoning for cooking and food-related queries
- **Web Search Integration**: Access current culinary information via Tavily

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

Set your API keys as environment variables in `.env`:
```bash
ANTHROPIC_API_KEY=your_anthropic_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Usage

### Run the agent with a query:
```bash
uipath run agent '{"messages": [{"type": "human", "content": "How do I make a perfect omelette?"}]}'
```

### Debug mode with interactive console:
```bash
uipath debug agent '{"messages": [{"type": "human", "content": "Tell me about sourdough starters"}]}'
```

### Example queries:
```bash
# Recipe request
uipath run agent '{"messages": [{"type": "human", "content": "Give me a recipe for Thai green curry"}]}'

# Ingredient substitution
uipath run agent '{"messages": [{"type": "human", "content": "What can I use instead of eggs in baking?"}]}'

# Cooking technique
uipath run agent '{"messages": [{"type": "human", "content": "Explain how to properly caramelize onions"}]}'

# Food science
uipath run agent '{"messages": [{"type": "human", "content": "Why does adding salt to pasta water matter?"}]}'

# Approve tool call
uipath run agent '{"decisions": [{"type": "approve"}]}' --resume
```

## Human-in-the-Loop

The agent uses `HumanInTheLoopMiddleware` to request approval before executing Tavily searches. When a search is triggered:

1. The agent pauses and requests approval
2. You can approve, edit, or reject the tool call
3. The agent continues based on your decision

This ensures you maintain control over external API calls and can review what information the agent will fetch.

## Agent Configuration

The agent is configured with:
- **Model**: `claude-3-7-sonnet-latest` (Claude 3.7 Sonnet)
- **Tools**: `TavilySearch` with max 5 results
- **Middleware**: Human-in-the-Loop with interrupts on `tavily_search`
- **System Prompt**: Specialized for culinary assistance

See `agent.py` for full implementation details.
