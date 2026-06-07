# Company Research Agent

An agent that researches companies and develops outreach strategies using Tavily search.

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
uipath run agent '{"company_name": "Microsoft"}'
```

### Input Format

```json
{
    "company_name": "Microsoft"
}
```

### Output Format

The agent returns a structured report with:
- Company Overview
- Organizational Structure
- Key Decision-Makers
- Outreach Strategy
