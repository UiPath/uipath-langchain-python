# Company Research Agent

A UiPath coded agent that conducts comprehensive company research and generates tailored strategies for effective communication and engagement with companies and their key personnel.

## Overview

This agent uses LangGraph and AI to analyze companies, identify key decision-makers, and create customized engagement strategies. It's designed to help sales teams, partnership managers, and business development professionals prepare for strategic engagements.

## Features

- **Comprehensive Company Research**: Gathers detailed information about company history, products, market position, and culture
- **Industry Analysis**: Analyzes competitive landscape and market trends
- **Key Personnel Identification**: Identifies 3-7 key decision-makers with background information and communication recommendations
- **Tailored Engagement Strategies**: Generates 4-6 specific strategies for different engagement scenarios
- **Communication Recommendations**: Provides actionable guidance on how to approach the company
- **Next Steps**: Delivers clear action items for implementation

## Agent Architecture

The agent uses a **multi-step sequential processing pattern** with 6 nodes:

1. **Research Node**: Conducts comprehensive research using AI tools
2. **Analysis Node**: Analyzes research data to create structured company analysis
3. **Personnel Identification Node**: Extracts and profiles key personnel
4. **Strategy Generation Node**: Creates tailored engagement strategies
5. **Recommendations Node**: Generates overall communication guidance
6. **Output Formatting Node**: Structures the final deliverable

### Technology Stack

- **LangGraph**: Orchestrates the multi-step agent workflow
- **UiPath LLM Gateway**: Provides AI capabilities via GPT-4o
- **LangChain**: Enables tool integration and structured outputs
- **Pydantic**: Ensures type-safe data models

## Installation

### 1. Prerequisites

- Python 3.10 or higher
- UV package manager (or pip)
- Tavily API key for web search

### 2. Get Tavily API Key

1. Go to [tavily.com](https://tavily.com)
2. Sign up for a free account
3. Get your API key from the dashboard
4. Free tier includes 1,000 searches/month

### 3. Install Dependencies

```bash
# Navigate to the project directory
cd company-research-agent

# Install dependencies
uv sync
```

### 4. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Tavily API key
# TAVILY_API_KEY=tvly-your-actual-api-key-here
```

On Windows:
```bash
copy .env.example .env
# Then edit .env in your text editor
```

## Usage

### Basic Usage

Run the agent with a company name:

```bash
uv run uipath run agent '{"company_name": "Microsoft"}'
```

### Save Output to File

```bash
uv run uipath run agent '{"company_name": "Microsoft"}' --output-file output.json
```

### With Different Companies

```bash
uv run uipath run agent '{"company_name": "Tesla"}'
uv run uipath run agent '{"company_name": "Amazon"}'
uv run uipath run agent '{"company_name": "Salesforce"}'
```

## Output Structure

The agent returns a comprehensive JSON output with:

```json
{
  "company_overview": "Detailed overview of the company...",
  "industry_context": "Competitive landscape analysis...",
  "key_personnel": [
    {
      "name": "John Doe",
      "role": "CEO",
      "background": "Background information...",
      "communication_style": "Recommended approach..."
    }
  ],
  "engagement_strategies": [
    {
      "strategy_name": "Executive Engagement Strategy",
      "description": "Detailed strategy description...",
      "target_audience": "C-level executives",
      "key_points": ["Point 1", "Point 2", "Point 3"],
      "timing_recommendations": "When to implement..."
    }
  ],
  "communication_recommendations": "Overall communication approach guide...",
  "next_steps": [
    "Action item 1",
    "Action item 2"
  ]
}
```

## Example Output

When researching Microsoft, the agent identified:

- **3 Key Personnel**: Satya Nadella (CEO), Brad Smith (President), Amy Hood (CFO)
- **6 Engagement Strategies**:
  - Executive Engagement Strategy
  - Technical/Product Strategy
  - Partnership Development Strategy
  - Value Proposition Strategy
  - Communication Channel Strategy
  - Follow-up and Nurture Strategy

## Deployment

### Deploy to UiPath Automation Cloud

```bash
# Pack and publish to your workspace
uv run uipath deploy
```

### Publish to Tenant Feed

```bash
uv run uipath publish --tenant
```

### Invoke Remotely

After publishing to your workspace:

```bash
uv run uipath invoke agent '{"company_name": "Tesla"}'
```

## Development

### Run with Debug Mode

```bash
uv run uipath run agent '{"company_name": "Microsoft"}' --debug
```

### Run Interactive Development Interface

```bash
uv run uipath dev
```

## Customization

### Web Search Configuration

The agent uses **Tavily** for real-time web search by default. Tavily provides:
- AI-powered search with context understanding
- Advanced search depth for comprehensive results
- Built-in answer generation from search results
- 1,000 free searches per month

To adjust search parameters, modify `main.py:38-44`:

```python
response = tavily_client.search(
    query=query,
    search_depth="advanced",  # Options: "basic", "advanced"
    max_results=5,            # Increase for more results (uses more quota)
    include_answer=True,      # Get AI-generated summary
    include_raw_content=False,
)
```

### Adding Alternative Search Providers

You can add additional search tools alongside Tavily:

**SerpAPI (Google Search):**
```python
from langchain_community.utilities import SerpAPIWrapper

serp_search = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))

@tool
def google_search(query: str) -> str:
    """Search Google for company information."""
    return serp_search.run(query)
```

**Bing Search:**
```python
from langchain_community.utilities import BingSearchAPIWrapper

bing_search = BingSearchAPIWrapper(bing_subscription_key=os.getenv("BING_API_KEY"))

@tool
def bing_search(query: str) -> str:
    """Search Bing for company information."""
    return bing_search.run(query)
```

Then add the new tools to the ReAct agent:
```python
research_agent = create_react_agent(
    llm,
    tools=[web_search, google_search, bing_search],  # Multiple tools
    prompt="..."
)
```

### Adding UiPath Context Grounding

To search your internal knowledge base:

```python
from uipath_langchain.retrievers import ContextGroundingRetriever

retriever = ContextGroundingRetriever(
    index_name="CompanyKnowledgeBase",
    number_of_results=3
)

@tool
def search_internal_kb(query: str) -> str:
    """Search internal knowledge base for company information."""
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])
```

### Modifying LLM Model

Change the model in `main.py:11`:

```python
# Use a different model
llm = UiPathChat(model="gpt-4o-2024-08-06", temperature=0.7)

# Or use a smaller, faster model
llm = UiPathChat(model="gpt-4o-mini-2024-07-18", temperature=0.5)
```

### Adjusting Personnel Count

Modify the prompt in `main.py:150`:

```python
# Change from "Identify 3-7 key personnel" to:
Identify 5-10 key personnel including:
```

### Adding More Strategy Types

Update the prompt in `main.py:188-194` to request additional strategies:

```python
Create 6-8 specific engagement strategies that include:
1. Executive Engagement Strategy
2. Technical/Product Strategy
3. Partnership Development Strategy
4. Value Proposition Strategy
5. Communication Channel Strategy
6. Follow-up and Nurture Strategy
7. Content Marketing Strategy  # New
8. Event-based Engagement Strategy  # New
```

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# UiPath credentials (if running locally)
UIPATH_BASE_URL=https://cloud.uipath.com/yourorg/yourtenantname
UIPATH_CLIENT_ID=your_client_id
UIPATH_CLIENT_SECRET=your_client_secret

# Or use OAuth token
UIPATH_FOLDER_PATH=Shared
```

### Authenticate with UiPath

```bash
uv run uipath auth login
```

## Troubleshooting

### Missing Tavily API Key

If you see an error like:
```
ValueError: TAVILY_API_KEY environment variable is not set
```

**Solution:**
1. Make sure you created a `.env` file (copy from `.env.example`)
2. Add your Tavily API key: `TAVILY_API_KEY=tvly-your-key`
3. Get a free key from [tavily.com](https://tavily.com)

### Unicode Encoding Error

If you see a `UnicodeEncodeError` at the end of execution, this is a known issue with the Windows console encoding. The agent actually completed successfully. Use the `--output-file` flag to save results:

```bash
uv run uipath run agent '{"company_name": "Company"}' --output-file output.json
```

### Empty Personnel or Strategies

If the output contains empty arrays for personnel or strategies, check:
1. The LLM is responding properly (check HTTP request logs)
2. The structured output parsing is working (add debug logging)
3. Try running with a different company name

### Rate Limiting

If you encounter rate limiting errors, adjust the LLM temperature or add retry logic:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def research_company(state: State) -> State:
    # ... existing code
```

## Project Structure

```
company-research-agent/
   main.py                 # Main agent implementation
   uipath.json            # UiPath configuration and schemas
   pyproject.toml         # Python dependencies
   README.md              # This file
   AGENTS.md              # UiPath agent patterns reference
   CLAUDE.md              # Custom instructions for Claude Code
   .env                   # Environment variables (not in repo)
```

## Best Practices

1. **Always use async/await**: All LLM calls should be asynchronous
2. **Handle dict and object responses**: Use the pattern shown in nodes 3 and 4
3. **Provide clear prompts**: The quality of output depends on prompt clarity
4. **Use structured outputs**: Leverage Pydantic models for type safety
5. **Test with various companies**: Different companies may have different data availability

## Contributing

To improve this agent:

1. Add more sophisticated web search integration
2. Enhance error handling and retry logic
3. Add more customization options
4. Improve persona-based communication style recommendations
5. Add evaluation sets for testing output quality

## License

This is a UiPath showcase project for demonstration purposes.

## Support

For issues or questions:
- Review the [UiPath Python SDK documentation](https://uipath.github.io/uipath-python/)
- Check the [AGENTS.md](AGENTS.md) for pattern reference
- Consult the [UiPath Community Forums](https://forum.uipath.com/)

---

**Built with UiPath Python SDK v2.1.78**
