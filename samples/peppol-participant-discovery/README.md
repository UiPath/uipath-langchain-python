# PEPPOL Participant Discovery

UiPath agent that extracts company information from email signatures and discovers PEPPOL participants using a two-stage lookup with LLM refinement.

## Features

- **Two-stage PEPPOL lookup**: Naive search → LLM refinement → Refined search
- **LangGraph pipeline**: 7-node workflow with conditional routing
- **Email signature parsing**: Extract company name, country, email, domain
- **Automatic queue integration**: Queue discovered participants for downstream processing

## Input Schema

```json
{
  "signature_text": "string (optional)",
  "fetch_from_api": "boolean (default: false)",
  "perform_peppol_lookup": "boolean (default: true)"
}
```

## Output Schema

```json
{
  "signature_id": "string",
  "source": "string",
  "company_name": "string",
  "country": "string",
  "email": "string",
  "domain": "string",
  "peppol_found": "boolean",
  "peppol_participant_id": "string | null",
  "peppol_entities": "array | null",
  "search_method": "naive | refined | null",
  "refined_company_name": "string | null",
  "validation_status": "valid | not_found | error",
  "confidence_score": "number (0.0-1.0)",
  "error": "string | null"
}
```

## Setup

```bash
cd samples/peppol-participant-discovery
uv sync
cp .env.example .env
# Edit .env with your credentials
```

### Required Environment Variables

- `UIPATH_URL` - Your UiPath Orchestrator URL
- `UIPATH_ACCESS_TOKEN` - Access token for authentication
- `OPENAI_API_KEY` - OpenAI API key (or compatible provider)
- `OPENAI_BASE_URL` - API base URL (e.g., OpenRouter)
- `MODEL` - Model to use (e.g., `anthropic/claude-3-haiku`)

### Optional Variables

- `API_ENDPOINT` - Company Data Hub API endpoint (for fetching test signatures)
- `COMPANYDATAHUB_API_KEY` - API key for Company Data Hub

## Usage

### Run with Manual Input

```bash
uv run uipath run src/peppol_participant_discovery/agent.py '{"signature_text": "John Doe, CEO\nAcme Corp\n123 Main St, Berlin 10115 DE"}'
```

### Run with API Fetch

```bash
uv run uipath run src/peppol_participant_discovery/agent.py '{"fetch_from_api": true}'
```

### Skip PEPPOL Lookup (Extract Only)

```bash
uv run uipath run src/peppol_participant_discovery/agent.py '{"signature_text": "Test Corp, Berlin DE", "perform_peppol_lookup": false}'
```

## Architecture

### LangGraph Pipeline

7-node state machine:
1. **Extract Data** - Parse email signature
2. **Naive PEPPOL Search** - Direct lookup with extracted company name
3. **Refine Company Name** - LLM refinement if naive search fails
4. **Refined PEPPOL Search** - Retry with refined name
5. **Fetch Participant Details** - Get full PEPPOL entity data
6. **Finalize Results** - Prepare output
7. **Queue Integration** - Write successful matches to UiPath queue

### Search Strategy

- **Naive Search**: Direct company name lookup (confidence: 1.0)
- **Refined Search**: LLM removes legal suffixes, standardizes format (confidence: 0.7)

### Queue Integration

When a PEPPOL participant is found, the agent automatically creates a queue item in the `mailsig-to-peppol` queue with:
- Slugified reference (e.g., `bounce-gmbh-de`)
- Full signature payload
- Extracted company data
- PEPPOL participant ID and entities
- Search method used

## Project Structure

```
peppol-participant-discovery/
├── src/
│   └── peppol_participant_discovery/
│       ├── agent.py                 # Main entry point
│       └── lib/
│           ├── config.py            # Settings management
│           ├── models.py            # Pydantic models
│           ├── api/                 # API clients (Company Hub, PEPPOL)
│           ├── extractors/          # Data extraction logic
│           ├── llm/                 # LLM integration & prompts
│           └── pipeline/            # LangGraph workflow
├── pyproject.toml
├── uipath.json
├── .env.example
└── README.md
```

## Development

### Update Schema

After modifying Input/Output models:

```bash
uv run uipath init src/peppol_participant_discovery/agent.py --infer-bindings
```

## Related Resources

- [PEPPOL Network](https://peppol.org/)
- [UiPath Documentation](https://docs.uipath.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
