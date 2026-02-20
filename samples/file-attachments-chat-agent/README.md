# File Attachments Chat Agent

An AI assistant that reads and analyzes file attachments shared in the conversation.

## Requirements

- Python 3.11+
- OpenAI API key

## Installation

```bash
uv venv -p 3.11 .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

Set your API key as an environment variable in .env

```bash
OPENAI_API_KEY=your_openai_api_key
```

## Usage

**1.** Upload the file to Orchestrator using the [attachments API](https://uipath.github.io/uipath-python/core/attachments/).

**2.** Run the agent, passing the attachment ID and file metadata returned by the upload:

```bash
uipath run agent '{
  "messages": [
    {
      "type": "human",
      "content": [
        { "type": "text", "text": "Summarize this document." },
        { "type": "text", "text": "<uip:attachments>[{\"id\": \"{orchestrator_attachment_id}\", \"full_name\": \"{file_name}\", \"mime_type\": \"{file_mime_type}\"}]</uip:attachments>" }
      ]
    }
  ]
}'
```
