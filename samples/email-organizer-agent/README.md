# Email Organizer Agent

Automate the organization of your Outlook inbox using AI-powered rule suggestions and UiPath integration.

## Features

- Fetches emails and folders from Outlook using Microsoft Graph API
- Suggests new and improved rules for organizing emails
- Human-in-the-loop approval for rule suggestions
- Automatically creates folders and rules in Outlook
- Extensible and configurable via `pyproject.toml` and environment variables

## Getting Started

### 1. Clone the Repository

```sh
git clone https://github.com/your-org/uipath-langchain-python.git
cd uipath-langchain-python/samples/email-organizer-agent
```

### 2. Set Up Python Environment

```sh
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Or use `pyproject.toml` with pip:
```sh
pip install .
```

### 3. Configure Environment

- Copy `.env.example` to `.env` and fill in your secrets (API keys, tokens, etc.)
- Update `pyproject.toml` as needed
For `OUTLOOK_CONNECTION_KEY=connection_key`, follow this guide: [UiPath Integration Service Connections](https://docs.uipath.com/integration-service/automation-cloud/latest/user-guide/connections)

### 4. Run the Agent

```sh
uipath run agent --file ./input.json
```
### 5. Resume

To approve the rules and commit them use:
```sh
uipath run agent true --resume
```

### Deployment Guide

To run the email-organizer-agent on the UiPath Cloud Platform, follow this guide:
[Ticket Classification Sample Deployment](https://github.com/UiPath/uipath-langchain-python/tree/main/samples/ticket-classification)
