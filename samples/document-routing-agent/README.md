# Document Routing Agent

An intelligent document routing system that automatically classifies and routes documents to appropriate departments based on content analysis, without using LLMs.

## Overview

This agent demonstrates the use of the UiPath LangChain SDK with deterministic rule-based routing. It analyzes document content, metadata, and patterns to make routing decisions with confidence scoring.

## How It Works

1. **Document Preparation**: Generates a unique routing ID and initializes processing
2. **Content Analysis**: Scans document for keywords, document types, and patterns
3. **Score Calculation**: Computes routing scores for each department based on:
   - Keyword matches (up to 50 points)
   - Document type matches (up to 30 points)
   - Sender department alignment (up to 20 points)
   - Tag matches (up to 10 points)
4. **Routing Decision**: Selects the highest-scoring department with confidence assessment
5. **Priority Assignment**: Calculates priority (0-100) based on urgency and department
6. **Processing Time Estimation**: Provides estimated processing time based on priority

## Routing Rules

### Departments and Keywords

- **HR**: employee, vacation, salary, benefits, leave, onboarding, performance, training
- **Finance**: invoice, payment, budget, expense, revenue, tax, audit, financial
- **Legal**: contract, agreement, compliance, legal, lawsuit, regulation, policy, terms
- **IT**: software, hardware, system, network, security, database, server, technical
- **Operations**: production, logistics, supply, inventory, quality, process, workflow, maintenance

### Document Types

Each department has preferred document types (e.g., Finance handles invoices, HR handles leave requests).

## Agent Setup and Publishing


1. **Set Up Python Environment**

```bash
# Install UV package manager
pip install uv

# Create and activate virtual environment
uv venv -p 3.11 .venv

# Windows
.venv\Scripts\activate

# Unix-like Systems
source .venv/bin/activate

# Install dependencies
uv sync
```

2. **UiPath Authentication**

```bash
uipath auth
```

> **Note:** After successful authentication in the browser, select the tenant for publishing the agent package.

```
👇 Select tenant:
  0: DefaultTenant
  1: Tenant2
  2: Tenant3
...
Select tenant: 2
```

3. **Package and Publish**

```bash
# Create and publish the package
uipath pack
uipath publish
```

Select the feed to publish your package:

```
👇 Select package feed:
  0: Orchestrator Tenant Processes Feed
  1: Orchestrator Folder1 Feed
  2: Orchestrator Folder2 Feed
  3: Orchestrator Personal Workspace Feed
  ...
Select feed number: 3
```

> Note: When publishing to personal workspace feed, the process will be auto-created for you.

## Running Locally

```bash
# setup environment first, then run:
uipath init
uipath run agent --file ./test_cases/input/<name_of_your_test_case>.json
```

## Input Schema

```json
{
    "document_id": "string",
    "document_type": "string",
    "title": "string",
    "content": "string",
    "sender_email": "string",
    "sender_department": "string (optional)",
    "urgency_level": "low|medium|high|critical",
    "tags": ["array", "of", "strings"]
}
```

## Output Schema

```json
{
    "document_id": "string",
    "routing_id": "string",
    "assigned_department": "string",
    "routing_confidence": "low|medium|high",
    "priority_score": 0-100,
    "routing_reasons": ["array", "of", "reasons"],
    "secondary_departments": ["array", "of", "departments"],
    "processing_timestamp": "ISO timestamp",
    "estimated_processing_time": "string"
}
```

## Test Cases

The agent includes 5 deterministic test cases:

1. **Finance Invoice**: High-urgency software license invoice → Routes to Finance
2. **HR Leave Request**: Employee vacation request → Routes to HR
3. **Legal Contract**: Critical service agreement amendment → Routes to Legal
4. **IT Incident**: Critical database server outage → Routes to IT
5. **Operations Work Order**: Low-priority maintenance request → Routes to Operations

