# Human-in-the-Loop Inbox Server

A FastAPI server for managing human-in-the-loop workflows with job submissions and inbox message approvals.

## Requirements

- Python 3.11+
- UiPath Orchestrator access

## Installation

```bash
uv venv -p 3.11 .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

Configure your UiPath credentials in a `.env` file:

```bash
UIPATH_URL=https://cloud.uipath.com/<organization>/<tenant>
UIPATH_ACCESS_TOKEN=your_access_token
```

## Usage

Start the server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The web interface will be available at `http://localhost:8000`

### API Endpoints

```bash
# Submit a job
POST /api/submit-job
Body: {"topic": "your research topic"}

# Get job status
GET /api/job/{job_id}

# Get inbox messages
GET /api/inbox?status=pending

# Approve message
POST /api/approve-message/{message_id}
Body: {"human_feedback": "approved"}

# Webhook for job events (configured in UiPath Orchestrator)
POST /webhook/job-complete
```

## Webhook Configuration

Set up webhooks in UiPath Orchestrator pointing to:
```
http://your-server:8000/webhook/job-complete
```

Subscribe to: Job Started, Job Suspended, Job Completed, Job Faulted
