# Calibration Dispatcher Agent

A production-grade autonomous agent for medical device calibration scheduling using UiPath SDK, LangGraph, and Context Grounding.

## Overview

This agent automates the complex process of scheduling medical equipment calibration visits across multiple healthcare facilities. It demonstrates advanced UiPath integration patterns including:

- **LangGraph StateGraph** workflow with Human-in-the-Loop (HITL) via Action Center
- **Context Grounding** for policy retrieval using Orchestrator Storage Buckets
- **Data Fabric** for equipment, clinic, and technician management
- **Google Maps API** integration for route optimization
- **MCP Server** integration for RPA workflow execution
- **Dynamic constraint management** with manager override capabilities

### Business Value

- **99% faster planning**: 2-4 hours manual scheduling → 2-3 minutes automated
- **27% route reduction**: Optimized waypoint sequencing via Google Maps
- **100% error elimination**: Automated constraint enforcement and SLA management

## Features

### Core Capabilities

1. **Intelligent Route Planning**
   - Priority-based device grouping (OVERDUE, URGENT, SCHEDULED)
   - SLA-aware scheduling (24h/48h/72h response times)
   - City-based clustering for efficient routing
   - Technician specialization matching

2. **Human-in-the-Loop Approval**
   - Action Center integration with revision support
   - Manager override for constraints
   - Approval, Rejection, and Change Request workflows
   - Automatic revision tracking (max 3 iterations)

3. **Context Grounding RAG**
   - Policy retrieval from Orchestrator Storage Buckets
   - Calibration rules, routing guidelines, service procedures
   - Constraint extraction and enforcement
   - Fallback to default values if retrieval fails

4. **Route Optimization**
   - Google Maps API waypoint optimization
   - Distance and duration calculations
   - Traffic-aware routing
   - Multi-city support

5. **Notification & Tracking**
   - Email notifications via RPA workflows
   - Slack integration (optional)
   - Service order creation in Data Fabric
   - Audit trail for all decisions

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Calibration Dispatcher Agent              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │  Equipment   │────────►│   Analysis   │                 │
│  │  Status      │         │   & Grouping │                 │
│  └──────────────┘         └──────┬───────┘                 │
│         │                         │                          │
│         │                         ▼                          │
│         │                 ┌──────────────┐                  │
│         │                 │   Context    │                  │
│         │                 │  Grounding   │◄─────Storage     │
│         │                 │    (RAG)     │      Bucket      │
│         │                 └──────┬───────┘                  │
│         │                         │                          │
│         ▼                         ▼                          │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │  Route       │────────►│  Human       │                 │
│  │  Optimization│         │  Approval    │                 │
│  └──────────────┘         │  (HITL)      │                 │
│                           └──────┬───────┘                  │
│                                  │                           │
│                                  ▼                           │
│                           ┌──────────────┐                  │
│                           │     RPA      │                  │
│                           │  Workflows   │◄─────MCP Server  │
│                           └──────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

## Technology Stack

- **Python 3.11+** with LangChain/LangGraph
- **UiPath SDK** for platform integration
- **UiPath Context Grounding** for RAG pattern
- **UiPath Data Fabric** for entity management
- **UiPath Action Center** for HITL workflows
- **UiPath MCP Server** for RPA integration
- **Google Maps API** for route optimization
- **OpenAI GPT-4** via UiPath LLM Gateway

## Quick Start

### Prerequisites

- UiPath Automation Cloud account with:
  - Data Service enabled
  - AI Trust Layer access (Context Grounding)
  - Action Center application deployed
  - (Optional) MCP Server with RPA workflows
- Python 3.11 or newer
- Google Maps API key (optional, for route optimization)
- UiPath CLI installed and authenticated

### Installation

1. **Clone and Navigate**

```bash
git clone https://github.com/UiPath/uipath-langchain-python.git
cd uipath-langchain/samples/calibration-dispatcher-agent
```

2. **Create Virtual Environment**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Authenticate with UiPath**

```bash
uipath auth
```

Select your tenant and organization when prompted.

## Configuration

All environment-specific settings are centralized in `config.py`. Update these values according to your UiPath environment:

### Critical Configuration

1. **Data Fabric Entity IDs** (Required)

```python
# In config.py or .env
EQUIPMENT_ENTITY_ID="your-equipment-entity-id"
CLINICS_ENTITY_ID="your-clinics-entity-id"
TECHNICIANS_ENTITY_ID="your-technicians-entity-id"
```

Find these IDs in: **Orchestrator > Data Service > Entities > [Entity Name] > Details**

2. **Context Grounding Index** (Required)

```python
# In config.py or .env
CONTEXT_GROUNDING_INDEX_NAME="Calibration Procedures"
```

Create this index in: **Orchestrator > Tenant > Indexes**

3. **Folder Path** (Required)

```python
# In config.py or .env
UIPATH_FOLDER_PATH="Calibration Services"
```

## Setup Guide

- Data Fabric entities and sample data
- Orchestrator Storage Buckets for Context Grounding
- Index creation and management
- Google Maps API configuration
- Action Center application deployment
- MCP Server integration (optional)

## Running the Agent

### Production Mode

```bash
# With full UiPath infrastructure
python3 main.py
```

Expected workflow:
1. Analyzes equipment status from Data Fabric
2. Groups devices by city and priority
3. Retrieves routing constraints from Context Grounding
4. Generates optimized routes with Google Maps
5. Presents routes for approval in Action Center
6. Executes RPA workflows (email, Slack, Data Fabric updates)

### Mock Mode (Local Testing)

For quick testing without full UiPath setup:

```python
# In config.py or .env
USE_MOCK_DATA=true
AUTO_APPROVE_IN_LOCAL=true
USE_MCP=false
```

**Note**: Mock mode relaxes configuration validation and skips Action Center/MCP integration, but still requires Data Fabric with imported CSV data (see Setup section).

Then run:

```bash
python3 main.py
```

## Project Structure

```
calibration-dispatcher-agent/
│
├── 📄 Core Application Files
│   ├── main.py                    # Main agent logic (LangGraph workflow)
│   ├── config.py                  # Centralized configuration
│   ├── mcp_bridge.py              # Async-to-sync MCP tool bridge
│   ├── requirements.txt           # Python dependencies
│   ├── .env.example               # Environment variables template
│   ├── .gitignore                 # Git exclusions
│   └── README.md                  # This file
│
├── 📁 data/                       # Sample data and schema
│   ├── README.md                  # Data directory documentation
│   ├── Schema.json                # Data Fabric entity definitions
│   ├── devices_for_data_fabric.csv  # Sample equipment (20 devices)
│   ├── locations.csv              # Sample clinics (20 locations)
│   └── technicians.csv            # Sample technicians (5 techs)
│
│
└── 📁 policies/                   # Policy documents for Context Grounding
    ├── README.md                  # Policies documentation
    ├── Calibration_Rules_Document.pdf        # Rules, intervals, SLAs
    ├── Routing_Guidelines_Document.pdf       # Route optimization
    └── Service_Procedures_Document.pdf       # Service procedures
```

## Business Logic

### Priority Classification

Devices are classified based on days until next calibration due:

| Status | Audiometer | Tympanometer | Priority | Action |
|--------|-----------|--------------|----------|---------|
| **OVERDUE** | Past due | Past due | Critical | Immediate scheduling |
| **URGENT** | ≤ 14 days | ≤ 7 days | High | Schedule within 48h |
| **SCHEDULED** | > 14 days | > 7 days | Normal | Regular scheduling |

### SLA Requirements

Response times based on clinic classification:

| Clinic Type | SLA | Example |
|------------|-----|---------|
| **Hospital** | 24 hours | Regional hospitals |
| **Specialist Clinic** | 48 hours | Audiology centers |
| **General Practice** | 72 hours | Family clinics |

### Routing Constraints

**Standard Mode:**
- Max 4 visits per route
- Max 200 km total distance
- Max 8 hours total work time

**OVERDUE Override (Emergency Mode):**
- Max 5 visits per route
- Max 300 km total distance  
- Max 12 hours total work time (includes overtime)

Constraints are retrieved from Context Grounding policies and can be overridden by manager notes.

### Technician Specialization

Devices are matched to technicians based on specializations:

| Device Type | Required Specialization |
|------------|------------------------|
| Audiometer | Audiometry or All |
| Tympanometer | Tympanometry or All |

## Extending the Sample

### Adding New Device Types

1. Update `devices_for_data_fabric.csv` with new device records
2. Add specialization mapping in `config.py`:
   ```python
   DEVICE_TO_SPECIALIZATION = {
       "Audiometer": {"Audiometry", "All"},
       "Tympanometer": {"Tympanometry", "All"},
       "Spirometer": {"Respiratory", "All"},  # New device type
   }
   ```
3. Add service time estimation in `config.py`:
   ```python
   SERVICE_TIME_SPIROMETER = float(os.getenv("SERVICE_TIME_SPIROMETER", "1.0"))
   ```

### Adding New Cities

1. Update `locations.csv` with clinic records in the new city
2. Add city coordinates in `config.py`:
   ```python
   CITY_COORDS = {
       "Warsaw": (52.2297, 21.0122),
       "Poznan": (52.4064, 16.9252),
       "Lodz": (51.7592, 19.4560),  # New city
   }
   ```

### Creating Custom Tools

Add new LangChain tools to extend agent capabilities:

```python
@tool
def check_parts_availability(device_type: str) -> dict:
    """Check if spare parts are available for device calibration."""
    # Your implementation
    return {"available": True, "lead_time_days": 2}
```

Then include in the agent's tool list.

## Troubleshooting

### Common Issues

**Configuration Validation Errors**

If you see "Configuration Errors" when running the agent:
- Verify entity IDs are correct (not placeholder `00000000-...`)
- Check that Context Grounding index exists
- Confirm UiPath authentication is valid

**Context Grounding Not Found**

If policy retrieval fails:
- Verify index name matches configuration
- Check that storage bucket contains policy PDFs
- Ensure index has been created and synchronized
- Confirm folder permissions allow access

**Google Maps API Errors**

If route optimization fails:
- Verify API key is valid and active
- Check that Distance Matrix API is enabled
- Ensure billing is configured in Google Cloud Console
- Routes will fall back to straight-line distance if API unavailable

**Action Center Task Not Created**

If HITL approval doesn't work:
- Verify Action Center application is deployed
- Check field names match configuration (`SelectedOutcome`, `ManagerComments`)
- Ensure user has permissions to create tasks
- Try `AUTO_APPROVE_IN_LOCAL=true` for local testing


## Support

For issues or questions:
- Review UiPath SDK documentation
- Contact your UiPath representative

## Acknowledgments

This sample demonstrates patterns from the UiPath Specialist Coded Agent Challenge 2025 (4th place solution).
