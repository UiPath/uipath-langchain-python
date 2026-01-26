# Policy Documents

This directory contains calibration policy documents used by the agent's Context Grounding retrieval system.

## Files

### Calibration_Rules_Document.pdf (302 KB)
Defines calibration intervals, SLA requirements, and priority classification rules:
- Device calibration intervals (365 days for both Audiometers and Tympanometers)
- Status classification (OVERDUE, URGENT, SCHEDULED, ACTIVE)
- SLA thresholds by clinic type (24h/48h/72h)
- Priority matrix and escalation procedures
- Technician specialization requirements
- Cost and time estimates per device type

### Routing_Guidelines_Document.pdf (444 KB)
Field service routing and scheduling optimization rules:
- Daily capacity constraints (max 4 visits, 200km, 8 hours)
- Route optimization principles (nearest neighbor with constraints)
- Geographic clustering rules (city-based prioritization)
- Technician assignment logic (specialization + proximity)
- Multi-device clinic optimization
- Traffic and seasonal considerations
- Google Maps API integration guidelines

### Service_Procedures_Document.pdf (286 KB)
Detailed calibration execution procedures:
- Pre-service preparation checklists
- Step-by-step calibration procedures for each device type
- Quality assurance standards and acceptance criteria
- Troubleshooting common issues
- Safety protocols (electrical, acoustic, hygiene)
- Post-service requirements and documentation
- Technician training requirements

## Usage in Agent

**December 2025 Update**: These documents are now uploaded to **Orchestrator Storage Buckets** and indexed via **Context Grounding Indexes** for RAG (Retrieval-Augmented Generation) pattern.

### Setup Process

1. **Upload to Storage Bucket**:
   - Navigate to **Orchestrator > Tenant > Storage Buckets**
   - Create or select bucket: "calibration-policies"
   - Upload all 3 PDF files to the bucket
   - Verify files appear in bucket file list

2. **Create Context Grounding Index**:
   - Navigate to **Orchestrator > Tenant > Indexes** (AI Trust Layer)
   - Click **Create Index**
   - Name: "Calibration Procedures"
   - Source: **Orchestrator Storage Bucket**
   - Select bucket: "calibration-policies"
   - File types: **PDF**
   - Click **Create** and wait for indexing (5-10 minutes)

3. **Agent Queries Index**:
   - Agent uses ContextGroundingRetriever to query the index
   - Retrieves relevant policy sections based on current task
   - Extracts constraints (max visits, distance, hours)
   - Applies rules to route optimization

### Example Queries

The agent makes queries like:
- "What is the calibration interval for audiometers?"
- "What are the SLA requirements for hospitals?"
- "What is the maximum number of visits per route?"
- "What are the routing constraints for OVERDUE devices?"

Context Grounding returns relevant excerpts which the agent parses to enforce business rules.

## Content Summary

### Key Rules Extracted

| Policy Area | Key Constraints |
|------------|-----------------|
| Calibration Intervals | 365 days (both device types) |
| Status Thresholds | ≤14 days (Audiometer), ≤7 days (Tympanometer) for URGENT |
| Daily Limits | 4 visits, 200km, 8 hours (standard) |
| OVERDUE Override | 5 visits, 300km, 12 hours (emergency) |
| Service Duration | 2.0h (Audiometer), 1.5h (Tympanometer) |
| Specialization | Audiometry/All for Audiometers, Tympanometry/All for Tympanometers |

### Deterministic vs LLM Processing

The agent uses a hybrid approach:
- **LLM Processing**: Initial policy retrieval and constraint extraction
- **Deterministic Logic**: Date calculations, priority sorting, route optimization
- **Fallback Values**: If Context Grounding fails, uses hardcoded defaults from config.py

This ensures reliable operation even if policy retrieval has issues.

## Customization

To adapt policies for your use case:

1. **Modify PDFs**: Edit policy documents with your business rules
2. **Re-upload**: Replace files in Context Grounding index
3. **Update Fallbacks**: Adjust default values in `config.py`
4. **Test**: Verify agent extracts correct constraints

The agent's prompts are designed to be flexible - minor policy changes should work without code modifications.

## Content Format

Documents are structured with:
- Clear section headers
- Numbered lists for rules
- Tables for reference values
- Examples for clarity

This structure optimizes Context Grounding retrieval accuracy.
