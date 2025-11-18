# Customer Support Ticket Routing Agent

A LangGraph agent with 10 nodes for testing breakpoints and step-by-step debugging scenarios.

## Installation

```bash
uv venv -p 3.11 .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

## Usage

```bash
uipath run agent '{"ticket_id": "T-12345", "customer_message": "This is urgent! Your service is terrible!", "customer_tier": "premium"}'
```

### Input

```json
{
    "ticket_id": "T-12345",
    "customer_message": "Your service is not working!",
    "customer_tier": "standard"
}
```

- `customer_tier`: `"standard"` or `"premium"` (default: `"standard"`)

### Output

```json
{
    "ticket_id": "T-12345",
    "category": "technical",
    "priority": "high",
    "assigned_department": "Engineering",
    "requires_escalation": false,
    "estimated_resolution_time": 24,
    "response_template": "Thank you for contacting us..."
}
```

## Debugging

The agent has 10 nodes ideal for breakpoint testing:

1. `analyze_sentiment`
2. `classify_category`
3. `check_urgency`
4. `determine_priority`
5. `check_escalation`
6. `route_to_department`
7. `escalate_to_manager` (conditional)
8. `assign_standard_queue` (conditional)
9. `generate_response`
10. `finalize_ticket`

Set breakpoints on any node to step through the routing logic.

## Test Cases

```bash
# Escalation scenario
uipath run agent '{"ticket_id": "T-001", "customer_message": "Terrible service! I demand a refund!", "customer_tier": "premium"}'

# Standard routing
uipath run agent '{"ticket_id": "T-002", "customer_message": "How do I reset my password?"}'

# Technical issue
uipath run agent '{"ticket_id": "T-003", "customer_message": "The app is broken and not loading"}'
```
