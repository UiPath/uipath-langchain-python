# Simplified Expense Approval Agent

A streamlined expense approval workflow that validates expenses and always suspends for manager review via API trigger.

## Features

- **Basic Validation**: Checks receipt, date, and amount limits
- **Always Suspends**: Every expense goes to manager for review
- **API Trigger Resume**: Manager approves/rejects via API
- **Clear Audit Trail**: Processing notes track all decisions
- **No LLM Required**: Pure business logic

## How It Works

### Workflow Steps

1. **Prepare Input**: Generates unique expense ID (EXP-YYYYMMDD-XXXXXXXX)
2. **Validate Expense**: Simple checks for receipt, date, and limits
3. **Manager Review**: Always suspends for approval via API trigger
4. **Finalize**: Returns final decision with notes

### Validation Rules

- **Receipt Required**: For expenses over $25
- **Date Limit**: Must be within 90 days
- **Category Limits**:
  - Travel: $2000
  - Meals: $150
  - Supplies: $500
  - Training: $5000
  - Equipment: $3000
  - Other: $300

## Setup

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

## Deployment

### 1. Package and Publish

```bash
uipath auth
uipath pack
uipath publish
```

### 2. Monitor in Orchestrator

- View suspended expenses awaiting approval
- Check processing logs and audit trails
- Approve/reject directly from Orchestrator UI or call `/orchestrator_/api/JobTriggers/DeliverPayload/{inbox_id}` to resume

## Running Locally

```bash
# setup environment first, then run:
uipath init
uipath run agent --file ./test_cases/input/<name_of_your_test_case>.json

# to approve the expense and commit it use:
uipath run agent --file ./test_cases/input/<name_of_payload>.json --resume
```


## Usage Examples

### Example 1: Valid Expense
```json
{
  "employee_id": "EMP002",
  "employee_name": "John Doe",
  "department": "engineering",
  "expense_amount": 45.50,
  "expense_category": "meals",
  "expense_description": "Team lunch during sprint planning",
  "receipt_attached": true,
  "expense_date": "2025-09-20"
}
```
Result: Validates successfully, suspends for manager review

### Example 2: Expense with Warnings
```json
{
  "employee_id": "EMP003",
  "employee_name": "Alice Johnson",
  "department": "hr",
  "expense_amount": 2500.00,
  "expense_category": "travel",
  "expense_description": "Flight and hotel for conference",
  "receipt_attached": true,
  "expense_date": "2025-09-10"
}
```
Result: Warning (exceeds $2000 limit), suspends for manager review

## Resume Options

When the agent suspends for manager review, you have two options:

### Option 1: Orchestrator UI
In the "Human review required" popup, simply type:
- `true` to approve
- `false` to reject

### Option 2: API Call
Resume via API with JSON payload:
```json
{
  "approved": true
}
```

or

```json
{
  "approved": false
}
```

## Output Format

```json
{
  "expense_id": "EXP-20251001-ABC123",
  "status": "approved",
  "approval_level": "manager",
  "final_amount": 45.5,
  "processing_notes": [
    "Expense report EXP-20251001-ABC123 created at 2025-10-01T11:53:07.354154",
    "Approved on 2025-10-01 11:54",
    "Reimbursement by 2025-10-06"
  ],
  "reimbursement_eta": "2025-10-06"
}
```

## Test Cases

The `test_cases/` folder contains comprehensive test scenarios with **deterministic outputs**:
- `input/` - Test input files with various expense scenarios
- `expected_output/` - Expected results for each test case

**Static Values for Reliable Testing:**
- Expense IDs: Generated from employee_id hash (e.g., EMP001 → EXP-20251001-B86D07B6)
- Creation time: Always 2025-10-01T10:00:00.000000
- Approval time: Always 2025-10-01 10:05
- Reimbursement date: Always 2025-10-06 for approved expenses

This ensures the same input always produces the same output, making tests reliable and repeatable.
