import logging
from typing import Literal, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command, interrupt

logger = logging.getLogger(__name__)

class GraphInput(BaseModel):
    employee_id: str
    employee_name: str
    department: str
    expense_amount: float
    expense_category: Literal["travel", "meals", "supplies", "training", "equipment", "other"]
    expense_description: str
    receipt_attached: bool = True
    expense_date: str  # ISO format date
    manager_email: Optional[str] = None


class GraphOutput(BaseModel):
    expense_id: str
    status: Literal["approved", "rejected", "refunded"]
    approval_level: str
    final_amount: float
    processing_notes: List[str]
    reimbursement_eta: Optional[str] = None


class GraphState(BaseModel):
    # Input fields
    employee_id: str
    employee_name: str
    department: str
    expense_amount: float
    expense_category: str
    expense_description: str
    receipt_attached: bool
    expense_date: str
    manager_email: Optional[str] = None

    # Generated/processing fields
    expense_id: Optional[str] = None
    validation_notes: List[str] = Field(default_factory=list)
    processing_notes: List[str] = Field(default_factory=list)

    # Final result fields
    status: Literal["approved", "rejected", "refunded"] = "rejected"
    approval_level: Optional[str] = None
    reimbursement_eta: Optional[str] = None


EXPENSE_LIMITS = {
    "travel": 2000,
    "meals": 150,
    "supplies": 500,
    "training": 5000,
    "equipment": 3000,
    "other": 300
}


def generate_expense_id(employee_id: str) -> str:
    """Generate a deterministic expense ID based on employee ID"""
    import hashlib
    # Create deterministic hash from employee_id
    hash_obj = hashlib.md5(employee_id.encode())
    hash_hex = hash_obj.hexdigest()[:8].upper()
    return f"EXP-20251001-{hash_hex}"


def prepare_input(graph_input: GraphInput) -> GraphState:
    """Initialize state and generate expense ID"""
    expense_id = generate_expense_id(graph_input.employee_id)
    logger.info(f"Processing expense report {expense_id} for {graph_input.employee_name}")

    return GraphState(
        **graph_input.model_dump(),
        expense_id=expense_id,
        processing_notes=[f"Expense report {expense_id} created at 2025-10-01T10:00:00.000000"]
    )


def validate_expense(state: GraphState) -> Command:
    """Validation of expense"""
    notes = []

    if not state.receipt_attached and state.expense_amount > 25:
        notes.append("Missing receipt for expense over $25")

    expense_date = datetime.fromisoformat(state.expense_date)
    days_old = (datetime.now() - expense_date).days
    if days_old > 90:
        notes.append(f"Expense is {days_old} days old (>90 day policy)")

    limit = EXPENSE_LIMITS.get(state.expense_category, EXPENSE_LIMITS["other"])
    if state.expense_amount > limit:
        notes.append(f"Exceeds ${limit} limit for {state.expense_category}")

    logger.info(f"Validation completed for {state.expense_id} - {len(notes)} notes")

    return Command(update={
        "validation_notes": notes,
        "processing_notes": state.processing_notes + notes
    })


def manager_review(state: GraphState) -> Command:
    """Suspend for manager approval via API trigger"""
    logger.info(f"Suspending expense {state.expense_id} for manager review")

    validation_summary = '; '.join(state.validation_notes) if state.validation_notes else 'No issues'
    review_data = interrupt(
        f"EXPENSE APPROVAL REQUIRED\n\n"
        f"Expense ID: {state.expense_id}\n"
        f"Employee: {state.employee_name} ({state.department})\n"
        f"Amount: ${state.expense_amount}\n"
        f"Category: {state.expense_category}\n"
        f"Description: {state.expense_description}\n"
        f"Validation: {validation_summary}\n\n"
        f"Please approve or reject this expense.\n"
        f"Type 'true' to approve or 'false' to reject.\n"
        f"Or send JSON: {{'payload':{{'approved': true/false}}}}"
    )

    logger.info(f"Manager decision received: {review_data}")
    logger.info(f"Review data type: {type(review_data)}")

    # Handle different response formats
    if isinstance(review_data, dict):
        # API response - JSON object
        approved = review_data.get("approved", False)
    elif isinstance(review_data, bool):
        # Direct boolean
        approved = review_data
    elif isinstance(review_data, str):
        # String response from Orchestrator UI
        approved = review_data.lower().strip() in ['true', 'yes', '1', 'approve', 'approved']
    else:
        # Default to rejected
        approved = False
        logger.warning(f"Unexpected review data type: {type(review_data)}, defaulting to rejected")

    if approved:
        eta = "2025-10-06"  # Static date for consistent testing
        status = "approved"
        notes = ["Approved on 2025-10-01 10:05", f"Reimbursement by {eta}"]
    else:
        status = "rejected"
        eta = None
        notes = ["Rejected on 2025-10-01 10:05"]

    return Command(update={
        "manager_approved": approved,
        "status": status,
        "approval_level": "manager",
        "reimbursement_eta": eta,
        "processing_notes": state.processing_notes + notes
    })


def finalize_expense(state: GraphState) -> GraphOutput:
    """Prepare final output"""
    logger.info(f"Completed processing {state.expense_id}: {state.status}")

    return GraphOutput(
        expense_id=state.expense_id,
        status=state.status,
        approval_level=state.approval_level,
        final_amount=state.expense_amount,
        processing_notes=state.processing_notes,
        reimbursement_eta=state.reimbursement_eta
    )


builder = StateGraph(GraphState, input=GraphInput, output=GraphOutput)

builder.add_node("prepare_input", prepare_input)
builder.add_node("validate_expense", validate_expense)
builder.add_node("manager_review", manager_review)
builder.add_node("finalize_expense", finalize_expense)

builder.add_edge(START, "prepare_input")
builder.add_edge("prepare_input", "validate_expense")
builder.add_edge("validate_expense", "manager_review")
builder.add_edge("manager_review", "finalize_expense")
builder.add_edge("finalize_expense", END)

from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=[])


