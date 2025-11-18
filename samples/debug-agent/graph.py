from enum import Enum
from typing import Literal
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel

class TicketPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketCategory(Enum):
    TECHNICAL = "technical"
    BILLING = "billing"
    GENERAL = "general"
    COMPLAINT = "complaint"


class TicketInput(BaseModel):
    """Input model for the ticket routing system."""
    ticket_id: str
    customer_message: str
    customer_tier: str = "standard"


class TicketOutput(BaseModel):
    """Output model with routing decisions and metadata."""
    ticket_id: str
    category: TicketCategory
    priority: TicketPriority
    assigned_department: str
    requires_escalation: bool
    estimated_resolution_time: int
    response_template: str


class State(BaseModel):
    """Internal state for processing."""
    ticket_id: str
    customer_message: str
    customer_tier: str = "standard"
    sentiment: str = ""
    category: TicketCategory | None = None
    priority: TicketPriority | None = None
    has_urgency_keywords: bool = False
    requires_escalation: bool = False
    assigned_department: str = ""
    estimated_resolution_time: int = 0
    response_template: str = ""


async def analyze_sentiment(state: State) -> State:
    """Analyze customer sentiment from message."""
    negative_keywords = ["angry", "frustrated", "terrible", "worst", "unacceptable"]
    state.sentiment = "negative" if any(kw in state.customer_message.lower() for kw in negative_keywords) else "positive"
    return state


async def classify_category(state: State) -> State:
    """Classify ticket into categories."""
    message = state.customer_message.lower()
    if any(word in message for word in ["bug", "error", "not working", "broken"]):
        state.category = TicketCategory.TECHNICAL
    elif any(word in message for word in ["charge", "payment", "invoice", "refund"]):
        state.category = TicketCategory.BILLING
    elif any(word in message for word in ["disappointed", "unhappy", "poor service"]):
        state.category = TicketCategory.COMPLAINT
    else:
        state.category = TicketCategory.GENERAL
    return state


async def check_urgency(state: State) -> State:
    """Check for urgency keywords."""
    urgency_keywords = ["urgent", "asap", "immediately", "emergency", "critical"]
    state.has_urgency_keywords = any(kw in state.customer_message.lower() for kw in urgency_keywords)
    return state


async def determine_priority(state: State) -> State:
    """Determine ticket priority based on multiple factors."""
    if state.has_urgency_keywords or state.customer_tier == "premium":
        state.priority = TicketPriority.HIGH
    elif state.sentiment == "negative":
        state.priority = TicketPriority.MEDIUM
    else:
        state.priority = TicketPriority.LOW

    # Critical cases
    if state.category == TicketCategory.COMPLAINT and state.customer_tier == "premium":
        state.priority = TicketPriority.CRITICAL

    return state


async def check_escalation(state: State) -> State:
    """Determine if ticket needs escalation."""
    state.requires_escalation = (
        state.priority in [TicketPriority.CRITICAL, TicketPriority.HIGH]
        and state.category == TicketCategory.COMPLAINT
    )
    return state


async def route_to_department(state: State) -> State:
    """Route ticket to appropriate department."""
    if state.category == TicketCategory.TECHNICAL:
        state.assigned_department = "Engineering"
    elif state.category == TicketCategory.BILLING:
        state.assigned_department = "Finance"
    elif state.category == TicketCategory.COMPLAINT:
        state.assigned_department = "Customer Success"
    else:
        state.assigned_department = "General Support"
    return state


async def escalate_to_manager(state: State) -> State:
    """Escalate to manager for high-priority issues."""
    state.assigned_department = f"{state.assigned_department} - Manager"
    state.estimated_resolution_time = 2  # hours
    return state


async def assign_standard_queue(state: State) -> State:
    """Assign to standard support queue."""
    state.estimated_resolution_time = 24  # hours
    return state


async def generate_response(state: State) -> State:
    """Generate automated response template."""
    state.response_template = (
        f"Thank you for contacting us. Your ticket #{state.ticket_id} has been "
        f"assigned to {state.assigned_department} with {state.priority.value} priority. "
        f"Expected resolution time: {state.estimated_resolution_time} hours."
    )
    return state


async def finalize_ticket(state: State) -> TicketOutput:
    """Final processing and return output model."""
    return TicketOutput(
        ticket_id=state.ticket_id,
        category=state.category,
        priority=state.priority,
        assigned_department=state.assigned_department,
        requires_escalation=state.requires_escalation,
        estimated_resolution_time=state.estimated_resolution_time,
        response_template=state.response_template
    )


def should_escalate(state: State) -> Literal["escalate_to_manager", "assign_standard_queue"]:
    """Routing function to determine escalation path."""
    return "escalate_to_manager" if state.requires_escalation else "assign_standard_queue"


builder = StateGraph(
    state_schema=State,
    input=TicketInput,
    output=TicketOutput
)

builder.add_node("analyze_sentiment", analyze_sentiment)
builder.add_node("classify_category", classify_category)
builder.add_node("check_urgency", check_urgency)
builder.add_node("determine_priority", determine_priority)
builder.add_node("check_escalation", check_escalation)
builder.add_node("route_to_department", route_to_department)
builder.add_node("escalate_to_manager", escalate_to_manager)
builder.add_node("assign_standard_queue", assign_standard_queue)
builder.add_node("generate_response", generate_response)
builder.add_node("finalize_ticket", finalize_ticket)

builder.add_edge(START, "analyze_sentiment")
builder.add_edge("analyze_sentiment", "classify_category")
builder.add_edge("classify_category", "check_urgency")
builder.add_edge("check_urgency", "determine_priority")
builder.add_edge("determine_priority", "check_escalation")
builder.add_edge("check_escalation", "route_to_department")

builder.add_conditional_edges(
    "route_to_department",
    should_escalate,
    {
        "escalate_to_manager": "escalate_to_manager",
        "assign_standard_queue": "assign_standard_queue"
    }
)

builder.add_edge("escalate_to_manager", "generate_response")
builder.add_edge("assign_standard_queue", "generate_response")
builder.add_edge("generate_response", "finalize_ticket")
builder.add_edge("finalize_ticket", END)

graph = builder.compile()
