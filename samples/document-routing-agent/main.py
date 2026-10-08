import logging
import hashlib
from typing import Literal, Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from uipath.tracing import traced

logger = logging.getLogger(__name__)

# deterministic document routing rules
ROUTING_RULES = {
    "hr": {
        "keywords": ["employee", "vacation", "salary", "benefits", "leave", "onboarding", "performance", "training"],
        "document_types": ["employment_contract", "leave_request", "performance_review", "training_certificate"],
        "priority_boost": 1
    },
    "finance": {
        "keywords": ["invoice", "payment", "budget", "expense", "revenue", "tax", "audit", "financial"],
        "document_types": ["invoice", "purchase_order", "expense_report", "financial_statement"],
        "priority_boost": 2
    },
    "legal": {
        "keywords": ["contract", "agreement", "compliance", "legal", "lawsuit", "regulation", "policy", "terms"],
        "document_types": ["contract", "legal_notice", "compliance_report", "policy_document"],
        "priority_boost": 3
    },
    "it": {
        "keywords": ["software", "hardware", "system", "network", "security", "database", "server", "technical"],
        "document_types": ["technical_spec", "incident_report", "change_request", "security_audit"],
        "priority_boost": 0
    },
    "operations": {
        "keywords": ["production", "logistics", "supply", "inventory", "quality", "process", "workflow", "maintenance"],
        "document_types": ["work_order", "quality_report", "inventory_list", "maintenance_log"],
        "priority_boost": 1
    }
}

CONFIDENCE_THRESHOLDS = {
    "high": 0.8,
    "medium": 0.5,
    "low": 0.3
}


class GraphInput(BaseModel):
    document_id: str = Field(description="Unique document identifier")
    document_type: str = Field(description="Type of document")
    title: str = Field(description="Document title")
    content: str = Field(description="Document content or description")
    sender_email: str = Field(description="Email of document sender")
    sender_department: Optional[str] = Field(default=None, description="Department of sender")
    urgency_level: Literal["low", "medium", "high", "critical"] = Field(default="medium")
    tags: List[str] = Field(default_factory=list, description="Document tags/labels")


class GraphOutput(BaseModel):
    document_id: str
    routing_id: str
    assigned_department: str
    routing_confidence: Literal["low", "medium", "high"]
    priority_score: int = Field(ge=0, le=100)
    routing_reasons: List[str]
    secondary_departments: List[str]
    processing_timestamp: str
    estimated_processing_time: str


class GraphState(BaseModel):
    # input
    document_id: str
    document_type: str
    title: str
    content: str
    sender_email: str
    sender_department: Optional[str] = None
    urgency_level: str
    tags: List[str]

    # processing
    routing_id: Optional[str] = None
    content_analysis: Dict[str, Any] = Field(default_factory=dict)
    department_scores: Dict[str, float] = Field(default_factory=dict)
    assigned_department: Optional[str] = None
    routing_confidence: Optional[str] = None
    priority_score: Optional[int] = None
    routing_reasons: List[str] = Field(default_factory=list)
    secondary_departments: List[str] = Field(default_factory=list)
    processing_timestamp: Optional[str] = None
    estimated_processing_time: Optional[str] = None


@traced(name="generate_routing_id")
def generate_routing_id(document_id: str) -> str:
    """Generate deterministic routing ID"""
    hash_obj = hashlib.md5(document_id.encode())
    hash_hex = hash_obj.hexdigest()[:8].upper()
    return f"RTG-{hash_hex}"


@traced(name="prepare_document")
def prepare_document(state: GraphInput) -> GraphState:
    """Initialize state with document data"""
    routing_id = generate_routing_id(state.document_id)
    timestamp = "2025-01-02T10:00:00Z"  # Fixed for deterministic output

    logger.info(f"Preparing document {state.document_id} with routing ID {routing_id}")

    return GraphState(
        **state.model_dump(),
        routing_id=routing_id,
        processing_timestamp=timestamp,
        routing_reasons=[f"Document received at {timestamp}"]
    )


@traced(name="analyze_content")
def analyze_content(state: GraphState) -> GraphState:
    """Analyze document content for routing patterns"""
    logger.info(f"Analyzing content for document {state.document_id}")

    full_text = f"{state.title} {state.content} {' '.join(state.tags)}".lower()

    keyword_matches = {}
    for dept, rules in ROUTING_RULES.items():
        matches = []
        for keyword in rules["keywords"]:
            if keyword in full_text:
                matches.append(keyword)
        keyword_matches[dept] = matches

    doc_type_matches = {}
    for dept, rules in ROUTING_RULES.items():
        doc_type_matches[dept] = state.document_type in rules["document_types"]

    state.content_analysis = {
        "keyword_matches": keyword_matches,
        "doc_type_matches": doc_type_matches,
        "text_length": len(full_text),
        "has_sender_dept": state.sender_department is not None
    }

    state.routing_reasons.append(f"Analyzed {len(full_text)} characters of content")

    return state


@traced(name="calculate_routing_scores")
def calculate_routing_scores(state: GraphState) -> GraphState:
    """Calculate routing scores for each department"""
    logger.info(f"Calculating routing scores for document {state.document_id}")

    scores = {}

    for dept in ROUTING_RULES.keys():
        score = 0.0

        keyword_matches = state.content_analysis["keyword_matches"][dept]
        if keyword_matches:
            keyword_score = min(len(keyword_matches) * 10, 50)
            score += keyword_score

        if state.content_analysis["doc_type_matches"][dept]:
            score += 30

        if state.sender_department and state.sender_department.lower() == dept:
            score += 20

        tag_matches = sum(1 for tag in state.tags if dept in tag.lower())
        score += min(tag_matches * 5, 10)

        scores[dept] = min(score / 100.0, 1.0)

    state.department_scores = scores

    sorted_depts = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_depts = [f"{dept}:{score:.2f}" for dept, score in sorted_depts[:3]]
    state.routing_reasons.append(f"Top departments: {', '.join(top_depts)}")

    return state


@traced(name="determine_routing")
def determine_routing(state: GraphState) -> GraphState:
    """Determine final routing based on scores"""
    logger.info(f"Determining final routing for document {state.document_id}")

    sorted_scores = sorted(state.department_scores.items(), key=lambda x: x[1], reverse=True)

    if not sorted_scores:
        state.assigned_department = "operations"
        state.routing_confidence = "low"
        state.routing_reasons.append("No matching patterns found, defaulting to operations")
    else:
        top_dept, top_score = sorted_scores[0]
        state.assigned_department = top_dept

        if top_score >= CONFIDENCE_THRESHOLDS["high"]:
            state.routing_confidence = "high"
        elif top_score >= CONFIDENCE_THRESHOLDS["medium"]:
            state.routing_confidence = "medium"
        else:
            state.routing_confidence = "low"

        keyword_matches = state.content_analysis["keyword_matches"][top_dept]
        if keyword_matches:
            state.routing_reasons.append(f"Matched keywords: {', '.join(keyword_matches[:3])}")
        if state.content_analysis["doc_type_matches"][top_dept]:
            state.routing_reasons.append(f"Document type '{state.document_type}' matches department")

        state.secondary_departments = [
            dept for dept, score in sorted_scores[1:4]
            if score >= CONFIDENCE_THRESHOLDS["low"]
        ]

    return state


@traced(name="calculate_priority")
def calculate_priority(state: GraphState) -> GraphState:
    """Calculate priority score based on urgency and department"""
    logger.info(f"Calculating priority for document {state.document_id}")

    urgency_scores = {
        "low": 10,
        "medium": 30,
        "high": 60,
        "critical": 90
    }
    base_priority = urgency_scores.get(state.urgency_level, 30)

    dept_boost = ROUTING_RULES.get(state.assigned_department, {}).get("priority_boost", 0) * 5

    confidence_boost = {
        "high": 10,
        "medium": 5,
        "low": 0
    }.get(state.routing_confidence, 0)

    state.priority_score = min(base_priority + dept_boost + confidence_boost, 100)

    if state.priority_score >= 80:
        state.estimated_processing_time = "2-4 hours"
    elif state.priority_score >= 50:
        state.estimated_processing_time = "4-8 hours"
    elif state.priority_score >= 30:
        state.estimated_processing_time = "1-2 business days"
    else:
        state.estimated_processing_time = "2-3 business days"

    state.routing_reasons.append(f"Priority score: {state.priority_score} ({state.estimated_processing_time})")

    return state


@traced(name="finalize_routing")
def finalize_routing(state: GraphState) -> GraphOutput:
    """Prepare final output with routing decision"""
    logger.info(f"Finalizing routing for document {state.document_id} to {state.assigned_department}")

    return GraphOutput(
        document_id=state.document_id,
        routing_id=state.routing_id,
        assigned_department=state.assigned_department,
        routing_confidence=state.routing_confidence,
        priority_score=state.priority_score,
        routing_reasons=state.routing_reasons,
        secondary_departments=state.secondary_departments,
        processing_timestamp=state.processing_timestamp,
        estimated_processing_time=state.estimated_processing_time
    )


builder = StateGraph(GraphState, input=GraphInput, output=GraphOutput)

builder.add_node("prepare_document", prepare_document)
builder.add_node("analyze_content", analyze_content)
builder.add_node("calculate_routing_scores", calculate_routing_scores)
builder.add_node("determine_routing", determine_routing)
builder.add_node("calculate_priority", calculate_priority)
builder.add_node("finalize_routing", finalize_routing)

builder.add_edge(START, "prepare_document")
builder.add_edge("prepare_document", "analyze_content")
builder.add_edge("analyze_content", "calculate_routing_scores")
builder.add_edge("calculate_routing_scores", "determine_routing")
builder.add_edge("determine_routing", "calculate_priority")
builder.add_edge("calculate_priority", "finalize_routing")
builder.add_edge("finalize_routing", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
