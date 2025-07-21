"""Ticket Classification Agent using LangGraph and UiPath Human-in-the-Loop."""

import logging
import os
from typing import Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from uipath import UiPath
from uipath.models import CreateAction
from uipath_langchain.chat import UiPathAzureChatOpenAI, UiPathChat

# Configuration
logger = logging.getLogger(__name__)
uipath = UiPath()

# Constants
DEFAULT_MODEL = "gpt-4o-2024-08-06"
DEFAULT_CONFIDENCE = 0.0
APP_FOLDER_PATH_PLACEHOLDER = "FOLDER_PATH_PLACEHOLDER"

# Ticket categories
TicketCategory = Literal["security", "error", "system", "billing", "performance"]
NextNode = Literal["classify", "notify_team"]

# Data Models
class GraphInput(BaseModel):
    """Input model for the ticket classification graph."""
    message: str
    ticket_id: str
    assignee: Optional[str] = None


class GraphOutput(BaseModel):
    """Output model for the ticket classification graph."""
    label: str
    confidence: float


class GraphState(MessagesState):
    """State model for the ticket classification workflow."""
    message: str
    ticket_id: str
    assignee: Optional[str]
    label: Optional[str] = None
    confidence: Optional[float] = None
    last_predicted_category: Optional[str] = None
    human_approval: Optional[bool] = None


class TicketClassification(BaseModel):
    """Model for ticket classification results."""
    label: TicketCategory = Field(
        description="The classification label for the support ticket"
    )
    confidence: float = Field(
        description="Confidence score for the classification", ge=0.0, le=1.0
    )


# Configuration and parsers
output_parser = PydanticOutputParser(pydantic_object=TicketClassification)

SYSTEM_MESSAGE_TEMPLATE = """You are a support ticket classifier. Classify tickets into exactly one category and provide a confidence score.

{format_instructions}

Categories:
- security: Security issues, access problems, auth failures
- error: Runtime errors, exceptions, unexpected behavior
- system: Core infrastructure or system-level problems
- billing: Payment and subscription related issues
- performance: Speed and resource usage concerns

Respond with the classification in the requested JSON format."""


def get_environment_flag(env_var: str) -> bool:
    """Get boolean value from environment variable."""
    return os.getenv(env_var, "false").lower() == "true"


def create_system_message() -> str:
    """Create the system message for ticket classification."""
    return SYSTEM_MESSAGE_TEMPLATE.format(
        format_instructions=output_parser.get_format_instructions()
    )

# Node Functions
def prepare_input(graph_input: GraphInput) -> GraphState:
    """Prepare the initial state from graph input."""
    return GraphState(
        message=graph_input.message,
        ticket_id=graph_input.ticket_id,
        assignee=graph_input.assignee,
        messages=[
            SystemMessage(content=create_system_message()),
            HumanMessage(content=graph_input.message)
        ],
        last_predicted_category=None,
        human_approval=None,
    )


def decide_next_node(state: GraphState) -> NextNode:
    """Decide the next node based on human approval status."""
    if state["human_approval"] is True:
        return "notify_team"
    return "classify"

async def classify(state: GraphState) -> Command:
    """Classify the support ticket using LLM."""
    if os.getenv("USE_AZURE_CHAT", "false").lower() == "true":
        llm = UiPathAzureChatOpenAI(model=DEFAULT_MODEL)
    else:
        llm = UiPathChat(model=DEFAULT_MODEL)

    # Add rejection message if there was a previous prediction
    if state.get("last_predicted_category"):
        predicted_category = state["last_predicted_category"]
        rejection_message = (
            f"The ticket is 100% not part of the category '{predicted_category}'. "
            "Choose another one."
        )
        state["messages"].append(HumanMessage(content=rejection_message))

    chain = llm | output_parser

    try:
        result = await chain.ainvoke(state["messages"])
        logger.info(
            f"Ticket classified with label: {result.label}, "
            f"confidence score: {result.confidence}"
        )

        return Command(
            update={
                "confidence": result.confidence,
                "label": result.label,
                "last_predicted_category": result.label,
                "messages": state["messages"],
            }
        )
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        return Command(
            update={
                "label": "error",
                "confidence": DEFAULT_CONFIDENCE,
            }
        )

def create_approval_message(ticket_id: str, ticket_message: str, label: str, confidence: float) -> str:
    """Create formatted message for human approval."""
    return (
        f"This is how I classified the ticket: '{ticket_id}', "
        f"with message '{ticket_message}' \n"
        f"Label: '{label}' "
        f"Confidence: '{confidence}'"
    )


async def wait_for_human(state: GraphState) -> Command:
    """Wait for human approval of the classification."""
    # Extract state information
    ticket_id = state["ticket_id"]
    ticket_message = state["messages"][1].content
    label = state["label"]
    confidence = state["confidence"]
    is_resume = state.get("human_approval") is not None
    
    
    
    if not is_resume:
        logger.info("Waiting for human approval via regular interrupt")
    interrupt_message = (
        "Please review the classification of the ticket, then use "
        "`uipath run agent '{\"Answer\": true}' --resume` to continue."
    )
    action_data = interrupt(interrupt_message)
    human_approved = bool(action_data)
   
    return Command(
        update={
            "human_approval": human_approved,
        }
    )

async def notify_team(state: GraphState) -> GraphOutput:
    """Send team notification and return final output."""
    logger.info("Sending team email notification")
    return GraphOutput(label=state["label"], confidence=state["confidence"])


def build_graph() -> StateGraph:
    """Build and compile the ticket classification graph."""
    builder = StateGraph(GraphState, input=GraphInput, output=GraphOutput)
    
    # Add nodes
    builder.add_node("prepare_input", prepare_input)
    builder.add_node("classify", classify)
    builder.add_node("human_approval_node", wait_for_human)
    builder.add_node("notify_team", notify_team)
    
    # Add edges
    builder.add_edge(START, "prepare_input")
    builder.add_edge("prepare_input", "classify")
    builder.add_edge("classify", "human_approval_node")
    builder.add_conditional_edges("human_approval_node", decide_next_node)
    builder.add_edge("notify_team", END)
    
    # Compile with memory checkpointer
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# Create the compiled graph
graph = build_graph()
