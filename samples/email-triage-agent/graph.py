"""Long-running support inbox triage agent.

Watches a UiPath Integration Services Outlook connection for emails whose
subject matches the value passed in as agent input. Each match:

1. Resumes the suspended job with the enriched Microsoft Graph `Message`
   as the resume value of `WaitIntegrationEvent`.
2. The LLM classifies the email into severity, category, a one-sentence
   summary, and a polite acknowledgement draft.
3. The agent replies to the original email with the LLM-drafted
   acknowledgement (via Microsoft Graph, using the connection's OAuth token).
4. The result is logged, transient state is cleared, and the agent loops
   back to suspend on the next matching email.

The graph has no terminal node — the agent stays SUSPENDED on the Outlook
trigger forever, briefly waking to triage and reply to each matching email
and then re-suspending. Cancel the job manually when you're done with it.

Demonstrates one suspend/resume primitive in a long-running agent:
- `WaitIntegrationEvent` — suspend until an external IS connector event fires.
"""

import logging
from enum import Enum
from typing import Any, Optional

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from uipath.platform import UiPath
from uipath.platform.common import WaitIntegrationEvent
from uipath_langchain.chat import UiPathChat

logger = logging.getLogger(__name__)

GRAPH_API_BASE = "https://graph.microsoft.com/v1.0"
OUTLOOK_CONNECTOR = "uipath-microsoft-outlook365"

# Placeholder connection key bound to a real connection via bindings.json.
# `connections.retrieve_async` is decorated with @resource_override("connection",
# resource_identifier="key"), so at run time the decorator inspects the binding
# overwrite context and substitutes the deployer-selected connection's real key.
OUTLOOK_CONNECTION_KEY = "<your-outlook-connection>"


class Severity(str, Enum):
    P0_CRITICAL = "P0_critical"
    P1_HIGH = "P1_high"
    P2_NORMAL = "P2_normal"
    P3_LOW = "P3_low"


class Category(str, Enum):
    BUG = "bug"
    FEATURE_REQUEST = "feature_request"
    HOWTO = "howto"
    BILLING = "billing"
    SPAM = "spam"
    OTHER = "other"


class Triage(BaseModel):
    severity: Severity = Field(
        description=(
            "P0 = production outage / data loss, "
            "P1 = major workflow impact, "
            "P2 = normal request or single-user impact, "
            "P3 = low / cosmetic / general question."
        )
    )
    category: Category
    summary: str = Field(description="One-sentence summary in the customer's voice.")
    suggested_response: str = Field(
        description="Polite acknowledgement reply confirming receipt and next steps."
    )


class GraphInput(BaseModel):
    subject: str = Field(
        description="The exact email subject to watch for. The IS trigger filters incoming emails by this value."
    )


class GraphState(BaseModel):
    subject: str = ""
    email: Optional[dict[str, Any]] = None
    triage: Optional[Triage] = None
    reply_sent: Optional[bool] = None
    reply_body: Optional[str] = None
    triage_count: int = 0


llm = UiPathChat(model="gpt-4.1-mini-2025-04-14")


def _email_str(email: dict[str, Any], *path: str, default: str = "") -> str:
    current: Any = email
    for p in path:
        if not isinstance(current, dict):
            return default
        current = current.get(p)
    return current if isinstance(current, str) else default


async def _send_outlook_reply(message_id: str, body: str) -> None:
    """Reply to an Outlook message via Microsoft Graph, using the OAuth token
    issued for the UiPath Outlook connection that received the trigger.
    """
    sdk = UiPath()
    connection = await sdk.connections.retrieve_async(OUTLOOK_CONNECTION_KEY)
    if connection.id is None:
        raise RuntimeError(
            f"Outlook connection {OUTLOOK_CONNECTION_KEY!r} could not be resolved."
        )

    token = await sdk.connections.retrieve_token_async(connection.id)

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{GRAPH_API_BASE}/me/messages/{message_id}/reply",
            headers={
                "Authorization": f"Bearer {token.access_token}",
                "Content-Type": "application/json",
            },
            json={"comment": body},
        )
        response.raise_for_status()


async def wait_for_email(state: GraphState) -> dict[str, Any]:
    sdk = UiPath()
    connection = await sdk.connections.retrieve_async(OUTLOOK_CONNECTION_KEY)
    folder_path = (
        connection.folder.get("path") if isinstance(connection.folder, dict) else None
    )
    logger.info(
        "Waiting for next email on '%s' (folder='%s') with subject=%r (triaged so far: %d)...",
        connection.name,
        folder_path,
        state.subject,
        state.triage_count,
    )
    email = interrupt(
        WaitIntegrationEvent(
            connector=OUTLOOK_CONNECTOR,
            connection_name=connection.name or "",
            connection_folder_path=folder_path,
            operation="EMAIL_RECEIVED",
            object_name="Message",
            filter_expression=f"(subject=='{state.subject}')",
        )
    )
    sender = _email_str(email, "from", "emailAddress", "address", default="?")
    logger.info("Received email from %s: %s", sender, _email_str(email, "subject"))
    return {"email": email}


async def triage_email(state: GraphState) -> dict[str, Any]:
    email = state.email or {}
    sender = _email_str(email, "from", "emailAddress", "address", default="unknown")
    subject = _email_str(email, "subject")
    body = _email_str(email, "bodyPreview") or _email_str(email, "body", "content")

    triage_llm = llm.with_structured_output(Triage)
    result: Triage = await triage_llm.ainvoke(
        [
            SystemMessage(
                "You are a support triage assistant. Read the customer email and "
                "produce a structured triage result.\n\n"
                "Severity guidelines:\n"
                "- P0: production outage, data loss, or anything blocking critical work.\n"
                "- P1: major workflow impact; affects many users.\n"
                "- P2: normal request or single-user impact.\n"
                "- P3: low priority, cosmetic, or general question.\n\n"
                "Always draft a polite acknowledgement confirming receipt and "
                "setting expectations for next steps."
            ),
            HumanMessage(f"From: {sender}\nSubject: {subject}\n\n{body}"),
        ]
    )
    logger.info(
        "Triage: severity=%s category=%s",
        result.severity.value,
        result.category.value,
    )
    return {"triage": result}


async def send_reply(state: GraphState) -> dict[str, Any]:
    email = state.email or {}
    triage = state.triage
    message_id = email.get("id") if isinstance(email, dict) else None
    body = triage.suggested_response if triage else None

    if not body:
        logger.warning("No reply body resolved — skipping send.")
        return {"reply_sent": False, "reply_body": None}

    if not message_id:
        logger.warning("Email payload had no 'id' field — cannot send reply.")
        return {"reply_sent": False, "reply_body": body}

    try:
        await _send_outlook_reply(message_id, body)
        logger.info("Reply sent.")
        return {"reply_sent": True, "reply_body": body}
    except Exception:
        logger.exception("Failed to send Outlook reply.")
        return {"reply_sent": False, "reply_body": body}


async def finalize(state: GraphState) -> dict[str, Any]:
    triage = state.triage
    assert triage is not None
    email = state.email or {}
    sender = _email_str(email, "from", "emailAddress", "address", default="unknown")
    subject = _email_str(email, "subject")

    logger.info(
        "Triaged email #%d from %s (subject=%r, severity=%s, category=%s, reply_sent=%s)",
        state.triage_count + 1,
        sender,
        subject,
        triage.severity.value,
        triage.category.value,
        bool(state.reply_sent),
    )
    return {
        "triage_count": state.triage_count + 1,
        "email": None,
        "triage": None,
        "reply_sent": None,
        "reply_body": None,
    }


builder = StateGraph(GraphState, input_schema=GraphInput)
builder.add_node("wait_for_email", wait_for_email)
builder.add_node("triage_email", triage_email)
builder.add_node("send_reply", send_reply)
builder.add_node("finalize", finalize)

builder.add_edge(START, "wait_for_email")
builder.add_edge("wait_for_email", "triage_email")
builder.add_edge("triage_email", "send_reply")
builder.add_edge("send_reply", "finalize")
builder.add_edge("finalize", "wait_for_email")

graph = builder.compile()
