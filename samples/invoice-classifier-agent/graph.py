import io

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from uipath.platform import UiPath
from uipath.platform.attachments import Attachment
from uipath.platform.common import CreateTask

from pii_masker import mask_pii

from classifier import classify


@dataclass
class InvoiceApprovalResult:
    approved: bool

class InvoiceApprovalRequest(BaseModel):
    attachment: Attachment

class InvoiceApprovalState(InvoiceApprovalRequest):
    attachment_bytes: bytes | None = None
    approval: str | None = None

async def download(request: InvoiceApprovalRequest) -> InvoiceApprovalState:
    assert request.attachment.mime_type.startswith("image/"), "Only image invoices are supported."
    uipath = UiPath()
    buffer = bytes()
    async with uipath.attachments.open_async(attachment=request.attachment) as (attachment, response):
        async for raw_bytes in response.aiter_raw():
            attachment_bytes = io.BytesIO(raw_bytes).read()
            buffer = buffer + attachment_bytes
    return InvoiceApprovalState(attachment=request.attachment, attachment_bytes=buffer)

async def mask_pii_node(state: InvoiceApprovalState) -> InvoiceApprovalState:
    attachment_bytes = mask_pii(state.attachment_bytes, state.attachment.mime_type)
    return InvoiceApprovalState(attachment=state.attachment, attachment_bytes=attachment_bytes)

async def auto_approval(state: InvoiceApprovalState) -> InvoiceApprovalState:
    state.approval = (await classify(state.attachment_bytes, state.attachment.mime_type)).auto_approval_status
    return state

async def escalation(state: InvoiceApprovalState) -> InvoiceApprovalState:
    state.approval = "approved"
    # state.approval = interrupt(
    #     CreateTask(
    #         app_name="AppName",
    #         app_folder_path="MyFolderPath",
    #         title="Escalate Issue",
    #         data={"attachment": state.attachment},
    #         assignee="user@example.com"
    #     )
    # )
    return state



def llm_result_router(state: InvoiceApprovalState) -> str:
    if state.approval in ("approved", "rejected"):
        return "respond"
    else:
        return "escalation"


async def respond(state: InvoiceApprovalState) -> InvoiceApprovalResult:
    return InvoiceApprovalResult(approved=state.approval == "approved")

async def is_approved(attachment: Attachment) -> InvoiceApprovalResult:
    return InvoiceApprovalResult(approved=False)

builder = StateGraph(state_schema=InvoiceApprovalState, input=InvoiceApprovalRequest, output=InvoiceApprovalResult)

builder.add_node("download", download)
builder.add_node("mask_pii", mask_pii_node)
builder.add_node("auto_approval", auto_approval)
builder.add_node("escalation", escalation)
builder.add_node("respond", respond)


builder.add_edge(START, "download")
builder.add_edge("download", "mask_pii")
builder.add_edge("mask_pii", "auto_approval")
builder.add_conditional_edges("auto_approval", llm_result_router)
builder.add_edge("escalation", "respond")

builder.add_edge("auto_approval", END)


graph = builder.compile()
