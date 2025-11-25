from pydantic import BaseModel
from datetime import datetime

class JobRequest(BaseModel):
    topic: str
    callback_url: str | None = None

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class InboxMessageResponse(BaseModel):
    id: str
    job_id: str
    content: str
    status: str
    human_feedback: str | None = None
    received_at: datetime
    approved_at: datetime | None = None

    class Config:
        orm_mode = True

class MessageApprovalRequest(BaseModel):
    human_feedback: str

class MessageApprovalResponse(BaseModel):
    status: str
    message: str

class JobBasicResponse(BaseModel):
    id: str
    topic: str
    status: str
    created_at: datetime
    completed_at: datetime | None = None

    class Config:
        orm_mode = True

class JobDetailResponse(BaseModel):
    id: str
    topic: str
    status: str
    output: str | None = None
    agent_job_id: int | None = None
    agent_job_key: str | None = None
    human_feedback: str | None = None
    created_at: datetime
    completed_at: datetime | None = None
    error_message: str | None = None
    messages: list[InboxMessageResponse] = []

    class Config:
        orm_mode = True
