import base64
import io
from enum import Enum, auto

from pydantic import BaseModel

from uipath_langchain.chat import UiPathChatOpenAI
from PIL import Image


class InvoiceClass(str, Enum):
    """
    Python Enum for common item and service types used in invoice classification
    for accounting purposes (General Ledger accounts).
    """
    # Auto-accepted
    TRAVEL_TRANSPORTATION = auto()  # Airfare, trains, taxis, rideshare, public transit, mileage
    LODGING = auto()  # Hotels, short-term rentals during business trips
    MEALS = auto()  # Business meals, per diem meal expenses
    OFFICE_SUPPLIES = auto()  # Stationery, minor equipment, work materials
    PROFESSIONAL_FEES = auto()  # Training, conferences, certifications, professional dues
    VEHICLE_EXPENSE = auto()  # Fuel, tolls, parking for business use (non-mileage)

    # Auto-rejected
    ENTERTAINMENT = auto()  # Client entertainment, event tickets when business-related
    IT_EQUIPMENT_SOFTWARE = auto()  # Laptops, peripherals, software, SaaS tools
    HOME_OFFICE = auto()  # Approved remote-work expenses, internet share, desk, chair
    COMMUNICATION = auto()  # Work phone, data plans, conferencing services

    # Manual approval
    MOVING_RELOCATION = auto()  # Approved relocation-related expenses
    MISC_BUSINESS_EXPENSE = auto()  # Other approved but infrequent business costs
    UNKNOWN = auto()

    def __str__(self):
        # Provides a human-readable string representation
        return self.name.replace('_', ' ').title()

    @property
    def auto_approval_status(self) -> str:
        auto_approved = {InvoiceClass.TRAVEL_TRANSPORTATION, InvoiceClass.LODGING, InvoiceClass.MEALS, InvoiceClass.OFFICE_SUPPLIES, InvoiceClass.PROFESSIONAL_FEES, InvoiceClass.VEHICLE_EXPENSE}
        auto_rejected = {InvoiceClass.ENTERTAINMENT, InvoiceClass.IT_EQUIPMENT_SOFTWARE, InvoiceClass.HOME_OFFICE, InvoiceClass.COMMUNICATION}
        if self in auto_approved: return "approved"
        elif self in auto_rejected: return "rejected"
        else: return "review"

def bytes_to_base64_data_url(image_bytes: bytes, mime_type: str) -> str:
    """
    Converts raw image bytes into a Base64 encoded data URL string.
    Example mime_type: 'image/jpeg', 'image/png'
    """
    base64_encoded_bytes = base64.b64encode(image_bytes)
    base64_string = base64_encoded_bytes.decode("utf-8")
    return f"data:{mime_type};base64,{base64_string}"

async def classify(image: bytes, mime_type: str) -> InvoiceClass:
    llm = UiPathChatOpenAI(model_name="gpt-5-mini-2025-08-07")

    class ClassificationOutput(BaseModel):
        invoice_classification: InvoiceClass

    base64_image_url = bytes_to_base64_data_url(image, mime_type)

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": ""
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": base64_image_url}
                }
            ]
        }
    ]
    return (await llm.with_structured_output(ClassificationOutput).ainvoke(messages)).invoice_classification

