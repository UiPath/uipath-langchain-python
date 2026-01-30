"""Base span attribute classes."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class ErrorDetails(BaseModel):
    """Error details captured in span attributes."""

    model_config = ConfigDict(populate_by_name=True)

    message: str = Field(..., alias="message")
    type: str = Field(..., alias="type")
    stack_trace: Optional[str] = Field(None, alias="stackTrace")


class BaseSpanAttributes(BaseModel, ABC):
    """Abstract base class for all span attributes.

    Uses polymorphic JSON serialization via the `type` property.
    Each subclass must implement the `type` property to identify span category.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="allow",  # Allow extra fields for forward compatibility
    )

    error: Optional[ErrorDetails] = Field(None, alias="error")
    license_ref_id: Optional[str] = Field(None, alias="licenseRefId")

    @property
    @abstractmethod
    def type(self) -> str: ...

    def to_otel_attributes(self) -> Dict[str, Any]:
        """Convert to OpenTelemetry attribute dict.

        None values excluded. Complex objects kept as-is for span processor.
        """
        attrs: Dict[str, Any] = {
            "type": self.type,
            "span_type": self.type,
            "uipath.custom_instrumentation": True,
        }
        data = self.model_dump(by_alias=True, exclude_none=True, exclude={"error"})
        attrs.update(data)

        if self.error:
            attrs["error"] = self.error.model_dump(by_alias=True)
        return attrs
