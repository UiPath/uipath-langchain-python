"""Governance-related span attribute classes."""

from abc import ABC
from typing import Optional

from pydantic import ConfigDict, Field

from .base import BaseSpanAttributes
from .types import SpanType


class GovernanceSpanAttributes(BaseSpanAttributes, ABC):
    """Abstract base for governance spans."""

    model_config = ConfigDict(populate_by_name=True)

    policy_name: Optional[str] = Field(None, alias="policyName")
    action: Optional[str] = Field(None, alias="action")
    assigned_to: Optional[str] = Field(None, alias="assignedTo")
    reason: Optional[str] = Field(None, alias="reason")


class PreGovernanceSpanAttributes(GovernanceSpanAttributes):
    """Attributes for pre-governance spans."""

    @property
    def type(self) -> str:
        return SpanType.PRE_GOVERNANCE


class PostGovernanceSpanAttributes(GovernanceSpanAttributes):
    """Attributes for post-governance spans."""

    @property
    def type(self) -> str:
        return SpanType.POST_GOVERNANCE


class ToolPreGovernanceSpanAttributes(BaseSpanAttributes):
    """Attributes for tool pre-governance spans."""

    model_config = ConfigDict(populate_by_name=True)

    policy_name: Optional[str] = Field(None, alias="policyName")
    action: Optional[str] = Field(None, alias="action")
    assigned_to: Optional[str] = Field(None, alias="assignedTo")
    reason: Optional[str] = Field(None, alias="reason")

    @property
    def type(self) -> str:
        return SpanType.TOOL_PRE_GOVERNANCE


class ToolPostGovernanceSpanAttributes(BaseSpanAttributes):
    """Attributes for tool post-governance spans."""

    model_config = ConfigDict(populate_by_name=True)

    policy_name: Optional[str] = Field(None, alias="policyName")
    action: Optional[str] = Field(None, alias="action")
    reason: Optional[str] = Field(None, alias="reason")

    @property
    def type(self) -> str:
        return SpanType.TOOL_POST_GOVERNANCE
