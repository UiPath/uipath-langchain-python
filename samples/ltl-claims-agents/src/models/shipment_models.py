"""
Pydantic models for shipment data and consistency validation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field


class ShipmentStatus(str, Enum):
    """Shipment status values."""
    PENDING = "pending"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    DELAYED = "delayed"
    DAMAGED = "damaged"
    LOST = "lost"
    RETURNED = "returned"


class ConsistencyCheckType(str, Enum):
    """Types of consistency checks."""
    CARRIER_MATCH = "carrier_match"
    TRACKING_NUMBER = "tracking_number"
    SHIPMENT_DATE = "shipment_date"
    DELIVERY_DATE = "delivery_date"
    ORIGIN_DESTINATION = "origin_destination"
    WEIGHT_VALUE = "weight_value"
    DAMAGE_REPORT = "damage_report"


class ConsistencyCheckResult(BaseModel):
    """Result of a single consistency check."""
    check_type: ConsistencyCheckType = Field(description="Type of consistency check")
    passed: bool = Field(description="Whether the check passed")
    severity: str = Field(description="Severity: info, warning, error, critical")
    claim_value: Optional[Any] = Field(default=None, description="Value from claim")
    shipment_value: Optional[Any] = Field(default=None, description="Value from shipment")
    discrepancy: Optional[str] = Field(default=None, description="Description of discrepancy if any")
    impact_on_risk: float = Field(default=0.0, ge=0.0, le=1.0, description="Impact on risk score")


class ShipmentData(BaseModel):
    """Shipment data from Data Fabric LTLShipments entity - matches actual schema."""
    # UiPath fields
    Id: Optional[str] = Field(default=None, description="UiPath auto-generated ID")
    
    # Required fields
    shipmentId: str = Field(description="Unique shipment identifier")
    shipper: str = Field(description="Shipper name")
    carrier: str = Field(description="Carrier name")
    proNumber: str = Field(description="PRO number")
    originCity: str = Field(description="Origin city")
    originState: str = Field(description="Origin state (2-letter code)")
    originZip: str = Field(description="Origin ZIP code")
    destinationCity: str = Field(description="Destination city")
    destinationState: str = Field(description="Destination state (2-letter code)")
    destinationZip: str = Field(description="Destination ZIP code")
    pickupDate: str = Field(description="Pickup date (ISO string)")
    status: str = Field(description="Shipment status")
    weightLbs: float = Field(description="Weight in pounds")
    
    # Optional fields
    consignee: Optional[str] = Field(default=None, description="Consignee name")
    bolNumber: Optional[str] = Field(default=None, description="Bill of Lading number")
    poNumber: Optional[str] = Field(default=None, description="Purchase Order number")
    deliveryDate: Optional[str] = Field(default=None, description="Delivery date (ISO string)")
    nmfcClass: Optional[str] = Field(default=None, description="NMFC freight class")
    declaredValueUsd: Optional[float] = Field(default=None, description="Declared value in USD")
    packagingType: Optional[str] = Field(default=None, description="Type of packaging")
    pieces: Optional[int] = Field(default=None, description="Number of pieces")
    hazmat: Optional[bool] = Field(default=None, description="Whether shipment contains hazmat")
    damageReported: Optional[bool] = Field(default=False, description="Whether damage was reported")
    claimReferenceId: Optional[str] = Field(default=None, description="Reference to claim if exists")
    notes: Optional[str] = Field(default=None, description="Additional notes")


class ClaimShipmentData(BaseModel):
    """Claim data relevant for shipment cross-referencing."""
    claim_id: str = Field(description="Claim ID")
    shipmentId: str = Field(description="Referenced shipment ID from claim")
    carrier: str = Field(description="Carrier name from claim")
    
    # Claim details
    amount: float = Field(description="Claimed amount")
    type: str = Field(description="Type of claim/damage")
    description: Optional[str] = Field(default=None, description="Damage description")
    
    # Dates from claim
    submittedDate: Optional[str] = Field(default=None, description="Date claim was submitted (ISO string)")
    
    # Customer info
    shipper: Optional[str] = Field(default=None, description="Shipper name from claim")
    FullName: Optional[str] = Field(default=None, description="Customer full name")
    EmailAddress: Optional[str] = Field(default=None, description="Customer email")
    Phone: Optional[str] = Field(default=None, description="Customer phone")


class ShipmentConsistencyResult(BaseModel):
    """Complete result of shipment consistency validation."""
    claim_id: str = Field(description="Claim ID")
    shipment_id: str = Field(description="Shipment ID")
    
    # Overall results
    is_consistent: bool = Field(description="Whether claim and shipment data are consistent")
    consistency_score: float = Field(ge=0.0, le=1.0, description="Overall consistency score (0-1)")
    risk_adjustment: float = Field(description="Risk score adjustment based on inconsistencies")
    
    # Individual checks
    checks: List[ConsistencyCheckResult] = Field(default_factory=list, description="Individual consistency checks")
    
    # Discrepancies
    critical_discrepancies: List[str] = Field(default_factory=list, description="Critical discrepancies found")
    warnings: List[str] = Field(default_factory=list, description="Warning-level discrepancies")
    
    # Data availability
    shipment_found: bool = Field(description="Whether shipment data was found")
    missing_fields: List[str] = Field(default_factory=list, description="Missing required fields")
    
    # Recommendations
    requires_investigation: bool = Field(description="Whether discrepancies require investigation")
    investigation_priority: str = Field(description="Priority: low, medium, high, critical")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions")
    
    validated_at: datetime = Field(default_factory=datetime.now, description="Validation timestamp")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of consistency validation."""
        return {
            "claim_id": self.claim_id,
            "shipment_id": self.shipment_id,
            "is_consistent": self.is_consistent,
            "consistency_score": round(self.consistency_score, 3),
            "risk_adjustment": round(self.risk_adjustment, 3),
            "shipment_found": self.shipment_found,
            "critical_discrepancies_count": len(self.critical_discrepancies),
            "warnings_count": len(self.warnings),
            "requires_investigation": self.requires_investigation,
            "investigation_priority": self.investigation_priority,
            "checks_passed": sum(1 for check in self.checks if check.passed),
            "checks_failed": sum(1 for check in self.checks if not check.passed)
        }
