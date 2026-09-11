"""
Pydantic models for risk assessment and decision making.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Risk level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DamageType(str, Enum):
    """Types of damage that can occur in LTL shipping."""
    PHYSICAL_DAMAGE = "physical_damage"
    WATER_DAMAGE = "water_damage"
    THEFT = "theft"
    LOSS = "loss"
    CONTAMINATION = "contamination"
    TEMPERATURE_DAMAGE = "temperature_damage"
    CONCEALED_DAMAGE = "concealed_damage"
    SHORTAGE = "shortage"
    OTHER = "other"


class DecisionType(str, Enum):
    """Types of decisions that can be made on a claim."""
    AUTO_APPROVE = "auto_approve"
    AUTO_REJECT = "auto_reject"
    HUMAN_REVIEW = "human_review"
    ADDITIONAL_INFO_REQUIRED = "additional_info_required"


class RiskFactor(BaseModel):
    """Individual risk factor with score and weight."""
    name: str = Field(description="Name of the risk factor")
    score: float = Field(ge=0.0, le=1.0, description="Risk score (0-1)")
    weight: float = Field(ge=0.0, le=1.0, description="Weight of this factor (0-1)")
    description: str = Field(description="Description of why this score was assigned")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in this assessment")


class AmountRiskAssessment(BaseModel):
    """Risk assessment based on claim amount."""
    claim_amount: float = Field(description="Claim amount in dollars")
    risk_score: float = Field(ge=0.0, le=1.0, description="Amount-based risk score")
    threshold_exceeded: bool = Field(description="Whether amount exceeds high-risk threshold")
    amount_category: str = Field(description="Category: small, medium, large, very_large")
    reasoning: str = Field(description="Explanation of the risk assessment")


class DamageTypeRiskAssessment(BaseModel):
    """Risk assessment based on damage type."""
    damage_type: DamageType = Field(description="Type of damage")
    risk_score: float = Field(ge=0.0, le=1.0, description="Damage type risk score")
    is_high_risk_type: bool = Field(description="Whether this damage type is high risk")
    typical_fraud_indicator: bool = Field(description="Whether this type is commonly associated with fraud")
    reasoning: str = Field(description="Explanation of the risk assessment")


class HistoricalPatternAssessment(BaseModel):
    """Risk assessment based on historical patterns."""
    customer_claim_count: int = Field(default=0, description="Number of previous claims by this customer")
    customer_approval_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Historical approval rate")
    carrier_claim_count: int = Field(default=0, description="Number of claims for this carrier")
    carrier_issue_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Carrier's issue rate")
    similar_claims_found: int = Field(default=0, description="Number of similar historical claims")
    risk_score: float = Field(ge=0.0, le=1.0, description="Historical pattern risk score")
    reasoning: str = Field(description="Explanation of the risk assessment")


class RiskAssessmentResult(BaseModel):
    """Complete risk assessment result for a claim."""
    claim_id: str = Field(description="Claim ID being assessed")
    overall_risk_score: float = Field(ge=0.0, le=1.0, description="Overall weighted risk score")
    risk_level: RiskLevel = Field(description="Categorized risk level")
    
    # Individual risk assessments
    amount_risk: AmountRiskAssessment = Field(description="Amount-based risk assessment")
    damage_type_risk: DamageTypeRiskAssessment = Field(description="Damage type risk assessment")
    historical_risk: HistoricalPatternAssessment = Field(description="Historical pattern assessment")
    
    # Risk factors breakdown
    risk_factors: List[RiskFactor] = Field(default_factory=list, description="Individual risk factors")
    
    # Decision recommendation
    recommended_decision: DecisionType = Field(description="Recommended decision based on risk")
    decision_confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the recommendation")
    decision_reasoning: str = Field(description="Explanation of the recommended decision")
    
    # Flags and alerts
    requires_human_review: bool = Field(description="Whether human review is required")
    fraud_indicators: List[str] = Field(default_factory=list, description="Potential fraud indicators")
    data_quality_issues: List[str] = Field(default_factory=list, description="Data quality concerns")
    
    # Metadata
    assessed_at: datetime = Field(default_factory=datetime.now, description="Assessment timestamp")
    assessment_version: str = Field(default="1.0", description="Risk assessment algorithm version")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the risk assessment."""
        return {
            "claim_id": self.claim_id,
            "overall_risk_score": round(self.overall_risk_score, 3),
            "risk_level": self.risk_level.value,
            "recommended_decision": self.recommended_decision.value,
            "decision_confidence": round(self.decision_confidence, 3),
            "requires_human_review": self.requires_human_review,
            "fraud_indicators_count": len(self.fraud_indicators),
            "data_quality_issues_count": len(self.data_quality_issues),
            "key_factors": [
                {
                    "name": factor.name,
                    "score": round(factor.score, 3),
                    "weight": round(factor.weight, 3)
                }
                for factor in sorted(self.risk_factors, key=lambda x: x.score * x.weight, reverse=True)[:3]
            ]
        }


class RiskThresholds(BaseModel):
    """Configurable risk thresholds for decision making."""
    auto_approve_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Max risk for auto-approval")
    human_review_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Min risk for human review")
    auto_reject_threshold: float = Field(default=0.9, ge=0.0, le=1.0, description="Min risk for auto-rejection")
    
    high_amount_threshold: float = Field(default=5000.0, description="Amount threshold for high risk")
    critical_amount_threshold: float = Field(default=10000.0, description="Amount threshold for critical risk")
    
    min_confidence_for_auto_decision: float = Field(default=0.8, ge=0.0, le=1.0, description="Min confidence for automation")


class RiskScoringWeights(BaseModel):
    """Configurable weights for different risk factors."""
    amount_weight: float = Field(default=0.35, ge=0.0, le=1.0, description="Weight for amount-based risk")
    damage_type_weight: float = Field(default=0.25, ge=0.0, le=1.0, description="Weight for damage type risk")
    historical_weight: float = Field(default=0.20, ge=0.0, le=1.0, description="Weight for historical patterns")
    consistency_weight: float = Field(default=0.15, ge=0.0, le=1.0, description="Weight for data consistency")
    policy_weight: float = Field(default=0.05, ge=0.0, le=1.0, description="Weight for policy compliance")
    
    def validate_weights(self) -> bool:
        """Validate that weights sum to approximately 1.0."""
        total = (
            self.amount_weight +
            self.damage_type_weight +
            self.historical_weight +
            self.consistency_weight +
            self.policy_weight
        )
        return abs(total - 1.0) < 0.01
