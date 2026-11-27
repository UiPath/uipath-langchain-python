"""
Risk assessment service for LTL claims processing.
Implements risk calculation algorithms based on amount, damage type, and historical patterns.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..models.risk_models import (
    RiskLevel,
    DamageType,
    DecisionType,
    RiskFactor,
    AmountRiskAssessment,
    DamageTypeRiskAssessment,
    HistoricalPatternAssessment,
    RiskAssessmentResult,
    RiskThresholds,
    RiskScoringWeights
)
from ..models.shipment_models import (
    ShipmentData,
    ClaimShipmentData,
    ShipmentConsistencyResult,
    ConsistencyCheckResult,
    ConsistencyCheckType
)
from ..config.settings import settings
from .uipath_service import UiPathService, UiPathServiceError
from .context_grounding_service import context_grounding_service

logger = logging.getLogger(__name__)


class RiskAssessor:
    """
    Service for assessing risk in LTL claims.
    Implements configurable risk scoring algorithms.
    """
    
    def __init__(
        self,
        thresholds: Optional[RiskThresholds] = None,
        weights: Optional[RiskScoringWeights] = None
    ):
        """
        Initialize risk assessor with configurable thresholds and weights.
        
        Args:
            thresholds: Risk thresholds for decision making
            weights: Weights for different risk factors
        """
        self.thresholds = thresholds or RiskThresholds()
        self.weights = weights or RiskScoringWeights()
        
        # Validate weights
        if not self.weights.validate_weights():
            logger.warning("Risk scoring weights do not sum to 1.0, normalizing...")
            self._normalize_weights()
        
        logger.info(f"RiskAssessor initialized with thresholds: {self.thresholds.model_dump()}")
        logger.info(f"Risk scoring weights: {self.weights.model_dump()}")
    
    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = (
            self.weights.amount_weight +
            self.weights.damage_type_weight +
            self.weights.historical_weight +
            self.weights.consistency_weight +
            self.weights.policy_weight
        )
        
        if total > 0:
            self.weights.amount_weight /= total
            self.weights.damage_type_weight /= total
            self.weights.historical_weight /= total
            self.weights.consistency_weight /= total
            self.weights.policy_weight /= total
    
    async def assess_claim_risk(
        self,
        claim_id: str,
        claim_amount: float,
        damage_type: str,
        customer_history: Optional[Dict[str, Any]] = None,
        carrier_history: Optional[Dict[str, Any]] = None,
        similar_claims: Optional[List[Dict[str, Any]]] = None
    ) -> RiskAssessmentResult:
        """
        Perform complete risk assessment on a claim.
        
        Args:
            claim_id: Unique claim identifier
            claim_amount: Claim amount in dollars
            damage_type: Type of damage (string or DamageType enum)
            customer_history: Optional customer claim history
            carrier_history: Optional carrier performance history
            similar_claims: Optional list of similar historical claims
            
        Returns:
            Complete risk assessment result
        """
        logger.info(f"🎯 Assessing risk for claim {claim_id}: ${claim_amount}, {damage_type}")
        
        try:
            # Convert damage type string to enum
            damage_type_enum = self._parse_damage_type(damage_type)
            
            # Perform individual risk assessments
            amount_risk = self.calculate_amount_risk(claim_amount)
            damage_risk = self.calculate_damage_type_risk(damage_type_enum)
            historical_risk = self.calculate_historical_risk(
                customer_history=customer_history,
                carrier_history=carrier_history,
                similar_claims=similar_claims
            )
            
            # Build risk factors list
            risk_factors = [
                RiskFactor(
                    name="claim_amount",
                    score=amount_risk.risk_score,
                    weight=self.weights.amount_weight,
                    description=amount_risk.reasoning,
                    confidence=1.0
                ),
                RiskFactor(
                    name="damage_type",
                    score=damage_risk.risk_score,
                    weight=self.weights.damage_type_weight,
                    description=damage_risk.reasoning,
                    confidence=1.0
                ),
                RiskFactor(
                    name="historical_patterns",
                    score=historical_risk.risk_score,
                    weight=self.weights.historical_weight,
                    description=historical_risk.reasoning,
                    confidence=0.8 if customer_history or carrier_history else 0.5
                )
            ]
            
            # Calculate overall risk score (weighted average)
            overall_risk_score = sum(
                factor.score * factor.weight 
                for factor in risk_factors
            )
            
            # Determine risk level
            risk_level = self._categorize_risk_level(overall_risk_score)
            
            # Identify fraud indicators
            fraud_indicators = self._identify_fraud_indicators(
                amount_risk=amount_risk,
                damage_risk=damage_risk,
                historical_risk=historical_risk
            )
            
            # Make decision recommendation
            recommended_decision, decision_confidence, decision_reasoning = self._recommend_decision(
                overall_risk_score=overall_risk_score,
                risk_factors=risk_factors,
                fraud_indicators=fraud_indicators
            )
            
            # Build result
            result = RiskAssessmentResult(
                claim_id=claim_id,
                overall_risk_score=overall_risk_score,
                risk_level=risk_level,
                amount_risk=amount_risk,
                damage_type_risk=damage_risk,
                historical_risk=historical_risk,
                risk_factors=risk_factors,
                recommended_decision=recommended_decision,
                decision_confidence=decision_confidence,
                decision_reasoning=decision_reasoning,
                requires_human_review=(recommended_decision == DecisionType.HUMAN_REVIEW),
                fraud_indicators=fraud_indicators,
                data_quality_issues=[],
                assessed_at=datetime.now()
            )
            
            logger.info(
                f"✅ Risk assessment complete for {claim_id}: "
                f"Score={overall_risk_score:.3f}, Level={risk_level.value}, "
                f"Decision={recommended_decision.value}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Risk assessment failed for claim {claim_id}: {e}")
            raise
    
    def calculate_amount_risk(self, claim_amount: float) -> AmountRiskAssessment:
        """
        Calculate risk score based on claim amount.
        
        Args:
            claim_amount: Claim amount in dollars
            
        Returns:
            Amount-based risk assessment
        """
        # Categorize amount
        if claim_amount < 1000:
            amount_category = "small"
            base_risk = 0.1
        elif claim_amount < 2500:
            amount_category = "medium"
            base_risk = 0.3
        elif claim_amount < self.thresholds.high_amount_threshold:
            amount_category = "large"
            base_risk = 0.5
        elif claim_amount < self.thresholds.critical_amount_threshold:
            amount_category = "very_large"
            base_risk = 0.7
        else:
            amount_category = "critical"
            base_risk = 0.9
        
        # Apply progressive scaling for very high amounts
        if claim_amount >= self.thresholds.critical_amount_threshold:
            # Scale up to 1.0 for amounts significantly above critical threshold
            excess_ratio = (claim_amount - self.thresholds.critical_amount_threshold) / self.thresholds.critical_amount_threshold
            risk_score = min(0.9 + (excess_ratio * 0.1), 1.0)
        else:
            risk_score = base_risk
        
        threshold_exceeded = claim_amount >= self.thresholds.high_amount_threshold
        
        reasoning = (
            f"Claim amount ${claim_amount:,.2f} categorized as '{amount_category}'. "
            f"{'Exceeds' if threshold_exceeded else 'Below'} high-risk threshold "
            f"(${self.thresholds.high_amount_threshold:,.2f})."
        )
        
        return AmountRiskAssessment(
            claim_amount=claim_amount,
            risk_score=risk_score,
            threshold_exceeded=threshold_exceeded,
            amount_category=amount_category,
            reasoning=reasoning
        )
    
    def calculate_damage_type_risk(self, damage_type: DamageType) -> DamageTypeRiskAssessment:
        """
        Calculate risk score based on damage type.
        
        Args:
            damage_type: Type of damage
            
        Returns:
            Damage type risk assessment
        """
        # Risk scores for different damage types
        damage_risk_scores = {
            DamageType.PHYSICAL_DAMAGE: 0.3,  # Common, verifiable
            DamageType.WATER_DAMAGE: 0.4,     # Moderate risk
            DamageType.THEFT: 0.8,             # High risk, fraud indicator
            DamageType.LOSS: 0.7,              # High risk, hard to verify
            DamageType.CONTAMINATION: 0.5,     # Moderate risk
            DamageType.TEMPERATURE_DAMAGE: 0.4, # Moderate risk
            DamageType.CONCEALED_DAMAGE: 0.6,  # Higher risk, discovered later
            DamageType.SHORTAGE: 0.5,          # Moderate risk
            DamageType.OTHER: 0.5              # Unknown, moderate risk
        }
        
        # Fraud indicators for damage types
        fraud_indicator_types = {
            DamageType.THEFT,
            DamageType.LOSS,
            DamageType.CONCEALED_DAMAGE
        }
        
        risk_score = damage_risk_scores.get(damage_type, 0.5)
        is_high_risk_type = risk_score >= 0.6
        typical_fraud_indicator = damage_type in fraud_indicator_types
        
        reasoning = (
            f"Damage type '{damage_type.value}' has risk score {risk_score:.2f}. "
            f"{'High-risk type' if is_high_risk_type else 'Standard risk type'}. "
            f"{'Common fraud indicator' if typical_fraud_indicator else 'Not typically associated with fraud'}."
        )
        
        return DamageTypeRiskAssessment(
            damage_type=damage_type,
            risk_score=risk_score,
            is_high_risk_type=is_high_risk_type,
            typical_fraud_indicator=typical_fraud_indicator,
            reasoning=reasoning
        )
    
    def calculate_historical_risk(
        self,
        customer_history: Optional[Dict[str, Any]] = None,
        carrier_history: Optional[Dict[str, Any]] = None,
        similar_claims: Optional[List[Dict[str, Any]]] = None
    ) -> HistoricalPatternAssessment:
        """
        Calculate risk score based on historical patterns.
        
        Args:
            customer_history: Customer's claim history
            carrier_history: Carrier's performance history
            similar_claims: Similar historical claims
            
        Returns:
            Historical pattern risk assessment
        """
        # Extract customer metrics
        customer_claim_count = 0
        customer_approval_rate = 0.5  # Neutral default
        
        if customer_history:
            customer_claim_count = customer_history.get('total_claims', 0)
            approved_claims = customer_history.get('approved_claims', 0)
            if customer_claim_count > 0:
                customer_approval_rate = approved_claims / customer_claim_count
        
        # Extract carrier metrics
        carrier_claim_count = 0
        carrier_issue_rate = 0.3  # Neutral default
        
        if carrier_history:
            carrier_claim_count = carrier_history.get('total_claims', 0)
            total_shipments = carrier_history.get('total_shipments', 1)
            if total_shipments > 0:
                carrier_issue_rate = carrier_claim_count / total_shipments
        
        # Similar claims analysis
        similar_claims_found = len(similar_claims) if similar_claims else 0
        
        # Calculate risk score based on patterns
        risk_components = []
        
        # Customer pattern risk
        if customer_claim_count > 0:
            # High claim frequency is risky
            if customer_claim_count > 10:
                customer_risk = 0.7
            elif customer_claim_count > 5:
                customer_risk = 0.5
            elif customer_claim_count > 2:
                customer_risk = 0.3
            else:
                customer_risk = 0.2
            
            # Low approval rate increases risk
            if customer_approval_rate < 0.3:
                customer_risk = min(customer_risk + 0.3, 1.0)
            elif customer_approval_rate < 0.5:
                customer_risk = min(customer_risk + 0.1, 1.0)
            
            risk_components.append(customer_risk)
        
        # Carrier pattern risk
        if carrier_claim_count > 0:
            # High issue rate for carrier
            if carrier_issue_rate > 0.1:
                carrier_risk = 0.6
            elif carrier_issue_rate > 0.05:
                carrier_risk = 0.4
            else:
                carrier_risk = 0.2
            
            risk_components.append(carrier_risk)
        
        # Similar claims pattern
        if similar_claims_found > 0:
            # Many similar claims might indicate a pattern
            if similar_claims_found > 5:
                similar_risk = 0.6
            elif similar_claims_found > 2:
                similar_risk = 0.4
            else:
                similar_risk = 0.2
            
            risk_components.append(similar_risk)
        
        # Calculate average risk or use neutral default
        risk_score = sum(risk_components) / len(risk_components) if risk_components else 0.4
        
        reasoning = (
            f"Customer has {customer_claim_count} previous claims "
            f"with {customer_approval_rate:.1%} approval rate. "
            f"Carrier has {carrier_claim_count} claims with {carrier_issue_rate:.1%} issue rate. "
            f"Found {similar_claims_found} similar historical claims."
        )
        
        return HistoricalPatternAssessment(
            customer_claim_count=customer_claim_count,
            customer_approval_rate=customer_approval_rate,
            carrier_claim_count=carrier_claim_count,
            carrier_issue_rate=carrier_issue_rate,
            similar_claims_found=similar_claims_found,
            risk_score=risk_score,
            reasoning=reasoning
        )
    
    def _parse_damage_type(self, damage_type: str) -> DamageType:
        """Parse damage type string to enum."""
        damage_type_lower = damage_type.lower().replace(" ", "_").replace("-", "_")
        
        # Try direct match
        for dt in DamageType:
            if dt.value == damage_type_lower:
                return dt
        
        # Try fuzzy matching
        if "theft" in damage_type_lower or "stolen" in damage_type_lower:
            return DamageType.THEFT
        elif "loss" in damage_type_lower or "lost" in damage_type_lower or "missing" in damage_type_lower:
            return DamageType.LOSS
        elif "water" in damage_type_lower or "wet" in damage_type_lower or "moisture" in damage_type_lower:
            return DamageType.WATER_DAMAGE
        elif "temperature" in damage_type_lower or "frozen" in damage_type_lower or "heat" in damage_type_lower:
            return DamageType.TEMPERATURE_DAMAGE
        elif "concealed" in damage_type_lower or "hidden" in damage_type_lower:
            return DamageType.CONCEALED_DAMAGE
        elif "shortage" in damage_type_lower or "short" in damage_type_lower:
            return DamageType.SHORTAGE
        elif "contamination" in damage_type_lower or "contaminated" in damage_type_lower:
            return DamageType.CONTAMINATION
        elif "physical" in damage_type_lower or "damage" in damage_type_lower or "broken" in damage_type_lower:
            return DamageType.PHYSICAL_DAMAGE
        else:
            return DamageType.OTHER
    
    def _categorize_risk_level(self, risk_score: float) -> RiskLevel:
        """Categorize numeric risk score into risk level."""
        if risk_score >= 0.8:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _identify_fraud_indicators(
        self,
        amount_risk: AmountRiskAssessment,
        damage_risk: DamageTypeRiskAssessment,
        historical_risk: HistoricalPatternAssessment
    ) -> List[str]:
        """Identify potential fraud indicators."""
        indicators = []
        
        # High amount
        if amount_risk.threshold_exceeded:
            indicators.append(f"High claim amount: ${amount_risk.claim_amount:,.2f}")
        
        # Fraud-prone damage type
        if damage_risk.typical_fraud_indicator:
            indicators.append(f"Fraud-prone damage type: {damage_risk.damage_type.value}")
        
        # Frequent claimant
        if historical_risk.customer_claim_count > 5:
            indicators.append(f"Frequent claimant: {historical_risk.customer_claim_count} previous claims")
        
        # Low approval rate
        if historical_risk.customer_approval_rate < 0.3 and historical_risk.customer_claim_count > 0:
            indicators.append(f"Low approval rate: {historical_risk.customer_approval_rate:.1%}")
        
        return indicators
    
    def _recommend_decision(
        self,
        overall_risk_score: float,
        risk_factors: List[RiskFactor],
        fraud_indicators: List[str]
    ) -> tuple[DecisionType, float, str]:
        """
        Recommend a decision based on risk assessment.
        
        Returns:
            Tuple of (decision_type, confidence, reasoning)
        """
        # Calculate average confidence from risk factors
        avg_confidence = sum(f.confidence for f in risk_factors) / len(risk_factors) if risk_factors else 0.5
        
        # Decision logic based on thresholds
        if overall_risk_score <= self.thresholds.auto_approve_threshold:
            # Low risk - auto approve
            if avg_confidence >= self.thresholds.min_confidence_for_auto_decision:
                decision = DecisionType.AUTO_APPROVE
                confidence = avg_confidence
                reasoning = (
                    f"Low risk score ({overall_risk_score:.3f}) below auto-approval threshold "
                    f"({self.thresholds.auto_approve_threshold}). High confidence in assessment."
                )
            else:
                decision = DecisionType.HUMAN_REVIEW
                confidence = avg_confidence
                reasoning = (
                    f"Low risk score but confidence ({avg_confidence:.3f}) below threshold "
                    f"({self.thresholds.min_confidence_for_auto_decision}). Requires review."
                )
        
        elif overall_risk_score >= self.thresholds.auto_reject_threshold:
            # Very high risk - consider auto reject
            if len(fraud_indicators) >= 2 and avg_confidence >= self.thresholds.min_confidence_for_auto_decision:
                decision = DecisionType.AUTO_REJECT
                confidence = avg_confidence
                reasoning = (
                    f"Critical risk score ({overall_risk_score:.3f}) with {len(fraud_indicators)} "
                    f"fraud indicators. Recommended for rejection."
                )
            else:
                decision = DecisionType.HUMAN_REVIEW
                confidence = avg_confidence
                reasoning = (
                    f"Critical risk score ({overall_risk_score:.3f}) requires human review "
                    f"before rejection decision."
                )
        
        elif overall_risk_score >= self.thresholds.human_review_threshold:
            # High risk - human review
            decision = DecisionType.HUMAN_REVIEW
            confidence = avg_confidence
            reasoning = (
                f"High risk score ({overall_risk_score:.3f}) above review threshold "
                f"({self.thresholds.human_review_threshold}). Requires human assessment."
            )
        
        else:
            # Medium risk - human review for safety
            decision = DecisionType.HUMAN_REVIEW
            confidence = avg_confidence * 0.9  # Slightly lower confidence for medium risk
            reasoning = (
                f"Medium risk score ({overall_risk_score:.3f}). "
                f"Requires human review for final decision."
            )
        
        return decision, confidence, reasoning
    
    async def validate_shipment_consistency(
        self,
        claim_data: ClaimShipmentData,
        shipment_data: Optional[ShipmentData] = None
    ) -> ShipmentConsistencyResult:
        """
        Validate consistency between claim and shipment data.
        
        Args:
            claim_data: Claim data for cross-referencing
            shipment_data: Shipment data from Data Fabric (will fetch if not provided)
            
        Returns:
            Shipment consistency validation result
        """
        logger.info(f"🔍 Validating shipment consistency for claim {claim_data.claim_id}")
        
        # Fetch shipment data if not provided
        if not shipment_data:
            try:
                async with UiPathService() as uipath:
                    shipment_dict = await uipath.get_shipment_data(claim_data.shipmentId)
                    if shipment_dict:
                        shipment_data = ShipmentData(**shipment_dict)
            except Exception as e:
                logger.error(f"Failed to fetch shipment data: {e}")
        
        # Check if shipment was found
        shipment_found = shipment_data is not None
        
        if not shipment_found:
            return ShipmentConsistencyResult(
                claim_id=claim_data.claim_id,
                shipment_id=claim_data.shipmentId,
                is_consistent=False,
                consistency_score=0.0,
                risk_adjustment=0.5,  # Moderate risk increase for missing shipment
                checks=[],
                critical_discrepancies=["Shipment data not found in system"],
                warnings=[],
                shipment_found=False,
                missing_fields=["all"],
                requires_investigation=True,
                investigation_priority="high",
                recommended_actions=[
                    "Verify shipment ID is correct",
                    "Check if shipment exists in carrier system",
                    "Request shipment documentation from customer"
                ],
                validated_at=datetime.now()
            )
        
        # Perform individual consistency checks
        checks = []
        critical_discrepancies = []
        warnings = []
        missing_fields = []
        
        # Check 1: Carrier match
        carrier_check = self._check_carrier_match(claim_data, shipment_data)
        checks.append(carrier_check)
        if not carrier_check.passed and carrier_check.severity == "critical":
            critical_discrepancies.append(carrier_check.discrepancy)
        elif not carrier_check.passed:
            warnings.append(carrier_check.discrepancy)
        
        # Check 2: Shipper match
        shipper_check = self._check_shipper_match(claim_data, shipment_data)
        checks.append(shipper_check)
        if not shipper_check.passed and shipper_check.severity == "error":
            warnings.append(shipper_check.discrepancy)
        
        # Check 3: Shipment dates validation
        date_check = self._check_shipment_dates(claim_data, shipment_data)
        checks.append(date_check)
        if not date_check.passed and date_check.severity in ["error", "critical"]:
            warnings.append(date_check.discrepancy)
        
        # Check 4: Declared value vs claim amount
        value_check = self._check_value_consistency(claim_data, shipment_data)
        checks.append(value_check)
        if not value_check.passed and value_check.severity == "critical":
            critical_discrepancies.append(value_check.discrepancy)
        elif not value_check.passed:
            warnings.append(value_check.discrepancy)
        
        # Check 5: Damage report status
        damage_check = self._check_damage_report_status(claim_data, shipment_data)
        checks.append(damage_check)
        if not damage_check.passed:
            warnings.append(damage_check.discrepancy)
        
        # Check 6: Shipment status validation
        status_check = self._check_shipment_status(claim_data, shipment_data)
        checks.append(status_check)
        if not status_check.passed and status_check.severity == "error":
            warnings.append(status_check.discrepancy)
        
        # Calculate consistency score
        passed_checks = sum(1 for check in checks if check.passed)
        total_checks = len(checks)
        consistency_score = passed_checks / total_checks if total_checks > 0 else 0.0
        
        # Calculate risk adjustment based on discrepancies
        risk_adjustment = sum(check.impact_on_risk for check in checks if not check.passed)
        risk_adjustment = min(risk_adjustment, 1.0)  # Cap at 1.0
        
        # Determine if investigation is required
        requires_investigation = len(critical_discrepancies) > 0 or consistency_score < 0.7
        
        # Determine investigation priority
        if len(critical_discrepancies) >= 2:
            investigation_priority = "critical"
        elif len(critical_discrepancies) >= 1:
            investigation_priority = "high"
        elif consistency_score < 0.7:
            investigation_priority = "medium"
        else:
            investigation_priority = "low"
        
        # Generate recommended actions
        recommended_actions = self._generate_consistency_actions(
            checks=checks,
            critical_discrepancies=critical_discrepancies,
            warnings=warnings
        )
        
        result = ShipmentConsistencyResult(
            claim_id=claim_data.claim_id,
            shipment_id=claim_data.shipmentId,
            is_consistent=(consistency_score >= 0.8 and len(critical_discrepancies) == 0),
            consistency_score=consistency_score,
            risk_adjustment=risk_adjustment,
            checks=checks,
            critical_discrepancies=critical_discrepancies,
            warnings=warnings,
            shipment_found=True,
            missing_fields=missing_fields,
            requires_investigation=requires_investigation,
            investigation_priority=investigation_priority,
            recommended_actions=recommended_actions,
            validated_at=datetime.now()
        )
        
        logger.info(
            f"✅ Shipment consistency validation complete: "
            f"Score={consistency_score:.3f}, Risk+={risk_adjustment:.3f}, "
            f"Critical={len(critical_discrepancies)}, Warnings={len(warnings)}"
        )
        
        return result
    
    def _check_carrier_match(
        self,
        claim_data: ClaimShipmentData,
        shipment_data: ShipmentData
    ) -> ConsistencyCheckResult:
        """Check if carrier names match between claim and shipment."""
        claim_carrier = claim_data.carrier.lower().strip()
        shipment_carrier = shipment_data.carrier.lower().strip()
        
        # Exact match
        if claim_carrier == shipment_carrier:
            return ConsistencyCheckResult(
                check_type=ConsistencyCheckType.CARRIER_MATCH,
                passed=True,
                severity="info",
                claim_value=claim_data.carrier,
                shipment_value=shipment_data.carrier,
                discrepancy=None,
                impact_on_risk=0.0
            )
        
        # Fuzzy match (contains)
        if claim_carrier in shipment_carrier or shipment_carrier in claim_carrier:
            return ConsistencyCheckResult(
                check_type=ConsistencyCheckType.CARRIER_MATCH,
                passed=True,
                severity="warning",
                claim_value=claim_data.carrier,
                shipment_value=shipment_data.carrier,
                discrepancy=f"Carrier names similar but not exact: '{claim_data.carrier}' vs '{shipment_data.carrier}'",
                impact_on_risk=0.1
            )
        
        # No match - critical
        return ConsistencyCheckResult(
            check_type=ConsistencyCheckType.CARRIER_MATCH,
            passed=False,
            severity="critical",
            claim_value=claim_data.carrier,
            shipment_value=shipment_data.carrier,
            discrepancy=f"Carrier mismatch: Claim='{claim_data.carrier}', Shipment='{shipment_data.carrier}'",
            impact_on_risk=0.4
        )
    
    def _check_shipper_match(
        self,
        claim_data: ClaimShipmentData,
        shipment_data: ShipmentData
    ) -> ConsistencyCheckResult:
        """Check if shipper names match."""
        claim_shipper = (claim_data.shipper or "").lower().strip()
        shipment_shipper = shipment_data.shipper.lower().strip()
        
        if not claim_shipper:
            return ConsistencyCheckResult(
                check_type=ConsistencyCheckType.ORIGIN_DESTINATION,
                passed=True,
                severity="info",
                claim_value=None,
                shipment_value=shipment_data.shipper,
                discrepancy=None,
                impact_on_risk=0.0
            )
        
        if claim_shipper == shipment_shipper or claim_shipper in shipment_shipper or shipment_shipper in claim_shipper:
            return ConsistencyCheckResult(
                check_type=ConsistencyCheckType.ORIGIN_DESTINATION,
                passed=True,
                severity="info",
                claim_value=claim_data.shipper,
                shipment_value=shipment_data.shipper,
                discrepancy=None,
                impact_on_risk=0.0
            )
        
        return ConsistencyCheckResult(
            check_type=ConsistencyCheckType.ORIGIN_DESTINATION,
            passed=False,
            severity="error",
            claim_value=claim_data.shipper,
            shipment_value=shipment_data.shipper,
            discrepancy=f"Shipper mismatch: Claim='{claim_data.shipper}', Shipment='{shipment_data.shipper}'",
            impact_on_risk=0.2
        )
    
    def _check_shipment_dates(
        self,
        claim_data: ClaimShipmentData,
        shipment_data: ShipmentData
    ) -> ConsistencyCheckResult:
        """Validate shipment dates are logical."""
        try:
            from dateutil import parser
            
            pickup_date = parser.parse(shipment_data.pickupDate)
            delivery_date = parser.parse(shipment_data.deliveryDate) if shipment_data.deliveryDate else None
            claim_date = parser.parse(claim_data.submittedDate) if claim_data.submittedDate else None
            
            # Check if claim was filed before shipment pickup (suspicious)
            if claim_date and claim_date < pickup_date:
                return ConsistencyCheckResult(
                    check_type=ConsistencyCheckType.SHIPMENT_DATE,
                    passed=False,
                    severity="critical",
                    claim_value=claim_data.submittedDate,
                    shipment_value=shipment_data.pickupDate,
                    discrepancy=f"Claim filed before shipment pickup: Claim={claim_date.date()}, Pickup={pickup_date.date()}",
                    impact_on_risk=0.5
                )
            
            # Check if delivery date is before pickup (data error)
            if delivery_date and delivery_date < pickup_date:
                return ConsistencyCheckResult(
                    check_type=ConsistencyCheckType.DELIVERY_DATE,
                    passed=False,
                    severity="error",
                    claim_value=None,
                    shipment_value=f"Pickup={pickup_date.date()}, Delivery={delivery_date.date()}",
                    discrepancy="Delivery date before pickup date - data integrity issue",
                    impact_on_risk=0.3
                )
            
            return ConsistencyCheckResult(
                check_type=ConsistencyCheckType.SHIPMENT_DATE,
                passed=True,
                severity="info",
                claim_value=claim_data.submittedDate,
                shipment_value=shipment_data.pickupDate,
                discrepancy=None,
                impact_on_risk=0.0
            )
            
        except Exception as e:
            logger.warning(f"Date validation error: {e}")
            return ConsistencyCheckResult(
                check_type=ConsistencyCheckType.SHIPMENT_DATE,
                passed=True,
                severity="warning",
                claim_value=claim_data.submittedDate,
                shipment_value=shipment_data.pickupDate,
                discrepancy="Unable to validate dates",
                impact_on_risk=0.1
            )
    
    def _check_value_consistency(
        self,
        claim_data: ClaimShipmentData,
        shipment_data: ShipmentData
    ) -> ConsistencyCheckResult:
        """Check if claim amount is consistent with declared value."""
        if not shipment_data.declaredValueUsd:
            return ConsistencyCheckResult(
                check_type=ConsistencyCheckType.WEIGHT_VALUE,
                passed=True,
                severity="info",
                claim_value=claim_data.amount,
                shipment_value=None,
                discrepancy="No declared value on shipment",
                impact_on_risk=0.0
            )
        
        claim_amount = claim_data.amount
        declared_value = shipment_data.declaredValueUsd
        
        # Claim significantly exceeds declared value (suspicious)
        if claim_amount > declared_value * 1.5:
            return ConsistencyCheckResult(
                check_type=ConsistencyCheckType.WEIGHT_VALUE,
                passed=False,
                severity="critical",
                claim_value=claim_amount,
                shipment_value=declared_value,
                discrepancy=f"Claim amount (${claim_amount:,.2f}) exceeds declared value (${declared_value:,.2f}) by {((claim_amount/declared_value - 1) * 100):.1f}%",
                impact_on_risk=0.4
            )
        
        # Claim moderately exceeds declared value
        if claim_amount > declared_value * 1.1:
            return ConsistencyCheckResult(
                check_type=ConsistencyCheckType.WEIGHT_VALUE,
                passed=False,
                severity="warning",
                claim_value=claim_amount,
                shipment_value=declared_value,
                discrepancy=f"Claim amount (${claim_amount:,.2f}) exceeds declared value (${declared_value:,.2f}) by {((claim_amount/declared_value - 1) * 100):.1f}%",
                impact_on_risk=0.2
            )
        
        return ConsistencyCheckResult(
            check_type=ConsistencyCheckType.WEIGHT_VALUE,
            passed=True,
            severity="info",
            claim_value=claim_amount,
            shipment_value=declared_value,
            discrepancy=None,
            impact_on_risk=0.0
        )
    
    def _check_damage_report_status(
        self,
        claim_data: ClaimShipmentData,
        shipment_data: ShipmentData
    ) -> ConsistencyCheckResult:
        """Check if damage was previously reported on shipment."""
        damage_reported = shipment_data.damageReported or False
        
        if damage_reported:
            return ConsistencyCheckResult(
                check_type=ConsistencyCheckType.DAMAGE_REPORT,
                passed=True,
                severity="info",
                claim_value="Claim filed",
                shipment_value="Damage previously reported",
                discrepancy=None,
                impact_on_risk=-0.1  # Reduces risk slightly
            )
        
        return ConsistencyCheckResult(
            check_type=ConsistencyCheckType.DAMAGE_REPORT,
            passed=False,
            severity="warning",
            claim_value="Claim filed",
            shipment_value="No prior damage report",
            discrepancy="Damage not previously reported on shipment record",
            impact_on_risk=0.15
        )
    
    def _check_shipment_status(
        self,
        claim_data: ClaimShipmentData,
        shipment_data: ShipmentData
    ) -> ConsistencyCheckResult:
        """Validate shipment status is appropriate for claim."""
        status = shipment_data.status.lower()
        
        # Valid statuses for claims
        valid_claim_statuses = ["delivered", "damaged", "lost", "delayed"]
        
        if any(valid_status in status for valid_status in valid_claim_statuses):
            return ConsistencyCheckResult(
                check_type=ConsistencyCheckType.TRACKING_NUMBER,
                passed=True,
                severity="info",
                claim_value="Claim filed",
                shipment_value=shipment_data.status,
                discrepancy=None,
                impact_on_risk=0.0
            )
        
        # Claim filed for in-transit shipment (suspicious)
        if "transit" in status or "pending" in status:
            return ConsistencyCheckResult(
                check_type=ConsistencyCheckType.TRACKING_NUMBER,
                passed=False,
                severity="error",
                claim_value="Claim filed",
                shipment_value=shipment_data.status,
                discrepancy=f"Claim filed for shipment still in transit (status: {shipment_data.status})",
                impact_on_risk=0.3
            )
        
        return ConsistencyCheckResult(
            check_type=ConsistencyCheckType.TRACKING_NUMBER,
            passed=True,
            severity="warning",
            claim_value="Claim filed",
            shipment_value=shipment_data.status,
            discrepancy=f"Unusual shipment status for claim: {shipment_data.status}",
            impact_on_risk=0.1
        )
    
    def _generate_consistency_actions(
        self,
        checks: List[ConsistencyCheckResult],
        critical_discrepancies: List[str],
        warnings: List[str]
    ) -> List[str]:
        """Generate recommended actions based on consistency checks."""
        actions = []
        
        if critical_discrepancies:
            actions.append("Escalate to senior reviewer for critical discrepancies")
            actions.append("Request additional documentation from customer")
        
        # Check for specific issues
        for check in checks:
            if not check.passed:
                if check.check_type == ConsistencyCheckType.CARRIER_MATCH:
                    actions.append("Verify carrier information with customer and shipment records")
                elif check.check_type == ConsistencyCheckType.WEIGHT_VALUE:
                    actions.append("Request proof of value (invoice, receipt) from customer")
                elif check.check_type == ConsistencyCheckType.SHIPMENT_DATE:
                    actions.append("Verify timeline of events with customer")
                elif check.check_type == ConsistencyCheckType.DAMAGE_REPORT:
                    actions.append("Check if damage was reported to carrier at delivery")
        
        if not actions:
            actions.append("Proceed with standard claim processing")
        
        return list(set(actions))  # Remove duplicates
    
    async def apply_policy_rules(
        self,
        claim_id: str,
        claim_amount: float,
        damage_type: str,
        carrier: str,
        risk_assessment: RiskAssessmentResult,
        consistency_result: Optional[ShipmentConsistencyResult] = None
    ) -> Dict[str, Any]:
        """
        Apply company policies using Context Grounding to refine risk assessment.
        
        Args:
            claim_id: Claim identifier
            claim_amount: Claim amount
            damage_type: Type of damage
            carrier: Carrier name
            risk_assessment: Initial risk assessment
            consistency_result: Optional shipment consistency validation
            
        Returns:
            Policy evaluation result with recommendations
        """
        logger.info(f"📋 Applying policy rules for claim {claim_id}")
        
        try:
            # Build policy query based on claim characteristics
            policy_query = self._build_policy_query(
                claim_amount=claim_amount,
                damage_type=damage_type,
                carrier=carrier,
                risk_level=risk_assessment.risk_level.value,
                has_discrepancies=consistency_result and not consistency_result.is_consistent if consistency_result else False
            )
            
            # Search policy knowledge base
            policy_results = await context_grounding_service.search_knowledge_base(
                query=policy_query,
                knowledge_type="policies",
                max_results=5
            )
            
            if not policy_results:
                logger.warning("No relevant policies found, using default rules")
                return self._apply_default_policies(risk_assessment, consistency_result)
            
            # Extract policy guidance
            policy_guidance = self._extract_policy_guidance(policy_results)
            
            # Apply policy rules to adjust risk and decision
            adjusted_assessment = self._adjust_risk_with_policies(
                risk_assessment=risk_assessment,
                policy_guidance=policy_guidance,
                consistency_result=consistency_result
            )
            
            logger.info(
                f"✅ Policy rules applied: "
                f"Original risk={risk_assessment.overall_risk_score:.3f}, "
                f"Adjusted risk={adjusted_assessment['adjusted_risk_score']:.3f}"
            )
            
            return adjusted_assessment
            
        except Exception as e:
            logger.error(f"❌ Policy application failed: {e}")
            # Fallback to default policies
            return self._apply_default_policies(risk_assessment, consistency_result)
    
    def _build_policy_query(
        self,
        claim_amount: float,
        damage_type: str,
        carrier: str,
        risk_level: str,
        has_discrepancies: bool
    ) -> str:
        """Build a policy search query based on claim characteristics."""
        query_parts = [
            f"LTL freight claim policy",
            f"damage type {damage_type}",
            f"claim amount ${claim_amount:,.0f}",
            f"{risk_level} risk"
        ]
        
        if has_discrepancies:
            query_parts.append("discrepancies inconsistencies")
        
        if claim_amount > self.thresholds.high_amount_threshold:
            query_parts.append("high value claim approval requirements")
        
        return " ".join(query_parts)
    
    def _extract_policy_guidance(self, policy_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract actionable guidance from policy search results."""
        guidance = {
            "approval_thresholds": {},
            "required_documentation": [],
            "escalation_rules": [],
            "special_conditions": [],
            "relevant_policies": []
        }
        
        for result in policy_results:
            content = result.get("content", "").lower()
            score = result.get("score", 0.0)
            
            # Store relevant policy excerpts
            if score >= 0.5:
                guidance["relevant_policies"].append({
                    "content": result.get("content", ""),
                    "score": score,
                    "source": result.get("source", "unknown")
                })
            
            # Extract approval thresholds
            if "approval" in content and "threshold" in content:
                guidance["approval_thresholds"]["found"] = True
            
            # Extract documentation requirements
            if "documentation" in content or "evidence" in content or "proof" in content:
                guidance["required_documentation"].append(result.get("content", ""))
            
            # Extract escalation rules
            if "escalate" in content or "senior" in content or "manager" in content:
                guidance["escalation_rules"].append(result.get("content", ""))
            
            # Extract special conditions
            if "exception" in content or "special" in content or "condition" in content:
                guidance["special_conditions"].append(result.get("content", ""))
        
        return guidance
    
    def _adjust_risk_with_policies(
        self,
        risk_assessment: RiskAssessmentResult,
        policy_guidance: Dict[str, Any],
        consistency_result: Optional[ShipmentConsistencyResult]
    ) -> Dict[str, Any]:
        """Adjust risk assessment based on policy guidance."""
        original_risk = risk_assessment.overall_risk_score
        adjusted_risk = original_risk
        policy_adjustments = []
        
        # Apply policy-based adjustments
        
        # 1. Documentation requirements
        if policy_guidance["required_documentation"]:
            # Increase risk slightly if extensive documentation is required
            adjusted_risk = min(adjusted_risk + 0.05, 1.0)
            policy_adjustments.append({
                "type": "documentation_required",
                "adjustment": 0.05,
                "reason": "Policy requires additional documentation"
            })
        
        # 2. Escalation rules
        if policy_guidance["escalation_rules"]:
            # Flag for human review if escalation is mentioned
            policy_adjustments.append({
                "type": "escalation_required",
                "adjustment": 0.0,
                "reason": "Policy requires escalation for this claim type"
            })
        
        # 3. Special conditions
        if policy_guidance["special_conditions"]:
            # Moderate risk increase for special conditions
            adjusted_risk = min(adjusted_risk + 0.1, 1.0)
            policy_adjustments.append({
                "type": "special_conditions",
                "adjustment": 0.1,
                "reason": "Special policy conditions apply"
            })
        
        # 4. Consistency check impact
        if consistency_result:
            consistency_adjustment = consistency_result.risk_adjustment
            adjusted_risk = min(adjusted_risk + consistency_adjustment, 1.0)
            policy_adjustments.append({
                "type": "consistency_check",
                "adjustment": consistency_adjustment,
                "reason": f"Shipment consistency score: {consistency_result.consistency_score:.2f}"
            })
        
        # Determine final decision with policy context
        final_decision = self._determine_policy_based_decision(
            adjusted_risk=adjusted_risk,
            original_decision=risk_assessment.recommended_decision,
            policy_guidance=policy_guidance,
            consistency_result=consistency_result
        )
        
        return {
            "original_risk_score": original_risk,
            "adjusted_risk_score": adjusted_risk,
            "risk_adjustment_total": adjusted_risk - original_risk,
            "policy_adjustments": policy_adjustments,
            "original_decision": risk_assessment.recommended_decision.value,
            "final_decision": final_decision,
            "policy_guidance_applied": len(policy_guidance["relevant_policies"]) > 0,
            "relevant_policies_count": len(policy_guidance["relevant_policies"]),
            "requires_escalation": len(policy_guidance["escalation_rules"]) > 0,
            "documentation_required": len(policy_guidance["required_documentation"]) > 0,
            "policy_excerpts": [
                {
                    "content": p["content"][:200] + "..." if len(p["content"]) > 200 else p["content"],
                    "score": p["score"]
                }
                for p in policy_guidance["relevant_policies"][:3]
            ]
        }
    
    def _determine_policy_based_decision(
        self,
        adjusted_risk: float,
        original_decision: DecisionType,
        policy_guidance: Dict[str, Any],
        consistency_result: Optional[ShipmentConsistencyResult]
    ) -> str:
        """Determine final decision considering policy guidance."""
        
        # Force human review if escalation is required by policy
        if policy_guidance["escalation_rules"]:
            return DecisionType.HUMAN_REVIEW.value
        
        # Force human review if critical discrepancies exist
        if consistency_result and len(consistency_result.critical_discrepancies) > 0:
            return DecisionType.HUMAN_REVIEW.value
        
        # Apply standard risk thresholds with adjusted risk
        if adjusted_risk <= self.thresholds.auto_approve_threshold:
            return DecisionType.AUTO_APPROVE.value
        elif adjusted_risk >= self.thresholds.auto_reject_threshold:
            # Still require human review for rejection unless very clear
            if adjusted_risk >= 0.95:
                return DecisionType.AUTO_REJECT.value
            else:
                return DecisionType.HUMAN_REVIEW.value
        else:
            return DecisionType.HUMAN_REVIEW.value
    
    def _apply_default_policies(
        self,
        risk_assessment: RiskAssessmentResult,
        consistency_result: Optional[ShipmentConsistencyResult]
    ) -> Dict[str, Any]:
        """Apply default policy rules when Context Grounding is unavailable."""
        logger.info("Applying default policy rules")
        
        original_risk = risk_assessment.overall_risk_score
        adjusted_risk = original_risk
        policy_adjustments = []
        
        # Default rule: Add consistency risk if available
        if consistency_result:
            consistency_adjustment = consistency_result.risk_adjustment
            adjusted_risk = min(adjusted_risk + consistency_adjustment, 1.0)
            policy_adjustments.append({
                "type": "consistency_check",
                "adjustment": consistency_adjustment,
                "reason": f"Shipment consistency score: {consistency_result.consistency_score:.2f}"
            })
        
        # Default rule: High amounts require review
        if risk_assessment.amount_risk.threshold_exceeded:
            policy_adjustments.append({
                "type": "high_amount",
                "adjustment": 0.0,
                "reason": "High claim amount requires human review"
            })
        
        # Determine decision
        if adjusted_risk <= self.thresholds.auto_approve_threshold and not risk_assessment.amount_risk.threshold_exceeded:
            final_decision = DecisionType.AUTO_APPROVE.value
        elif adjusted_risk >= self.thresholds.auto_reject_threshold:
            final_decision = DecisionType.HUMAN_REVIEW.value  # Conservative default
        else:
            final_decision = DecisionType.HUMAN_REVIEW.value
        
        return {
            "original_risk_score": original_risk,
            "adjusted_risk_score": adjusted_risk,
            "risk_adjustment_total": adjusted_risk - original_risk,
            "policy_adjustments": policy_adjustments,
            "original_decision": risk_assessment.recommended_decision.value,
            "final_decision": final_decision,
            "policy_guidance_applied": False,
            "relevant_policies_count": 0,
            "requires_escalation": False,
            "documentation_required": False,
            "policy_excerpts": [],
            "note": "Default policies applied - Context Grounding unavailable"
        }
    
    async def get_policy_recommendations(
        self,
        claim_type: str,
        damage_type: str,
        carrier: str
    ) -> List[Dict[str, Any]]:
        """
        Get specific policy recommendations for a claim scenario.
        
        Args:
            claim_type: Type of claim
            damage_type: Type of damage
            carrier: Carrier name
            
        Returns:
            List of relevant policy recommendations
        """
        logger.info(f"📚 Fetching policy recommendations for {claim_type} claim")
        
        try:
            # Build specific policy query
            query = f"LTL freight claim policy {claim_type} {damage_type} carrier {carrier} requirements procedures"
            
            # Search policies
            results = await context_grounding_service.search_knowledge_base(
                query=query,
                knowledge_type="policies",
                max_results=10
            )
            
            # Format recommendations
            recommendations = []
            for result in results:
                if result.get("score", 0) >= 0.4:
                    recommendations.append({
                        "content": result.get("content", ""),
                        "relevance_score": result.get("score", 0),
                        "source": result.get("source", "unknown"),
                        "knowledge_type": result.get("knowledge_type", "policy")
                    })
            
            logger.info(f"✅ Found {len(recommendations)} policy recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch policy recommendations: {e}")
            return []


# Global risk assessor instance with default configuration
risk_assessor = RiskAssessor()
