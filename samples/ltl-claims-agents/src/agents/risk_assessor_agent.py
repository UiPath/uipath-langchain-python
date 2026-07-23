"""
Risk Assessor Sub-Agent for LTL Claims Processing
Specialized agent for risk analysis and scoring operations.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

from uipath_langchain.chat.models import UiPathChat

from ..services.uipath_service import UiPathService
from .config import RiskAssessorConfig
from .exceptions import OrchestratorError


logger = logging.getLogger(__name__)


class RiskAssessmentError(OrchestratorError):
    """
    Raised when risk assessment fails.
    
    Additional Attributes:
        risk_factors: Risk factors identified before failure
    """
    
    def __init__(
        self,
        message: str,
        claim_id: Optional[str] = None,
        risk_factors: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, claim_id=claim_id, step_name="assess_risk", **kwargs)
        self.risk_factors = risk_factors or []
    
    def is_critical(self) -> bool:
        """Risk assessment errors are not critical - can use default medium risk."""
        return False


class RiskAssessorAgent:
    """
    Specialized agent for risk assessment operations.
    
    Responsibilities:
    - Analyze claim data for risk factors
    - Calculate weighted risk scores
    - Categorize risk levels (low, medium, high)
    - Provide risk reasoning and recommendations
    
    Implements Requirements 5.1, 5.2, 5.3, 5.4, 11.1
    """
    
    # Risk thresholds
    LOW_RISK_THRESHOLD = 0.4
    HIGH_RISK_THRESHOLD = 0.7
    
    # Risk factor weights
    WEIGHT_HIGH_AMOUNT = 0.25
    WEIGHT_CLAIM_TYPE = 0.20
    WEIGHT_LOW_CONFIDENCE = 0.20
    WEIGHT_MISSING_DOCS = 0.15
    WEIGHT_POLICY_VIOLATIONS = 0.20
    
    # Partial weight for document errors (less severe than missing docs)
    DOCUMENT_ERROR_WEIGHT_MULTIPLIER = 0.5
    
    def __init__(self, uipath_service: UiPathService, config: Optional[RiskAssessorConfig] = None):
        """
        Initialize the risk assessor agent.
        
        Args:
            uipath_service: Authenticated UiPath service instance
            config: Optional configuration object (uses defaults if not provided)
        """
        self.uipath_service = uipath_service
        self.config = config or RiskAssessorConfig()
        
        # Use UiPath Chat model (gpt-4o-mini for efficiency in risk analysis)
        self.llm = UiPathChat(
            model="gpt-4o-mini-2024-07-18",
            temperature=0,
            max_tokens=2000,
            timeout=30,
            max_retries=2
        )
        
        logger.info("[RISK_ASSESSOR] Initialized risk assessor agent")
    
    @staticmethod
    def _extract_claim_id(state: Dict[str, Any]) -> str:
        """Extract claim ID from state, handling both field name formats."""
        return state.get('claim_id') or state.get('ObjectClaimId', 'UNKNOWN')
    
    def _collect_risk_factors(self, state: Dict[str, Any]) -> Tuple[List[str], Dict[str, float]]:
        """
        Collect all risk factors and their scores from the state.
        
        Args:
            state: Current GraphState containing claim data
            
        Returns:
            Tuple of (risk_factors list, risk_scores dict)
        """
        risk_factors = []
        risk_scores = {}
        
        # Define assessment methods to run
        assessments = [
            ("amount", self._assess_claim_amount),
            ("type", self._assess_claim_type),
            ("confidence", self._assess_extraction_confidence),
            ("documents", self._assess_missing_documents),
            ("policy", self._assess_policy_violations),
        ]
        
        # Run each assessment and collect results
        for key, assessment_method in assessments:
            result = assessment_method(state)
            if result["has_risk"]:
                risk_factors.append(result["factor"])
                risk_scores[key] = result["score"]
        
        return risk_factors, risk_scores
    
    async def assess_risk(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive risk assessment on the claim.
        
        Analyzes multiple risk factors including:
        - Claim amount (high amounts increase risk)
        - Claim type (loss/theft are higher risk)
        - Extraction confidence (low confidence increases risk)
        - Missing documents (incomplete data increases risk)
        - Policy violations (violations increase risk)
        
        Args:
            state: Current GraphState containing claim data
            
        Returns:
            Dictionary with:
                - risk_score: Numerical score from 0.0 to 1.0
                - risk_level: Categorization (low, medium, high)
                - risk_factors: List of identified risk factors
                - risk_reasoning: Explanation of risk assessment
                
        Implements Requirements 5.1, 5.2, 5.3, 5.4
        """
        from langgraph.prebuilt import create_react_agent
        from langchain_core.messages import HumanMessage
        
        claim_id = self._extract_claim_id(state)
        
        logger.info(f"[RISK_ASSESSOR] Starting risk assessment for claim: {claim_id}")
        
        try:
            # Collect all risk factors using rule-based assessment (Requirement 5.2)
            risk_factors, risk_scores = self._collect_risk_factors(state)
            
            # Calculate weighted risk score (Requirement 5.3)
            risk_score = self._calculate_risk_score(risk_scores)
            
            # Categorize risk level (Requirement 5.4)
            risk_level = self._categorize_risk_level(risk_score)
            
            # Build prompt
            system_prompt = (
                "As a risk assessment specialist, analyze the provided claim data and risk factors. "
                "Provide clear reasoning for the risk level determination. "
                "Consider claim amount, type, document quality, and policy compliance."
            )
            
            # Build risk analysis prompt
            risk_analysis_prompt = (
                f"Analyze risk for claim {claim_id}:\n"
                f"- Risk Score: {risk_score:.3f}\n"
                f"- Risk Level: {risk_level}\n"
                f"- Risk Factors: {', '.join(risk_factors) if risk_factors else 'None'}\n"
                f"- Claim Amount: ${state.get('claim_amount', 0):,.2f}\n"
                f"- Claim Type: {state.get('claim_type', 'unknown')}\n"
                f"Provide detailed reasoning for this risk assessment."
            )
            
            # Debug logging
            logger.debug(f"[RISK_ASSESSOR] System prompt: {system_prompt}")
            logger.debug(f"[RISK_ASSESSOR] Risk analysis prompt: {risk_analysis_prompt}")
            logger.debug(f"[RISK_ASSESSOR] Risk factors identified: {risk_factors}")
            logger.debug(f"[RISK_ASSESSOR] Risk scores breakdown: {risk_scores}")
            
            # Use react agent for enhanced risk reasoning
            risk_agent = create_react_agent(
                self.llm,
                tools=[],  # Risk assessment is primarily analytical, no tools needed
                prompt=system_prompt
            )
            
            # Invoke agent for reasoning
            logger.debug(f"[RISK_ASSESSOR] Invoking risk reasoning agent for claim {claim_id}")
            result = await risk_agent.ainvoke({
                "messages": [HumanMessage(content=risk_analysis_prompt)]
            })
            
            # Debug: Log all messages in the result
            logger.debug(f"[RISK_ASSESSOR] Agent returned {len(result['messages'])} messages")
            for i, msg in enumerate(result["messages"]):
                msg_type = type(msg).__name__
                logger.debug(f"[RISK_ASSESSOR] Message {i} ({msg_type}): {str(msg.content)[:150]}...")
            
            # Extract reasoning from agent response
            risk_reasoning = result["messages"][-1].content
            logger.debug(f"[RISK_ASSESSOR] Agent reasoning: {risk_reasoning[:200]}...")
            
            logger.info(
                f"[RISK_ASSESSOR] Risk assessment complete for claim {claim_id}: "
                f"score={risk_score:.3f}, level={risk_level}, "
                f"factors={len(risk_factors)}"
            )
            
            return {
                "risk_score": risk_score,
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "risk_reasoning": risk_reasoning,
                "risk_scores_breakdown": risk_scores
            }
            
        except Exception as e:
            logger.error(f"[RISK_ASSESSOR] Risk assessment failed for claim {claim_id}: {e}")
            
            # Return default medium risk on failure (Requirement 5.1)
            return self._get_default_risk_assessment(claim_id, str(e))
    
    @staticmethod
    def _create_risk_result(has_risk: bool, factor: Optional[str] = None, score: float = 0.0) -> Dict[str, Any]:
        """Helper method to create consistent risk assessment results."""
        return {"has_risk": has_risk, "factor": factor, "score": score}
    
    def _assess_claim_amount(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess risk based on claim amount.
        
        High claim amounts (>$10,000) are considered higher risk.
        
        Args:
            state: Current GraphState
            
        Returns:
            Dictionary with has_risk, factor, and score
        """
        claim_amount = state.get('claim_amount') or state.get('ClaimAmount', 0)
        
        if not claim_amount:
            return self._create_risk_result(False)
        
        # High amount threshold from config
        if claim_amount > self.config.high_amount_threshold:
            return self._create_risk_result(
                True,
                f"High claim amount: ${claim_amount:,.2f}",
                self.WEIGHT_HIGH_AMOUNT
            )
        
        return self._create_risk_result(False)
    
    def _assess_claim_type(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess risk based on claim type.
        
        Loss and theft claims are considered higher risk.
        
        Args:
            state: Current GraphState
            
        Returns:
            Dictionary with has_risk, factor, and score
        """
        claim_type = (state.get('claim_type') or state.get('ClaimType', '')).lower()
        
        if not claim_type:
            return self._create_risk_result(False)
        
        # High-risk claim types from config
        if claim_type in self.config.high_risk_claim_types:
            return self._create_risk_result(
                True,
                f"High-risk claim type: {claim_type}",
                self.WEIGHT_CLAIM_TYPE
            )
        
        return self._create_risk_result(False)
    
    def _assess_extraction_confidence(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess risk based on extraction confidence scores.
        
        Low confidence in extracted data increases risk.
        
        Args:
            state: Current GraphState
            
        Returns:
            Dictionary with has_risk, factor, and score
        """
        extraction_confidence = state.get('extraction_confidence', {})
        
        if not extraction_confidence:
            return self._create_risk_result(False)
        
        # Calculate average confidence
        confidence_values = list(extraction_confidence.values())
        avg_confidence = sum(confidence_values) / len(confidence_values)
        
        # Low confidence threshold from config
        if avg_confidence < self.config.low_confidence_threshold:
            low_confidence_count = sum(
                1 for c in confidence_values 
                if c < self.config.low_confidence_threshold
            )
            
            return self._create_risk_result(
                True,
                f"Low extraction confidence: {avg_confidence:.2%} average "
                f"({low_confidence_count} fields below threshold)",
                self.WEIGHT_LOW_CONFIDENCE
            )
        
        return self._create_risk_result(False)
    
    def _assess_missing_documents(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess risk based on missing or incomplete documents.
        
        Missing required documents increases risk.
        
        Args:
            state: Current GraphState
            
        Returns:
            Dictionary with has_risk, factor, and score
        """
        # Check if documents were expected but not downloaded
        shipping_docs = state.get('shipping_documents', [])
        damage_evidence = state.get('damage_evidence', [])
        downloaded_docs = state.get('downloaded_documents', [])
        
        expected_doc_count = len(shipping_docs) + len(damage_evidence)
        actual_doc_count = len(downloaded_docs)
        
        if expected_doc_count > 0 and actual_doc_count < expected_doc_count:
            missing_count = expected_doc_count - actual_doc_count
            
            return self._create_risk_result(
                True,
                f"Missing documents: {missing_count} of {expected_doc_count} expected documents not available",
                self.WEIGHT_MISSING_DOCS
            )
        
        # Check for extraction errors
        errors = state.get('errors', [])
        doc_errors = [e for e in errors if 'document' in str(e).lower()]
        
        if doc_errors:
            return self._create_risk_result(
                True,
                f"Document processing errors: {len(doc_errors)} errors encountered",
                self.WEIGHT_MISSING_DOCS * self.DOCUMENT_ERROR_WEIGHT_MULTIPLIER
            )
        
        return self._create_risk_result(False)
    
    def _assess_policy_violations(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess risk based on policy violations.
        
        Policy violations significantly increase risk.
        
        Args:
            state: Current GraphState
            
        Returns:
            Dictionary with has_risk, factor, and score
        """
        policy_violations = state.get('policy_violations', [])
        
        if policy_violations:
            return self._create_risk_result(
                True,
                f"Policy violations detected: {len(policy_violations)} violations found",
                self.WEIGHT_POLICY_VIOLATIONS
            )
        
        return self._create_risk_result(False)
    
    def _calculate_risk_score(self, risk_scores: Dict[str, float]) -> float:
        """
        Calculate weighted risk score from individual risk components.
        
        Args:
            risk_scores: Dictionary of risk component scores
            
        Returns:
            Overall risk score from 0.0 to 1.0
            
        Implements Requirement 5.3
        """
        # Sum all risk scores
        total_score = sum(risk_scores.values())
        
        # Normalize to 0.0-1.0 range
        normalized_score = min(total_score, 1.0)
        
        logger.debug(
            f"[RISK_ASSESSOR] Risk score calculation: "
            f"components={risk_scores}, total={normalized_score:.3f}"
        )
        
        return normalized_score
    
    def _categorize_risk_level(self, risk_score: float) -> str:
        """
        Categorize risk score into low, medium, or high.
        
        Args:
            risk_score: Numerical risk score (0.0 to 1.0)
            
        Returns:
            Risk level: 'low', 'medium', or 'high'
            
        Implements Requirement 5.4
        """
        if risk_score >= self.HIGH_RISK_THRESHOLD:
            return "high"
        elif risk_score >= self.LOW_RISK_THRESHOLD:
            return "medium"
        else:
            return "low"
    
    def _generate_risk_reasoning(
        self,
        risk_score: float,
        risk_level: str,
        risk_factors: List[str],
        risk_scores: Dict[str, float]
    ) -> str:
        """
        Generate human-readable explanation of risk assessment.
        
        Args:
            risk_score: Overall risk score
            risk_level: Categorized risk level
            risk_factors: List of identified risk factors
            risk_scores: Breakdown of risk component scores
            
        Returns:
            Risk reasoning explanation
        """
        if not risk_factors:
            return (
                f"Risk assessment: {risk_level} risk (score: {risk_score:.3f}). "
                f"No significant risk factors identified."
            )
        
        reasoning_parts = [
            f"Risk assessment: {risk_level} risk (score: {risk_score:.3f}).",
            f"Identified {len(risk_factors)} risk factor(s):"
        ]
        
        # Add each risk factor
        for factor in risk_factors:
            reasoning_parts.append(f"  - {factor}")
        
        # Add recommendation based on risk level
        if risk_level == "high":
            reasoning_parts.append(
                "Recommendation: Mandatory human review required due to high risk level."
            )
        elif risk_level == "medium":
            reasoning_parts.append(
                "Recommendation: Consider additional validation or review."
            )
        else:
            reasoning_parts.append(
                "Recommendation: Standard processing can proceed."
            )
        
        return "\n".join(reasoning_parts)
    
    def _get_default_risk_assessment(self, claim_id: str, error_msg: str) -> Dict[str, Any]:
        """
        Get default medium risk assessment when assessment fails.
        
        Args:
            claim_id: Claim identifier
            error_msg: Error message
            
        Returns:
            Default risk assessment
            
        Implements Requirement 5.1 (graceful degradation)
        """
        logger.warning(
            f"[RISK_ASSESSOR] Using default medium risk for claim {claim_id} "
            f"due to assessment failure: {error_msg}"
        )
        
        return {
            "risk_score": 0.5,
            "risk_level": "medium",
            "risk_factors": [
                "Risk assessment failed - defaulting to medium risk for safety"
            ],
            "risk_reasoning": (
                f"Risk assessment: medium risk (score: 0.500). "
                f"Unable to complete full risk assessment due to error: {error_msg}. "
                f"Defaulting to medium risk level for safety. "
                f"Recommendation: Manual review recommended due to incomplete assessment."
            ),
            "risk_scores_breakdown": {"default": 0.5}
        }
