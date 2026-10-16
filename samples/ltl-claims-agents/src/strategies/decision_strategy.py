"""Decision-making strategies for claim processing."""

import json
import re
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

from ..config.constants import DecisionConstants, ThresholdConstants, RiskLevelConstants

logger = logging.getLogger(__name__)


class DecisionStrategy(ABC):
    """Abstract base class for decision-making strategies."""
    
    @abstractmethod
    async def make_decision(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a decision based on claim state.
        
        Args:
            state_data: Dictionary containing claim state information
            
        Returns:
            Dictionary with keys: decision, confidence, reasoning
        """
        pass


class LLMDecisionStrategy(DecisionStrategy):
    """LLM-based decision strategy using language model reasoning."""
    
    def __init__(self, llm):
        """
        Initialize LLM decision strategy.
        
        Args:
            llm: Language model instance for decision making
        """
        self.llm = llm
    
    async def make_decision(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make decision using LLM reasoning.
        
        Args:
            state_data: Dictionary containing claim state information
            
        Returns:
            Dictionary with decision, confidence, and reasoning
        """
        try:
            # Build prompt
            messages = self._build_prompt(state_data)
            
            # Get LLM response
            response = await self.llm.ainvoke(messages)
            
            # Parse response
            decision_data = self._parse_response(response.content)
            
            logger.info(
                f"LLM decision for claim {state_data.get('claim_id')}: "
                f"{decision_data['decision']} (confidence: {decision_data['confidence']:.2%})"
            )
            
            return decision_data
            
        except Exception as e:
            logger.error(f"LLM decision failed: {e}", exc_info=True)
            # Fallback to pending with low confidence
            return {
                "decision": DecisionConstants.PENDING,
                "confidence": 0.3,
                "reasoning": f"Unable to make automated decision due to error: {str(e)}. Manual review required."
            }
    
    def _build_prompt(self, state_data: Dict[str, Any]) -> List[BaseMessage]:
        """
        Build LLM prompt for decision making.
        
        Args:
            state_data: Claim state data
            
        Returns:
            List of messages for LLM
        """
        system_prompt = """You are an expert claims adjudicator for LTL freight claims. 
Analyze the claim information and make a final decision.

Consider:
- Claim amount and type
- Risk assessment results (score and level)
- Policy compliance status
- Document extraction confidence
- Any human review decisions
- Historical patterns and precedents from similar claims

When historical context is available, use it to inform your decision:
- Look at how similar claims were decided in the past
- Consider the confidence levels and outcomes of similar claims
- Use decision patterns to understand typical outcomes for this claim type
- Be consistent with historical precedents unless there's a good reason to deviate

Provide:
1. Decision: approved, denied, or pending
2. Confidence: 0.0 to 1.0 (be conservative - use pending if uncertain)
3. Reasoning: Clear, professional explanation of your decision

Format your response as JSON:
{
  "decision": "approved|denied|pending",
  "confidence": 0.85,
  "reasoning": "explanation here"
}"""
        
        # Extract relevant data with safe defaults
        claim_id = state_data.get('claim_id', 'UNKNOWN')
        claim_type = state_data.get('claim_type', 'unknown')
        claim_amount = state_data.get('claim_amount', 0.0)
        risk_level = state_data.get('risk_level', 'unknown')
        risk_score = state_data.get('risk_score', 0.5)
        risk_factors = state_data.get('risk_factors', [])
        policy_compliant = state_data.get('policy_compliant')
        policy_violations = state_data.get('policy_violations', [])
        data_fabric_validated = state_data.get('data_fabric_validated', False)
        downloaded_documents = state_data.get('downloaded_documents', [])
        extraction_confidence = state_data.get('extraction_confidence', {})
        human_decision = state_data.get('human_decision')
        historical_context = state_data.get('historical_context', [])
        decision_patterns = state_data.get('decision_patterns')
        
        # Calculate average extraction confidence
        avg_extraction_confidence = "N/A"
        if extraction_confidence:
            avg_conf = sum(extraction_confidence.values()) / len(extraction_confidence)
            avg_extraction_confidence = f"{avg_conf:.2%}"
        
        # Build user prompt with current claim information
        user_prompt = f"""Claim Information:
- Claim ID: {claim_id}
- Claim Type: {claim_type}
- Claim Amount: ${claim_amount:,.2f}
- Risk Level: {risk_level} (score: {risk_score:.3f})
- Risk Factors: {', '.join(risk_factors) if risk_factors else 'None identified'}
- Policy Compliant: {policy_compliant if policy_compliant is not None else 'Not evaluated'}
- Policy Violations: {', '.join(policy_violations) if policy_violations else 'None'}
- Data Fabric Validated: {data_fabric_validated}
- Documents Processed: {len(downloaded_documents)}
- Average Extraction Confidence: {avg_extraction_confidence}
- Human Review Decision: {human_decision if human_decision else 'Not required'}
"""
        
        # Add historical context if available
        if historical_context and len(historical_context) > 0:
            user_prompt += f"\nHistorical Context - Similar Claims ({len(historical_context)} found):\n"
            for i, claim in enumerate(historical_context[:3], 1):  # Show top 3
                user_prompt += f"""
{i}. Claim {claim['claim_id']} (Similarity: {claim['similarity_score']:.1%})
   - Type: {claim['claim_type']}, Amount: ${claim['claim_amount']:,.2f}
   - Carrier: {claim['carrier']}
   - Decision: {claim['decision']} (Confidence: {claim['confidence']:.1%})
   - Outcome: {claim['outcome']}
"""
        
        # Add decision patterns if available
        if decision_patterns and decision_patterns.get('total_claims', 0) > 0:
            user_prompt += f"\nDecision Patterns for {claim_type} Claims (Last 90 days):\n"
            user_prompt += f"- Total Claims: {decision_patterns['total_claims']}\n"
            user_prompt += f"- Most Common Decision: {decision_patterns.get('most_common_decision', 'N/A')}\n"
            user_prompt += f"- Average Confidence: {decision_patterns.get('average_confidence', 0):.1%}\n"
            user_prompt += f"- Average Claim Amount: ${decision_patterns.get('average_claim_amount', 0):,.2f}\n"
            
            if 'decision_distribution' in decision_patterns:
                user_prompt += "- Decision Distribution:\n"
                for decision, percentage in decision_patterns['decision_distribution'].items():
                    user_prompt += f"  * {decision}: {percentage:.1f}%\n"
        
        user_prompt += "\nBased on this information and historical precedents, make your decision:"
        
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
    
    def _parse_response(self, response_content: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured decision data.
        
        Args:
            response_content: Raw LLM response text
            
        Returns:
            Dictionary with decision, confidence, and reasoning
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response_content, re.DOTALL)
            
            if json_match:
                decision_data = json.loads(json_match.group())
                
                # Validate and normalize decision
                decision = decision_data.get("decision", DecisionConstants.PENDING).lower()
                if decision not in DecisionConstants.VALID_DECISIONS:
                    logger.warning(f"Invalid decision '{decision}', defaulting to pending")
                    decision = DecisionConstants.PENDING
                
                # Validate confidence
                confidence = float(decision_data.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                
                # Get reasoning
                reasoning = decision_data.get("reasoning", "Decision made based on available information")
                
                return {
                    "decision": decision,
                    "confidence": confidence,
                    "reasoning": reasoning
                }
            else:
                # No JSON found, use entire response as reasoning
                logger.warning("No JSON found in LLM response, using fallback parsing")
                return {
                    "decision": DecisionConstants.PENDING,
                    "confidence": 0.5,
                    "reasoning": response_content[:500]  # Limit length
                }
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {
                "decision": DecisionConstants.PENDING,
                "confidence": 0.3,
                "reasoning": f"Failed to parse decision response: {str(e)}"
            }


class RuleBasedDecisionStrategy(DecisionStrategy):
    """Rule-based decision strategy as fallback when LLM is unavailable."""
    
    async def make_decision(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make decision using predefined rules.
        
        Args:
            state_data: Dictionary containing claim state information
            
        Returns:
            Dictionary with decision, confidence, and reasoning
        """
        claim_id = state_data.get('claim_id', 'UNKNOWN')
        claim_amount = state_data.get('claim_amount', 0.0)
        risk_level = state_data.get('risk_level', RiskLevelConstants.MEDIUM)
        policy_compliant = state_data.get('policy_compliant')
        policy_violations = state_data.get('policy_violations', [])
        
        reasoning_parts = []
        
        # Rule 1: Policy violations = deny
        if policy_violations and len(policy_violations) > 0:
            return {
                "decision": DecisionConstants.DENIED,
                "confidence": 0.9,
                "reasoning": f"Claim denied due to policy violations: {', '.join(policy_violations)}"
            }
        
        # Rule 2: Not policy compliant = pending
        if policy_compliant is False:
            return {
                "decision": DecisionConstants.PENDING,
                "confidence": 0.7,
                "reasoning": "Claim requires manual review due to policy compliance concerns"
            }
        
        # Rule 3: High risk = pending
        if risk_level == RiskLevelConstants.HIGH:
            return {
                "decision": DecisionConstants.PENDING,
                "confidence": 0.8,
                "reasoning": "Claim flagged for manual review due to high risk level"
            }
        
        # Rule 4: Low risk + policy compliant + reasonable amount = approve
        if (risk_level == RiskLevelConstants.LOW and 
            policy_compliant is True and 
            claim_amount <= 5000.0):
            return {
                "decision": DecisionConstants.APPROVED,
                "confidence": 0.85,
                "reasoning": "Claim approved: low risk, policy compliant, and within auto-approval threshold"
            }
        
        # Rule 5: Medium risk + policy compliant + small amount = approve
        if (risk_level == RiskLevelConstants.MEDIUM and 
            policy_compliant is True and 
            claim_amount <= 2000.0):
            return {
                "decision": DecisionConstants.APPROVED,
                "confidence": 0.75,
                "reasoning": "Claim approved: medium risk but small amount and policy compliant"
            }
        
        # Default: pending for manual review
        return {
            "decision": DecisionConstants.PENDING,
            "confidence": 0.6,
            "reasoning": "Claim requires manual review - does not meet auto-approval criteria"
        }


class HybridDecisionStrategy(DecisionStrategy):
    """Hybrid strategy that uses LLM with rule-based fallback."""
    
    def __init__(self, llm):
        """
        Initialize hybrid decision strategy.
        
        Args:
            llm: Language model instance for decision making
        """
        self.llm_strategy = LLMDecisionStrategy(llm)
        self.rule_strategy = RuleBasedDecisionStrategy()
    
    async def make_decision(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make decision using LLM, falling back to rules if needed.
        
        Args:
            state_data: Dictionary containing claim state information
            
        Returns:
            Dictionary with decision, confidence, and reasoning
        """
        try:
            # Try LLM first
            decision_data = await self.llm_strategy.make_decision(state_data)
            
            # If LLM has low confidence, validate with rules
            if decision_data["confidence"] < ThresholdConstants.CONFIDENCE_THRESHOLD:
                logger.info("LLM confidence low, validating with rule-based strategy")
                rule_decision = await self.rule_strategy.make_decision(state_data)
                
                # If rules agree, boost confidence slightly
                if rule_decision["decision"] == decision_data["decision"]:
                    decision_data["confidence"] = min(0.85, decision_data["confidence"] + 0.15)
                    decision_data["reasoning"] += " (Validated by rule-based system)"
            
            return decision_data
            
        except Exception as e:
            logger.error(f"Hybrid decision failed, falling back to rules: {e}")
            return await self.rule_strategy.make_decision(state_data)
