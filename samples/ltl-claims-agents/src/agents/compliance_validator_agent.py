"""
Compliance Validator Sub-Agent for LTL Claims Processing
Specialized agent for policy validation and compliance checking operations.
"""

import logging
import json
from typing import Dict, Any, List, Optional

from uipath_langchain.chat.models import UiPathChat

from ..services.uipath_service import UiPathService
from .config import ComplianceValidatorConfig
from .exceptions import OrchestratorError


logger = logging.getLogger(__name__)


class ComplianceValidationError(OrchestratorError):
    """
    Raised when compliance validation fails.
    
    Additional Attributes:
        violations: Policy violations identified before failure
    """
    
    def __init__(
        self,
        message: str,
        claim_id: Optional[str] = None,
        violations: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, claim_id=claim_id, step_name="validate_policy", **kwargs)
        self.violations = violations or []
    
    def is_critical(self) -> bool:
        """Compliance validation errors are not critical - can proceed with manual review flag."""
        return False


class ComplianceValidatorAgent:
    """
    Specialized agent for policy compliance validation operations.
    
    Responsibilities:
    - Search claims knowledge base for relevant policies
    - Search carrier information for liability rules
    - Validate claim against policy limits and conditions
    - Detect policy violations
    - Provide compliance recommendations
    
    Implements Requirements 6.1, 6.2, 6.3, 6.4, 11.1
    """
    
    def __init__(self, uipath_service: UiPathService, config: Optional[ComplianceValidatorConfig] = None):
        """
        Initialize the compliance validator agent.
        
        Args:
            uipath_service: Authenticated UiPath service instance
            config: Optional configuration object (uses defaults if not provided)
        """
        self.uipath_service = uipath_service
        self.config = config or ComplianceValidatorConfig()
        
        # Use UiPath Chat model (gpt-4o-mini for efficiency in policy analysis)
        self.llm = UiPathChat(
            model="gpt-4o-mini-2024-07-18",
            temperature=0,
            max_tokens=2000,
            timeout=30,
            max_retries=2
        )
        
        logger.info("[COMPLIANCE_VALIDATOR] Initialized compliance validator agent")
    
    @staticmethod
    def _extract_claim_id(state: Dict[str, Any]) -> str:
        """Extract claim ID from state, handling both field name formats."""
        return state.get('claim_id') or state.get('ObjectClaimId', 'UNKNOWN')
    
    async def validate_policy(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate claim against policies and carrier liability rules.
        
        Main entry point that coordinates the complete compliance validation:
        1. Search for relevant policies based on claim type
        2. Search for carrier liability information
        3. Validate claim amount against policy limits
        4. Validate carrier liability
        5. Check for policy violations
        
        Args:
            state: Current GraphState containing claim data
            
        Returns:
            Dictionary with:
                - policy_compliant: Boolean indicating if claim is compliant
                - policy_violations: List of identified violations
                - policy_data: Retrieved policy information
                - carrier_data: Retrieved carrier information
                - compliance_reasoning: Explanation of compliance assessment
                
        Implements Requirements 6.1, 6.2, 6.3, 6.4
        """
        from langgraph.prebuilt import create_react_agent
        from langchain_core.messages import HumanMessage
        
        claim_id = self._extract_claim_id(state)
        
        logger.info(f"[COMPLIANCE_VALIDATOR] Starting policy validation for claim: {claim_id}")
        
        try:
            # Get compliance validation tools
            from ..tools.context_grounding_tool import search_claims_knowledge, search_claim_procedures, search_carrier_information
            
            # Build prompt
            system_prompt = (
                "As a compliance validation specialist, your task is to verify that claims meet all policy requirements and carrier liability rules. "
                "Search for relevant policies and carrier information using the available tools. "
                "Identify any violations or non-compliance issues. Be thorough and precise in your analysis."
            )
            
            # Build validation instructions
            claim_type = state.get('claim_type', 'unknown')
            claim_amount = state.get('claim_amount', 0)
            carrier = state.get('carrier', 'Unknown')
            
            validation_instructions = (
                f"Validate claim {claim_id} for compliance:\n"
                f"- Claim Type: {claim_type}\n"
                f"- Claim Amount: ${claim_amount:,.2f}\n"
                f"- Carrier: {carrier}\n\n"
                f"Search for relevant policies and carrier liability information. "
                f"Check if the claim amount is within policy limits and if the carrier is liable for this type of claim."
            )
            
            # Debug logging
            logger.debug(f"[COMPLIANCE_VALIDATOR] System prompt: {system_prompt}")
            logger.debug(f"[COMPLIANCE_VALIDATOR] Validation instructions: {validation_instructions}")
            logger.debug(f"[COMPLIANCE_VALIDATOR] Available tools: search_claims_knowledge, search_claim_procedures, search_carrier_information")
            
            # Create react agent for compliance validation
            compliance_agent = create_react_agent(
                self.llm,
                tools=[search_claims_knowledge, search_claim_procedures, search_carrier_information],
                prompt=system_prompt
            )
            
            # Invoke agent
            logger.debug(f"[COMPLIANCE_VALIDATOR] Invoking compliance validation agent for claim {claim_id}")
            result = await compliance_agent.ainvoke({
                "messages": [HumanMessage(content=validation_instructions)]
            })
            
            # Debug: Log all messages in the result
            logger.debug(f"[COMPLIANCE_VALIDATOR] Agent returned {len(result['messages'])} messages")
            for i, msg in enumerate(result["messages"]):
                msg_type = type(msg).__name__
                logger.debug(f"[COMPLIANCE_VALIDATOR] Message {i} ({msg_type}): {str(msg.content)[:150]}...")
                # Log tool calls if present
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        logger.debug(f"[COMPLIANCE_VALIDATOR] Tool called: {tool_call.get('name', 'unknown')}")
            
            # Step 1: Search for relevant policies (Requirement 6.1)
            policy_data = await self._search_policies(state)
            
            # Step 2: Search for carrier information (Requirement 6.2)
            carrier_data = await self._search_carrier_info(state)
            
            # Step 3: Validate claim against policies (Requirement 6.3)
            violations = self._check_policy_violations(state, policy_data, carrier_data)
            
            # Step 4: Determine compliance status (Requirement 6.4)
            policy_compliant = len(violations) == 0
            
            # Step 5: Extract compliance reasoning from agent response
            compliance_reasoning = result["messages"][-1].content
            
            logger.info(
                f"[COMPLIANCE_VALIDATOR] Validation complete for claim {claim_id}: "
                f"compliant={policy_compliant}, violations={len(violations)}"
            )
            
            return {
                "policy_compliant": policy_compliant,
                "policy_violations": violations,
                "policy_data": policy_data,
                "carrier_data": carrier_data,
                "compliance_reasoning": compliance_reasoning
            }
            
        except Exception as e:
            logger.error(f"[COMPLIANCE_VALIDATOR] Policy validation failed for claim {claim_id}: {e}")
            
            # Return default result with manual review flag (Requirement 6.1)
            return self._get_default_validation_result(claim_id, str(e))
    
    async def _search_policies(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search claims knowledge base for relevant policies.
        
        Uses context grounding to find policies related to:
        - Claim type (damage, loss, shortage, etc.)
        - Claim amount limits
        - Documentation requirements
        - Processing procedures
        
        Args:
            state: Current GraphState with claim information
            
        Returns:
            Dictionary with policy information
            
        Implements Requirement 6.1
        """
        claim_id = self._extract_claim_id(state)
        claim_type = (state.get('claim_type') or state.get('ClaimType', 'unknown')).lower()
        claim_amount = state.get('claim_amount') or state.get('ClaimAmount', 0)
        
        logger.info(
            f"[COMPLIANCE_VALIDATOR] Searching policies for claim {claim_id}: "
            f"type={claim_type}, amount=${claim_amount:,.2f}"
        )
        
        try:
            # Import context grounding tools
            from ..tools.context_grounding_tool import search_claims_knowledge, search_claim_procedures
            
            # Search for general policies
            policy_query = f"policy limits requirements for {claim_type} claims amount ${claim_amount}"
            policy_results = await search_claims_knowledge.ainvoke({"query": policy_query})
            
            # Search for specific procedures
            procedure_results = await search_claim_procedures.ainvoke({"claim_type": claim_type})
            
            # Parse and structure results
            policy_data = {
                "claim_type": claim_type,
                "policy_results": policy_results,
                "procedure_results": procedure_results,
                "max_claim_amount": self._extract_max_claim_amount(policy_results),
                "required_documents": self._extract_required_documents(procedure_results),
                "search_successful": True
            }
            
            logger.info(
                f"[COMPLIANCE_VALIDATOR] Policy search complete for claim {claim_id}: "
                f"max_amount={policy_data.get('max_claim_amount', 'N/A')}"
            )
            
            return policy_data
            
        except Exception as e:
            logger.error(f"[COMPLIANCE_VALIDATOR] Policy search failed for claim {claim_id}: {e}")
            return {
                "claim_type": claim_type,
                "policy_results": f"Error searching policies: {str(e)}",
                "procedure_results": "",
                "max_claim_amount": None,
                "required_documents": [],
                "search_successful": False,
                "error": str(e)
            }
    
    async def _search_carrier_info(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for carrier liability information.
        
        Uses context grounding to find carrier-specific information:
        - Liability limits
        - Coverage policies
        - Historical claim data
        - Contact information
        
        Args:
            state: Current GraphState with carrier information
            
        Returns:
            Dictionary with carrier information
            
        Implements Requirement 6.2
        """
        claim_id = self._extract_claim_id(state)
        carrier = state.get('carrier', 'Unknown')
        
        if not carrier or carrier == 'Unknown':
            logger.warning(f"[COMPLIANCE_VALIDATOR] No carrier specified for claim {claim_id}")
            return {
                "carrier": carrier,
                "carrier_results": "No carrier information available",
                "liable": None,
                "liability_limit": None,
                "search_successful": False
            }
        
        logger.info(f"[COMPLIANCE_VALIDATOR] Searching carrier info for {carrier}")
        
        try:
            # Import context grounding tool
            from ..tools.context_grounding_tool import search_carrier_information
            
            # Search for carrier information
            carrier_results = await search_carrier_information.ainvoke({"carrier_name": carrier})
            
            # Parse carrier data
            carrier_data = {
                "carrier": carrier,
                "carrier_results": carrier_results,
                "liable": self._extract_carrier_liability(carrier_results),
                "liability_limit": self._extract_liability_limit(carrier_results),
                "search_successful": True
            }
            
            logger.info(
                f"[COMPLIANCE_VALIDATOR] Carrier search complete for {carrier}: "
                f"liable={carrier_data.get('liable', 'N/A')}, "
                f"limit={carrier_data.get('liability_limit', 'N/A')}"
            )
            
            return carrier_data
            
        except Exception as e:
            logger.error(f"[COMPLIANCE_VALIDATOR] Carrier search failed for {carrier}: {e}")
            return {
                "carrier": carrier,
                "carrier_results": f"Error searching carrier info: {str(e)}",
                "liable": None,
                "liability_limit": None,
                "search_successful": False,
                "error": str(e)
            }
    
    def _check_policy_violations(
        self,
        state: Dict[str, Any],
        policy_data: Dict[str, Any],
        carrier_data: Dict[str, Any]
    ) -> List[str]:
        """
        Check for policy violations based on claim data and retrieved policies.
        
        Checks performed:
        - Claim amount vs policy limits
        - Carrier liability
        - Required documentation
        - Claim type restrictions
        
        Args:
            state: Current GraphState
            policy_data: Retrieved policy information
            carrier_data: Retrieved carrier information
            
        Returns:
            List of violation descriptions
            
        Implements Requirement 6.3
        """
        claim_id = self._extract_claim_id(state)
        violations = []
        
        logger.info(f"[COMPLIANCE_VALIDATOR] Checking policy violations for claim {claim_id}")
        
        # Check 1: Claim amount vs policy limit
        claim_amount = state.get('claim_amount') or state.get('ClaimAmount', 0)
        max_claim_amount = policy_data.get('max_claim_amount')
        
        if max_claim_amount and claim_amount > max_claim_amount:
            violation = (
                f"Claim amount ${claim_amount:,.2f} exceeds policy limit of ${max_claim_amount:,.2f}"
            )
            violations.append(violation)
            logger.warning(f"[COMPLIANCE_VALIDATOR] Violation detected: {violation}")
        
        # Check 2: Carrier liability
        carrier_liable = carrier_data.get('liable')
        carrier = carrier_data.get('carrier', 'Unknown')
        claim_type = (state.get('claim_type') or state.get('ClaimType', 'unknown')).lower()
        
        if carrier_liable is False:
            violation = f"Carrier {carrier} is not liable for {claim_type} claims"
            violations.append(violation)
            logger.warning(f"[COMPLIANCE_VALIDATOR] Violation detected: {violation}")
        
        # Check 3: Claim amount vs carrier liability limit
        liability_limit = carrier_data.get('liability_limit')
        if liability_limit and claim_amount > liability_limit:
            violation = (
                f"Claim amount ${claim_amount:,.2f} exceeds carrier liability limit of ${liability_limit:,.2f}"
            )
            violations.append(violation)
            logger.warning(f"[COMPLIANCE_VALIDATOR] Violation detected: {violation}")
        
        # Check 4: Required documentation
        required_docs = policy_data.get('required_documents', [])
        downloaded_docs = state.get('downloaded_documents', [])
        
        if required_docs and len(downloaded_docs) < len(required_docs):
            violation = (
                f"Missing required documentation: {len(required_docs)} documents required, "
                f"only {len(downloaded_docs)} available"
            )
            violations.append(violation)
            logger.warning(f"[COMPLIANCE_VALIDATOR] Violation detected: {violation}")
        
        # Check 5: Policy violations from state (if already identified)
        existing_violations = state.get('policy_violations', [])
        if existing_violations:
            violations.extend(existing_violations)
        
        if violations:
            logger.info(
                f"[COMPLIANCE_VALIDATOR] Found {len(violations)} policy violations for claim {claim_id}"
            )
        else:
            logger.info(f"[COMPLIANCE_VALIDATOR] No policy violations found for claim {claim_id}")
        
        return violations
    
    def _generate_compliance_reasoning(
        self,
        policy_compliant: bool,
        violations: List[str],
        policy_data: Dict[str, Any],
        carrier_data: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable explanation of compliance assessment.
        
        Args:
            policy_compliant: Whether claim is compliant
            violations: List of violations
            policy_data: Policy information
            carrier_data: Carrier information
            
        Returns:
            Compliance reasoning explanation
        """
        if policy_compliant:
            return (
                f"Compliance assessment: COMPLIANT. "
                f"Claim meets all policy requirements and carrier liability conditions. "
                f"No violations detected. Processing can proceed normally."
            )
        
        reasoning_parts = [
            f"Compliance assessment: NON-COMPLIANT. "
            f"Identified {len(violations)} policy violation(s):"
        ]
        
        # Add each violation
        for i, violation in enumerate(violations, 1):
            reasoning_parts.append(f"  {i}. {violation}")
        
        # Add recommendation
        reasoning_parts.append(
            "\nRecommendation: Manual review required due to policy violations. "
            "Claim should be escalated for supervisor approval."
        )
        
        return "\n".join(reasoning_parts)
    
    def _extract_max_claim_amount(self, policy_results: str) -> Optional[float]:
        """
        Extract maximum claim amount from policy search results.
        
        Args:
            policy_results: Policy search results text
            
        Returns:
            Maximum claim amount or None if not found
        """
        # Simple extraction - look for common patterns
        # In production, this would use more sophisticated NLP
        import re
        
        # Look for patterns like "$10,000" or "$10000" or "10000"
        patterns = [
            r'\$?([\d,]+(?:\.\d{2})?)\s*(?:limit|maximum|max)',
            r'(?:limit|maximum|max)\s*(?:of|:)?\s*\$?([\d,]+(?:\.\d{2})?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, policy_results, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    return float(amount_str)
                except ValueError:
                    continue
        
        # Default limit if not found
        return self.config.default_max_claim_amount
    
    def _extract_required_documents(self, procedure_results: str) -> List[str]:
        """
        Extract required documents from procedure search results.
        
        Args:
            procedure_results: Procedure search results text
            
        Returns:
            List of required document types
        """
        # Simple extraction - look for document mentions
        # In production, this would use more sophisticated NLP
        required_docs = []
        
        doc_keywords = [
            'bill of lading', 'bol', 'invoice', 'proof of delivery', 'pod',
            'damage report', 'photos', 'inspection report', 'freight bill'
        ]
        
        procedure_lower = procedure_results.lower()
        for keyword in doc_keywords:
            if keyword in procedure_lower:
                required_docs.append(keyword)
        
        return required_docs
    
    def _extract_carrier_liability(self, carrier_results: str) -> Optional[bool]:
        """
        Extract carrier liability status from search results.
        
        Args:
            carrier_results: Carrier search results text
            
        Returns:
            True if liable, False if not liable, None if unknown
        """
        # Simple extraction - look for liability indicators
        carrier_lower = carrier_results.lower()
        
        # Negative indicators
        if any(phrase in carrier_lower for phrase in ['not liable', 'no liability', 'not responsible']):
            return False
        
        # Positive indicators
        if any(phrase in carrier_lower for phrase in ['liable', 'responsible', 'coverage']):
            return True
        
        # Unknown
        return None
    
    def _extract_liability_limit(self, carrier_results: str) -> Optional[float]:
        """
        Extract carrier liability limit from search results.
        
        Args:
            carrier_results: Carrier search results text
            
        Returns:
            Liability limit amount or None if not found
        """
        # Similar to max claim amount extraction
        import re
        
        patterns = [
            r'liability\s*limit\s*(?:of|:)?\s*\$?([\d,]+(?:\.\d{2})?)',
            r'\$?([\d,]+(?:\.\d{2})?)\s*liability\s*limit',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, carrier_results, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    return float(amount_str)
                except ValueError:
                    continue
        
        return None
    
    def _get_default_validation_result(self, claim_id: str, error_msg: str) -> Dict[str, Any]:
        """
        Get default validation result when validation fails.
        
        Args:
            claim_id: Claim identifier
            error_msg: Error message
            
        Returns:
            Default validation result with manual review flag
            
        Implements Requirement 6.1 (graceful degradation)
        """
        logger.warning(
            f"[COMPLIANCE_VALIDATOR] Using default validation result for claim {claim_id} "
            f"due to validation failure: {error_msg}"
        )
        
        return {
            "policy_compliant": None,
            "policy_violations": [
                "Policy validation failed - manual review required"
            ],
            "policy_data": {
                "search_successful": False,
                "error": error_msg
            },
            "carrier_data": {
                "search_successful": False,
                "error": error_msg
            },
            "compliance_reasoning": (
                f"Compliance assessment: UNKNOWN. "
                f"Unable to complete policy validation due to error: {error_msg}. "
                f"Recommendation: Manual review required to verify compliance."
            )
        }
