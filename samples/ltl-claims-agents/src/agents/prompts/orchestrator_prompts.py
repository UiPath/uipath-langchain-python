"""
Prompt builders for orchestrator agent operations.
"""

from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class ClaimContext:
    """Structured claim context for prompt building."""
    claim_id: str
    claim_type: str
    claim_amount: float
    carrier: str
    shipment_id: str
    has_shipping_docs: bool
    has_damage_evidence: bool
    validation_errors: List[str]
    data_fabric_validated: bool
    completed_steps: List[str]
    
    def format_for_prompt(self) -> str:
        """Format claim context for inclusion in prompts."""
        return f"""Claim ID: {self.claim_id}
Claim Type: {self.claim_type}
Claim Amount: ${self.claim_amount:,.2f}
Carrier: {self.carrier}
Shipment ID: {self.shipment_id or 'Not provided'}

Documents Available:
- Shipping Documents: {'Yes' if self.has_shipping_docs else 'No'}
- Damage Evidence: {'Yes' if self.has_damage_evidence else 'No'}

Current Status:
- Data Fabric Validated: {self.data_fabric_validated}
- Validation Errors: {len(self.validation_errors)} errors
- Completed Steps: {', '.join(self.completed_steps) or 'None'}"""


class ClaimContextExtractor:
    """Extract claim context from state dictionary."""
    
    @staticmethod
    def extract(state: Dict[str, Any]) -> ClaimContext:
        """
        Extract structured claim context from state.
        
        Args:
            state: Current GraphState dictionary
            
        Returns:
            ClaimContext with extracted information
        """
        return ClaimContext(
            claim_id=state.get('claim_id') or state.get('ObjectClaimId', 'UNKNOWN'),
            claim_type=state.get('claim_type') or state.get('ClaimType', 'unknown'),
            claim_amount=state.get('claim_amount') or state.get('ClaimAmount', 0),
            carrier=state.get('carrier', 'unknown'),
            shipment_id=state.get('shipment_id'),
            has_shipping_docs=bool(state.get('shipping_documents')),
            has_damage_evidence=bool(state.get('damage_evidence')),
            validation_errors=state.get('validation_errors', []),
            data_fabric_validated=state.get('data_fabric_validated', False),
            completed_steps=state.get('completed_steps', [])
        )


class OrchestratorPrompts:
    """Centralized prompt builder for orchestrator agent."""
    
    @staticmethod
    def build_plan_system_prompt() -> str:
        """
        Build the system prompt for plan generation.
        
        Returns:
            System prompt string
        """
        return """You are an expert claims processing orchestrator for LTL freight claims.

Your role is to create a step-by-step execution plan to process a freight claim efficiently and accurately.

Consider these processing stages:
1. Data Validation - Verify claim and shipment data in Data Fabric
2. Document Processing - Download and extract data from shipping documents and damage evidence
3. Risk Assessment - Analyze claim for risk factors and fraud indicators
4. Policy Validation - Check compliance with company policies and carrier liability
5. Decision Making - Make final approval/denial decision
6. System Updates - Update queue status and Data Fabric with results

Available capabilities:
- Query Data Fabric for claim and shipment validation
- Download documents from storage buckets
- Extract data using Document Understanding (IXP)
- Search knowledge base for policies and procedures
- Assess risk factors and calculate risk scores
- Create Action Center tasks for human review
- Update queue transactions and Data Fabric records

CRITICAL - Document Download Instructions:
When downloading documents, you MUST use the EXACT document metadata from the claim input.
DO NOT construct file paths yourself. The claim input contains fields like 'shipping_documents'
and 'damage_evidence' with complete metadata including the correct 'path' field.

Example: If claim has shipping_documents=[{"bucketId": 99943, "path": "/claims/xxx/documents/BOL.pdf", "fileName": "BOL.pdf"}],
pass this EXACT metadata to download_multiple_documents. DO NOT use "shipping_documents/BOL.pdf" as the path.

Create a concise, ordered plan with 5-8 steps that covers the essential processing stages.
Each step should be a clear, actionable task.

Return ONLY the plan steps, one per line, numbered. No additional explanation."""
    
    @staticmethod
    def build_plan_user_prompt(state: Dict[str, Any]) -> str:
        """
        Build the user prompt with claim context for plan generation.
        
        Args:
            state: Current GraphState
            
        Returns:
            User prompt string
        """
        claim_context = ClaimContextExtractor.extract(state)
        
        return f"""Create a processing plan for this freight claim:

{claim_context.format_for_prompt()}

Create an execution plan that:
1. Addresses any validation errors
2. Processes available documents if present
3. Performs risk assessment
4. Validates against policies
5. Makes a final decision
6. Updates all systems

Return the plan as numbered steps."""
    
    @staticmethod
    def build_replan_system_prompt() -> str:
        """
        Build the system prompt for replanning.
        
        Returns:
            System prompt string
        """
        return """You are an expert claims processing orchestrator adapting to processing challenges.

Your role is to create a revised execution plan that addresses identified issues and completes the claim processing.

Consider:
- What steps have already been completed successfully
- What errors or issues have been encountered
- What information is still needed
- How to work around failures or missing data
- When to escalate to human review

CRITICAL - Document Download Instructions:
When downloading documents, use the EXACT document metadata from the claim input.
The claim input contains 'shipping_documents' and 'damage_evidence' arrays with complete
metadata including the 'path' field. Pass this metadata directly to download_multiple_documents.
DO NOT construct paths from field names (e.g., don't use "shipping_documents/file.pdf").

Create a revised plan that:
1. Skips already completed steps
2. Addresses identified issues
3. Works around failures when possible
4. Includes escalation if needed
5. Completes remaining processing stages

Return ONLY the revised plan steps, one per line, numbered. No additional explanation."""
    
    @staticmethod
    def build_replan_user_prompt(state: Dict[str, Any], reflection: Dict[str, Any]) -> str:
        """
        Build user prompt for replanning.
        
        Args:
            state: Current GraphState
            reflection: Reflection results
            
        Returns:
            User prompt string
        """
        claim_id = state.get('claim_id') or state.get('ObjectClaimId', 'UNKNOWN')
        completed_steps = state.get('completed_steps', [])
        errors = state.get('errors', [])
        
        # Format issues and recommendations
        issues_text = '\n'.join(f"- {issue}" for issue in reflection.get('issues_identified', []))
        if not issues_text:
            issues_text = '- None'
        
        recommendations_text = '\n'.join(f"- {rec}" for rec in reflection.get('recommendations', []))
        if not recommendations_text:
            recommendations_text = '- None'
        
        # Format recent errors
        recent_errors = errors[-3:] if errors else []
        errors_text = '\n'.join(f"- {error.get('error', 'Unknown error')}" for error in recent_errors)
        if not errors_text:
            errors_text = '- None'
        
        return f"""Create a revised plan for claim {claim_id}:

Current Progress:
- Completed Steps: {', '.join(completed_steps) if completed_steps else 'None'}
- Progress Assessment: {reflection.get('progress_assessment', 'unknown')}
- Confidence Level: {reflection.get('confidence_level', 0.5):.2f}

Issues Identified:
{issues_text}

Recommendations:
{recommendations_text}

Recent Errors:
{errors_text}

Current State:
- Data Fabric Validated: {state.get('data_fabric_validated', False)}
- Documents Downloaded: {len(state.get('downloaded_documents', []))} files
- Risk Level: {state.get('risk_level', 'unknown')}
- Policy Compliant: {state.get('policy_compliant', 'unknown')}

Create a revised plan that:
1. Skips already completed steps
2. Addresses the identified issues
3. Completes remaining necessary processing
4. Includes human escalation if confidence is too low

Return the revised plan as numbered steps."""
