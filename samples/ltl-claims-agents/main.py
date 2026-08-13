"""
LTL Claims Processing Agent - Main Entry Point

This module implements a production-grade React-style LangGraph agent for LTL Claims Processing.
The agent follows a plan-execute-observe-reflect pattern with multi-agent coordination.

Architecture:
- Main orchestrator coordinates specialized sub-agents
- Document processing, risk assessment, and compliance validation sub-agents
- Human-in-the-loop escalation via Action Center
- Comprehensive error handling and logging
- Integration with UiPath services (Data Fabric, IXP, Context Grounding, etc.)

Usage:
    # Run with file input
    uv run uipath run main.py --file input.json
    
    # Run with inline JSON
    uv run uipath run main.py '{"claim_id": "CLM-12345", "claim_type": "damage", "claim_amount": 1500.0}'
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, model_validator

from src.config.constants import (
    ThresholdConstants,
    DecisionConstants,
    RiskLevelConstants,
    PriorityConstants,
    ClaimTypeConstants,
    FieldMappingConstants,
    ValidationConstants
)

# Simple logging configuration for UiPath Orchestrator
# Check for debug flag from environment
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() in ("true", "1", "yes")
ENABLE_DEBUG_LOGGING = os.getenv("ENABLE_DEBUG_LOGGING", "false").lower() in ("true", "1", "yes")

# Configure basic logging
if DEBUG_MODE or ENABLE_DEBUG_LOGGING:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Suppress noisy libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpcore.http11').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ============================================================================
# STATE MODELS
# ============================================================================

class GraphState(BaseModel):
    """
    Comprehensive agent state throughout processing.
    
    This model tracks all data and metadata as the claim moves through
    the processing pipeline. Automatically normalizes UiPath queue format
    field names to standard format.
    
    The state is passed between all nodes in the LangGraph workflow and
    accumulates information at each step.
    """
    
    # ========================================================================
    # INPUT FIELDS - Normalized format
    # ========================================================================
    
    # Core claim information
    claim_id: Optional[str] = Field(
        default=None,
        description="Unique claim identifier"
    )
    claim_type: Optional[str] = Field(
        default=None,
        description="Type of claim: damage, loss, shortage, delay, other"
    )
    claim_amount: Optional[float] = Field(
        default=None,
        description="Claimed amount in USD",
        ge=ValidationConstants.MIN_CLAIM_AMOUNT,
        le=ValidationConstants.MAX_CLAIM_AMOUNT
    )
    
    # Shipment information
    shipment_id: Optional[str] = Field(
        default=None,
        description="Associated shipment identifier"
    )
    
    # Carrier information
    carrier: Optional[str] = Field(
        default=None,
        description="Carrier name"
    )
    
    # Customer information
    customer_name: Optional[str] = Field(
        default=None,
        description="Customer full name"
    )
    customer_email: Optional[str] = Field(
        default=None,
        description="Customer email address"
    )
    customer_phone: Optional[str] = Field(
        default=None,
        description="Customer phone number"
    )
    
    # Claim details
    description: Optional[str] = Field(
        default=None,
        description="Detailed claim description",
        max_length=ValidationConstants.MAX_DESCRIPTION_LENGTH
    )
    submission_source: Optional[str] = Field(
        default=None,
        description="Source of claim submission"
    )
    submitted_at: Optional[str] = Field(
        default=None,
        description="Submission timestamp (ISO format)"
    )
    
    # ========================================================================
    # DOCUMENT REFERENCES
    # ========================================================================
    
    shipping_documents: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of shipping document references with bucket/path info"
    )
    damage_evidence: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of damage evidence file references"
    )
    
    # ========================================================================
    # PROCESSING METADATA
    # ========================================================================
    
    transaction_key: Optional[str] = Field(
        default=None,
        description="UiPath queue transaction key for status updates"
    )
    queue_name: Optional[str] = Field(
        default=None,
        description="Source queue name for queue-based processing"
    )
    processing_priority: str = Field(
        default=PriorityConstants.NORMAL,
        description="Processing priority: Low, Normal, High, Critical"
    )
    
    # ========================================================================
    # FIELD NORMALIZATION
    # ========================================================================
    
    @model_validator(mode='before')
    @classmethod
    def normalize_queue_fields(cls, data: Any) -> Any:
        """
        Normalize UiPath queue format fields to standard format.
        
        This allows the agent to accept both standard field names and
        UiPath queue format field names, automatically converting to
        the standard format for internal processing.
        """
        if not isinstance(data, dict):
            return data
        
        # Create a copy to avoid modifying original
        normalized = dict(data)
        
        # Map UiPath queue format to standard format
        for queue_field, standard_field in FieldMappingConstants.QUEUE_TO_STANDARD.items():
            if queue_field in normalized and standard_field not in normalized:
                normalized[standard_field] = normalized[queue_field]
        
        return normalized
    
    # ========================================================================
    # AGENT STATE - Plan-Execute-Observe-Reflect
    # ========================================================================
    
    plan: Optional[List[str]] = Field(
        default=None,
        description="Ordered list of planned execution steps"
    )
    
    current_step: int = Field(
        default=0,
        description="Current step number in the plan"
    )
    
    completed_steps: List[str] = Field(
        default_factory=list,
        description="List of successfully completed step names"
    )
    
    observations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Observations from each step for reflection"
    )
    
    # ========================================================================
    # VALIDATION RESULTS
    # ========================================================================
    
    data_fabric_validated: bool = Field(
        default=False,
        description="Whether Data Fabric validation succeeded"
    )
    
    validation_errors: List[str] = Field(
        default_factory=list,
        description="List of validation error messages"
    )
    
    # ========================================================================
    # DOCUMENT PROCESSING RESULTS
    # ========================================================================
    
    downloaded_documents: List[str] = Field(
        default_factory=list,
        description="List of local file paths for downloaded documents"
    )
    
    extracted_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured data extracted from documents via IXP"
    )
    
    extraction_confidence: Dict[str, float] = Field(
        default_factory=dict,
        description="Confidence scores for each extracted field (0.0-1.0)"
    )
    
    # ========================================================================
    # RISK ASSESSMENT RESULTS
    # ========================================================================
    
    risk_score: Optional[float] = Field(
        default=None,
        description="Calculated risk score (0.0-1.0)"
    )
    
    risk_level: Optional[str] = Field(
        default=None,
        description="Risk categorization: low, medium, high"
    )
    
    risk_factors: List[str] = Field(
        default_factory=list,
        description="List of identified risk factors"
    )
    
    # ========================================================================
    # POLICY VALIDATION RESULTS
    # ========================================================================
    
    policy_compliant: Optional[bool] = Field(
        default=None,
        description="Whether claim complies with all policies"
    )
    
    policy_violations: List[str] = Field(
        default_factory=list,
        description="List of policy violations detected"
    )
    
    # ========================================================================
    # DECISION MAKING
    # ========================================================================
    
    decision: Optional[str] = Field(
        default=None,
        description="Final decision: approved, denied, pending"
    )
    
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence in the decision (0.0-1.0)"
    )
    
    reasoning: str = Field(
        default="",
        description="Human-readable explanation of the decision"
    )
    
    reasoning_steps: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed reasoning chain with thought process"
    )
    
    # ========================================================================
    # HUMAN REVIEW / ESCALATION
    # ========================================================================
    
    requires_human_review: bool = Field(
        default=False,
        description="Whether human review is required"
    )
    
    human_review_reason: Optional[str] = Field(
        default=None,
        description="Reason for human review escalation"
    )
    
    action_center_task_id: Optional[str] = Field(
        default=None,
        description="UiPath Action Center task ID if escalated"
    )
    
    human_decision: Optional[str] = Field(
        default=None,
        description="Decision provided by human reviewer"
    )
    
    # ========================================================================
    # MEMORY AND HISTORICAL CONTEXT
    # ========================================================================
    
    historical_context: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Historical context from similar past claims"
    )
    
    similar_claims_count: int = Field(
        default=0,
        description="Number of similar historical claims found"
    )
    
    decision_patterns: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Decision patterns for this claim type"
    )
    
    # ========================================================================
    # METADATA AND AUDIT TRAIL
    # ========================================================================
    
    tools_used: List[str] = Field(
        default_factory=list,
        description="List of tool names used during processing"
    )
    
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of errors encountered with context"
    )
    
    start_time: Optional[datetime] = Field(
        default=None,
        description="Processing start timestamp"
    )
    
    end_time: Optional[datetime] = Field(
        default=None,
        description="Processing end timestamp"
    )
    
    class Config:
        """Pydantic model configuration."""
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class GraphOutput(BaseModel):
    """
    Structured output from the agent after processing completion.
    
    This model represents the final result returned to the caller,
    whether that's a UiPath queue, file output, or API response.
    
    It includes the decision, reasoning, audit trail, and all relevant
    metadata for downstream systems and reporting.
    """
    
    # ========================================================================
    # CORE RESULTS
    # ========================================================================
    
    success: bool = Field(
        ...,
        description="Whether processing completed successfully"
    )
    
    claim_id: str = Field(
        ...,
        description="Unique claim identifier"
    )
    
    decision: str = Field(
        ...,
        description="Final decision: approved, denied, pending"
    )
    
    confidence: float = Field(
        ...,
        description="Confidence in the decision (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    
    reasoning: str = Field(
        ...,
        description="Human-readable explanation of the decision"
    )
    
    # ========================================================================
    # DETAILED REASONING AND AUDIT
    # ========================================================================
    
    reasoning_steps: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed thought process and reasoning chain"
    )
    
    tools_used: List[str] = Field(
        default_factory=list,
        description="List of tools invoked during processing"
    )
    
    # ========================================================================
    # HUMAN REVIEW
    # ========================================================================
    
    human_review_required: bool = Field(
        ...,
        description="Whether human review was required"
    )
    
    action_center_task_id: Optional[str] = Field(
        default=None,
        description="Action Center task ID if escalated"
    )
    
    # ========================================================================
    # PERFORMANCE METRICS
    # ========================================================================
    
    processing_duration_seconds: Optional[float] = Field(
        default=None,
        description="Total processing time in seconds"
    )
    
    timestamp: str = Field(
        ...,
        description="Completion timestamp (ISO format)"
    )
    
    # ========================================================================
    # ERROR HANDLING
    # ========================================================================
    
    error: Optional[str] = Field(
        default=None,
        description="Error message if processing failed"
    )
    
    # ========================================================================
    # ADDITIONAL DETAILS
    # ========================================================================
    
    risk_level: Optional[str] = Field(
        default=None,
        description="Risk assessment result: low, medium, high"
    )
    
    policy_compliant: Optional[bool] = Field(
        default=None,
        description="Whether claim complies with policies"
    )
    
    data_fabric_updated: bool = Field(
        default=False,
        description="Whether Data Fabric was updated with results"
    )
    
    queue_updated: bool = Field(
        default=False,
        description="Whether queue transaction was updated"
    )
    
    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


# ============================================================================
# IMPORTS FOR GRAPH IMPLEMENTATION
# ============================================================================

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from uipath_langchain.chat.models import UiPathChat

from src.services.uipath_service import UiPathService
from src.services.processing_history_service import ProcessingHistoryService
from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.document_processor_agent import DocumentProcessorAgent
from src.agents.risk_assessor_agent import RiskAssessorAgent
from src.agents.compliance_validator_agent import ComplianceValidatorAgent
from src.utils.validators import InputValidator
from src.config.settings import settings
from src.utils.node_decorators import node_wrapper, log_execution_time
from src.strategies.decision_strategy import HybridDecisionStrategy


# ============================================================================
# NODE FUNCTIONS
# ============================================================================


async def _record_processing_start(state: GraphState) -> None:
    """
    Record processing started event in history.
    
    This is a non-critical operation - failures are logged but don't stop processing.
    
    Args:
        state: Current graph state
    """
    try:
        async with UiPathService() as uipath_service:
            # Create history service with shared UiPath client
            history_service = ProcessingHistoryService(uipath_service._client)
            
            await history_service.record_processing_started(
                claim_id=state.claim_id,
                claim_data={
                    "claim_type": state.claim_type,
                    "claim_amount": state.claim_amount,
                    "carrier": state.carrier,
                    "customer_name": state.customer_name,
                    "submission_source": state.submission_source
                }
            )
            logger.debug(f"Recorded processing started for claim {state.claim_id}")
    except Exception as e:
        logger.warning(f"Failed to record processing history: {e}")
        state.errors.append({
            "step": "initialize_input",
            "error": f"History recording failed: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "critical": False
        })


@node_wrapper("initialize_input", mark_completed=False)
async def initialize_input_node(state: GraphState) -> GraphState:
    """
    Initialize processing and normalize input data.
    
    Sets start time, validates input fields, and loads historical context
    from long-term memory if enabled.
    
    Implements Requirements 1.1, 1.2, 1.4, 1.5, 15.1, 15.2
    """
    # Set start time
    state.start_time = datetime.now()
    
    # Record processing started in history (non-critical)
    await _record_processing_start(state)
    
    # Additional validation using validator
    raw_data = state.model_dump()
    normalized = InputValidator.validate_and_normalize(raw_data)
    
    # Update state with any additional normalized fields
    for key, value in normalized.items():
        if hasattr(state, key) and value is not None:
            setattr(state, key, value)
    
    # Load historical context if memory is enabled
    if settings.enable_long_term_memory:
        try:
            from src.memory.long_term_memory import ClaimMemoryStore
            
            logger.info(f"Loading historical context for claim: {state.claim_id}")
            
            # Initialize memory store
            memory_store = ClaimMemoryStore(
                connection_string=settings.memory_connection_string,
                store_type=settings.memory_store_type
            )
            
            # Retrieve similar claims if we have enough information
            if state.claim_type and state.claim_amount and state.carrier:
                similar_claims = await memory_store.retrieve_similar_claims(
                    claim_type=state.claim_type,
                    claim_amount=state.claim_amount,
                    carrier=state.carrier,
                    limit=5
                )
                
                if similar_claims:
                    state.historical_context = similar_claims
                    state.similar_claims_count = len(similar_claims)
                    
                    logger.info(
                        f"Loaded {len(similar_claims)} similar historical claims "
                        f"for context (avg similarity: "
                        f"{sum(c['similarity_score'] for c in similar_claims) / len(similar_claims):.2%})"
                    )
                    
                    # Add observation about historical context
                    state.observations.append({
                        "step": "initialize_input",
                        "observation": f"Found {len(similar_claims)} similar historical claims for context",
                        "timestamp": datetime.now().isoformat(),
                        "details": {
                            "similar_claims": [
                                {
                                    "claim_id": c["claim_id"],
                                    "decision": c["decision"],
                                    "confidence": c["confidence"],
                                    "similarity": c["similarity_score"]
                                }
                                for c in similar_claims
                            ]
                        }
                    })
                else:
                    logger.info("No similar historical claims found")
                
                # Get decision patterns for this claim type
                decision_patterns = await memory_store.get_decision_patterns(
                    claim_type=state.claim_type,
                    time_window_days=90
                )
                
                if decision_patterns.get("total_claims", 0) > 0:
                    state.decision_patterns = decision_patterns
                    logger.info(
                        f"Loaded decision patterns: {decision_patterns['total_claims']} "
                        f"claims in last 90 days, most common decision: "
                        f"{decision_patterns.get('most_common_decision', 'N/A')}"
                    )
            else:
                logger.warning(
                    "Insufficient claim information to retrieve historical context "
                    "(need claim_type, claim_amount, and carrier)"
                )
                
        except Exception as e:
            logger.warning(f"Failed to load historical context: {e}")
            logger.info("Continuing without historical context")
            # Don't fail the entire process if memory loading fails
            state.errors.append({
                "step": "initialize_input",
                "error": f"Memory loading failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "critical": False
            })
    else:
        logger.debug("Long-term memory disabled, skipping historical context loading")
    
    return state


@node_wrapper("create_plan")
async def create_plan_node(state: GraphState) -> GraphState:
    """
    Create execution plan using orchestrator agent.
    
    Uses async context manager for automatic resource cleanup.
    
    Implements Requirement 10.1
    """
    try:
        async with UiPathService() as uipath_service:
            # Create orchestrator agent
            orchestrator = OrchestratorAgent(uipath_service=uipath_service)
            
            # Generate plan
            plan = await orchestrator.create_plan(state.model_dump())
            state.plan = plan
            state.current_step = 0
        
    except Exception as e:
        # Use fallback plan on error
        logger.warning(f"Using fallback plan due to error: {e}")
        state.plan = [
            "Validate data in Data Fabric",
            "Download and process documents",
            "Assess risk",
            "Validate policy compliance",
            "Make decision",
            "Update systems"
        ]
        state.current_step = 0
        raise  # Re-raise to be caught by decorator
    
    return state


@node_wrapper("validate_data")
async def validate_data_node(state: GraphState) -> GraphState:
    """
    Validate claim and shipment data using Data Fabric.
    
    Queries Data Fabric to validate claim_id and shipment_id,
    enriches state with Data Fabric information.
    
    Implements Requirements 2.1, 2.2, 2.3, 2.4
    """
    async with UiPathService() as uipath_service:
        # Validate claim_id in Data Fabric
        claim_data = await uipath_service.get_claim_by_id(state.claim_id)
        
        if claim_data:
            state.data_fabric_validated = True
            # Enrich state with Data Fabric information
            if 'description' in claim_data and not state.description:
                state.description = claim_data.get('description')
        else:
            state.validation_errors.append(f"Claim {state.claim_id} not found in Data Fabric")
        
        # Validate shipment_id if provided
        if state.shipment_id:
            shipment_data = await uipath_service.get_shipment_data(state.shipment_id)
            if not shipment_data:
                state.validation_errors.append(f"Shipment {state.shipment_id} not found in Data Fabric")
        
        state.tools_used.append("query_data_fabric")
        
        # Record step completion in processing history
        try:
            history_service = ProcessingHistoryService(uipath_service._client)
            await history_service.record_step_completed(
                claim_id=state.claim_id,
                step_name="validate_data",
                step_data={
                    "data_fabric_validated": state.data_fabric_validated,
                    "validation_errors": state.validation_errors,
                    "shipment_validated": bool(state.shipment_id)
                }
            )
        except Exception as e:
            logger.warning(f"Failed to record step completion: {e}")
    
    return state


@node_wrapper("download_documents")
@log_execution_time
async def download_documents_node(state: GraphState) -> GraphState:
    """
    Download and extract data from documents.
    
    Uses DocumentProcessorAgent to download documents from storage
    and extract structured data using IXP.
    
    Implements Requirements 3.1, 3.2, 3.4, 3.5, 4.1, 4.2, 4.3
    """
    # Check if documents are referenced
    has_docs = bool(state.shipping_documents or state.damage_evidence)
    
    if not has_docs:
        logger.info("No documents to process")
        return state
    
    async with UiPathService() as uipath_service:
        # Create document processor agent
        doc_processor = DocumentProcessorAgent(uipath_service=uipath_service)
        
        # Process documents (download and extract)
        results = await doc_processor.process_documents(state.model_dump())
        
        # Store results in state
        state.downloaded_documents = results.get("downloaded", [])
        state.extracted_data = results.get("extracted", {})
        state.extraction_confidence = results.get("confidence", {})
        
        # Handle errors
        if results.get("errors"):
            for error in results["errors"]:
                # Convert error to string if it's a dictionary
                error_message = str(error) if isinstance(error, dict) else error
                state.errors.append({
                    "step": "download_documents",
                    "error": error_message,
                    "timestamp": datetime.now().isoformat()
                })
        
        state.tools_used.append("download_multiple_documents")
        
        # Record step completion in processing history
        try:
            history_service = ProcessingHistoryService(uipath_service._client)
            await history_service.record_step_completed(
                claim_id=state.claim_id,
                step_name="download_documents",
                step_data={
                    "documents_downloaded": len(state.downloaded_documents),
                    "extracted_fields": list(state.extracted_data.keys()),
                    "avg_confidence": sum(state.extraction_confidence.values()) / len(state.extraction_confidence) if state.extraction_confidence else 0.0,
                    "errors": len(results.get("errors", []))
                }
            )
        except Exception as e:
            logger.warning(f"Failed to record step completion: {e}")
    
    return state


@node_wrapper("assess_risk")
async def assess_risk_node(state: GraphState) -> GraphState:
    """
    Perform risk analysis on the claim.
    
    Uses RiskAssessorAgent to analyze risk factors,
    calculate risk score, and categorize risk level.
    
    Implements Requirements 5.1, 5.2, 5.3, 5.4
    """
    try:
        async with UiPathService() as uipath_service:
            # Create risk assessor agent
            risk_assessor = RiskAssessorAgent(uipath_service=uipath_service)
            
            # Perform risk assessment
            risk_results = await risk_assessor.assess_risk(state.model_dump())
            
            # Store results in state
            state.risk_score = risk_results.get("risk_score")
            state.risk_level = risk_results.get("risk_level")
            state.risk_factors = risk_results.get("risk_factors", [])
            
            # Add reasoning to observations
            if risk_results.get("risk_reasoning"):
                state.observations.append({
                    "step": "assess_risk",
                    "observation": risk_results["risk_reasoning"],
                    "timestamp": datetime.now().isoformat()
                })
            
            # Record step completion in processing history
            try:
                history_service = ProcessingHistoryService(uipath_service._client)
                await history_service.record_step_completed(
                    claim_id=state.claim_id,
                    step_name="assess_risk",
                    step_data={
                        "risk_score": state.risk_score,
                        "risk_level": state.risk_level,
                        "risk_factors": state.risk_factors
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to record step completion: {e}")
        
    except Exception as e:
        # Use default medium risk on error
        logger.warning(f"Using default risk assessment due to error: {e}")
        state.risk_score = ThresholdConstants.DEFAULT_RISK_SCORE
        state.risk_level = RiskLevelConstants.MEDIUM
        state.risk_factors = ["Risk assessment failed - defaulting to medium risk"]
        raise  # Re-raise to be caught by decorator
    
    return state


@node_wrapper("validate_policy")
async def validate_policy_node(state: GraphState) -> GraphState:
    """
    Validate claim against policies and carrier liability.
    
    Uses ComplianceValidatorAgent to check policy compliance,
    search for relevant policies, and validate carrier liability.
    
    Implements Requirements 6.1, 6.2, 6.3, 6.4
    """
    try:
        async with UiPathService() as uipath_service:
            # Create compliance validator agent
            compliance_validator = ComplianceValidatorAgent(uipath_service=uipath_service)
            
            # Perform policy validation
            compliance_results = await compliance_validator.validate_policy(state.model_dump())
            
            # Store results in state
            state.policy_compliant = compliance_results.get("policy_compliant")
            state.policy_violations = compliance_results.get("policy_violations", [])
            
            # Add reasoning to observations
            if compliance_results.get("compliance_reasoning"):
                state.observations.append({
                    "step": "validate_policy",
                    "observation": compliance_results["compliance_reasoning"],
                    "timestamp": datetime.now().isoformat()
                })
            
            state.tools_used.append("search_claims_knowledge")
            
            # Record step completion in processing history
            try:
                history_service = ProcessingHistoryService(uipath_service._client)
                await history_service.record_step_completed(
                    claim_id=state.claim_id,
                    step_name="validate_policy",
                    step_data={
                        "policy_compliant": state.policy_compliant,
                        "policy_violations": state.policy_violations
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to record step completion: {e}")
        
    except Exception as e:
        # Flag for manual review on error
        logger.warning(f"Policy validation failed, flagging for manual review: {e}")
        state.policy_compliant = None
        state.policy_violations = ["Policy validation failed - manual review required"]
        raise  # Re-raise to be caught by decorator
    
    return state


@node_wrapper("evaluate_progress", mark_completed=False)
async def evaluate_progress_node(state: GraphState) -> GraphState:
    """
    Evaluate progress and determine if human review is needed.
    
    Reflects on completed steps, checks confidence levels,
    risk levels, and policy violations to determine escalation.
    
    Implements Requirements 10.3, 7.1, 7.2, 7.3
    """
    # Check if confidence is below threshold
    if state.confidence and state.confidence < ThresholdConstants.CONFIDENCE_THRESHOLD:
        state.requires_human_review = True
        state.human_review_reason = f"Low confidence decision: {state.confidence:.2%}"
    
    # Check if risk level is high
    if state.risk_level == RiskLevelConstants.HIGH:
        state.requires_human_review = True
        state.human_review_reason = "High risk claim detected"
    
    # Check if policy violations exist
    if state.policy_violations and len(state.policy_violations) > 0:
        state.requires_human_review = True
        state.human_review_reason = f"Policy violations detected: {len(state.policy_violations)} violations"
    
    # Check for critical errors
    critical_errors = [e for e in state.errors if e.get('critical', False)]
    if critical_errors:
        state.requires_human_review = True
        state.human_review_reason = f"Critical errors encountered: {len(critical_errors)} errors"
    
    # Add observation
    state.observations.append({
        "step": "evaluate_progress",
        "observation": f"Requires human review: {state.requires_human_review}. Reason: {state.human_review_reason or 'N/A'}",
        "timestamp": datetime.now().isoformat()
    })
    
    return state


@node_wrapper("escalate_to_human")
async def escalate_to_human_node(state: GraphState) -> GraphState:
    """
    Create Action Center task for human review using LangGraph interrupt.
    
    Uses interrupt mechanism to pause execution and wait for human decision.
    When resumed, extracts human decision and updates state accordingly.
    
    Implements Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8
    """
    # Check if Action Center is enabled
    if not settings.enable_action_center:
        logger.warning("[ACTION_CENTER] Action Center disabled in settings, skipping escalation")
        state.human_decision = "auto_proceed"
        return state
    
    logger.info(f"[ACTION_CENTER] Creating Action Center task for claim: {state.claim_id}")
    
    try:
        async with UiPathService() as uipath_service:
            # Record escalation in processing history
            try:
                history_service = ProcessingHistoryService(uipath_service._client)
                await history_service.record_escalation(
                    claim_id=state.claim_id,
                    reason=state.human_review_reason or "Human review required",
                    action_center_task_id=None,  # Will be set after interrupt
                    escalation_data={
                        "confidence": state.confidence,
                        "decision": state.decision,
                        "risk_level": state.risk_level
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to record escalation in history: {e}")
        
        # Get assignee from state or use default
        assignee = getattr(state, 'assigned_reviewer', None) or "claims_team@company.com"
        
        # Get folder path from environment or use default
        app_folder_path = os.getenv("FOLDER_PATH", "Shared")
        
        # Build data payload for Action Center task
        task_data = {
            "claim_id": state.claim_id,
            "agent_recommendation": state.decision or "pending",
            "confidence": state.confidence or 0.0,
            "reasoning": state.reasoning or "Processing in progress",
            "reasoning_steps": state.reasoning_steps,
            "risk_factors": state.risk_factors,
            "extracted_data": state.extracted_data
        }
        
        logger.info(f"[ACTION_CENTER] Pausing execution for human review - Claim: {state.claim_id}")
        
        # Use interrupt to pause execution and create Action Center task
        # This will pause the graph execution and wait for external input
        action_data = interrupt({
            "type": "action",
            "app_name": "ltl_claims_review_app",
            "title": f"Review Claim {state.claim_id} - Approval Required",
            "data": task_data,
            "app_version": 1,
            "assignee": assignee,
            "app_folder_path": app_folder_path
        })
        
        logger.info(f"[ACTION_CENTER] Execution resumed with human decision for claim: {state.claim_id}")
        
        # When execution resumes, action_data contains the human response
        # Extract Answer field (boolean) from action_data
        if action_data and isinstance(action_data, dict):
            answer = action_data.get("Answer")
            
            # Set human_decision based on Answer field
            if isinstance(answer, bool):
                if answer is True:
                    state.human_decision = "approved"
                    logger.info(f"[ACTION_CENTER] Human approved the recommendation")
                else:
                    state.human_decision = "rejected"
                    logger.info(f"[ACTION_CENTER] Human rejected the recommendation")
                    
                    # Check for AlternativeDecision field
                    alternative_decision = action_data.get("AlternativeDecision")
                    if alternative_decision:
                        state.decision = alternative_decision
                        logger.info(f"[ACTION_CENTER] Using alternative decision: {alternative_decision}")
            else:
                logger.warning(f"[ACTION_CENTER] Invalid Answer field type: {type(answer)}, defaulting to auto_proceed")
                state.human_decision = "auto_proceed"
            
            # Store action_center_task_id if available
            if "action_key" in action_data:
                state.action_center_task_id = action_data["action_key"]
            elif "task_id" in action_data:
                state.action_center_task_id = action_data["task_id"]
        else:
            logger.warning(f"[ACTION_CENTER] No action data received, defaulting to auto_proceed")
            state.human_decision = "auto_proceed"
        
        # Record human decision in processing history
        try:
            async with UiPathService() as uipath_service:
                history_service = ProcessingHistoryService(uipath_service._client)
                await history_service.record_human_decision(
                    claim_id=state.claim_id,
                    human_decision=state.human_decision,
                    action_center_task_id=state.action_center_task_id
                )
        except Exception as e:
            logger.warning(f"Failed to record human decision in history: {e}")
        
    except Exception as e:
        # Continue without human review on error
        logger.warning(f"Escalation failed, continuing without human review: {e}")
        state.human_decision = "auto_proceed"
        raise  # Re-raise to be caught by decorator
    
    return state


@node_wrapper("make_decision")
async def make_decision_node(state: GraphState) -> GraphState:
    """
    Make final decision on the claim.
    
    Uses hybrid decision strategy (LLM with rule-based fallback)
    to analyze all gathered information and make a final decision.
    
    Implements Requirements 10.1, 10.2, 10.4
    """
    # Create UiPath Chat model for decision making
    llm = UiPathChat(
        model="gpt-4o-2024-08-06",
        temperature=0,
        max_tokens=4000,
        timeout=30,
        max_retries=2
    )
    
    # Create decision strategy
    strategy = HybridDecisionStrategy(llm)
    
    # Make decision using strategy
    decision_data = await strategy.make_decision(state.model_dump())
    
    # Update state with decision
    state.decision = decision_data["decision"]
    state.confidence = decision_data["confidence"]
    state.reasoning = decision_data["reasoning"]
    
    # Add reasoning step
    state.reasoning_steps.append({
        "step": "make_decision",
        "reasoning": state.reasoning,
        "confidence": state.confidence,
        "timestamp": datetime.now().isoformat()
    })
    
    # Record decision in processing history
    try:
        async with UiPathService() as uipath_service:
            history_service = ProcessingHistoryService(uipath_service._client)
            await history_service.record_decision_made(
                claim_id=state.claim_id,
                decision=state.decision,
                confidence=state.confidence,
                reasoning=state.reasoning,
                reasoning_steps=state.reasoning_steps
            )
    except Exception as e:
        logger.warning(f"Failed to record decision in history: {e}")
    
    return state


@node_wrapper("update_systems")
async def update_systems_node(state: GraphState) -> GraphState:
    """
    Update queue and Data Fabric with processing results.
    
    Updates queue transaction status and stores results in Data Fabric.
    
    Implements Requirements 8.1, 8.2, 8.3, 9.1, 9.2, 9.3
    """
    async with UiPathService() as uipath_service:
        # Update queue transaction if transaction_key exists
        if state.transaction_key:
            try:
                from uipath import UiPath
                sdk = UiPath()
                
                # Determine status based on decision
                if state.decision == DecisionConstants.APPROVED:
                    status = "Successful"
                elif state.decision == DecisionConstants.DENIED:
                    status = "Failed"
                else:
                    status = "Failed"  # Pending or error cases
                
                # Prepare output data
                output_data = {
                    "decision": state.decision,
                    "confidence": state.confidence,
                    "reasoning": state.reasoning,
                    "risk_level": state.risk_level,
                    "risk_score": state.risk_score,
                    "policy_compliant": state.policy_compliant,
                    "human_review_required": state.requires_human_review,
                    "processing_duration_seconds": (datetime.now() - state.start_time).total_seconds() if state.start_time else None,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Complete transaction with final status
                completion_result = {
                    "Status": status,
                    "OutputData": json.dumps(output_data)
                }
                
                # Add error message if there are errors
                if state.errors:
                    error_messages = [e.get("error", str(e)) if isinstance(e, dict) else str(e) for e in state.errors]
                    completion_result["ErrorMessage"] = "; ".join(error_messages[:3])  # First 3 errors
                
                await sdk.queues.complete_transaction_item_async(
                    transaction_key=state.transaction_key,
                    result=completion_result
                )
                
                logger.info(f"Queue transaction completed: {status}")
                state.tools_used.append("complete_queue_transaction")
                
            except Exception as e:
                logger.error(f" Queue update failed: {e}")
                state.errors.append({
                    "step": "update_queue",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Update Data Fabric with results
        try:
            additional_data = {
                "Status": state.decision,
                "ProcessingHistory": {
                    "decision": state.decision,
                    "confidence": state.confidence,
                    "reasoning": state.reasoning,
                    "risk_level": state.risk_level,
                    "risk_score": state.risk_score,
                    "policy_compliant": state.policy_compliant,
                    "processed_at": datetime.now().isoformat()
                }
            }
            
            await uipath_service.update_claim_status(
                claim_id=state.claim_id,
                status=state.decision,
                additional_data=additional_data
            )
            
        except Exception as e:
            logger.error(f"Data Fabric update failed: {e}")
            state.errors.append({
                "step": "update_data_fabric",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    return state


@node_wrapper("finalize_output", mark_completed=False)
async def finalize_output_node(state: GraphState) -> GraphOutput:
    """
    Finalize processing and return structured output.
    
    Sets end time, calculates duration, stores outcome in memory,
    and builds final output.
    
    Implements Requirements 13.1, 13.2, 13.3, 13.4, 13.5, 15.3
    """
    # Set end time
    state.end_time = datetime.now()
    
    # Calculate processing duration
    duration = None
    if state.start_time and state.end_time:
        duration = (state.end_time - state.start_time).total_seconds()
    
    # Store outcome in long-term memory if enabled
    if settings.enable_long_term_memory and state.claim_id:
        try:
            from src.memory.long_term_memory import ClaimMemoryStore
            
            logger.info(f"Storing claim outcome in memory: {state.claim_id}")
            
            # Initialize memory store
            memory_store = ClaimMemoryStore(
                connection_string=settings.memory_connection_string,
                store_type=settings.memory_store_type
            )
            
            # Prepare claim data for storage
            claim_data = {
                "ClaimId": state.claim_id,
                "ClaimType": state.claim_type,
                "ClaimAmount": state.claim_amount,
                "Carrier": state.carrier,
                "CustomerName": state.customer_name,
                "RiskLevel": state.risk_level,
                "RiskScore": state.risk_score,
                "PolicyCompliant": state.policy_compliant,
                "DataFabricValidated": state.data_fabric_validated,
                "DocumentsProcessed": len(state.downloaded_documents),
                "ExtractionConfidence": state.extraction_confidence,
                "HumanReviewRequired": state.requires_human_review
            }
            
            # Determine outcome based on success and decision
            if len(state.errors) > 0:
                outcome = "failed"
            elif state.decision == DecisionConstants.APPROVED:
                outcome = "approved"
            elif state.decision == DecisionConstants.DENIED:
                outcome = "denied"
            else:
                outcome = "pending"
            
            # Save to memory
            await memory_store.save_claim_session(
                claim_id=state.claim_id,
                claim_data=claim_data,
                reasoning_steps=state.reasoning_steps,
                decision=state.decision or DecisionConstants.PENDING,
                confidence=state.confidence or 0.0,
                outcome=outcome
            )
            
            logger.info(
                f"Claim outcome stored in memory: {state.claim_id} "
                f"(Decision: {state.decision}, Outcome: {outcome})"
            )
            
        except Exception as e:
            logger.warning(f"Failed to store outcome in memory: {e}")
            # Don't fail the entire process if memory storage fails
            logger.info("Continuing without memory storage")
    else:
        if not settings.enable_long_term_memory:
            logger.debug("Long-term memory disabled, skipping outcome storage")
        elif not state.claim_id:
            logger.warning("No claim_id available, cannot store outcome in memory")
    
    # Build output
    # Extract error message as string from error dictionary
    error_message = None
    if state.errors:
        first_error = state.errors[0]
        if isinstance(first_error, dict):
            # Format error dictionary into a readable string
            error_message = first_error.get("error", str(first_error))
            if isinstance(error_message, dict):
                # If error value is also a dict, convert to string
                error_message = str(error_message)
        else:
            error_message = str(first_error)
    
    output = GraphOutput(
        success=len(state.errors) == 0,
        claim_id=state.claim_id or "UNKNOWN",
        decision=state.decision or DecisionConstants.PENDING,
        confidence=state.confidence or 0.0,
        reasoning=state.reasoning or "Processing incomplete",
        reasoning_steps=state.reasoning_steps,
        tools_used=state.tools_used,
        human_review_required=state.requires_human_review,
        action_center_task_id=state.action_center_task_id,
        processing_duration_seconds=duration,
        error=error_message,
        timestamp=datetime.now().isoformat(),
        risk_level=state.risk_level,
        policy_compliant=state.policy_compliant,
        data_fabric_updated="update_systems" in state.completed_steps,
        queue_updated=bool(state.transaction_key)
    )
    
    return output



# ============================================================================
# CONDITIONAL ROUTING
# ============================================================================

def should_escalate(state: GraphState) -> str:
    """
    Decide if human escalation is needed.
    
    Returns "escalate" if human review is required, otherwise "decide".
    
    Implements Requirements 7.1, 10.3
    """
    if state.requires_human_review:
        return "escalate"
    return "decide"


# ============================================================================
# GRAPH DEFINITION
# ============================================================================

try:
    # Build the state graph
    builder = StateGraph(GraphState, output=GraphOutput)
    
    # Add all nodes
    builder.add_node("initialize_input", initialize_input_node)
    builder.add_node("create_plan", create_plan_node)
    builder.add_node("validate_data", validate_data_node)
    builder.add_node("download_documents", download_documents_node)
    builder.add_node("assess_risk", assess_risk_node)
    builder.add_node("validate_policy", validate_policy_node)
    builder.add_node("evaluate_progress", evaluate_progress_node)
    builder.add_node("escalate_to_human", escalate_to_human_node)
    builder.add_node("make_decision", make_decision_node)
    builder.add_node("update_systems", update_systems_node)
    builder.add_node("finalize_output", finalize_output_node)
    
    # Define edges - main workflow
    builder.add_edge(START, "initialize_input")
    builder.add_edge("initialize_input", "create_plan")
    builder.add_edge("create_plan", "validate_data")
    builder.add_edge("validate_data", "download_documents")
    builder.add_edge("download_documents", "assess_risk")
    builder.add_edge("assess_risk", "validate_policy")
    builder.add_edge("validate_policy", "evaluate_progress")
    
    # Conditional routing for human escalation
    builder.add_conditional_edges(
        "evaluate_progress",
        should_escalate,
        {
            "escalate": "escalate_to_human",
            "decide": "make_decision"
        }
    )
    
    # Continue from escalation to decision
    builder.add_edge("escalate_to_human", "make_decision")
    
    # Final steps
    builder.add_edge("make_decision", "update_systems")
    builder.add_edge("update_systems", "finalize_output")
    builder.add_edge("finalize_output", END)
    
    # Compile and export
    graph = builder.compile()
    
    logger.info("[GRAPH] LTL Claims Processing Agent graph compiled successfully")
    
except Exception as e:
    logger.error(f"[GRAPH] Failed to compile graph: {e}")
    raise


# ============================================================================
# UIPATH AGENT BINDINGS
# ============================================================================

# Alias GraphState and GraphOutput for UiPath agent bindings
# This allows the UiPath SDK to recognize the input/output schemas
Input = GraphState
Output = GraphOutput


# Main function for UiPath agent execution
async def main(input_data: Input) -> Output:
    """
    Main entry point for the LTL Claims Processing Agent.
    
    This function invokes the LangGraph workflow with the provided input data.
    
    Args:
        input_data: GraphState with claim information
        
    Returns:
        GraphOutput with processing results
    """
    try:
        # Invoke the graph with input data using async API
        result = await graph.ainvoke(input_data.model_dump())
        return result
    except Exception as e:
        logger.error(f"[MAIN] Agent execution failed: {e}", exc_info=True)
        # Return error output
        return GraphOutput(
            success=False,
            claim_id=input_data.claim_id or "UNKNOWN",
            decision="error",
            confidence=0.0,
            reasoning=f"Agent execution failed: {str(e)}",
            reasoning_steps=[],
            tools_used=[],
            human_review_required=False,
            error=str(e),
            timestamp=datetime.now().isoformat()
        )
