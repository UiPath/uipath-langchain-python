"""
Agent Memory and State Models for LTL Claims Processing
Implements comprehensive state management, reasoning chains, and audit trails.
"""

import json
from typing import Dict, Any, List, Optional, Union, TypedDict, Annotated
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass, field


class ConfidenceLevel(Enum):
    """Confidence levels for agent decisions."""
    VERY_HIGH = "very_high"  # 0.9+
    HIGH = "high"           # 0.7-0.89
    MEDIUM = "medium"       # 0.5-0.69
    LOW = "low"            # 0.3-0.49
    VERY_LOW = "very_low"  # <0.3


class ProcessingPhase(Enum):
    """Different phases of claim processing."""
    INITIALIZATION = "initialization"
    INFORMATION_GATHERING = "information_gathering"
    DOCUMENT_ANALYSIS = "document_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    POLICY_APPLICATION = "policy_application"
    DECISION_MAKING = "decision_making"
    EXECUTION = "execution"
    ESCALATION = "escalation"
    FINALIZATION = "finalization"


class UncertaintyType(Enum):
    """Types of uncertainty that can arise during processing."""
    DATA_INCOMPLETE = "data_incomplete"
    DATA_INCONSISTENT = "data_inconsistent"
    LOW_EXTRACTION_CONFIDENCE = "low_extraction_confidence"
    POLICY_AMBIGUITY = "policy_ambiguity"
    HIGH_RISK_INDICATORS = "high_risk_indicators"
    MODEL_DISAGREEMENT = "model_disagreement"
    EXTERNAL_SERVICE_FAILURE = "external_service_failure"
    TIMEOUT_EXCEEDED = "timeout_exceeded"


class EscalationTrigger(Enum):
    """Triggers that cause escalation to human review."""
    LOW_CONFIDENCE = "low_confidence"
    HIGH_RISK = "high_risk"
    POLICY_VIOLATION = "policy_violation"
    FRAUD_SUSPECTED = "fraud_suspected"
    SYSTEM_ERROR = "system_error"
    TIMEOUT = "timeout"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    INCONSISTENT_DATA = "inconsistent_data"


class NotificationChannel(Enum):
    """Available notification channels."""
    EMAIL = "email"


class NotificationStatus(Enum):
    """Notification delivery status."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    BOUNCED = "bounced"
    RETRY_SCHEDULED = "retry_scheduled"


@dataclass
class ReasoningStep:
    """Individual step in the agent's reasoning chain."""
    step_number: int
    timestamp: datetime
    phase: ProcessingPhase
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    confidence: float = 0.5
    reasoning_chain: List[str] = field(default_factory=list)
    tool_used: Optional[str] = None
    execution_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class ToolExecution:
    """Record of tool execution with results and performance."""
    tool_name: str
    timestamp: datetime
    input_parameters: Dict[str, Any]
    output_result: Any
    execution_time: float
    success: bool
    confidence: float = 0.0
    error_message: Optional[str] = None
    retry_count: int = 0


@dataclass
class DecisionFactor:
    """Individual factor contributing to a decision."""
    factor_name: str
    factor_value: Any
    weight: float
    confidence: float
    source: str  # Which tool/analysis provided this factor
    timestamp: datetime


@dataclass
class UncertaintyArea:
    """Area of uncertainty identified during processing."""
    uncertainty_type: UncertaintyType
    description: str
    confidence_impact: float
    resolution_suggestions: List[str]
    timestamp: datetime
    resolved: bool = False
    resolution_method: Optional[str] = None


@dataclass
class AuditEntry:
    """Individual audit trail entry."""
    timestamp: datetime
    action: str
    actor: str  # agent, human, system
    details: Dict[str, Any]
    phase: ProcessingPhase
    confidence_before: float
    confidence_after: float
    decision_impact: Optional[str] = None


@dataclass
class NotificationRecord:
    """Record of a notification sent during claim processing."""
    notification_id: str
    claim_id: str
    recipient: str
    channel: NotificationChannel
    template_name: str
    subject: str
    content: str
    priority: str
    timestamp: datetime
    status: NotificationStatus
    delivery_attempts: int = 0
    last_attempt: Optional[datetime] = None
    error_message: Optional[str] = None
    provider_message_id: Optional[str] = None


@dataclass
class DeliveryRecord:
    """Record of notification delivery tracking."""
    notification_id: str
    status: NotificationStatus
    timestamp: datetime
    provider_response: Optional[str] = None
    delivery_timestamp: Optional[datetime] = None
    bounce_reason: Optional[str] = None
    retry_count: int = 0
    next_retry: Optional[datetime] = None


@dataclass
class MemoryContext:
    """Historical context from long-term memory for claim processing."""
    similar_claims: List[Dict[str, Any]] = field(default_factory=list)
    decision_patterns: Dict[str, Any] = field(default_factory=dict)
    common_risk_factors: List[str] = field(default_factory=list)
    total_similar_claims: int = 0
    average_confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "similar_claims": self.similar_claims,
            "decision_patterns": self.decision_patterns,
            "common_risk_factors": self.common_risk_factors,
            "total_similar_claims": self.total_similar_claims,
            "average_confidence": self.average_confidence
        }
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the memory context."""
        if not self.similar_claims:
            return "No similar historical claims found."
        
        summary_parts = [
            f"Found {self.total_similar_claims} similar historical claims:",
        ]
        
        # Summarize similar claims
        for i, claim in enumerate(self.similar_claims[:3], 1):  # Top 3
            summary_parts.append(
                f"  {i}. Claim {claim.get('claim_id', 'unknown')}: "
                f"{claim.get('decision', 'unknown')} "
                f"(confidence: {claim.get('confidence', 0):.2f}, "
                f"similarity: {claim.get('similarity_score', 0):.2f})"
            )
        
        # Add decision patterns
        if self.decision_patterns:
            patterns = self.decision_patterns
            if patterns.get("total_claims", 0) > 0:
                summary_parts.append(
                    f"\nDecision patterns (last {patterns.get('time_window_days', 90)} days):"
                )
                summary_parts.append(
                    f"  - Total claims: {patterns['total_claims']}"
                )
                summary_parts.append(
                    f"  - Average confidence: {patterns.get('average_confidence', 0):.2f}"
                )
                if patterns.get("most_common_decision"):
                    summary_parts.append(
                        f"  - Most common decision: {patterns['most_common_decision']}"
                    )
        
        # Add risk factors
        if self.common_risk_factors:
            summary_parts.append(f"\nCommon risk factors identified:")
            for factor in self.common_risk_factors[:3]:  # Top 3
                summary_parts.append(f"  - {factor}")
        
        return "\n".join(summary_parts)


class AgentMemoryState(BaseModel):
    """
    Comprehensive agent memory state for dynamic information tracking.
    Implements the AgentMemoryState TypedDict from the design with full validation.
    """
    
    # Core claim context
    claim_id: str = Field(..., description="Unique identifier for the claim")
    claim_data: Dict[str, Any] = Field(default_factory=dict, description="Original claim data")
    queue_item: Optional[Dict[str, Any]] = Field(None, description="Queue item data if applicable")
    
    # Processing state
    current_phase: ProcessingPhase = Field(ProcessingPhase.INITIALIZATION, description="Current processing phase")
    processing_start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processing_complete: bool = Field(False, description="Whether processing is complete")
    
    # ReAct reasoning chain
    reasoning_steps: List[ReasoningStep] = Field(default_factory=list, description="Complete reasoning chain")
    current_step: int = Field(0, description="Current step number")
    
    # Dynamic planning and goals
    current_goal: str = Field("Process freight claim efficiently and accurately", description="Current processing goal")
    planned_actions: List[str] = Field(default_factory=list, description="Planned actions to take")
    completed_actions: List[str] = Field(default_factory=list, description="Successfully completed actions")
    failed_actions: List[Dict[str, Any]] = Field(default_factory=list, description="Failed actions with error details")
    
    # Information gathering and analysis
    gathered_information: Dict[str, Any] = Field(default_factory=dict, description="Information gathered during processing")
    confidence_levels: Dict[str, float] = Field(default_factory=dict, description="Confidence levels for different aspects")
    uncertainty_areas: List[UncertaintyArea] = Field(default_factory=list, description="Identified uncertainty areas")
    
    # Tool execution tracking
    tool_executions: List[ToolExecution] = Field(default_factory=list, description="Complete tool execution history")
    tools_used: List[str] = Field(default_factory=list, description="List of tools used")
    tool_performance: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Tool performance metrics")
    
    # Decision context and factors
    decision_factors: List[DecisionFactor] = Field(default_factory=list, description="Factors contributing to decisions")
    risk_indicators: List[str] = Field(default_factory=list, description="Identified risk indicators")
    policy_references: List[str] = Field(default_factory=list, description="Applied policy references")
    
    # Escalation and human interaction
    escalation_triggers: List[EscalationTrigger] = Field(default_factory=list, description="Triggers for escalation")
    human_feedback: Optional[Dict[str, Any]] = Field(None, description="Human feedback if provided")
    escalation_history: List[Dict[str, Any]] = Field(default_factory=list, description="History of escalations")
    
    # Audit and compliance
    audit_trail: List[AuditEntry] = Field(default_factory=list, description="Complete audit trail")
    compliance_checks: Dict[str, bool] = Field(default_factory=dict, description="Compliance check results")
    
    # Performance and quality metrics
    overall_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Overall processing confidence")
    quality_score: float = Field(0.5, ge=0.0, le=1.0, description="Processing quality score")
    efficiency_score: float = Field(0.5, ge=0.0, le=1.0, description="Processing efficiency score")
    
    # Final results
    final_result: Optional[Dict[str, Any]] = Field(None, description="Final processing result")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Enum: lambda v: v.value
        }
    
    @validator('overall_confidence', 'quality_score', 'efficiency_score')
    def validate_scores(cls, v):
        """Ensure scores are within valid range."""
        return max(0.0, min(1.0, v))
    
    def add_reasoning_step(
        self,
        thought: str,
        action: Optional[str] = None,
        action_input: Optional[Dict[str, Any]] = None,
        confidence: float = 0.5,
        phase: Optional[ProcessingPhase] = None
    ) -> ReasoningStep:
        """Add a new reasoning step to the chain."""
        
        step = ReasoningStep(
            step_number=len(self.reasoning_steps) + 1,
            timestamp=datetime.now(timezone.utc),
            phase=phase or self.current_phase,
            thought=thought,
            action=action,
            action_input=action_input,
            confidence=confidence,
            reasoning_chain=[step.thought for step in self.reasoning_steps[-3:]]  # Last 3 thoughts
        )
        
        self.reasoning_steps.append(step)
        self.current_step = step.step_number
        self.last_activity_time = step.timestamp
        
        return step
    
    def record_tool_execution(
        self,
        tool_name: str,
        input_parameters: Dict[str, Any],
        output_result: Any,
        execution_time: float,
        success: bool,
        confidence: float = 0.0,
        error_message: Optional[str] = None
    ) -> ToolExecution:
        """Record a tool execution with full details."""
        
        execution = ToolExecution(
            tool_name=tool_name,
            timestamp=datetime.now(timezone.utc),
            input_parameters=input_parameters,
            output_result=output_result,
            execution_time=execution_time,
            success=success,
            confidence=confidence,
            error_message=error_message
        )
        
        self.tool_executions.append(execution)
        
        if tool_name not in self.tools_used:
            self.tools_used.append(tool_name)
        
        # Update tool performance metrics
        if tool_name not in self.tool_performance:
            self.tool_performance[tool_name] = {
                "success_rate": 0.0,
                "avg_execution_time": 0.0,
                "avg_confidence": 0.0,
                "total_executions": 0
            }
        
        perf = self.tool_performance[tool_name]
        perf["total_executions"] += 1
        
        # Update running averages
        alpha = 1.0 / perf["total_executions"]  # Simple average
        perf["success_rate"] = perf["success_rate"] * (1 - alpha) + (1.0 if success else 0.0) * alpha
        perf["avg_execution_time"] = perf["avg_execution_time"] * (1 - alpha) + execution_time * alpha
        perf["avg_confidence"] = perf["avg_confidence"] * (1 - alpha) + confidence * alpha
        
        self.last_activity_time = execution.timestamp
        return execution
    
    def add_decision_factor(
        self,
        factor_name: str,
        factor_value: Any,
        weight: float,
        confidence: float,
        source: str
    ) -> DecisionFactor:
        """Add a decision factor to the analysis."""
        
        factor = DecisionFactor(
            factor_name=factor_name,
            factor_value=factor_value,
            weight=weight,
            confidence=confidence,
            source=source,
            timestamp=datetime.now(timezone.utc)
        )
        
        self.decision_factors.append(factor)
        return factor
    
    def add_uncertainty(
        self,
        uncertainty_type: UncertaintyType,
        description: str,
        confidence_impact: float,
        resolution_suggestions: List[str]
    ) -> UncertaintyArea:
        """Add an uncertainty area that needs resolution."""
        
        uncertainty = UncertaintyArea(
            uncertainty_type=uncertainty_type,
            description=description,
            confidence_impact=confidence_impact,
            resolution_suggestions=resolution_suggestions,
            timestamp=datetime.now(timezone.utc)
        )
        
        self.uncertainty_areas.append(uncertainty)
        
        # Adjust overall confidence based on uncertainty impact
        self.overall_confidence = max(0.0, self.overall_confidence - confidence_impact)
        
        return uncertainty
    
    def resolve_uncertainty(self, uncertainty_index: int, resolution_method: str):
        """Mark an uncertainty as resolved."""
        
        if 0 <= uncertainty_index < len(self.uncertainty_areas):
            uncertainty = self.uncertainty_areas[uncertainty_index]
            uncertainty.resolved = True
            uncertainty.resolution_method = resolution_method
            
            # Restore confidence impact
            self.overall_confidence = min(1.0, self.overall_confidence + uncertainty.confidence_impact)
    
    def add_audit_entry(
        self,
        action: str,
        actor: str,
        details: Dict[str, Any],
        decision_impact: Optional[str] = None
    ) -> AuditEntry:
        """Add an entry to the audit trail."""
        
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc),
            action=action,
            actor=actor,
            details=details,
            phase=self.current_phase,
            confidence_before=self.overall_confidence,
            confidence_after=self.overall_confidence,  # Will be updated if confidence changes
            decision_impact=decision_impact
        )
        
        self.audit_trail.append(entry)
        self.last_activity_time = entry.timestamp
        
        return entry
    
    def update_confidence(self, new_confidence: float, reason: str):
        """Update overall confidence with audit trail."""
        
        old_confidence = self.overall_confidence
        self.overall_confidence = max(0.0, min(1.0, new_confidence))
        
        # Update the last audit entry if it exists
        if self.audit_trail:
            self.audit_trail[-1].confidence_after = self.overall_confidence
        
        # Add confidence update to audit trail
        self.add_audit_entry(
            action="confidence_update",
            actor="agent",
            details={
                "old_confidence": old_confidence,
                "new_confidence": self.overall_confidence,
                "reason": reason,
                "change": self.overall_confidence - old_confidence
            }
        )
    
    def add_escalation_trigger(self, trigger: EscalationTrigger, reason: str):
        """Add an escalation trigger with reasoning."""
        
        if trigger not in self.escalation_triggers:
            self.escalation_triggers.append(trigger)
        
        self.escalation_history.append({
            "trigger": trigger.value,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence_at_trigger": self.overall_confidence
        })
        
        self.add_audit_entry(
            action="escalation_trigger_added",
            actor="agent",
            details={
                "trigger": trigger.value,
                "reason": reason
            },
            decision_impact="escalation_required"
        )
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get a comprehensive processing summary."""
        
        processing_duration = (self.last_activity_time - self.processing_start_time).total_seconds()
        
        return {
            "claim_id": self.claim_id,
            "processing_duration": processing_duration,
            "current_phase": self.current_phase.value,
            "total_steps": len(self.reasoning_steps),
            "completed_actions": len(self.completed_actions),
            "failed_actions": len(self.failed_actions),
            "tools_used": len(self.tools_used),
            "tool_executions": len(self.tool_executions),
            "overall_confidence": self.overall_confidence,
            "quality_score": self.quality_score,
            "efficiency_score": self.efficiency_score,
            "uncertainty_count": len([u for u in self.uncertainty_areas if not u.resolved]),
            "escalation_triggers": len(self.escalation_triggers),
            "audit_entries": len(self.audit_trail),
            "processing_complete": self.processing_complete,
            "human_review_required": len(self.escalation_triggers) > 0
        }
    
    def get_confidence_breakdown(self) -> Dict[str, float]:
        """Get detailed confidence breakdown by category."""
        
        breakdown = {
            "overall": self.overall_confidence,
            "quality": self.quality_score,
            "efficiency": self.efficiency_score
        }
        
        # Add confidence levels from different aspects
        breakdown.update(self.confidence_levels)
        
        # Calculate derived confidence metrics
        if self.reasoning_steps:
            breakdown["reasoning_confidence"] = sum(step.confidence for step in self.reasoning_steps) / len(self.reasoning_steps)
        
        if self.tool_executions:
            successful_tools = [t for t in self.tool_executions if t.success]
            if successful_tools:
                breakdown["tool_confidence"] = sum(t.confidence for t in successful_tools) / len(successful_tools)
        
        return breakdown
    
    def export_for_human_review(self) -> Dict[str, Any]:
        """Export state in format suitable for human review."""
        
        return {
            "claim_summary": {
                "claim_id": self.claim_id,
                "processing_duration": (self.last_activity_time - self.processing_start_time).total_seconds(),
                "current_phase": self.current_phase.value,
                "overall_confidence": self.overall_confidence,
                "escalation_triggers": [t.value for t in self.escalation_triggers]
            },
            "reasoning_chain": [
                {
                    "step": step.step_number,
                    "thought": step.thought,
                    "action": step.action,
                    "confidence": step.confidence,
                    "timestamp": step.timestamp.isoformat()
                }
                for step in self.reasoning_steps[-10:]  # Last 10 steps
            ],
            "key_findings": {
                "gathered_information": self.gathered_information,
                "decision_factors": [
                    {
                        "factor": f.factor_name,
                        "value": f.factor_value,
                        "confidence": f.confidence,
                        "source": f.source
                    }
                    for f in self.decision_factors
                ],
                "risk_indicators": self.risk_indicators,
                "policy_references": self.policy_references
            },
            "uncertainty_areas": [
                {
                    "type": u.uncertainty_type.value,
                    "description": u.description,
                    "impact": u.confidence_impact,
                    "suggestions": u.resolution_suggestions,
                    "resolved": u.resolved
                }
                for u in self.uncertainty_areas
            ],
            "tool_performance": self.tool_performance,
            "processing_summary": self.get_processing_summary()
        }


class AgentMemoryManager:
    """
    Manager for agent memory operations including persistence and retrieval.
    """
    
    def __init__(self):
        """Initialize the memory manager."""
        self.active_memories: Dict[str, AgentMemoryState] = {}
        self.memory_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def create_memory(self, claim_id: str, claim_data: Dict[str, Any]) -> AgentMemoryState:
        """Create a new agent memory state for a claim."""
        
        memory = AgentMemoryState(
            claim_id=claim_id,
            claim_data=claim_data
        )
        
        # Add initial audit entry
        memory.add_audit_entry(
            action="memory_created",
            actor="system",
            details={
                "claim_id": claim_id,
                "initialization_time": memory.processing_start_time.isoformat()
            }
        )
        
        self.active_memories[claim_id] = memory
        return memory
    
    def get_memory(self, claim_id: str) -> Optional[AgentMemoryState]:
        """Retrieve agent memory for a claim."""
        return self.active_memories.get(claim_id)
    
    def update_memory(self, claim_id: str, memory: AgentMemoryState):
        """Update agent memory state."""
        self.active_memories[claim_id] = memory
        memory.last_activity_time = datetime.now(timezone.utc)
    
    def archive_memory(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """Archive completed memory state."""
        
        memory = self.active_memories.get(claim_id)
        if not memory:
            return None
        
        # Export memory state
        archived_state = memory.dict()
        
        # Add to history
        if claim_id not in self.memory_history:
            self.memory_history[claim_id] = []
        
        self.memory_history[claim_id].append({
            "archived_at": datetime.now(timezone.utc).isoformat(),
            "state": archived_state
        })
        
        # Remove from active memories
        del self.active_memories[claim_id]
        
        return archived_state
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about memory usage."""
        
        active_count = len(self.active_memories)
        archived_count = sum(len(history) for history in self.memory_history.values())
        
        if active_count > 0:
            avg_confidence = sum(m.overall_confidence for m in self.active_memories.values()) / active_count
            avg_steps = sum(len(m.reasoning_steps) for m in self.active_memories.values()) / active_count
        else:
            avg_confidence = 0.0
            avg_steps = 0.0
        
        return {
            "active_memories": active_count,
            "archived_memories": archived_count,
            "average_confidence": avg_confidence,
            "average_reasoning_steps": avg_steps,
            "memory_phases": {
                phase.value: sum(1 for m in self.active_memories.values() if m.current_phase == phase)
                for phase in ProcessingPhase
            }
        }


# Global memory manager instance
memory_manager = AgentMemoryManager()