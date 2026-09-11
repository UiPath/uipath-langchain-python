"""
Orchestrator Agent for LTL Claims Processing
Implements the main supervisor agent that coordinates specialized sub-agents
and manages the plan-execute-observe-reflect cycle.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from functools import lru_cache

from langchain_core.messages import SystemMessage, HumanMessage
from uipath_langchain.chat.models import UiPathChat

from ..services.uipath_service import UiPathService
from ..tools.tools_registry import get_all_tools
from ..models.agent_models import ProcessingPhase, ReasoningStep
from ..utils.errors import ProcessingError
from .config import OrchestratorConfig
from .exceptions import PlanGenerationError, ReflectionError, ReplanningError
from .prompts.orchestrator_prompts import OrchestratorPrompts, ClaimContextExtractor


logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Main orchestrator agent that coordinates the overall claims processing workflow.
    
    Responsibilities:
    - Create execution plans based on claim data
    - Coordinate specialized sub-agents
    - Reflect on progress and adapt plans
    - Handle failures and replanning
    
    Implements the plan-execute-observe-reflect pattern from Requirements 10.1, 10.3, 10.4, 11.1, 11.4
    """
    
    # Plan parsing prefixes (compiled once for performance)
    _PLAN_PREFIXES = ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.',
                      '-', '*', '•', 'Step', 'step']
    
    def __init__(self, uipath_service: UiPathService, config: Optional[OrchestratorConfig] = None):
        """
        Initialize the orchestrator agent.
        
        Args:
            uipath_service: Authenticated UiPath service instance for SDK operations
            config: Optional configuration object (uses defaults if not provided)
        """
        self.uipath_service = uipath_service
        self.config = config or OrchestratorConfig()
        
        # Use UiPath Chat model for orchestration (Requirement 11.1)
        self.llm = UiPathChat(
            model="gpt-4o-2024-08-06",
            temperature=0,
            max_tokens=4000,
            timeout=30,
            max_retries=2
        )
        
        # Load all available tools from registry (Requirement 11.1)
        self.tools = get_all_tools()
        
        logger.info(f"[ORCHESTRATOR] Initialized with {len(self.tools)} tools available")
    
    @staticmethod
    def _extract_claim_id(state: Dict[str, Any]) -> str:
        """Extract claim ID from state, handling both field name formats."""
        return state.get('claim_id') or state.get('ObjectClaimId', 'UNKNOWN')
    
    def _log_agent_invocation_debug(self, result: Dict[str, Any], operation: str) -> None:
        """
        Log debug information about agent invocation results.
        
        Args:
            result: Agent invocation result containing messages
            operation: Name of the operation (e.g., 'planning', 'replanning')
        """
        messages = result.get("messages", [])
        logger.debug(f"[ORCHESTRATOR] {operation.capitalize()} agent returned {len(messages)} messages")
        
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            content_preview = str(msg.content)[:150] if hasattr(msg, 'content') else 'No content'
            logger.debug(f"[ORCHESTRATOR] Message {i} ({msg_type}): {content_preview}...")
            
            # Log tool calls if present
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get('name', 'unknown') if isinstance(tool_call, dict) else getattr(tool_call, 'name', 'unknown')
                    logger.debug(f"[ORCHESTRATOR] Tool called: {tool_name}")
    
    async def _invoke_react_agent(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        operation: str,
        claim_id: str
    ) -> Dict[str, Any]:
        """
        Create and invoke a react agent with the given prompts.
        
        Args:
            system_prompt: System prompt for the agent
            user_prompt: User prompt with specific task details
            operation: Name of the operation for logging (e.g., 'planning', 'replanning')
            claim_id: Claim ID for logging context
            
        Returns:
            Agent invocation result containing messages
        """
        from langgraph.prebuilt import create_react_agent
        from langchain_core.messages import HumanMessage
        
        # Debug logging
        logger.debug(f"[ORCHESTRATOR] System prompt: {system_prompt[:200]}...")
        logger.debug(f"[ORCHESTRATOR] User prompt: {user_prompt[:200]}...")
        logger.debug(f"[ORCHESTRATOR] Available tools: {[tool.name for tool in self.tools]}")
        
        # Create react agent with tools (no system prompt parameter in this version)
        agent = create_react_agent(
            self.llm,
            tools=self.tools
        )
        
        # Invoke agent with system prompt prepended to user message
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        logger.debug(f"[ORCHESTRATOR] Invoking {operation} agent for claim {claim_id}")
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=combined_prompt)]
        })
        
        # Debug logging
        self._log_agent_invocation_debug(result, operation)
        
        return result
    
    async def create_plan(self, state: Dict[str, Any]) -> List[str]:
        """
        Create an execution plan based on the current claim state.
        
        Analyzes the claim data and generates a step-by-step execution plan
        that will guide the processing workflow.
        
        Args:
            state: Current GraphState containing claim data and processing context
            
        Returns:
            Ordered list of plan steps to execute
            
        Implements Requirement 10.1: Plan creation using LLM
        """
        claim_id = self._extract_claim_id(state)
        
        logger.info(f"[ORCHESTRATOR] Creating execution plan for claim: {claim_id}")
        
        try:
            # Build prompts
            system_prompt = OrchestratorPrompts.build_plan_system_prompt()
            user_prompt = OrchestratorPrompts.build_plan_user_prompt(state)
            
            # Invoke react agent for planning
            result = await self._invoke_react_agent(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                operation="planning",
                claim_id=claim_id
            )
            
            # Extract plan from agent response
            plan_content = result["messages"][-1].content
            plan = self._parse_plan(plan_content)
            
            logger.info(f"[ORCHESTRATOR] Created plan with {len(plan)} steps for claim {claim_id}")
            logger.debug(f"[ORCHESTRATOR] Plan steps: {plan}")
            
            return plan
            
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Failed to create plan for claim {claim_id}: {e}")
            raise PlanGenerationError(
                message=f"Plan generation failed: {str(e)}",
                claim_id=claim_id,
                llm_error=e,
                retry_count=0
            ) from e
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _clean_plan_line(line: str) -> str:
        """Clean a single plan line (cached for repeated patterns)."""
        cleaned = line.strip()
        
        for prefix in OrchestratorAgent._PLAN_PREFIXES:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break
        
        return cleaned
    
    def _parse_plan(self, response: str) -> List[str]:
        """
        Parse the LLM response into a list of plan steps.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            List of plan steps
        """
        lines = response.strip().split('\n')
        
        plan = [
            self._clean_plan_line(line)
            for line in lines
            if line.strip() and len(line.strip()) > 10 and not line.strip().endswith(':')
        ]
        
        if not plan:
            logger.warning("[ORCHESTRATOR] Failed to parse plan from LLM response, using fallback")
            return self.config.get_default_plan_steps()
        
        return plan
    
    def _get_fallback_plan(self, state: Dict[str, Any]) -> List[str]:
        """
        Get a fallback plan when LLM plan generation fails.
        
        Args:
            state: Current GraphState
            
        Returns:
            Default fallback plan
        """
        logger.info("[ORCHESTRATOR] Using fallback plan")
        
        plan = self.config.get_default_plan_steps().copy()
        
        # If no documents available, remove document processing steps
        if not (state.get('shipping_documents') or state.get('damage_evidence')):
            plan = [step for step in plan if 'document' not in step.lower()]
        
        return plan
    
    def _initialize_reflection(self) -> Dict[str, Any]:
        """Initialize reflection dictionary with default values."""
        return {
            "progress_assessment": "on_track",
            "issues_identified": [],
            "recommendations": [],
            "replan_needed": False,
            "confidence_level": 0.8
        }
    
    def _assess_errors(self, state: Dict[str, Any], reflection: Dict[str, Any]) -> None:
        """Assess errors and update reflection."""
        errors = state.get('errors', [])
        if not errors:
            return
        
        reflection["issues_identified"].append(f"Encountered {len(errors)} errors during processing")
        reflection["confidence_level"] -= self.config.error_penalty * len(errors)
        
        critical_errors = [e for e in errors if e.get('critical', False)]
        if critical_errors:
            reflection["progress_assessment"] = "blocked"
            reflection["replan_needed"] = True
            reflection["recommendations"].append("Critical errors require replanning")
    
    def _assess_progress(self, state: Dict[str, Any], reflection: Dict[str, Any]) -> None:
        """Assess progress and update reflection."""
        completed_steps = state.get('completed_steps', [])
        if len(completed_steps) == 0 and state.get('current_step', 0) > 0:
            reflection["issues_identified"].append("No steps completed despite processing attempts")
            reflection["progress_assessment"] = "stalled"
            reflection["replan_needed"] = True
    
    def _assess_extraction_confidence(self, state: Dict[str, Any], reflection: Dict[str, Any]) -> None:
        """Assess extraction confidence and update reflection."""
        extraction_confidence = state.get('extraction_confidence', {})
        if not extraction_confidence:
            return
        
        low_confidence_fields = [
            k for k, v in extraction_confidence.items() 
            if v < self.config.low_confidence_threshold
        ]
        
        if low_confidence_fields:
            reflection["issues_identified"].append(
                f"Low confidence in {len(low_confidence_fields)} extracted fields"
            )
            reflection["recommendations"].append(
                "Consider human review for low-confidence extractions"
            )
            reflection["confidence_level"] -= self.config.low_confidence_field_penalty * len(low_confidence_fields)
    
    def _assess_risk_level(self, state: Dict[str, Any], reflection: Dict[str, Any]) -> None:
        """Assess risk level and update reflection."""
        risk_level = state.get('risk_level')
        if risk_level == 'high':
            reflection["issues_identified"].append("High risk level detected")
            reflection["recommendations"].append("Escalate to human review due to high risk")
            reflection["confidence_level"] -= self.config.high_risk_penalty
    
    def _assess_policy_compliance(self, state: Dict[str, Any], reflection: Dict[str, Any]) -> None:
        """Assess policy compliance and update reflection."""
        policy_violations = state.get('policy_violations', [])
        if policy_violations:
            reflection["issues_identified"].append(f"Found {len(policy_violations)} policy violations")
            reflection["recommendations"].append("Review policy violations before final decision")
            reflection["confidence_level"] -= self.config.policy_violation_penalty
    
    @staticmethod
    def _normalize_confidence(confidence: float) -> float:
        """Ensure confidence is within valid range [0.0, 1.0]."""
        return max(0.0, min(1.0, confidence))
    
    def _log_reflection_summary(self, reflection: Dict[str, Any]) -> None:
        """Log reflection summary."""
        logger.info(
            f"[ORCHESTRATOR] Reflection complete: {reflection['progress_assessment']}, "
            f"confidence: {reflection['confidence_level']:.2f}, "
            f"replan needed: {reflection['replan_needed']}"
        )
    
    async def reflect_on_progress(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on completed steps and evaluate progress.
        
        Analyzes what has been accomplished, identifies any issues,
        and determines if replanning is needed.
        
        Args:
            state: Current GraphState with completed steps and observations
            
        Returns:
            Reflection results with recommendations
            
        Implements Requirement 10.3: Reflection on progress
        """
        claim_id = self._extract_claim_id(state)
        
        logger.info(f"[ORCHESTRATOR] Reflecting on progress for claim {claim_id}")
        
        try:
            # Initialize reflection with rule-based assessment
            reflection = self._initialize_reflection()
            
            self._assess_errors(state, reflection)
            self._assess_progress(state, reflection)
            self._assess_extraction_confidence(state, reflection)
            self._assess_risk_level(state, reflection)
            self._assess_policy_compliance(state, reflection)
            
            reflection["confidence_level"] = self._normalize_confidence(reflection["confidence_level"])
            
            self._log_reflection_summary(reflection)
            
            return reflection
            
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Reflection failed for claim {claim_id}: {e}")
            raise ReflectionError(
                message=f"Reflection failed: {str(e)}",
                claim_id=claim_id,
                completed_steps=state.get('completed_steps', []),
                observations=state.get('observations', [])
            ) from e
    
    async def replan(self, state: Dict[str, Any], reflection: Dict[str, Any]) -> List[str]:
        """
        Adjust the plan based on observations and failures.
        
        Creates a new plan that addresses identified issues and adapts
        to the current processing state.
        
        Args:
            state: Current GraphState
            reflection: Reflection results from reflect_on_progress
            
        Returns:
            New execution plan
            
        Implements Requirement 10.4: Plan adaptation based on feedback
        """
        claim_id = self._extract_claim_id(state)
        
        logger.info(f"[ORCHESTRATOR] Replanning for claim {claim_id}")
        logger.debug(f"[ORCHESTRATOR] Reflection: {reflection}")
        
        try:
            # Build prompts
            system_prompt = OrchestratorPrompts.build_replan_system_prompt()
            user_prompt = OrchestratorPrompts.build_replan_user_prompt(state, reflection)
            
            # Additional debug logging for replanning context
            logger.debug(f"[ORCHESTRATOR] Reflection data: {reflection}")
            
            # Invoke react agent for replanning
            result = await self._invoke_react_agent(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                operation="replanning",
                claim_id=claim_id
            )
            
            # Extract revised plan from agent response
            plan_content = result["messages"][-1].content
            new_plan = self._parse_plan(plan_content)
            
            logger.info(f"[ORCHESTRATOR] Created revised plan with {len(new_plan)} steps")
            logger.debug(f"[ORCHESTRATOR] Revised plan: {new_plan}")
            
            return new_plan
            
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Failed to create revised plan: {e}")
            raise ReplanningError(
                message=f"Replanning failed: {str(e)}",
                claim_id=claim_id,
                original_plan=state.get('plan', []),
                reflection_data=reflection
            ) from e
    

    
    def _get_recovery_plan(self, state: Dict[str, Any], reflection: Dict[str, Any]) -> List[str]:
        """
        Get a recovery plan when replanning fails.
        
        Args:
            state: Current GraphState
            reflection: Reflection results
            
        Returns:
            Recovery plan
        """
        logger.info("[ORCHESTRATOR] Using recovery plan")
        
        completed_steps = state.get('completed_steps', [])
        plan = []
        
        # Only add steps that haven't been completed
        if self.config.STEP_VALIDATE_DATA not in completed_steps:
            plan.append("Validate claim data in Data Fabric")
        
        if self.config.STEP_DOWNLOAD_DOCUMENTS not in completed_steps and (
            state.get('shipping_documents') or state.get('damage_evidence')
        ):
            plan.append("Download available documents")
        
        if self.config.STEP_EXTRACT_DATA not in completed_steps and state.get('downloaded_documents'):
            plan.append("Extract data from downloaded documents")
        
        if self.config.STEP_ASSESS_RISK not in completed_steps:
            plan.append("Perform risk assessment")
        
        if self.config.STEP_VALIDATE_POLICY not in completed_steps:
            plan.append("Validate against policies")
        
        # If confidence is low or issues exist, escalate
        if reflection.get('confidence_level', 0) < self.config.escalation_threshold or reflection.get('issues_identified'):
            plan.append("Escalate to human review due to processing issues")
        
        plan.extend([
            "Make final decision with available information",
            "Update all systems with results"
        ])
        
        return plan
    
    def handle_step_failure(self, step_name: str, error: Exception, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a step failure and determine recovery action.
        
        Args:
            step_name: Name of the failed step
            error: Exception that occurred
            state: Current GraphState
            
        Returns:
            Recovery action recommendation
            
        Implements Requirement 10.3, 10.4: Failure handling and adaptation
        """
        claim_id = self._extract_claim_id(state)
        
        logger.warning(f"[ORCHESTRATOR] Step '{step_name}' failed for claim {claim_id}: {error}")
        
        recovery = {
            "action": "continue",  # continue, retry, skip, escalate, abort
            "reason": "",
            "retry_count": 0,
            "max_retries": self.config.max_step_retries
        }
        
        # Check if this step has failed before
        failed_actions = state.get('failed_actions', [])
        previous_failures = [f for f in failed_actions if f.get('step') == step_name]
        recovery["retry_count"] = len(previous_failures)
        
        # Determine recovery action based on step type and failure count
        if recovery["retry_count"] >= recovery["max_retries"]:
            # Too many retries, skip or escalate
            if step_name in self.config.critical_steps:
                # Critical steps - escalate
                recovery["action"] = "escalate"
                recovery["reason"] = f"Critical step '{step_name}' failed after {recovery['retry_count']} retries"
            else:
                # Non-critical steps - skip and continue
                recovery["action"] = "skip"
                recovery["reason"] = f"Skipping '{step_name}' after {recovery['retry_count']} failed attempts"
        else:
            # Retry with backoff
            recovery["action"] = "retry"
            recovery["reason"] = f"Retrying '{step_name}' (attempt {recovery['retry_count'] + 1}/{recovery['max_retries']})"
        
        logger.info(
            f"[ORCHESTRATOR] Recovery action for '{step_name}': {recovery['action']} - {recovery['reason']}"
        )
        
        return recovery
