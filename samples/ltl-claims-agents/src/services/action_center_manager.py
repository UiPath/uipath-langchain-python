"""Action Center Manager for human-in-the-loop review and validation."""

import logging
from typing import Dict, List, Optional, Any

from ..utils.field_normalizer import FieldNormalizer

logger = logging.getLogger(__name__)

# Constants for reasoning summary formatting (can be overridden via settings)
DEFAULT_MAX_REASONING_STEPS = 5
DEFAULT_MAX_THOUGHT_LENGTH = 100


class ActionCenterManager:
    """
    Manages Action Center task creation for human-in-the-loop workflows.
    
    Uses UiPath CreateAction model with interrupt() for agent escalation.
    """
    
    def __init__(self, uipath_service, settings):
        """
        Initialize Action Center Manager.
        
        Args:
            uipath_service: UiPathService instance
            settings: Settings instance with Action Center configuration
        """
        self.uipath_service = uipath_service
        self.settings = settings
        self.normalizer = FieldNormalizer()
        
        # Action Center configuration from settings
        self.app_name = getattr(settings, 'action_center_app_name', 'ClaimsTrackingApp')
        self.folder_path = getattr(settings, 'action_center_folder_path', 'Agents')
        self.assignee = getattr(settings, 'action_center_assignee', 'Claims_Reviewers')
        
        # Reasoning summary configuration
        self.max_reasoning_steps = getattr(settings, 'max_reasoning_steps', DEFAULT_MAX_REASONING_STEPS)
        self.max_thought_length = getattr(settings, 'max_thought_length', DEFAULT_MAX_THOUGHT_LENGTH)
        
        logger.info(
            f"ActionCenterManager initialized - "
            f"App: {self.app_name}, Folder: {self.folder_path}, "
            f"Assignee: {self.assignee}"
        )
    
    def _format_reasoning_summary(self, reasoning_steps: List[Dict[str, Any]]) -> str:
        """
        Format reasoning steps into a human-readable summary.
        
        Args:
            reasoning_steps: List of reasoning step dictionaries
            
        Returns:
            Formatted reasoning summary string
        """
        if not reasoning_steps:
            return "No reasoning steps available"
        
        return "\n".join([
            f"Step {step.get('step_number', i+1)}: {step.get('thought', 'N/A')[:self.max_thought_length]}"
            for i, step in enumerate(reasoning_steps[-self.max_reasoning_steps:])
        ])
    
    def _format_task_title(self, claim_type: str, claim_amount: float, confidence_score: float) -> str:
        """
        Format a descriptive task title for Action Center.
        
        Args:
            claim_type: Type of claim
            claim_amount: Claim amount
            confidence_score: Agent confidence score
            
        Returns:
            Formatted task title
        """
        return (
            f"Review {claim_type} - "
            f"${claim_amount:,.2f} - "
            f"Confidence: {confidence_score:.0%}"
        )
    
    async def _create_action(self, title: str, data: Dict[str, Any]) -> Any:
        """
        Create an action in Action Center using the UiPath SDK.
        
        This method properly delegates to the SDK through the service wrapper,
        ensuring authentication and error handling are properly managed.
        
        Args:
            title: Action title
            data: Action data payload
            
        Returns:
            Created action object
            
        Raises:
            RuntimeError: If action creation fails
        """
        # Ensure service is authenticated
        if not self.uipath_service._authenticated:
            await self.uipath_service.authenticate()
        
        # Create action through SDK
        action = await self.uipath_service._client.actions.create_async(
            title=title,
            data=data,
            app_name=self.app_name,
            app_folder_path=self.folder_path,
            assignee=self.assignee,
            app_version=1
        )
        
        return action
    
    async def create_review_task(
        self,
        claim_id: str,
        claim_data: Dict[str, Any],
        confidence_score: float,
        reasoning_steps: List[Dict[str, Any]],
        extracted_data: Dict[str, Any],
        risk_factors: List[str]
    ) -> Dict[str, Any]:
        """
        Create an Action Center task using UiPath SDK Actions service.
        
        This method creates an action in Action Center for human review.
        The action will be assigned to the configured assignee/group.
        
        Args:
            claim_id: Unique identifier for the claim
            claim_data: Complete claim data dictionary
            confidence_score: Agent's confidence score (0.0-1.0)
            reasoning_steps: List of agent reasoning steps
            extracted_data: Data extracted from documents
            risk_factors: List of identified risk factors
            
        Returns:
            Action response with action_key and other details
        """
        try:
            logger.info(f"[ACTION] Creating Action Center task for claim: {claim_id}")
            
            # Normalize claim data to PascalCase format using shared utility
            normalized_data = self.normalizer.standard_to_queue(claim_data)
            
            logger.debug(f"[ACTION] Normalized claim data keys: {list(normalized_data.keys())}")
            
            # Format reasoning summary for display
            reasoning_summary = self._format_reasoning_summary(reasoning_steps)
            
            # Prepare task data matching ClaimsTrackingApp input structure
            # All fields MUST be in PascalCase as expected by Action Center apps
            task_data = {
                "ClaimType": normalized_data.get("ClaimType", "Unknown"),
                "ClaimStatus": "Pending Review",
                "ClaimAmount": self.normalizer.safe_float(normalized_data.get("ClaimAmount", 0)),
                "CustomerName": normalized_data.get("CustomerName", "Unknown"),
                "CarrierName": normalized_data.get("Carrier", "Unknown"),
                "ShipmentId": normalized_data.get("ShipmentID", ""),
                "BolNumber": normalized_data.get("ShipmentID", ""),
                "ProNumber": normalized_data.get("ProNumber", ""),
                "ShipmentRoute": normalized_data.get("ShipmentRoute", ""),
                "DeclaredValue": self.normalizer.safe_float(normalized_data.get("DeclaredValue", 0)),
                "ShipmentWeight": self.normalizer.safe_float(normalized_data.get("ShipmentWeight", 0)),
                "NumberOfPieces": self.normalizer.safe_int(normalized_data.get("NumberOfPieces", 0)),
                "AgentReasoningSummary": reasoning_summary
            }
            
            logger.info(f"[ACTION] Task data prepared: {task_data}")
            
            # Create task title
            claim_amount = self.normalizer.safe_float(normalized_data.get("ClaimAmount", 0))
            claim_type = normalized_data.get("ClaimType", "Claim")
            task_title = self._format_task_title(claim_type, claim_amount, confidence_score)
            
            logger.info(
                f"[ACTION] Creating action - "
                f"App: {self.app_name}, Title: {task_title}"
            )
            
            # Create action using UiPath SDK through service wrapper
            action = await self._create_action(
                title=task_title,
                data=task_data
            )
            
            logger.info(
                f"[OK] Action created successfully - "
                f"Action Key: {action.key}, Claim: {claim_id}"
            )
            
            return {
                "action_key": str(action.key),
                "action_title": task_title,
                "claim_id": claim_id,
                "status": "pending_review",
                "created_at": action.created_at if hasattr(action, 'created_at') else None
            }
            
        except ValueError as e:
            logger.error(f"[ERROR] Invalid data for Action Center task (claim {claim_id}): {e}")
            raise ValueError(f"Invalid task data: {e}") from e
        except ConnectionError as e:
            logger.error(f"[ERROR] Connection failed while creating Action Center task (claim {claim_id}): {e}")
            raise ConnectionError(f"Failed to connect to Action Center: {e}") from e
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error creating Action Center task for claim {claim_id}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to create Action Center task: {e}") from e
