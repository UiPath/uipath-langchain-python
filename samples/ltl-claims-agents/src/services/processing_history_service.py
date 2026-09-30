"""Processing History Service for tracking claim processing events in Data Fabric."""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from uipath import UiPath

from ..config.settings import settings
from ..utils.retry import retry_with_backoff, RetryConfig


logger = logging.getLogger(__name__)


class ProcessingHistoryServiceError(Exception):
    """Custom exception for Processing History service errors."""
    pass


class ProcessingHistoryService:
    """
    Service for recording and managing claim processing history in Data Fabric.
    
    This service provides a clean separation of concerns for all processing history
    operations, using the LTLProcessingHistory entity in Data Fabric.
    
    Usage:
        # With dependency injection (recommended)
        async with UiPathService() as uipath_service:
            history_service = ProcessingHistoryService(uipath_service._client)
            await history_service.record_processing_started(claim_id, data)
        
        # Standalone usage
        async with ProcessingHistoryService.create() as history_service:
            await history_service.record_processing_started(claim_id, data)
    """
    
    def __init__(self, uipath_client: UiPath):
        """
        Initialize the Processing History Service.
        
        Args:
            uipath_client: UiPath SDK client instance (required for proper resource management)
        """
        self._client = uipath_client
        self._owns_client = False  # Track if we created the client
        
        # Configure retry behavior
        self._retry_config = RetryConfig(
            max_attempts=3,
            initial_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=True
        )
        
        # Transient errors that should trigger retry
        self._retryable_errors = (
            ConnectionError,
            TimeoutError,
        )
    
    @classmethod
    async def create(cls) -> 'ProcessingHistoryService':
        """
        Factory method for standalone usage.
        
        Creates a new UiPath client that will be cleaned up when the service is closed.
        
        Returns:
            ProcessingHistoryService instance
        """
        client = UiPath()
        service = cls(client)
        service._owns_client = True
        return service
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper cleanup."""
        if self._owns_client and self._client:
            # Cleanup client if we created it
            try:
                # UiPath SDK handles cleanup automatically
                self._client = None
            except Exception as e:
                logger.warning(f"Error during client cleanup: {e}")
        return False  # Don't suppress exceptions
    
    async def create_history_entry(
        self,
        claim_id: str,
        event_type: str,
        description: str,
        data: Optional[Dict[str, Any]] = None,
        status: str = "completed",
        agent_id: str = "Claims_Agent"
    ) -> str:
        """
        Create a processing history entry in Data Fabric.
        
        Args:
            claim_id: The claim ID this entry relates to
            event_type: Type of event (e.g., "processing_started", "step_completed")
            description: Human-readable description of the event
            data: Optional additional event data
            status: Event status (default: "completed")
            agent_id: ID of the agent that performed the action
            
        Returns:
            The ID of the created history entry
            
        Raises:
            ProcessingHistoryServiceError: If the operation fails
        """
        try:
            logger.debug(f"Creating history entry for claim {claim_id}: {event_type}")
            
            # Prepare history entry data for LTLProcessingHistory entity
            # Schema fields: claimId (UNIQUEIDENTIFIER), eventType, description, agentId, data, status
            
            # Add optional data field - include timestamp in the data field since there's no timestamp column
            data_with_timestamp = data.copy() if data else {}
            data_with_timestamp["timestamp"] = datetime.now(timezone.utc).isoformat()
            
            # Convert to string and truncate to fit 200 character limit
            data_str = str(data_with_timestamp)
            if len(data_str) > 195:  # Leave room for ellipsis
                data_str = data_str[:195] + "..."
            
            # Truncate description to fit 200 character limit
            description_truncated = description[:200] if len(description) > 200 else description
            
            # Create a simple namespace object (SDK expects objects with __dict__, not plain dicts)
            from types import SimpleNamespace
            history_record = SimpleNamespace(
                claimId=claim_id,  # Foreign key to LTLClaims (UNIQUEIDENTIFIER)
                eventType=event_type[:200] if len(event_type) > 200 else event_type,
                description=description_truncated,
                agentId=agent_id[:200] if len(agent_id) > 200 else agent_id,
                status=status[:200] if len(status) > 200 else status,
                data=data_str
            )
            
            # Insert record into Data Fabric - simple SDK call without retry
            # Use entity ID from settings
            entity_id = settings.uipath_processing_history_entity
            
            result = await self._client.entities.insert_records_async(
                entity_key=entity_id,
                records=[history_record]
            )
            
            # Handle different response types
            entry_id = "unknown"
            if result:
                if hasattr(result, 'successful_records') and result.successful_records:
                    entry_id = result.successful_records[0]
                elif isinstance(result, dict) and 'successful_records' in result:
                    entry_id = result['successful_records'][0] if result['successful_records'] else "unknown"
                elif isinstance(result, dict) and 'Id' in result:
                    entry_id = result['Id']
            
            logger.info(f"Created history entry {entry_id} for claim {claim_id}")
            return entry_id
            
        except Exception as e:
            logger.error(f"Failed to create history entry for claim {claim_id}: {str(e)}")
            raise ProcessingHistoryServiceError(f"Failed to create history entry: {str(e)}")
    
    async def record_processing_started(
        self,
        claim_id: str,
        claim_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record that processing has started for a claim.
        
        Args:
            claim_id: The claim ID
            claim_data: Optional claim data to include in the record
            
        Raises:
            Does not raise exceptions - logs errors instead to prevent processing failures
        """
        try:
            description = f"Agent started processing claim {claim_id}"
            data = {"claim_id": claim_id}
            if claim_data:
                data["claim_data"] = claim_data
            
            await self.create_history_entry(
                claim_id=claim_id,
                event_type="processing_started",
                description=description,
                data=data,
                status="in_progress"
            )
        except Exception as e:
            logger.error(f"Failed to record processing started for claim {claim_id}: {str(e)}")
    
    async def record_step_completed(
        self,
        claim_id: str,
        step_name: str,
        step_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record that a processing step has been completed.
        
        Args:
            claim_id: The claim ID
            step_name: Name of the completed step
            step_data: Optional data about the step execution
            
        Raises:
            Does not raise exceptions - logs errors instead
        """
        try:
            description = f"Completed step: {step_name}"
            data = {"step_name": step_name}
            if step_data:
                data["step_data"] = step_data
            
            await self.create_history_entry(
                claim_id=claim_id,
                event_type="step_completed",
                description=description,
                data=data,
                status="completed"
            )
        except Exception as e:
            logger.error(f"Failed to record step completed for claim {claim_id}: {str(e)}")
    
    async def record_decision_made(
        self,
        claim_id: str,
        decision: str,
        confidence: float,
        reasoning: str,
        reasoning_steps: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Record that a decision has been made for a claim.
        
        Args:
            claim_id: The claim ID
            decision: The decision made (e.g., "approved", "denied")
            confidence: Confidence score for the decision
            reasoning: Reasoning behind the decision
            reasoning_steps: Optional detailed reasoning steps
            
        Raises:
            Does not raise exceptions - logs errors instead
        """
        try:
            description = f"Decision made: {decision} (confidence: {confidence:.2f})"
            data = {
                "decision": decision,
                "confidence": confidence,
                "reasoning": reasoning
            }
            if reasoning_steps:
                data["reasoning_steps"] = reasoning_steps
            
            await self.create_history_entry(
                claim_id=claim_id,
                event_type="decision_made",
                description=description,
                data=data,
                status="completed"
            )
        except Exception as e:
            logger.error(f"Failed to record decision made for claim {claim_id}: {str(e)}")
    
    async def record_escalation(
        self,
        claim_id: str,
        reason: str,
        action_center_task_id: Optional[str] = None,
        escalation_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record that a claim has been escalated to human review.
        
        Args:
            claim_id: The claim ID
            reason: Reason for escalation
            action_center_task_id: Optional Action Center task ID
            escalation_data: Optional additional escalation data
            
        Raises:
            Does not raise exceptions - logs errors instead
        """
        try:
            description = f"Escalated to human review: {reason}"
            data = {"reason": reason}
            if action_center_task_id:
                data["action_center_task_id"] = action_center_task_id
            if escalation_data:
                data["escalation_data"] = escalation_data
            
            await self.create_history_entry(
                claim_id=claim_id,
                event_type="escalated_to_human",
                description=description,
                data=data,
                status="pending"
            )
        except Exception as e:
            logger.error(f"Failed to record escalation for claim {claim_id}: {str(e)}")
    
    async def record_human_decision(
        self,
        claim_id: str,
        human_decision: str,
        action_center_task_id: Optional[str] = None,
        reviewer_comments: Optional[str] = None
    ) -> None:
        """
        Record that a human decision has been received.
        
        Args:
            claim_id: The claim ID
            human_decision: The decision made by the human reviewer
            action_center_task_id: Optional Action Center task ID
            reviewer_comments: Optional comments from the reviewer
            
        Raises:
            Does not raise exceptions - logs errors instead
        """
        try:
            description = f"Human decision received: {human_decision}"
            data = {"human_decision": human_decision}
            if action_center_task_id:
                data["action_center_task_id"] = action_center_task_id
            if reviewer_comments:
                data["reviewer_comments"] = reviewer_comments
            
            await self.create_history_entry(
                claim_id=claim_id,
                event_type="human_decision_received",
                description=description,
                data=data,
                status="completed"
            )
        except Exception as e:
            logger.error(f"Failed to record human decision for claim {claim_id}: {str(e)}")
    
    async def record_error(
        self,
        claim_id: str,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
        step_name: Optional[str] = None
    ) -> None:
        """
        Record that an error occurred during processing.
        
        Args:
            claim_id: The claim ID
            error_message: Error message
            error_details: Optional detailed error information
            step_name: Optional name of the step where error occurred
            
        Raises:
            Does not raise exceptions - logs errors instead
        """
        try:
            description = f"Error occurred: {error_message}"
            if step_name:
                description = f"Error in {step_name}: {error_message}"
            
            data = {"error_message": error_message}
            if error_details:
                data["error_details"] = error_details
            if step_name:
                data["step_name"] = step_name
            
            await self.create_history_entry(
                claim_id=claim_id,
                event_type="error_occurred",
                description=description,
                data=data,
                status="failed"
            )
        except Exception as e:
            logger.error(f"Failed to record error for claim {claim_id}: {str(e)}")
    
    async def record_processing_completed(
        self,
        claim_id: str,
        final_status: str,
        processing_duration_seconds: Optional[float] = None,
        summary_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record that processing has been completed for a claim.
        
        Args:
            claim_id: The claim ID
            final_status: Final processing status
            processing_duration_seconds: Optional processing duration in seconds
            summary_data: Optional summary data about the processing
            
        Raises:
            Does not raise exceptions - logs errors instead
        """
        try:
            description = f"Processing completed with status: {final_status}"
            data = {"final_status": final_status}
            if processing_duration_seconds is not None:
                data["processing_duration_seconds"] = processing_duration_seconds
                description += f" (duration: {processing_duration_seconds:.2f}s)"
            if summary_data:
                data["summary_data"] = summary_data
            
            await self.create_history_entry(
                claim_id=claim_id,
                event_type="processing_completed",
                description=description,
                data=data,
                status="completed"
            )
        except Exception as e:
            logger.error(f"Failed to record processing completed for claim {claim_id}: {str(e)}")
    
    async def get_claim_history(
        self,
        claim_id: str,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve processing history for a specific claim.
        
        Args:
            claim_id: The claim ID
            event_type: Optional filter by event type
            limit: Maximum number of records to retrieve
            
        Returns:
            List of history entries
            
        Raises:
            ProcessingHistoryServiceError: If the operation fails
        """
        try:
            logger.debug(f"Retrieving history for claim {claim_id}")
            
            # Get all records from LTLProcessingHistory entity - simple SDK call
            # Use entity ID from settings
            entity_id = settings.uipath_processing_history_entity
            
            records = await self._client.entities.list_records_async(
                entity_key=entity_id,
                start=0,
                limit=limit
            )
            
            # Filter by claim_id and optionally by event_type
            history = []
            for record in records:
                # Handle EntityRecord objects from SDK
                if hasattr(record, 'claimId'):
                    # It's an EntityRecord object, access attributes directly
                    record_claim_id = getattr(record, 'claimId', None)
                    
                    if str(record_claim_id) == str(claim_id):
                        record_event_type = getattr(record, 'eventType', None)
                        if event_type is None or record_event_type == event_type:
                            # Convert to dict for easier handling
                            history.append({
                                'id': getattr(record, 'id', None),
                                'claimId': record_claim_id,
                                'eventType': record_event_type,
                                'description': getattr(record, 'description', None),
                                'agentId': getattr(record, 'agentId', None),
                                'data': getattr(record, 'data', None),
                                'status': getattr(record, 'status', None),
                                'CreateTime': getattr(record, 'CreateTime', None),
                                'UpdateTime': getattr(record, 'UpdateTime', None)
                            })
            
            logger.info(f"Retrieved {len(history)} history entries for claim {claim_id}")
            return history
            
        except Exception as e:
            logger.error(f"Failed to retrieve history for claim {claim_id}: {str(e)}")
            raise ProcessingHistoryServiceError(f"Failed to retrieve claim history: {str(e)}")
