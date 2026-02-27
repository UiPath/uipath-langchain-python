"""
Queue helper functions for retrieving and processing queue items.

This module provides utility functions for working with UiPath Orchestrator Queues,
including retrieving queue items and mapping queue data to GraphState format.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..config.constants import FieldMappingConstants
from ..services.uipath_service import UiPathService, UiPathServiceError


logger = logging.getLogger(__name__)


async def get_next_claim_from_queue(
    uipath_service: UiPathService,
    queue_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Retrieve the next claim from the UiPath queue and map to GraphState format.
    
    This function uses the proper UiPath StartTransaction API to:
    1. Retrieve and lock the next available queue item as a transaction
    2. Extract claim data from the specific_content field
    3. Map queue field names to GraphState field names using FieldMappingConstants
    4. Return a GraphState-compatible dictionary with transaction_key
    
    The StartTransaction API ensures proper transaction locking, preventing multiple
    processors from handling the same item simultaneously.
    
    Args:
        uipath_service: Authenticated UiPathService instance
        queue_name: Optional queue name (uses configured default if not provided)
        
    Returns:
        Dictionary with GraphState-compatible fields and transaction_key, or None if queue is empty
        
    Raises:
        UiPathServiceError: If queue retrieval fails
        
    Example:
        async with UiPathService() as uipath_service:
            claim_data = await get_next_claim_from_queue(uipath_service, "LTL_Claims_Processing")
            if claim_data:
                state = GraphState(**claim_data)
    """
    try:
        from ..config.settings import settings
        
        # Use configured queue name if not provided
        queue_name = queue_name or settings.queue_name
        
        logger.info(f"Starting transaction for next claim from queue: {queue_name}")
        
        # Start transaction using proper API - this locks the item for processing
        queue_item = await uipath_service.start_transaction(
            queue_name=queue_name
        )
        
        # Handle empty queue gracefully
        if not queue_item:
            logger.info("Queue is empty, no claims to process")
            return None
        
        # Extract specific_content which contains the claim data
        specific_content = queue_item.get('specific_content', {})
        
        if not specific_content:
            logger.warning(f"Queue item {queue_item.get('id')} has no specific_content")
            return None
        
        # Initialize result dictionary with queue metadata
        result = {
            'transaction_key': queue_item.get('transaction_key'),
            'queue_name': queue_item.get('queue_name')
        }
        
        # Map queue fields to GraphState fields using FieldMappingConstants
        import json
        
        for queue_field, standard_field in FieldMappingConstants.QUEUE_TO_STANDARD.items():
            if queue_field in specific_content:
                value = specific_content[queue_field]
                
                # Handle special cases for field types
                if standard_field in ['shipping_documents', 'damage_evidence']:
                    # These fields are stored as JSON strings in the queue
                    # Deserialize them back to lists
                    if isinstance(value, str):
                        try:
                            value = json.loads(value)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON for {standard_field}, using empty list")
                            value = []
                    elif not isinstance(value, list):
                        value = [value] if value else []
                elif standard_field == 'claim_amount':
                    # Ensure claim_amount is a float
                    try:
                        value = float(value) if value is not None else None
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid claim_amount value: {value}, setting to None")
                        value = None
                
                result[standard_field] = value
        
        # Also include any fields that are already in standard format
        for standard_field in FieldMappingConstants.QUEUE_TO_STANDARD.values():
            if standard_field in specific_content and standard_field not in result:
                result[standard_field] = specific_content[standard_field]
        
        logger.info(
            f"Successfully retrieved claim from queue: "
            f"claim_id={result.get('claim_id')}, "
            f"transaction_key={result.get('transaction_key')}"
        )
        
        return result
        
    except UiPathServiceError as e:
        logger.error(f"Failed to retrieve claim from queue: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error retrieving claim from queue: {e}")
        raise UiPathServiceError(f"Failed to retrieve claim from queue: {str(e)}")


async def get_multiple_claims_from_queue(
    uipath_service: UiPathService,
    queue_name: Optional[str] = None,
    max_items: int = 10
) -> List[Dict[str, Any]]:
    """
    Retrieve multiple claims from the UiPath queue.
    
    This is a batch version of get_next_claim_from_queue() for processing
    multiple queue items at once.
    
    Args:
        uipath_service: Authenticated UiPathService instance
        queue_name: Optional queue name (uses configured default if not provided)
        max_items: Maximum number of items to retrieve (default: 10)
        
    Returns:
        List of dictionaries with GraphState-compatible fields
        
    Raises:
        UiPathServiceError: If queue retrieval fails
    """
    try:
        logger.info(f"Retrieving up to {max_items} claims from queue: {queue_name or 'default'}")
        
        # Get queue items using existing UiPathService method
        queue_items = await uipath_service.get_queue_items(
            queue_name=queue_name,
            max_items=max_items
        )
        
        # Handle empty queue gracefully
        if not queue_items or len(queue_items) == 0:
            logger.info("Queue is empty, no claims to process")
            return []
        
        results = []
        
        # Process each queue item
        for queue_item in queue_items:
            # Extract specific_content which contains the claim data
            specific_content = queue_item.get('specific_content', {})
            
            if not specific_content:
                logger.warning(f"Queue item {queue_item.get('id')} has no specific_content, skipping")
                continue
            
            # Initialize result dictionary with queue metadata
            result = {
                'transaction_key': queue_item.get('transaction_key'),
                'queue_name': queue_item.get('queue_name')
            }
            
            # Map queue fields to GraphState fields using FieldMappingConstants
            import json
            
            for queue_field, standard_field in FieldMappingConstants.QUEUE_TO_STANDARD.items():
                if queue_field in specific_content:
                    value = specific_content[queue_field]
                    
                    # Handle special cases for field types
                    if standard_field in ['shipping_documents', 'damage_evidence']:
                        # These fields are stored as JSON strings in the queue
                        # Deserialize them back to lists
                        if isinstance(value, str):
                            try:
                                value = json.loads(value)
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse JSON for {standard_field}, using empty list")
                                value = []
                        elif not isinstance(value, list):
                            value = [value] if value else []
                    elif standard_field == 'claim_amount':
                        # Ensure claim_amount is a float
                        try:
                            value = float(value) if value is not None else None
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid claim_amount value: {value}, setting to None")
                            value = None
                    
                    result[standard_field] = value
            
            # Also include any fields that are already in standard format
            for standard_field in FieldMappingConstants.QUEUE_TO_STANDARD.values():
                if standard_field in specific_content and standard_field not in result:
                    result[standard_field] = specific_content[standard_field]
            
            results.append(result)
        
        logger.info(f"Successfully retrieved {len(results)} claims from queue")
        
        return results
        
    except UiPathServiceError as e:
        logger.error(f"Failed to retrieve claims from queue: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error retrieving claims from queue: {e}")
        raise UiPathServiceError(f"Failed to retrieve claims from queue: {str(e)}")
