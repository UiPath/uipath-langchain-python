"""
Data Fabric Tool for UiPath Data Fabric operations.
Uses actual UiPath SDK for real Data Fabric operations with proper @tool decorator.
"""

import logging
import json
import asyncio
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from uipath.tracing import traced

logger = logging.getLogger(__name__)

# Import settings for configuration
from ..config.settings import settings

# Global UiPath service instance
_uipath_service = None

# Entity name to ID mapping from settings
ENTITY_NAME_TO_ID = {
    "LTLClaims": settings.uipath_claims_entity,
    "LTLShipments": settings.uipath_shipments_entity,
    "LTLProcessingHistory": settings.uipath_processing_history_entity
}

@traced(name="map_entity_name_to_id", run_type="utility")
def _get_entity_id(entity_key: str) -> str:
    """Convert entity name to entity ID (UUID).
    
    Args:
        entity_key: Entity name or UUID
        
    Returns:
        Entity UUID string
    """
    # If it's already a UUID, return it
    if len(entity_key) == 36 and '-' in entity_key:
        return entity_key
    
    # Otherwise, look it up in the mapping
    entity_id = ENTITY_NAME_TO_ID.get(entity_key)
    if entity_id:
        logger.info(f"Mapped entity name '{entity_key}' to ID '{entity_id}'")
        return entity_id
    
    # If not found, return the original (might be an ID we don't know about)
    logger.warning(f"Entity '{entity_key}' not found in mapping, using as-is")
    return entity_key

@traced(name="get_uipath_data_fabric_service", run_type="setup")
async def _get_uipath_service():
    """Get UiPath service instance.
    
    Returns:
        Initialized UiPath SDK instance
        
    Raises:
        ImportError: If UiPath SDK is not available
        Exception: If service initialization fails
    """
    global _uipath_service
    if _uipath_service is None:
        try:
            # Import UiPath service
            from uipath import UiPath
            
            # Initialize with environment variables
            _uipath_service = UiPath()
            logger.info("UiPath Data Fabric service initialized")
        except ImportError:
            logger.error("UiPath SDK not available")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize UiPath service: {e}")
            raise
    
    return _uipath_service


class DataFabricInput(BaseModel):
    """Input schema for Data Fabric operations."""
    operation: str = Field(description="Operation to perform (get_records, get_claim, get_shipment, insert_record, update_record, log_agent_action)")
    entity_key: str = Field(description="Data Fabric entity name (LTLClaims, LTLShipments, LTLProcessingHistory)")
    record_data: Optional[Dict[str, Any]] = Field(default=None, description="Data to insert or update")
    record_id: Optional[str] = Field(default=None, description="ID of record to update")
    claim_id: Optional[str] = Field(default=None, description="Claim ID to retrieve or associate")
    shipment_id: Optional[str] = Field(default=None, description="Shipment ID to retrieve")
    start: int = Field(default=0, description="Starting index for pagination")
    limit: int = Field(default=100, description="Maximum number of records to return")


@tool
@traced(name="query_data_fabric", run_type="tool")
async def query_data_fabric(
    operation: str,
    entity_key: str,
    record_data: Optional[Dict[str, Any]] = None,
    record_id: Optional[str] = None,
    claim_id: Optional[str] = None,
    shipment_id: Optional[str] = None,
    start: int = 0,
    limit: int = 100
) -> str:
    """Interact with UiPath Data Fabric entities for claims and shipment data.
    
    This tool provides access to UiPath Data Fabric for storing and retrieving structured data
    like claims, shipments, and processing history. Use this for querying claim records, shipment
    information, and logging agent actions. This is ONLY for structured data entities, NOT for
    documents (documents are in storage buckets).
    
    Supported Operations:
        - get_records: List all records with pagination
        - get_claim: Get specific claim by claim_id (searches LTLClaims entity)
        - get_shipment: Get specific shipment by shipment_id (searches LTLShipments entity)
        - insert_record: Insert new record into entity
        - update_record: Update existing record by record_id
        - log_agent_action: Log agent action to LTLProcessingHistory (auto-creates record)
    
    Args:
        operation: The operation to perform (see supported operations above)
        entity_key: The Data Fabric entity name ('LTLClaims', 'LTLShipments', 'LTLProcessingHistory')
        record_data: Data to insert or update (required for insert/update/log operations)
        record_id: ID of record to update (required for update operations)
        claim_id: Claim ID to retrieve or associate (for get_claim, log_agent_action)
        shipment_id: Shipment ID to retrieve (for get_shipment)
        start: Starting index for pagination (default: 0)
        limit: Maximum number of records to return (default: 100)
    
    Returns:
        JSON string containing:
            - success: Boolean indicating operation success
            - operation: The operation performed
            - entity_key: The entity accessed
            - records/claim/shipment: Retrieved data (operation-dependent)
            - error: Error message if operation failed
    
    Examples:
        Get specific claim:
            {"operation": "get_claim", "entity_key": "LTLClaims", "claim_id": "CLM-2024-001"}
        
        Log agent action:
            {"operation": "log_agent_action", "entity_key": "LTLProcessingHistory", 
             "claim_id": "CLM-2024-001", "record_data": {"action": "document_extracted"}}
        
        Update claim status:
            {"operation": "update_record", "entity_key": "LTLClaims", 
             "record_id": "uuid-here", "record_data": {"Status": "Approved"}}
    
    Note: For document processing, use download_multiple_documents and extract_documents_batch tools.
    """
    try:
        logger.info(f"Data Fabric operation: {operation} on {entity_key}")
        
        # Convert entity name to entity ID
        entity_id = _get_entity_id(entity_key)
        
        service = await _get_uipath_service()
        
        if operation == "get_claim":
            # Get specific claim by ID from LTLClaims entity
            if not claim_id:
                result = {"success": False, "error": "claim_id required for get_claim operation"}
            else:
                # Use LTLClaims entity if not specified
                if entity_key not in ["LTLClaims", settings.uipath_claims_entity]:
                    logger.warning(f"get_claim operation should use LTLClaims entity, got {entity_key}")
                    entity_id = _get_entity_id("LTLClaims")
                
                # Get all records and search for the claim by ClaimId field
                records = await service.entities.list_records_async(
                    entity_key=entity_id,
                    start=0,
                    limit=1000  # Get enough records to find the claim
                )
                
                claim_found = None
                if records:
                    for record in records:
                        record_data = record.data if hasattr(record, 'data') else record.__dict__
                        # Check multiple possible ID fields
                        record_claim_id = (
                            record_data.get('ClaimId') or 
                            record_data.get('claimId') or
                            record_data.get('Id') or 
                            record_data.get('id')
                        )
                        if str(record_claim_id) == str(claim_id):
                            claim_found = record_data
                            break
                
                if claim_found:
                    result = {
                        "success": True,
                        "operation": operation,
                        "entity_key": entity_key,
                        "claim": claim_found,
                        "message": f"Found claim {claim_id} in LTLClaims entity"
                    }
                else:
                    result = {
                        "success": False,
                        "operation": operation,
                        "entity_key": entity_key,
                        "error": f"Claim {claim_id} not found in LTLClaims entity",
                        "claim": None
                    }
        
        elif operation == "get_shipment":
            # Get specific shipment by ID from LTLShipments entity
            if not shipment_id:
                result = {"success": False, "error": "shipment_id required for get_shipment operation"}
            else:
                # Use LTLShipments entity if not specified
                if entity_key not in ["LTLShipments", settings.uipath_shipments_entity]:
                    logger.warning(f"get_shipment operation should use LTLShipments entity, got {entity_key}")
                    entity_id = _get_entity_id("LTLShipments")
                
                # Get all records and search for the shipment
                records = await service.entities.list_records_async(
                    entity_key=entity_id,
                    start=0,
                    limit=1000
                )
                
                shipment_found = None
                if records:
                    for record in records:
                        record_data = record.data if hasattr(record, 'data') else record.__dict__
                        record_shipment_id = (
                            record_data.get('ShipmentId') or 
                            record_data.get('shipmentId') or
                            record_data.get('Id') or 
                            record_data.get('id')
                        )
                        if str(record_shipment_id) == str(shipment_id):
                            shipment_found = record_data
                            break
                
                if shipment_found:
                    result = {
                        "success": True,
                        "operation": operation,
                        "entity_key": entity_key,
                        "shipment": shipment_found,
                        "message": f"Found shipment {shipment_id} in LTLShipments entity"
                    }
                else:
                    result = {
                        "success": False,
                        "operation": operation,
                        "entity_key": entity_key,
                        "error": f"Shipment {shipment_id} not found in LTLShipments entity",
                        "shipment": None
                    }
        
        elif operation == "log_agent_action":
            # Log agent action to LTLProcessingHistory entity
            if not claim_id:
                result = {"success": False, "error": "claim_id required for log_agent_action operation"}
            elif not record_data:
                result = {"success": False, "error": "record_data required for log_agent_action operation"}
            else:
                # Use LTLProcessingHistory entity
                history_entity_id = _get_entity_id("LTLProcessingHistory")
                
                # Prepare processing history record with required fields
                history_record = {
                    "ClaimId": claim_id,
                    "Timestamp": datetime.utcnow().isoformat(),
                    "Action": record_data.get("action", "agent_action"),
                    "Actor": "AI Agent",
                    "Details": json.dumps(record_data.get("details", {})),
                    "Status": record_data.get("status", "completed"),
                    "Confidence": record_data.get("confidence", 0.0),
                    **record_data  # Include any additional fields
                }
                
                # Insert the history record
                insert_result = await service.entities.insert_records_async(
                    entity_key=history_entity_id,
                    records=[history_record]
                )
                
                # Extract record ID from result
                history_record_id = None
                if insert_result:
                    if hasattr(insert_result, 'successful_records') and insert_result.successful_records:
                        history_record_id = insert_result.successful_records[0]
                    elif isinstance(insert_result, dict):
                        history_record_id = insert_result.get("id")
                
                result = {
                    "success": True,
                    "operation": operation,
                    "entity_key": "LTLProcessingHistory",
                    "claim_id": claim_id,
                    "action_logged": True,
                    "record_id": history_record_id,
                    "message": f"Logged agent action for claim {claim_id} to processing history"
                }
        
        elif operation == "get_records":
            # Get records from entity using exact SDK signature
            # sdk.entities.list_records_async(entity_key: str, start: int, limit: int, schema: Optional[Type]=None)
            records = await service.entities.list_records_async(
                entity_key=entity_id,
                start=start,
                limit=limit
            )
            
            # Convert records to list of dicts
            records_list = []
            if records:
                for record in records:
                    if hasattr(record, 'data'):
                        records_list.append(record.data)
                    elif hasattr(record, '__dict__'):
                        records_list.append(record.__dict__)
                    else:
                        records_list.append(record)
            
            result = {
                "success": True,
                "operation": operation,
                "entity_key": entity_key,
                "records": records_list,
                "count": len(records_list)
            }
        
        elif operation == "insert_record":
            # Insert new record using exact SDK signature
            # sdk.entities.insert_records_async(entity_key: str, records: List[Dict], schema: Optional[Type]=None)
            if not record_data:
                result = {"success": False, "error": "record_data required for insert operation"}
            else:
                insert_result = await service.entities.insert_records_async(
                    entity_key=entity_id,
                    records=[record_data]
                )
                
                # Extract record ID from result
                record_id = None
                if insert_result:
                    if hasattr(insert_result, 'successful_records') and insert_result.successful_records:
                        record_id = insert_result.successful_records[0]
                    elif isinstance(insert_result, dict):
                        record_id = insert_result.get("id")
                
                result = {
                    "success": True,
                    "operation": operation,
                    "entity_key": entity_key,
                    "inserted": True,
                    "record_id": record_id
                }
        
        elif operation == "update_record":
            # Update existing record using exact SDK signature
            # sdk.entities.update_records_async(entity_key: str, records: List[Dict], schema: Optional[Type]=None)
            if not record_id:
                result = {"success": False, "error": "record_id required for update operation"}
            elif not record_data:
                result = {"success": False, "error": "record_data required for update operation"}
            else:
                # Include ID in record data for update
                update_data = {"Id": record_id, **record_data}
                
                update_result = await service.entities.update_records_async(
                    entity_key=entity_id,
                    records=[update_data]
                )
                
                result = {
                    "success": True,
                    "operation": operation,
                    "entity_key": entity_key,
                    "updated": True,
                    "record_id": record_id
                }
        
        else:
            result = {
                "success": False, 
                "error": f"Unknown operation: {operation}. Valid operations: get_records, get_claim, insert_record, update_record"
            }
        
        return json.dumps(result)
            
    except Exception as e:
        logger.error(f"❌ Real Data Fabric operation failed: {e}")
        result = {"success": False, "error": str(e)}
        return json.dumps(result)