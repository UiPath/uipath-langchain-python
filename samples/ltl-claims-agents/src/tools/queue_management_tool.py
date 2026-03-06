"""
Queue Management Tool for UiPath Queue operations.
Manages queue items, transactions, and status updates.
"""

import logging
import json
from typing import Any, Dict, Optional
from datetime import datetime

from langchain_core.tools import tool
from uipath.tracing import traced

logger = logging.getLogger(__name__)

# Import settings for configuration
from ..config.settings import settings

# Global UiPath service instance
_uipath_service = None

async def _get_uipath_service():
    """Get UiPath service instance."""
    global _uipath_service
    if _uipath_service is None:
        try:
            from uipath import UiPath
            
            _uipath_service = UiPath()
            logger.info("✅ UiPath Queue service initialized")
        except ImportError:
            logger.error("❌ UiPath SDK not available")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to initialize UiPath service: {e}")
            raise
    
    return _uipath_service


@tool
@traced(name="update_queue_transaction", run_type="tool")
async def update_queue_transaction(
    operation: str,
    transaction_key: Optional[str] = None,
    queue_name: Optional[str] = None,
    status: Optional[str] = None,
    output_data: Optional[Dict[str, Any]] = None,
    progress: Optional[str] = None,
    error_message: Optional[str] = None
) -> str:
    """Update UiPath queue transaction status and progress.
    
    This tool manages queue items for claims processing, updating status,
    progress, and output data as the claim moves through the workflow.
    
    Args:
        operation: Operation to perform ('set_status', 'update_progress', 'complete', 'fail')
        transaction_key: Transaction key/ID from queue item (required for most operations)
        queue_name: Queue name (optional, uses settings default)
        status: New status ('InProgress', 'Successful', 'Failed', 'Abandoned')
        output_data: Output data to attach to transaction (optional)
        progress: Progress message/percentage (optional)
        error_message: Error message for failed transactions (optional)
    
    Returns:
        JSON string with operation result
    
    Operations:
        - set_status: Set transaction status (InProgress, Successful, Failed)
        - update_progress: Update progress message during processing
        - complete: Mark transaction as successful with output data
        - fail: Mark transaction as failed with error message
    
    Example:
        {
            "operation": "update_progress",
            "transaction_key": "abc-123-txn",
            "progress": "Document extraction completed - 75%"
        }
    """
    try:
        logger.info(f"📋 Queue operation: {operation}")
        
        # Use default queue name if not provided
        queue_name = queue_name or settings.queue_name
        
        service = await _get_uipath_service()
        
        if operation == "set_status":
            if not transaction_key or not status:
                return json.dumps({
                    "success": False,
                    "error": "transaction_key and status required for set_status"
                })
            
            # Set transaction status using SDK
            # sdk.queues.set_transaction_status_async(transaction_key, status)
            await service.queues.set_transaction_status_async(
                transaction_key=transaction_key,
                status=status
            )
            
            result = {
                "success": True,
                "operation": "set_status",
                "transaction_key": transaction_key,
                "status": status,
                "queue_name": queue_name
            }
        
        elif operation == "update_progress":
            if not transaction_key or not progress:
                return json.dumps({
                    "success": False,
                    "error": "transaction_key and progress required for update_progress"
                })
            
            # Update transaction progress
            # sdk.queues.update_progress_of_transaction_item_async(transaction_key, progress)
            await service.queues.update_progress_of_transaction_item_async(
                transaction_key=transaction_key,
                progress=progress
            )
            
            result = {
                "success": True,
                "operation": "update_progress",
                "transaction_key": transaction_key,
                "progress": progress,
                "queue_name": queue_name
            }
        
        elif operation == "complete":
            if not transaction_key:
                return json.dumps({
                    "success": False,
                    "error": "transaction_key required for complete"
                })
            
            # Complete transaction with output data
            # sdk.queues.complete_transaction_item_async(transaction_key, output_data)
            await service.queues.complete_transaction_item_async(
                transaction_key=transaction_key,
                result={
                    "Status": "Successful",
                    "OutputData": output_data or {},
                    "CompletedAt": datetime.utcnow().isoformat()
                }
            )
            
            result = {
                "success": True,
                "operation": "complete",
                "transaction_key": transaction_key,
                "output_data": output_data,
                "queue_name": queue_name
            }
        
        elif operation == "fail":
            if not transaction_key:
                return json.dumps({
                    "success": False,
                    "error": "transaction_key required for fail"
                })
            
            # Fail transaction with error message
            # sdk.queues.fail_transaction_item_async(transaction_key, error_message)
            await service.queues.fail_transaction_item_async(
                transaction_key=transaction_key,
                result={
                    "Status": "Failed",
                    "ErrorMessage": error_message or "Processing failed",
                    "FailedAt": datetime.utcnow().isoformat()
                }
            )
            
            result = {
                "success": True,
                "operation": "fail",
                "transaction_key": transaction_key,
                "error_message": error_message,
                "queue_name": queue_name
            }
        
        else:
            result = {
                "success": False,
                "error": f"Unknown operation: {operation}. Valid: set_status, update_progress, complete, fail"
            }
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"❌ Queue operation failed: {e}")
        result = {
            "success": False,
            "error": str(e),
            "operation": operation,
            "transaction_key": transaction_key
        }
        return json.dumps(result)
