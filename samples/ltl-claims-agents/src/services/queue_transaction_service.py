"""
Queue Transaction Service - Refactored queue transaction operations.

This module provides clean, well-structured methods for UiPath queue transactions
with proper error handling, retry logic, and type safety.
"""

import logging
from typing import Dict, Any, Optional, Literal
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import httpx

from ..config.settings import settings
from ..utils.retry import retry_with_backoff, RetryConfig

logger = logging.getLogger(__name__)


class QueueTransactionError(Exception):
    """Exception raised for queue transaction errors."""
    pass


class QueueTransactionService:
    """
    Service for managing UiPath queue transactions.
    
    Provides methods for:
    - Starting transactions (retrieving and locking queue items)
    - Updating transaction progress
    - Completing transactions with success/failure status
    - Setting business and application exceptions
    
    Usage:
        async with QueueTransactionService(uipath_client) as service:
            transaction = await service.start_transaction("MyQueue")
            if transaction:
                await service.set_progress(transaction['transaction_key'], "Processing...")
                await service.complete_transaction(transaction['transaction_key'])
    """
    
    # Default timeout for API requests (seconds)
    DEFAULT_TIMEOUT = 30.0
    
    def __init__(self, uipath_client, timeout: float = DEFAULT_TIMEOUT):
        """
        Initialize the queue transaction service.
        
        Args:
            uipath_client: Authenticated UiPath SDK client
            timeout: Timeout for API requests in seconds (default: 30.0)
        """
        if not uipath_client:
            raise ValueError("uipath_client cannot be None")
            
        self._client = uipath_client
        self._timeout = timeout
        self._retry_config = RetryConfig(
            max_attempts=3,  # Fixed: was max_retries
            initial_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=True
        )
        self._retryable_errors = (
            httpx.HTTPStatusError,
            httpx.ConnectError,
            httpx.TimeoutException,
            ConnectionError,
            TimeoutError
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # No cleanup needed currently, but structure is in place
        return False
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for API requests.
        
        Returns:
            Dictionary of HTTP headers including authorization and folder context
        """
        headers = {
            "Authorization": f"Bearer {self._client.api_client.secret}",
            "Content-Type": "application/json"
        }
        
        if settings.uipath_folder_id:
            headers["X-UIPATH-OrganizationUnitId"] = str(settings.uipath_folder_id)
        
        return headers
    
    def _validate_transaction_key(self, transaction_key: str) -> None:
        """
        Validate transaction key is not empty.
        
        Args:
            transaction_key: The transaction key to validate
            
        Raises:
            ValueError: If transaction key is empty or None
        """
        if not transaction_key or not str(transaction_key).strip():
            raise ValueError("transaction_key cannot be empty")
    
    def _validate_queue_name(self, queue_name: str) -> None:
        """
        Validate queue name is not empty.
        
        Args:
            queue_name: The queue name to validate
            
        Raises:
            ValueError: If queue name is empty or None
        """
        if not queue_name or not str(queue_name).strip():
            raise ValueError("queue_name cannot be empty")
    
    async def start_transaction(
        self,
        queue_name: str,
        robot_identifier: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Start a queue transaction by retrieving and locking the next available item.
        
        Uses the UiPath API endpoint: /odata/Queues/UiPathODataSvc.StartTransaction
        
        Args:
            queue_name: Name of the queue to retrieve from
            robot_identifier: Optional robot identifier (UUID)
            
        Returns:
            Dictionary with transaction data:
            - id: Queue item ID
            - transaction_key: Transaction key for subsequent operations
            - queue_name: Queue name
            - status: Item status (typically 'InProgress')
            - priority: Item priority
            - specific_content: Item payload data
            - reference: Item reference string
            - creation_time: When item was created
            - defer_date: When item should be processed (if deferred)
            - due_date: When item is due
            - retry_number: Number of times item has been retried
            
            Returns None if no items are available in the queue.
            
        Raises:
            ValueError: If queue_name is empty
            QueueTransactionError: If the operation fails
        """
        self._validate_queue_name(queue_name)
        
        try:
            logger.info(f"Starting transaction for queue: {queue_name}")
            
            # Prepare request payload
            transaction_data = {
                "Name": queue_name,
                "SpecificContent": None  # None means get next available item
            }
            
            if robot_identifier:
                transaction_data["RobotIdentifier"] = robot_identifier
            
            request_body = {"transactionData": transaction_data}
            
            # Make API call
            base_url = self._client.api_client.base_url
            url = f"{base_url}/odata/Queues/UiPathODataSvc.StartTransaction"
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json=request_body,
                    headers=self._get_headers(),
                    timeout=self._timeout
                )
                
                # Handle 204 No Content (no items available)
                if response.status_code == 204:
                    logger.info(f"No items available in queue: {queue_name}")
                    return None
                
                response.raise_for_status()
                item_data = response.json()
                
                # Normalize transaction data
                result = {
                    'id': item_data.get('Id'),
                    'transaction_key': item_data.get('Key'),
                    'queue_name': queue_name,
                    'status': item_data.get('Status', 'InProgress'),
                    'priority': item_data.get('Priority', 'Normal'),
                    'specific_content': item_data.get('SpecificContent', {}),
                    'reference': item_data.get('Reference', ''),
                    'creation_time': item_data.get('CreationTime'),
                    'defer_date': item_data.get('DeferDate'),
                    'due_date': item_data.get('DueDate'),
                    'retry_number': item_data.get('RetryNumber', 0)
                }
                
                logger.info(
                    f"Transaction started: key={result['transaction_key']}, "
                    f"reference={result['reference']}"
                )
                
                return result
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 204:
                return None
            logger.error(f"HTTP error starting transaction: {e}", exc_info=True)
            raise QueueTransactionError(f"Failed to start transaction: {str(e)}") from e
        except Exception as e:
            logger.error(
                f"Failed to start transaction for queue {queue_name}: {str(e)}", 
                exc_info=True
            )
            raise QueueTransactionError(f"Failed to start transaction: {str(e)}") from e
    
    async def set_progress(
        self,
        transaction_key: str,
        progress: str
    ) -> bool:
        """
        Update the progress of an in-progress transaction.
        
        Uses the UiPath API endpoint: 
        /odata/QueueItems({key})/UiPathODataSvc.SetTransactionProgress
        
        Args:
            transaction_key: The transaction key from start_transaction()
            progress: Progress description (max 500 characters recommended)
            
        Returns:
            True if progress was updated successfully
            
        Raises:
            ValueError: If transaction_key or progress is empty
            QueueTransactionError: If the operation fails
        """
        self._validate_transaction_key(transaction_key)
        
        if not progress or not progress.strip():
            raise ValueError("progress cannot be empty")
        
        # Truncate progress if too long (UiPath has limits)
        max_progress_length = 500
        if len(progress) > max_progress_length:
            logger.warning(
                f"Progress message truncated from {len(progress)} to {max_progress_length} characters"
            )
            progress = progress[:max_progress_length]
        
        try:
            logger.debug(f"Setting progress for transaction {transaction_key}: {progress}")
            
            base_url = self._client.api_client.base_url
            url = f"{base_url}/odata/QueueItems({transaction_key})/UiPathODataSvc.SetTransactionProgress"
            
            request_body = {"Progress": progress}
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json=request_body,
                    headers=self._get_headers(),
                    timeout=self._timeout
                )
                
                response.raise_for_status()
                logger.debug(f"Progress updated for transaction {transaction_key}")
                return True
                
        except Exception as e:
            logger.error(
                f"Failed to set progress for {transaction_key}: {str(e)}", 
                exc_info=True
            )
            raise QueueTransactionError(f"Failed to set progress: {str(e)}") from e
    
    async def complete_transaction(
        self,
        transaction_key: str,
        status: Literal["Successful", "Failed"] = "Successful",
        output_data: Optional[Dict[str, Any]] = None,
        analytics_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Complete a transaction with success or failure status.
        
        Uses the SDK's complete_transaction_item_async method.
        
        Args:
            transaction_key: The transaction key from start_transaction()
            status: Transaction status - "Successful" or "Failed"
            output_data: Optional output data to store with the transaction
            analytics_data: Optional analytics/metrics data
            
        Returns:
            True if transaction was completed successfully
            
        Raises:
            ValueError: If transaction_key is empty or status is invalid
            QueueTransactionError: If the operation fails
        """
        self._validate_transaction_key(transaction_key)
        
        if status not in ("Successful", "Failed"):
            raise ValueError(f"Invalid status: {status}. Must be 'Successful' or 'Failed'")
        
        try:
            logger.info(f"Completing transaction {transaction_key} with status: {status}")
            
            from uipath.models.queues import TransactionItemResult
            
            # Prepare result data
            result_data = output_data.copy() if output_data else {}
            if analytics_data:
                result_data['_analytics'] = analytics_data
            
            transaction_result = TransactionItemResult(
                status=status,
                output_data=result_data
            )
            
            # Complete transaction using SDK with retry logic
            await retry_with_backoff(
                self._client.queues.complete_transaction_item_async,
                transaction_key=transaction_key,
                result=transaction_result,
                config=self._retry_config,
                error_types=self._retryable_errors,
                context={"operation": "complete_transaction", "transaction_key": transaction_key}
            )
            
            logger.info(f"Transaction {transaction_key} completed successfully with status: {status}")
            return True
            
        except Exception as e:
            logger.error(
                f"Failed to complete transaction {transaction_key}: {str(e)}", 
                exc_info=True
            )
            raise QueueTransactionError(f"Failed to complete transaction: {str(e)}") from e
    
    async def _set_exception(
        self,
        transaction_key: str,
        exception_type: str,
        exception_message: str,
        is_business_exception: bool,
        exception_details: Optional[str] = None,
        should_retry: bool = False,
        output_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Internal method to set exception data for a transaction.
        
        Args:
            transaction_key: The transaction key from start_transaction()
            exception_type: Type/category of the exception
            exception_message: Exception message or reason
            is_business_exception: True for business exceptions, False for application exceptions
            exception_details: Optional detailed exception information
            should_retry: Whether the item should be retried
            output_data: Optional additional data about the exception
            
        Returns:
            True if exception was set successfully
            
        Raises:
            QueueTransactionError: If the operation fails
        """
        # Prepare output data with exception details
        exception_data = output_data.copy() if output_data else {}
        exception_data.update({
            'ExceptionType': exception_type,
            'ExceptionMessage': exception_message,
            'ExceptionDetails': exception_details,
            'ExceptionTime': datetime.now(timezone.utc).isoformat(),
            'IsBusinessException': is_business_exception,
            'ShouldRetry': should_retry
        })
        
        # Complete transaction with Failed status
        return await self.complete_transaction(
            transaction_key=transaction_key,
            status="Failed",
            output_data=exception_data
        )
    
    async def set_business_exception(
        self,
        transaction_key: str,
        exception_type: str,
        exception_reason: str,
        output_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Mark a transaction as failed with a business exception.
        
        Business exceptions are expected failures that don't require retry
        (e.g., invalid data, business rule violations).
        
        Args:
            transaction_key: The transaction key from start_transaction()
            exception_type: Type/category of the business exception
            exception_reason: Detailed reason for the exception
            output_data: Optional additional data about the exception
            
        Returns:
            True if exception was set successfully
            
        Raises:
            ValueError: If transaction_key, exception_type, or exception_reason is empty
            QueueTransactionError: If the operation fails
        """
        self._validate_transaction_key(transaction_key)
        
        if not exception_type or not exception_type.strip():
            raise ValueError("exception_type cannot be empty")
        if not exception_reason or not exception_reason.strip():
            raise ValueError("exception_reason cannot be empty")
        
        try:
            logger.warning(
                f"Setting business exception for transaction {transaction_key}: "
                f"{exception_type} - {exception_reason}"
            )
            
            return await self._set_exception(
                transaction_key=transaction_key,
                exception_type=exception_type,
                exception_message=exception_reason,
                is_business_exception=True,
                should_retry=False,
                output_data=output_data
            )
            
        except QueueTransactionError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to set business exception for {transaction_key}: {str(e)}", 
                exc_info=True
            )
            raise QueueTransactionError(f"Failed to set business exception: {str(e)}") from e
    
    async def set_application_exception(
        self,
        transaction_key: str,
        exception_message: str,
        exception_details: Optional[str] = None,
        should_retry: bool = True
    ) -> bool:
        """
        Mark a transaction as failed with an application exception.
        
        Application exceptions are unexpected failures that may require retry
        (e.g., network errors, temporary service unavailability).
        
        Args:
            transaction_key: The transaction key from start_transaction()
            exception_message: Exception message
            exception_details: Optional detailed exception information (stack trace, etc.)
            should_retry: Whether the item should be retried (default: True)
            
        Returns:
            True if exception was set successfully
            
        Raises:
            ValueError: If transaction_key or exception_message is empty
            QueueTransactionError: If the operation fails
        """
        self._validate_transaction_key(transaction_key)
        
        if not exception_message or not exception_message.strip():
            raise ValueError("exception_message cannot be empty")
        
        try:
            logger.error(
                f"Setting application exception for transaction {transaction_key}: "
                f"{exception_message}"
            )
            
            return await self._set_exception(
                transaction_key=transaction_key,
                exception_type="ApplicationException",
                exception_message=exception_message,
                is_business_exception=False,
                exception_details=exception_details,
                should_retry=should_retry
            )
            
        except QueueTransactionError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to set application exception for {transaction_key}: {str(e)}", 
                exc_info=True
            )
            raise QueueTransactionError(f"Failed to set application exception: {str(e)}") from e
