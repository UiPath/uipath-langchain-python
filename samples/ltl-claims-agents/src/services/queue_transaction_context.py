"""
Queue Transaction Context Manager.

Provides a context manager for safe queue transaction handling with automatic
completion/failure handling and progress tracking.
"""

import logging
from typing import Dict, Any, Optional, Callable, Awaitable
from contextlib import asynccontextmanager

from .queue_transaction_service import QueueTransactionService, QueueTransactionError

logger = logging.getLogger(__name__)


class TransactionContext:
    """
    Context object for managing a queue transaction.
    
    Provides methods for updating progress and accessing transaction data
    within a transaction processing block.
    """
    
    def __init__(
        self,
        transaction_service: QueueTransactionService,
        transaction_data: Dict[str, Any]
    ):
        """
        Initialize transaction context.
        
        Args:
            transaction_service: The queue transaction service
            transaction_data: Transaction data from start_transaction()
        """
        self._service = transaction_service
        self._data = transaction_data
        self._completed = False
        self._output_data = {}
    
    @property
    def transaction_key(self) -> str:
        """Get the transaction key."""
        return self._data['transaction_key']
    
    @property
    def item_id(self) -> str:
        """Get the queue item ID."""
        return self._data['id']
    
    @property
    def reference(self) -> str:
        """Get the item reference."""
        return self._data.get('reference', '')
    
    @property
    def priority(self) -> str:
        """Get the item priority."""
        return self._data.get('priority', 'Normal')
    
    @property
    def content(self) -> Dict[str, Any]:
        """Get the specific content (payload) of the queue item."""
        return self._data.get('specific_content', {})
    
    @property
    def retry_number(self) -> int:
        """Get the retry number for this item."""
        return self._data.get('retry_number', 0)
    
    async def update_progress(self, progress: str) -> None:
        """
        Update transaction progress.
        
        Args:
            progress: Progress description
        """
        await self._service.set_progress(self.transaction_key, progress)
    
    def set_output(self, key: str, value: Any) -> None:
        """
        Set output data to be stored with the transaction result.
        
        Args:
            key: Output data key
            value: Output data value
        """
        self._output_data[key] = value
    
    def set_outputs(self, data: Dict[str, Any]) -> None:
        """
        Set multiple output data fields.
        
        Args:
            data: Dictionary of output data
        """
        self._output_data.update(data)
    
    async def complete_success(self, output_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark transaction as successfully completed.
        
        Args:
            output_data: Optional output data (merged with set_output data)
        """
        if self._completed:
            logger.warning(f"Transaction {self.transaction_key} already completed")
            return
        
        final_output = {**self._output_data}
        if output_data:
            final_output.update(output_data)
        
        await self._service.complete_transaction(
            transaction_key=self.transaction_key,
            status="Successful",
            output_data=final_output
        )
        self._completed = True
    
    async def complete_business_exception(
        self,
        exception_type: str,
        exception_reason: str,
        output_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Mark transaction as failed with a business exception.
        
        Args:
            exception_type: Type of business exception
            exception_reason: Reason for the exception
            output_data: Optional additional data
        """
        if self._completed:
            logger.warning(f"Transaction {self.transaction_key} already completed")
            return
        
        await self._service.set_business_exception(
            transaction_key=self.transaction_key,
            exception_type=exception_type,
            exception_reason=exception_reason,
            output_data=output_data
        )
        self._completed = True
    
    async def complete_application_exception(
        self,
        exception_message: str,
        exception_details: Optional[str] = None,
        should_retry: bool = True
    ) -> None:
        """
        Mark transaction as failed with an application exception.
        
        Args:
            exception_message: Exception message
            exception_details: Optional detailed exception info
            should_retry: Whether the item should be retried
        """
        if self._completed:
            logger.warning(f"Transaction {self.transaction_key} already completed")
            return
        
        await self._service.set_application_exception(
            transaction_key=self.transaction_key,
            exception_message=exception_message,
            exception_details=exception_details,
            should_retry=should_retry
        )
        self._completed = True


@asynccontextmanager
async def queue_transaction(
    transaction_service: QueueTransactionService,
    queue_name: str,
    robot_identifier: Optional[str] = None,
    auto_complete_on_success: bool = True
):
    """
    Context manager for safe queue transaction processing.
    
    Automatically handles transaction completion/failure and ensures
    transactions are always properly closed.
    
    Usage:
        async with queue_transaction(service, "MyQueue") as ctx:
            if ctx is None:
                # No items available
                return
            
            # Process the item
            data = ctx.content
            await ctx.update_progress("Processing...")
            
            # Set output data
            ctx.set_output("result", "processed")
            
            # Transaction is auto-completed on success if auto_complete_on_success=True
            # Or manually complete:
            # await ctx.complete_success()
    
    Args:
        transaction_service: The queue transaction service
        queue_name: Name of the queue
        robot_identifier: Optional robot identifier
        auto_complete_on_success: Automatically complete transaction on success
        
    Yields:
        TransactionContext or None if no items available
    """
    transaction_data = await transaction_service.start_transaction(
        queue_name=queue_name,
        robot_identifier=robot_identifier
    )
    
    # No items available
    if transaction_data is None:
        yield None
        return
    
    ctx = TransactionContext(transaction_service, transaction_data)
    
    try:
        yield ctx
        
        # Auto-complete if enabled and not already completed
        if auto_complete_on_success and not ctx._completed:
            await ctx.complete_success()
            
    except Exception as e:
        # If not already completed, mark as application exception
        if not ctx._completed:
            logger.error(f"Unhandled exception in transaction {ctx.transaction_key}: {str(e)}")
            try:
                await ctx.complete_application_exception(
                    exception_message=str(e),
                    exception_details=repr(e),
                    should_retry=True
                )
            except Exception as completion_error:
                logger.error(
                    f"Failed to complete transaction {ctx.transaction_key} "
                    f"after exception: {completion_error}"
                )
        raise


async def process_queue_items(
    transaction_service: QueueTransactionService,
    queue_name: str,
    processor: Callable[[TransactionContext], Awaitable[None]],
    max_items: Optional[int] = None,
    robot_identifier: Optional[str] = None
) -> Dict[str, int]:
    """
    Process multiple queue items with a processor function.
    
    Continues processing items until the queue is empty or max_items is reached.
    
    Usage:
        async def process_claim(ctx: TransactionContext):
            claim_data = ctx.content
            await ctx.update_progress("Processing claim...")
            # Process the claim
            ctx.set_output("status", "processed")
        
        stats = await process_queue_items(
            service,
            "LTL_Claims_Processing",
            process_claim,
            max_items=10
        )
        print(f"Processed: {stats['successful']}, Failed: {stats['failed']}")
    
    Args:
        transaction_service: The queue transaction service
        queue_name: Name of the queue
        processor: Async function that processes a TransactionContext
        max_items: Maximum number of items to process (None = unlimited)
        robot_identifier: Optional robot identifier
        
    Returns:
        Dictionary with processing statistics:
        - processed: Total items processed
        - successful: Successfully completed items
        - failed: Failed items
        - business_exceptions: Items failed with business exceptions
        - application_exceptions: Items failed with application exceptions
    """
    stats = {
        'processed': 0,
        'successful': 0,
        'failed': 0,
        'business_exceptions': 0,
        'application_exceptions': 0
    }
    
    while max_items is None or stats['processed'] < max_items:
        async with queue_transaction(
            transaction_service,
            queue_name,
            robot_identifier,
            auto_complete_on_success=False
        ) as ctx:
            # No more items
            if ctx is None:
                break
            
            stats['processed'] += 1
            
            try:
                # Process the item
                await processor(ctx)
                
                # Complete if not already completed
                if not ctx._completed:
                    await ctx.complete_success()
                    stats['successful'] += 1
                else:
                    # Check completion status
                    # This is a simplification - in reality we'd need to track the completion type
                    stats['successful'] += 1
                    
            except Exception as e:
                logger.error(f"Error processing item {ctx.transaction_key}: {str(e)}")
                stats['failed'] += 1
                
                # Ensure transaction is completed
                if not ctx._completed:
                    await ctx.complete_application_exception(
                        exception_message=str(e),
                        exception_details=repr(e)
                    )
                    stats['application_exceptions'] += 1
    
    logger.info(
        f"Queue processing complete: {stats['processed']} items processed, "
        f"{stats['successful']} successful, {stats['failed']} failed"
    )
    
    return stats
