"""
Performance metrics tracking for LTL Claims Agent System.

Provides comprehensive metrics collection and logging for monitoring agent performance.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict


logger = logging.getLogger(__name__)


@dataclass
class ProcessingMetrics:
    """
    Comprehensive processing metrics for claim processing.
    
    Tracks timing, resource usage, and operation counts throughout processing.
    """
    
    # Timing metrics
    processing_start_time: float = field(default_factory=time.time)
    processing_end_time: Optional[float] = None
    processing_duration_seconds: float = 0.0
    
    # Reasoning metrics
    recursion_steps: int = 0
    max_recursion_depth: int = 0
    reasoning_cycles: int = 0
    average_step_duration: float = 0.0
    
    # Tool execution metrics
    tool_executions: int = 0
    tool_execution_times: Dict[str, List[float]] = field(default_factory=dict)
    total_tool_time: float = 0.0
    
    # API call metrics
    api_calls: int = 0
    api_call_times: Dict[str, List[float]] = field(default_factory=dict)
    total_api_time: float = 0.0
    failed_api_calls: int = 0
    
    # Memory metrics
    memory_queries: int = 0
    memory_query_time: float = 0.0
    memory_hits: int = 0
    memory_misses: int = 0
    
    # Document processing metrics
    documents_downloaded: int = 0
    documents_extracted: int = 0
    document_download_time: float = 0.0
    document_extraction_time: float = 0.0
    
    # Queue operation metrics
    queue_operations: int = 0
    queue_operation_time: float = 0.0
    queue_updates: int = 0
    
    # Action Center metrics
    action_center_tasks_created: int = 0
    escalations: int = 0
    
    # Error metrics
    errors_encountered: int = 0
    recoverable_errors: int = 0
    fatal_errors: int = 0
    
    # Confidence metrics
    initial_confidence: float = 0.0
    final_confidence: float = 0.0
    confidence_changes: List[float] = field(default_factory=list)
    
    # Claim context
    claim_id: Optional[str] = None
    claim_type: Optional[str] = None
    claim_amount: Optional[float] = None
    
    def start_processing(self, claim_id: str, claim_type: Optional[str] = None, claim_amount: Optional[float] = None) -> None:
        """
        Mark the start of processing.
        
        Args:
            claim_id: Claim ID being processed
            claim_type: Type of claim
            claim_amount: Claim amount
        """
        self.processing_start_time = time.time()
        self.claim_id = claim_id
        self.claim_type = claim_type
        self.claim_amount = claim_amount
    
    def end_processing(self) -> None:
        """Mark the end of processing and calculate duration."""
        self.processing_end_time = time.time()
        self.processing_duration_seconds = self.processing_end_time - self.processing_start_time
        
        # Calculate average step duration
        if self.recursion_steps > 0:
            self.average_step_duration = self.processing_duration_seconds / self.recursion_steps
    
    def increment_recursion_step(self) -> None:
        """Increment recursion step counter."""
        self.recursion_steps += 1
        if self.recursion_steps > self.max_recursion_depth:
            self.max_recursion_depth = self.recursion_steps
    
    def record_tool_execution(self, tool_name: str, execution_time: float) -> None:
        """
        Record a tool execution.
        
        Args:
            tool_name: Name of the tool
            execution_time: Execution time in seconds
        """
        self.tool_executions += 1
        self.total_tool_time += execution_time
        
        if tool_name not in self.tool_execution_times:
            self.tool_execution_times[tool_name] = []
        self.tool_execution_times[tool_name].append(execution_time)
    
    def record_api_call(self, service: str, operation: str, call_time: float, failed: bool = False) -> None:
        """
        Record an API call.
        
        Args:
            service: Service name
            operation: Operation name
            call_time: Call time in seconds
            failed: Whether the call failed
        """
        self.api_calls += 1
        self.total_api_time += call_time
        
        if failed:
            self.failed_api_calls += 1
        
        api_key = f"{service}.{operation}"
        if api_key not in self.api_call_times:
            self.api_call_times[api_key] = []
        self.api_call_times[api_key].append(call_time)
    
    def record_memory_query(self, query_time: float, hit: bool = True) -> None:
        """
        Record a memory query.
        
        Args:
            query_time: Query time in seconds
            hit: Whether the query returned results
        """
        self.memory_queries += 1
        self.memory_query_time += query_time
        
        if hit:
            self.memory_hits += 1
        else:
            self.memory_misses += 1
    
    def record_document_download(self, download_time: float) -> None:
        """
        Record a document download.
        
        Args:
            download_time: Download time in seconds
        """
        self.documents_downloaded += 1
        self.document_download_time += download_time
    
    def record_document_extraction(self, extraction_time: float) -> None:
        """
        Record a document extraction.
        
        Args:
            extraction_time: Extraction time in seconds
        """
        self.documents_extracted += 1
        self.document_extraction_time += extraction_time
    
    def record_queue_operation(self, operation_time: float, is_update: bool = False) -> None:
        """
        Record a queue operation.
        
        Args:
            operation_time: Operation time in seconds
            is_update: Whether this is a progress update
        """
        self.queue_operations += 1
        self.queue_operation_time += operation_time
        
        if is_update:
            self.queue_updates += 1
    
    def record_action_center_task(self) -> None:
        """Record an Action Center task creation."""
        self.action_center_tasks_created += 1
    
    def record_escalation(self) -> None:
        """Record an escalation to human review."""
        self.escalations += 1
    
    def record_error(self, recoverable: bool = True) -> None:
        """
        Record an error.
        
        Args:
            recoverable: Whether the error was recoverable
        """
        self.errors_encountered += 1
        
        if recoverable:
            self.recoverable_errors += 1
        else:
            self.fatal_errors += 1
    
    def record_confidence_change(self, confidence: float) -> None:
        """
        Record a confidence level change.
        
        Args:
            confidence: New confidence level
        """
        if not self.confidence_changes:
            self.initial_confidence = confidence
        
        self.confidence_changes.append(confidence)
        self.final_confidence = confidence
    
    def get_tool_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for tool executions.
        
        Returns:
            Dictionary with tool statistics
        """
        stats = {}
        
        for tool_name, times in self.tool_execution_times.items():
            if times:
                stats[tool_name] = {
                    "count": len(times),
                    "total_time": sum(times),
                    "average_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times)
                }
        
        return stats
    
    def get_api_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for API calls.
        
        Returns:
            Dictionary with API call statistics
        """
        stats = {}
        
        for api_key, times in self.api_call_times.items():
            if times:
                stats[api_key] = {
                    "count": len(times),
                    "total_time": sum(times),
                    "average_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times)
                }
        
        return stats
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all metrics.
        
        Returns:
            Dictionary with metric summary
        """
        return {
            "claim_id": self.claim_id,
            "claim_type": self.claim_type,
            "claim_amount": self.claim_amount,
            "processing_duration_seconds": self.processing_duration_seconds,
            "recursion_steps": self.recursion_steps,
            "max_recursion_depth": self.max_recursion_depth,
            "average_step_duration": self.average_step_duration,
            "tool_executions": self.tool_executions,
            "total_tool_time": self.total_tool_time,
            "api_calls": self.api_calls,
            "total_api_time": self.total_api_time,
            "failed_api_calls": self.failed_api_calls,
            "memory_queries": self.memory_queries,
            "memory_query_time": self.memory_query_time,
            "memory_hit_rate": self.memory_hits / self.memory_queries if self.memory_queries > 0 else 0.0,
            "documents_downloaded": self.documents_downloaded,
            "documents_extracted": self.documents_extracted,
            "document_download_time": self.document_download_time,
            "document_extraction_time": self.document_extraction_time,
            "queue_operations": self.queue_operations,
            "queue_operation_time": self.queue_operation_time,
            "action_center_tasks_created": self.action_center_tasks_created,
            "escalations": self.escalations,
            "errors_encountered": self.errors_encountered,
            "recoverable_errors": self.recoverable_errors,
            "fatal_errors": self.fatal_errors,
            "initial_confidence": self.initial_confidence,
            "final_confidence": self.final_confidence,
            "confidence_improvement": self.final_confidence - self.initial_confidence
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to dictionary.
        
        Returns:
            Dictionary representation of metrics
        """
        return asdict(self)
    
    def log_metrics(self) -> None:
        """Log metrics summary."""
        summary = self.get_summary()
        
        logger.info(
            f"Processing metrics for claim {self.claim_id}",
            extra={
                "event": "processing_metrics",
                "metrics": summary,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
        # Log detailed tool statistics if any tools were used
        if self.tool_executions > 0:
            tool_stats = self.get_tool_statistics()
            logger.info(
                f"Tool execution statistics for claim {self.claim_id}",
                extra={
                    "event": "tool_statistics",
                    "claim_id": self.claim_id,
                    "tool_statistics": tool_stats,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        # Log detailed API statistics if any API calls were made
        if self.api_calls > 0:
            api_stats = self.get_api_statistics()
            logger.info(
                f"API call statistics for claim {self.claim_id}",
                extra={
                    "event": "api_statistics",
                    "claim_id": self.claim_id,
                    "api_statistics": api_stats,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )


def create_processing_metrics(claim_id: str, claim_type: Optional[str] = None, claim_amount: Optional[float] = None) -> ProcessingMetrics:
    """
    Create and initialize a ProcessingMetrics instance.
    
    Args:
        claim_id: Claim ID
        claim_type: Type of claim
        claim_amount: Claim amount
        
    Returns:
        Initialized ProcessingMetrics instance
    """
    metrics = ProcessingMetrics()
    metrics.start_processing(claim_id, claim_type, claim_amount)
    return metrics


__all__ = [
    "ProcessingMetrics",
    "create_processing_metrics"
]
