"""
Base Agent Class for LTL Claims Processing
Provides common functionality for all specialized agents.
"""

import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from ..services.uipath_service import UiPathService


logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all claims processing agents.
    
    Provides common functionality:
    - Claim ID extraction
    - Debug logging helpers
    - Configuration management
    - Error handling patterns
    """
    
    def __init__(self, uipath_service: UiPathService, config: Optional[Any] = None):
        """
        Initialize base agent.
        
        Args:
            uipath_service: Authenticated UiPath service instance
            config: Optional configuration object
        """
        self.uipath_service = uipath_service
        self.config = config
        self._agent_name = self.__class__.__name__.replace('Agent', '').upper()
    
    @staticmethod
    def _extract_claim_id(state: Dict[str, Any]) -> str:
        """
        Extract claim ID from state, handling both field name formats.
        
        Args:
            state: Current GraphState
            
        Returns:
            Claim ID string or 'UNKNOWN' if not found
        """
        return state.get('claim_id') or state.get('ObjectClaimId', 'UNKNOWN')
    
    def _log_debug(self, message: str, claim_id: Optional[str] = None) -> None:
        """
        Log debug message with agent context.
        
        Args:
            message: Debug message
            claim_id: Optional claim ID for context
        """
        if claim_id:
            logger.debug(f"[{self._agent_name}] [{claim_id}] {message}")
        else:
            logger.debug(f"[{self._agent_name}] {message}")
    
    def _log_info(self, message: str, claim_id: Optional[str] = None) -> None:
        """
        Log info message with agent context.
        
        Args:
            message: Info message
            claim_id: Optional claim ID for context
        """
        if claim_id:
            logger.info(f"[{self._agent_name}] [{claim_id}] {message}")
        else:
            logger.info(f"[{self._agent_name}] {message}")
    
    def _log_warning(self, message: str, claim_id: Optional[str] = None) -> None:
        """
        Log warning message with agent context.
        
        Args:
            message: Warning message
            claim_id: Optional claim ID for context
        """
        if claim_id:
            logger.warning(f"[{self._agent_name}] [{claim_id}] {message}")
        else:
            logger.warning(f"[{self._agent_name}] {message}")
    
    def _log_error(self, message: str, claim_id: Optional[str] = None, exc: Optional[Exception] = None) -> None:
        """
        Log error message with agent context.
        
        Args:
            message: Error message
            claim_id: Optional claim ID for context
            exc: Optional exception for stack trace
        """
        if claim_id:
            log_msg = f"[{self._agent_name}] [{claim_id}] {message}"
        else:
            log_msg = f"[{self._agent_name}] {message}"
        
        if exc:
            logger.error(log_msg, exc_info=exc)
        else:
            logger.error(log_msg)
    
    def _log_agent_invocation(self, result: Dict[str, Any], operation: str, claim_id: str) -> None:
        """
        Log debug information about agent invocation results.
        
        Args:
            result: Agent invocation result containing messages
            operation: Name of the operation (e.g., 'planning', 'replanning')
            claim_id: Claim ID for context
        """
        messages = result.get("messages", [])
        self._log_debug(
            f"{operation.capitalize()} agent returned {len(messages)} messages",
            claim_id
        )
        
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            content_preview = str(msg.content)[:150] if hasattr(msg, 'content') else 'No content'
            self._log_debug(f"Message {i} ({msg_type}): {content_preview}...", claim_id)
            
            # Log tool calls if present
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = (
                        tool_call.get('name', 'unknown') 
                        if isinstance(tool_call, dict) 
                        else getattr(tool_call, 'name', 'unknown')
                    )
                    self._log_debug(f"Tool called: {tool_name}", claim_id)
    
    @abstractmethod
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method - must be implemented by subclasses.
        
        Args:
            state: Current GraphState
            
        Returns:
            Processing results
        """
        pass
