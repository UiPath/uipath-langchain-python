"""
Context Grounding Tool for LTL Claims Knowledge Base.
Uses UiPath Context Grounding to search policies, procedures, and historical data.
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from functools import wraps

from langchain_core.tools import tool
from uipath.tracing import traced
from uipath_langchain.retrievers import ContextGroundingRetriever

logger = logging.getLogger(__name__)

# Configuration constants
MAX_CONTENT_LENGTH = 500
MAX_CARRIER_CONTENT_LENGTH = 400
MAX_RESULTS_TO_DISPLAY = 10

# Initialize Context Grounding Retriever
try:
    from src.config.settings import settings
    from uipath import UiPath
    
    # Check if Context Grounding is enabled in settings
    if not settings.enable_context_grounding:
        logger.info("[DISABLED] Context Grounding is disabled in settings (ENABLE_CONTEXT_GROUNDING=false)")
        claims_knowledge_retriever = None
        INDEX_NAME = None
    else:
        # Use index name from settings instead of hardcoded value
        INDEX_NAME = settings.context_grounding_index_name
        
        # Create SDK instance with proper authentication
        sdk = UiPath(
            base_url=settings.effective_base_url,
            secret=settings.uipath_access_token
        )
        
        # Context Grounding requires folder_key (UUID format), not folder_id
        # The folder_key should be the UUID of the folder, not the numeric ID
        folder_key = settings.uipath_folder_id if settings.uipath_folder_id else None
        
        claims_knowledge_retriever = ContextGroundingRetriever(
            index_name=INDEX_NAME,
            folder_key=folder_key,
            sdk=sdk  # Pass authenticated SDK instance
        )
        logger.info(f"[OK] Context Grounding retriever initialized for '{INDEX_NAME}' with folder_key={folder_key}")
except Exception as e:
    logger.warning(f"[DISABLED] Context Grounding not available: {e}")
    claims_knowledge_retriever = None
    INDEX_NAME = None


def _check_retriever_available() -> Optional[str]:
    """
    Check if the Context Grounding retriever is available.
    
    Returns:
        Error message if unavailable, None if available
    """
    if not claims_knowledge_retriever:
        return "Context Grounding service is not available. Cannot search knowledge base."
    return None


def _format_search_results(
    results: List[Any],
    query: str,
    max_content_length: int = MAX_CONTENT_LENGTH,
    prefix: str = ""
) -> str:
    """
    Format search results into a readable string.
    
    Args:
        results: List of document results from retriever
        query: Original search query
        max_content_length: Maximum length of content to include per result
        prefix: Optional prefix for the formatted response
        
    Returns:
        Formatted string with search results
    """
    if not results:
        logger.info(f"[EMPTY] No results found for query: {query}")
        return f"No relevant information found in the knowledge base for query: '{query}'"
    
    formatted_response = prefix if prefix else f"Found {len(results)} relevant documents:\n\n"
    
    for i, doc in enumerate(results[:MAX_RESULTS_TO_DISPLAY], 1):
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        source = metadata.get('source', 'Unknown source')
        
        # Truncate content if needed
        truncated_content = content[:max_content_length]
        if len(content) > max_content_length:
            truncated_content += "..."
        
        formatted_response += f"Document {i}:\n"
        formatted_response += f"Source: {source}\n"
        formatted_response += f"Content: {truncated_content}\n\n"
    
    logger.info(f"[OK] Formatted {len(results)} results from knowledge base")
    return formatted_response


def _safe_search(query: str, context: str = "knowledge base") -> str:
    """
    Perform a safe search with error handling.
    
    Args:
        query: Search query string
        context: Context description for logging
        
    Returns:
        Formatted search results or error message
    """
    try:
        # Check retriever availability
        error_msg = _check_retriever_available()
        if error_msg:
            return error_msg
        
        logger.info(f"[SEARCH] Searching {context}: {query}")
        
        # Perform search
        results = claims_knowledge_retriever.invoke(query)
        
        return results
        
    except Exception as e:
        error_msg = f"Error searching {context}: {str(e)}"
        logger.error(f"[ERROR] {error_msg}", exc_info=True)
        return error_msg


@tool
@traced(name="search_claims_knowledge", run_type="tool")
def search_claims_knowledge(query: str) -> str:
    """
    Search the LTL Claims knowledge base for policies, procedures, carrier information, and historical data.
    
    Use this tool to find information about:
    - Claims processing policies and procedures
    - Carrier liability rules and regulations
    - Damage assessment guidelines
    - Historical claim decisions and precedents
    - Freight handling best practices
    - Documentation requirements
    
    Args:
        query: Natural language search query describing what information you need
        
    Returns:
        Relevant information from the knowledge base with sources
        
    Example queries:
        - "What is the policy for handling damaged freight claims?"
        - "Speedy Freight Lines carrier liability limits"
        - "How to assess loss claims for missing shipments"
        - "Documentation required for damage claims"
    """
    results = _safe_search(query, context="claims knowledge base")
    
    # If results is a string, it's an error message
    if isinstance(results, str):
        return results
    
    return _format_search_results(results, query)


@tool
@traced(name="search_carrier_information", run_type="tool")
def search_carrier_information(carrier_name: str) -> str:
    """
    Search for specific carrier information including liability limits, policies, and historical data.
    
    Use this tool to find carrier-specific information such as:
    - Carrier liability limits and coverage
    - Carrier-specific claim procedures
    - Historical claim outcomes with this carrier
    - Carrier contact information
    - Carrier performance and reliability data
    
    Args:
        carrier_name: Name of the carrier to search for (e.g., "Speedy Freight Lines")
        
    Returns:
        Carrier-specific information from the knowledge base
    """
    # Create carrier-specific query
    query = f"carrier information liability limits policies procedures for {carrier_name}"
    
    results = _safe_search(query, context=f"carrier information for {carrier_name}")
    
    # If results is a string, it's an error message
    if isinstance(results, str):
        return results
    
    # Format with carrier-specific prefix
    prefix = f"Carrier Information for {carrier_name}:\n\n"
    return _format_search_results(
        results, 
        query, 
        max_content_length=MAX_CARRIER_CONTENT_LENGTH,
        prefix=prefix
    )


@tool
@traced(name="search_claim_procedures", run_type="tool")
def search_claim_procedures(claim_type: str) -> str:
    """
    Search for specific claim processing procedures based on claim type.
    
    Use this tool to find step-by-step procedures for:
    - Damage claims
    - Loss claims
    - Shortage claims
    - Concealed damage claims
    - Overcharge claims
    
    Args:
        claim_type: Type of claim (e.g., "damage", "loss", "shortage")
        
    Returns:
        Detailed procedures for processing the specified claim type
    """
    # Create procedure-specific query
    query = f"procedures steps process for {claim_type} claims documentation requirements"
    
    results = _safe_search(query, context=f"procedures for {claim_type} claims")
    
    # If results is a string, it's an error message
    if isinstance(results, str):
        return results
    
    # Format with procedure-specific prefix
    prefix = f"Procedures for {claim_type.title()} Claims:\n\n"
    return _format_search_results(results, query, prefix=prefix)
