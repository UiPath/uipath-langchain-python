"""
Tools Registry for ReAct Claims Processor.
Centralized place to import and manage all essential tools using @tool decorator.
"""

import logging
import sys
import os
from typing import List

from langchain_core.tools import BaseTool
from uipath.tracing import traced

logger = logging.getLogger(__name__)


def get_all_tools() -> List:
    """Get all available tools for the LTL Claims Processing Agent.
    
    Core Tools (Required):
    1. query_data_fabric - Query/update claim and shipment data in Data Fabric
    2. download_multiple_documents - Download documents from storage buckets
    3. extract_documents_batch - Extract data using Document Understanding (IXP)
    4. update_queue_transaction - Update UiPath queue transaction status
    
    Optional Tools:
    5. search_claims_knowledge - Search claims knowledge base (Context Grounding)
    6. search_carrier_information - Search carrier-specific information
    7. search_claim_procedures - Search claim processing procedures
   
    """
    tools = []
    
    # Add current directory to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # 1. Data Fabric Tool (REQUIRED)
    try:
        from . import data_fabric_tool
        tools.append(data_fabric_tool.query_data_fabric)
        logger.info("Loaded query_data_fabric tool")
    except Exception as e:
        logger.error(f"Failed to load query_data_fabric tool: {e}")
        # This is critical, but continue to see what else loads
    
    # 2. Document Download Tool (REQUIRED)
    try:
        from . import document_download_tool
        tools.append(document_download_tool.download_multiple_documents)
        logger.info("Loaded download_multiple_documents tool")
    except Exception as e:
        logger.error(f"Failed to load download_multiple_documents tool: {e}")
    
    # 3. Document Extraction Tool (REQUIRED)
    try:
        from . import document_extraction_tool
        tools.append(document_extraction_tool.extract_documents_batch)
        logger.info("Loaded extract_documents_batch tool")
    except Exception as e:
        logger.error(f"Failed to load extract_documents_batch tool: {e}")
    
    # 4. Queue Management Tool (REQUIRED)
    try:
        from . import queue_management_tool
        tools.append(queue_management_tool.update_queue_transaction)
        logger.info("Loaded update_queue_transaction tool")
    except Exception as e:
        logger.error(f"Failed to load queue_management tool: {e}")
    
    # 5-7. Context Grounding Tools (OPTIONAL - Knowledge Base)
    try:
        from . import context_grounding_tool
        tools.append(context_grounding_tool.search_claims_knowledge)
        tools.append(context_grounding_tool.search_carrier_information)
        tools.append(context_grounding_tool.search_claim_procedures)
        logger.info("Loaded 3 Context Grounding tools (knowledge search)")
    except Exception as e:
        logger.warning(f"Context Grounding tools not loaded: {e}")
        # Context Grounding is optional
  
   
    
    if not tools:
        logger.error("NO TOOLS LOADED! Agent cannot function without tools.")
        raise RuntimeError("Failed to load any tools")
    
    logger.info(f"Tools Registry loaded {len(tools)} tools successfully")
    
    # Validate all tools
    validation_errors = validate_all_tools(tools)
    if validation_errors:
        logger.warning(f"Tool validation found {len(validation_errors)} issues:")
        for error in validation_errors:
            logger.warning(f"  - {error}")
    else:
        logger.info("All tools validated successfully")
    
    return tools


def validate_all_tools(tools: List) -> List[str]:
    """
    Validate that all registered tools are properly decorated and configured.
    
    Checks performed:
    - Tool is a BaseTool instance
    - Tool has a proper description/docstring
    - Tool has invoke/ainvoke methods (for execution)
    - Tool has a name
    
    Args:
        tools: List of tools to validate
        
    Returns:
        List of validation error messages (empty if all valid)
    """
    errors = []
    
    for i, tool in enumerate(tools):
        tool_name = getattr(tool, 'name', f'Tool #{i+1}')
        
        # Check if tool is BaseTool instance
        if not isinstance(tool, BaseTool):
            errors.append(f"Tool '{tool_name}' is not a BaseTool instance (type: {type(tool).__name__})")
            continue
        
        # Check if tool has proper description
        if not hasattr(tool, 'description') or not tool.description:
            errors.append(f"Tool '{tool_name}' missing description/docstring")
        
        # Check if tool has invoke or ainvoke methods (BaseTool execution methods)
        if not (hasattr(tool, 'invoke') or hasattr(tool, 'ainvoke') or hasattr(tool, '_run') or hasattr(tool, '_arun')):
            errors.append(f"Tool '{tool_name}' missing execution methods (invoke/ainvoke/_run/_arun)")
        
        # Check if tool has a name
        if not hasattr(tool, 'name') or not tool.name:
            errors.append(f"Tool at index {i} missing name attribute")
    
    return errors