"""
Document Processor Sub-Agent for LTL Claims Processing
Specialized agent for document download and extraction operations.
"""

import logging
import os
from typing import Dict, Any, List, Optional
import json

from uipath_langchain.chat.models import UiPathChat

from ..services.uipath_service import UiPathService
from .config import DocumentProcessorConfig
from .exceptions import DocumentProcessingError


logger = logging.getLogger(__name__)


class DocumentProcessorAgent:
    """
    Specialized agent for document processing operations.
    
    Responsibilities:
    - Download documents from UiPath storage buckets
    - Extract structured data using UiPath IXP
    - Assess extraction confidence scores
    - Flag low-confidence fields for review
    
    Implements Requirements 3.1, 3.2, 4.1, 4.2, 4.3, 11.1
    """
    
    def __init__(self, uipath_service: UiPathService, config: Optional[DocumentProcessorConfig] = None):
        """
        Initialize the document processor agent.
        
        Args:
            uipath_service: Authenticated UiPath service instance
            config: Optional configuration object (uses defaults if not provided)
        """
        self.uipath_service = uipath_service
        self.config = config or DocumentProcessorConfig()
        
        # Use UiPath Chat model (gpt-4o-mini for efficiency in document analysis)
        self.llm = UiPathChat(
            model="gpt-4o-mini-2024-07-18",
            temperature=0,
            max_tokens=2000,
            timeout=30,
            max_retries=2
        )
        
        logger.info("[DOCUMENT_PROCESSOR] Initialized document processor agent")
    
    @staticmethod
    def _extract_claim_id(state: Dict[str, Any]) -> str:
        """Extract claim ID from state, handling both field name formats."""
        return state.get('claim_id') or state.get('ObjectClaimId', 'UNKNOWN')
    
    async def process_documents(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate document download and extraction workflow.
        
        Main entry point that coordinates the complete document processing pipeline:
        1. Download documents from storage
        2. Extract data using IXP
        3. Assess confidence scores
        4. Flag low-confidence fields
        
        Args:
            state: Current GraphState containing document references
            
        Returns:
            Dictionary with:
                - downloaded: List of downloaded file paths
                - extracted: Dictionary of extracted data by document
                - confidence: Dictionary of confidence scores by field
                - errors: List of any errors encountered
                
        Implements Requirements 3.1, 4.1, 11.4
        """
        from langgraph.prebuilt import create_react_agent
        from langchain_core.messages import HumanMessage
        
        claim_id = self._extract_claim_id(state)
        
        logger.info(f"[DOCUMENT_PROCESSOR] Starting document processing for claim: {claim_id}")
        
        results = {
            "downloaded": [],
            "extracted": {},
            "confidence": {},
            "errors": [],
            "low_confidence_fields": [],
            "needs_validation": False
        }
        
        # Early return if no documents to process
        shipping_docs = state.get('shipping_documents', [])
        damage_evidence = state.get('damage_evidence', [])
        total_docs = len(shipping_docs) + len(damage_evidence)
        
        if total_docs == 0:
            logger.info(f"[DOCUMENT_PROCESSOR] No documents to process for claim {claim_id}")
            return results
        
        try:
            # Get document processing tools
            from ..tools.document_download_tool import download_multiple_documents
            from ..tools.document_extraction_tool import extract_documents_batch
            
            # Build system prompt
            system_prompt = (
                "You are a document processing specialist for freight claims. "
                "Your task is to download and extract data from claim documents. "
                "Use the available tools to download documents from storage and extract structured data using IXP. "
                "Focus on accuracy and completeness. Report any issues encountered during processing.\n\n"
                "CRITICAL - Document Download Instructions:\n"
                "When downloading documents, use the EXACT document metadata from the claim input. "
                "The claim input contains 'shipping_documents' and 'damage_evidence' arrays with complete "
                "metadata including the 'path' field. Pass this metadata directly to download_multiple_documents. "
                "DO NOT construct paths from field names (e.g., don't use 'shipping_documents/file.pdf')."
            )
            
            # Build processing instructions
            processing_instructions = (
                f"Process documents for claim {claim_id}. "
                f"Documents to download: {len(state.get('shipping_documents', []) + state.get('damage_evidence', []))}. "
                f"After downloading, extract structured data from each document using IXP project '{self.config.ixp_project_name}'."
            )
            
            # Debug logging
            logger.debug(f"[DOCUMENT_PROCESSOR] System prompt: {system_prompt[:200]}...")
            logger.debug(f"[DOCUMENT_PROCESSOR] Processing instructions: {processing_instructions}")
            logger.debug(f"[DOCUMENT_PROCESSOR] Available tools: download_multiple_documents, extract_documents_batch")
            
            # Create react agent (no system prompt parameter in this version)
            doc_agent = create_react_agent(
                self.llm,
                tools=[download_multiple_documents, extract_documents_batch]
            )
            
            # Combine system prompt with user instructions
            combined_prompt = f"{system_prompt}\n\n{processing_instructions}"
            
            # Invoke agent
            logger.debug(f"[DOCUMENT_PROCESSOR] Invoking document processing agent for claim {claim_id}")
            result = await doc_agent.ainvoke({
                "messages": [HumanMessage(content=combined_prompt)]
            })
            
            # Debug: Log all messages in the result
            logger.debug(f"[DOCUMENT_PROCESSOR] Agent returned {len(result['messages'])} messages")
            for i, msg in enumerate(result["messages"]):
                msg_type = type(msg).__name__
                logger.debug(f"[DOCUMENT_PROCESSOR] Message {i} ({msg_type}): {str(msg.content)[:150]}...")
                # Log tool calls if present
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        logger.debug(f"[DOCUMENT_PROCESSOR] Tool called: {tool_call.get('name', 'unknown')}")
            
            # Parse agent results
            agent_response = result["messages"][-1].content
            logger.debug(f"[DOCUMENT_PROCESSOR] Agent final response: {agent_response[:200]}...")
            
            # Step 1: Download documents (fallback to direct call if agent didn't use tools)
            download_results = await self._download_documents(state)
            results["downloaded"] = download_results.get("files", [])
            
            if download_results.get("errors"):
                results["errors"].extend(download_results["errors"])
            
            # Step 2: Extract data from downloaded documents
            if results["downloaded"]:
                extraction_results = await self._extract_data(results["downloaded"])
                results["extracted"] = extraction_results.get("data", {})
                results["confidence"] = extraction_results.get("confidence", {})
                
                if extraction_results.get("errors"):
                    results["errors"].extend(extraction_results["errors"])
                
                # Step 3: Identify low-confidence fields
                low_confidence_fields = self._identify_low_confidence_fields(
                    results["confidence"]
                )
                results["low_confidence_fields"] = low_confidence_fields
                results["needs_validation"] = len(low_confidence_fields) > 0
                
                logger.info(
                    f"[DOCUMENT_PROCESSOR] Processing complete for claim {claim_id}: "
                    f"{len(results['downloaded'])} documents, "
                    f"{len(results['extracted'])} extracted, "
                    f"{len(low_confidence_fields)} low-confidence fields"
                )
            else:
                logger.warning(f"[DOCUMENT_PROCESSOR] No documents downloaded for claim {claim_id}")
            
            return results
            
        except Exception as e:
            logger.error(f"[DOCUMENT_PROCESSOR] Document processing failed for claim {claim_id}: {e}")
            results["errors"].append({
                "step": "process_documents",
                "error": str(e),
                "claim_id": claim_id
            })
            raise DocumentProcessingError(
                message=f"Document processing failed: {str(e)}",
                claim_id=claim_id,
                step="process_documents"
            ) from e



    
    async def _download_documents(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Download documents from UiPath storage buckets.
        
        Handles both shipping_documents and damage_evidence references,
        using the download_multiple_documents tool.
        
        Args:
            state: Current GraphState with document references
            
        Returns:
            Dictionary with:
                - files: List of downloaded file paths
                - errors: List of download errors
                
        Implements Requirements 3.1, 3.2, 3.4
        """
        claim_id = self._extract_claim_id(state)
        
        logger.info(f"[DOCUMENT_PROCESSOR] Downloading documents for claim: {claim_id}")
        
        # Collect all document references
        documents_to_download = []
        
        # Add shipping documents
        shipping_docs = state.get('shipping_documents', [])
        if shipping_docs:
            logger.info(f"[DOCUMENT_PROCESSOR] Found {len(shipping_docs)} shipping documents")
            documents_to_download.extend(shipping_docs)
        
        # Add damage evidence
        damage_evidence = state.get('damage_evidence', [])
        if damage_evidence:
            logger.info(f"[DOCUMENT_PROCESSOR] Found {len(damage_evidence)} damage evidence files")
            documents_to_download.extend(damage_evidence)
        
        if not documents_to_download:
            logger.warning(f"[DOCUMENT_PROCESSOR] No documents to download for claim {claim_id}")
            return {"files": [], "errors": []}
        
        try:
            # Import tool here to avoid circular imports
            from ..tools.document_download_tool import download_multiple_documents
            
            logger.info(f"[DOCUMENT_PROCESSOR] Downloading {len(documents_to_download)} documents")
            
            # Call the tool
            result_json = await download_multiple_documents.ainvoke({
                "claim_id": claim_id,
                "documents": documents_to_download,
                "max_concurrent": self.config.max_concurrent_downloads
            })
            
            # Parse result
            result = json.loads(result_json) if isinstance(result_json, str) else result_json
            
            # Extract file paths from successful downloads
            downloaded_files = []
            errors = []
            
            if result.get("success"):
                for doc in result.get("documents", []):
                    if doc.get("local_path"):
                        downloaded_files.append(doc["local_path"])
                
                # Track failed downloads
                for failed_doc in result.get("failed_documents", []):
                    errors.append({
                        "document": failed_doc.get("filename", "unknown"),
                        "error": failed_doc.get("error", "Download failed"),
                        "step": "download"
                    })
                
                logger.info(
                    f"[DOCUMENT_PROCESSOR] Downloaded {len(downloaded_files)} documents "
                    f"({len(errors)} failed)"
                )
            else:
                error_msg = result.get("error", "Download failed")
                logger.error(f"[DOCUMENT_PROCESSOR] Download failed: {error_msg}")
                errors.append({
                    "step": "download",
                    "error": error_msg
                })
            
            return {
                "files": downloaded_files,
                "errors": errors
            }
            
        except Exception as e:
            logger.error(f"[DOCUMENT_PROCESSOR] Document download failed: {e}")
            return {
                "files": [],
                "errors": [{
                    "step": "download",
                    "error": str(e)
                }]
            }
    
    async def _extract_data(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Extract structured data from documents using UiPath IXP.
        
        Processes each document through Document Understanding to extract
        fields with confidence scores. Flags fields below confidence threshold.
        
        Args:
            file_paths: List of local file paths to process
            
        Returns:
            Dictionary with:
                - data: Extracted data by document
                - confidence: Confidence scores by field
                - errors: List of extraction errors
                
        Implements Requirements 4.1, 4.2, 4.3, 4.4
        """
        logger.info(f"[DOCUMENT_PROCESSOR] Extracting data from {len(file_paths)} documents")
        
        if not file_paths:
            logger.warning("[DOCUMENT_PROCESSOR] No files to extract")
            return {"data": {}, "confidence": {}, "errors": []}
        
        try:
            # Prepare documents for extraction tool
            documents = [{"local_path": path} for path in file_paths]
            
            # Import tool here to avoid circular imports
            from ..tools.document_extraction_tool import extract_documents_batch
            
            # Get claim_id from first file path (format: claim_id_filename)
            first_filename = os.path.basename(file_paths[0])
            claim_id = first_filename.split('_')[0] if '_' in first_filename else 'UNKNOWN'
            
            logger.info(f"[DOCUMENT_PROCESSOR] Processing {len(documents)} documents with IXP")
            
            # Call the extraction tool
            result_json = await extract_documents_batch.ainvoke({
                "claim_id": claim_id,
                "documents": documents,
                "project_name": self.config.ixp_project_name,
                "cleanup_files": self.config.cleanup_after_extraction
            })
            
            # Parse result
            result = json.loads(result_json) if isinstance(result_json, str) else result_json
            
            # Process extraction results
            extracted_data = {}
            confidence_scores = {}
            errors = []
            
            if result.get("success"):
                for doc_result in result.get("documents", []):
                    doc_path = doc_result.get("document_path", "unknown")
                    doc_name = os.path.basename(doc_path)
                    
                    if doc_result.get("success"):
                        # Store extracted data
                        extracted_fields = doc_result.get("extracted_data", {})
                        extracted_data[doc_name] = extracted_fields
                        
                        # Extract confidence scores
                        for field_name, field_data in extracted_fields.items():
                            if isinstance(field_data, dict) and "confidence" in field_data:
                                confidence_scores[f"{doc_name}.{field_name}"] = field_data["confidence"]
                            elif isinstance(field_data, dict) and "value" in field_data:
                                # Handle nested structure
                                confidence_scores[f"{doc_name}.{field_name}"] = field_data.get("confidence", 0.0)
                        
                        logger.info(
                            f"[DOCUMENT_PROCESSOR] Extracted {len(extracted_fields)} fields from {doc_name} "
                            f"(confidence: {doc_result.get('confidence', 0):.2%})"
                        )
                    else:
                        # Track extraction failure
                        errors.append({
                            "document": doc_name,
                            "error": doc_result.get("error", "Extraction failed"),
                            "step": "extraction"
                        })
                
                logger.info(
                    f"[DOCUMENT_PROCESSOR] Extraction complete: "
                    f"{len(extracted_data)} documents processed, "
                    f"{len(confidence_scores)} fields extracted"
                )
            else:
                error_msg = result.get("error", "Extraction failed")
                logger.error(f"[DOCUMENT_PROCESSOR] Extraction failed: {error_msg}")
                errors.append({
                    "step": "extraction",
                    "error": error_msg
                })
            
            return {
                "data": extracted_data,
                "confidence": confidence_scores,
                "errors": errors
            }
            
        except Exception as e:
            logger.error(f"[DOCUMENT_PROCESSOR] Data extraction failed: {e}")
            return {
                "data": {},
                "confidence": {},
                "errors": [{
                    "step": "extraction",
                    "error": str(e)
                }]
            }
    
    def _identify_low_confidence_fields(self, confidence_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Identify fields with confidence below threshold.
        
        Flags fields that need manual review based on confidence scores.
        
        Args:
            confidence_scores: Dictionary mapping field names to confidence scores
            
        Returns:
            List of low-confidence fields with details
            
        Implements Requirement 4.3
        """
        low_confidence_fields = []
        
        for field_name, confidence in confidence_scores.items():
            if confidence < self.config.low_confidence_threshold:
                low_confidence_fields.append({
                    "field": field_name,
                    "confidence": confidence,
                    "threshold": self.config.low_confidence_threshold,
                    "requires_review": True
                })
                
                logger.warning(
                    f"[DOCUMENT_PROCESSOR] Low confidence field: {field_name} "
                    f"(confidence: {confidence:.2%}, threshold: {self.config.low_confidence_threshold:.2%})"
                )
        
        if low_confidence_fields:
            logger.info(
                f"[DOCUMENT_PROCESSOR] Identified {len(low_confidence_fields)} low-confidence fields "
                f"requiring manual review"
            )
        
        return low_confidence_fields
