"""
Document Extraction Tool for extracting data from documents using UiPath Document Understanding.
Uses actual UiPath IXP (Document Understanding) for real document processing with proper @tool decorator.
"""

import logging
import json
import os
import asyncio
from typing import Any, Dict, List

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from uipath.tracing import traced

logger = logging.getLogger(__name__)

# Import settings for configuration
from ..config.settings import settings

# Global UiPath service instance
_uipath_service = None

@traced(name="get_uipath_du_service", run_type="setup")
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
            logger.info("✅ UiPath Document Understanding service initialized")
        except ImportError:
            logger.error("❌ UiPath SDK not available")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to initialize UiPath service: {e}")
            raise
    
    return _uipath_service

@traced(name="cleanup_document_files", run_type="cleanup")
async def _cleanup_files(file_paths: List[str]):
    """Clean up downloaded files after processing.
    
    Args:
        file_paths: List of file paths to delete
    """
    cleaned_count = 0
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                cleaned_count += 1
                logger.info(f"🗑️ Real cleanup: {os.path.basename(file_path)}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to clean up file {file_path}: {e}")
    
    if cleaned_count > 0:
        logger.info(f"✅ Real cleanup completed: {cleaned_count} files removed")


class DocumentToExtract(BaseModel):
    """Input schema for document to extract."""
    document_path: str = Field(description="Local path to the document file")
    local_path: str = Field(default=None, description="Alternative field for local path")


class ExtractDocumentsInput(BaseModel):
    """Input schema for batch document extraction."""
    claim_id: str = Field(description="Claim ID for organizing extraction results")
    documents: List[Dict[str, Any]] = Field(
        description="List of documents with local_path or document_path for processing"
    )
    project_name: str = Field(
        default=None,
        description="UiPath Document Understanding project name (uses settings default if not provided)"
    )
    cleanup_files: bool = Field(
        default=False,
        description="Whether to delete files after extraction"
    )


@tool
@traced(name="extract_documents_batch", run_type="tool")
async def extract_documents_batch(
    claim_id: str,
    documents: List[Dict[str, Any]],
    project_name: str = None,
    cleanup_files: bool = False
) -> str:
    """Extract structured data from multiple documents using UiPath Document Understanding (IXP).
    
    This tool processes downloaded documents using UiPath's Document Understanding service (IXP)
    to extract structured data from damage photos, shipping documents, bills of lading, invoices,
    and other claim evidence. The tool uses machine learning models to identify and extract
    relevant fields with confidence scores.
    
    Args:
        claim_id: The claim ID for organizing extraction results (e.g., "CLM-2024-001")
        documents: List of document dictionaries, each containing:
            - document_path or local_path: Local file path to the document
        project_name: UiPath Document Understanding project name (optional, uses settings default)
        cleanup_files: Whether to delete files after extraction (default: False to preserve files)
    
    Returns:
        JSON string containing:
            - success: Boolean indicating overall operation success
            - claim_id: The claim ID provided
            - processed_count: Number of documents processed
            - high_confidence_count: Number of extractions with confidence >= 0.8
            - low_confidence_count: Number of extractions with confidence < 0.8
            - needs_validation: Boolean indicating if manual validation is needed
            - documents: List of extraction results with:
                - success: Boolean for individual document
                - document_path: Path to the processed document
                - extracted_data: Dictionary of extracted fields with values and confidence
                - confidence: Average confidence score (0.0 to 1.0)
                - extraction_status: Status (completed, failed, error)
            - files_cleaned_up: Number of files deleted (if cleanup_files=True)
            - uipath_ixp_used: Boolean indicating if UiPath IXP was used
    
    Example:
        Input: {
            "claim_id": "CLM-2024-001",
            "documents": [
                {"local_path": "downloads/CLM-2024-001_bol.pdf"}
            ],
            "cleanup_files": false
        }
        Output: {
            "success": true,
            "processed_count": 1,
            "high_confidence_count": 1,
            "documents": [{
                "extracted_data": {
                    "shipment_id": {"value": "SHP-001", "confidence": 0.95},
                    "carrier": {"value": "Speedy Freight", "confidence": 0.92}
                }
            }]
        }
    """
    files_to_cleanup = []
    try:
        # Use settings if project_name not provided
        if project_name is None:
            project_name = settings.uipath_du_project_name
        
        logger.info(f"🔍 Real UiPath IXP extracting data from {len(documents)} documents for claim {claim_id}")
        logger.info(f"📋 Using IXP project: {project_name} (tag: {settings.uipath_du_project_tag})")
        
        service = await _get_uipath_service()
        
        extracted_docs = []
        high_confidence_count = 0
        low_confidence_count = 0
        
        for doc in documents:
            document_path = doc.get("document_path") or doc.get("local_path")
            if not document_path:
                logger.warning(f"⚠️ No document path for: {doc}")
                continue
            
            # Normalize path - handle both absolute and relative paths
            # If path starts with / or \ but doesn't exist, try as relative path
            if document_path.startswith(('/', '\\')):
                # Try as absolute path first
                if not os.path.exists(document_path):
                    # Try as relative path from current directory
                    relative_path = document_path.lstrip('/\\')
                    if os.path.exists(relative_path):
                        document_path = relative_path
                        logger.info(f"📁 Using relative path: {document_path}")
                    else:
                        # Try with current working directory
                        cwd_path = os.path.join(os.getcwd(), relative_path)
                        if os.path.exists(cwd_path):
                            document_path = cwd_path
                            logger.info(f"📁 Using CWD path: {document_path}")
            
            # Verify file exists
            if not os.path.exists(document_path):
                logger.error(f"❌ File not found: {document_path}")
                extracted_docs.append({
                    **doc,
                    "success": False,
                    "error": f"File not found: {document_path}",
                    "confidence": 0.0
                })
                continue
            
            # Track files for cleanup
            files_to_cleanup.append(document_path)
            
            try:
                # Use actual UiPath Document Understanding with exact SDK signature
                logger.info(f"🔍 Processing document with UiPath IXP: {os.path.basename(document_path)}")
                
                # Extract data using UiPath Document Understanding
                from uipath.models.documents import ProjectType
                
                extraction_result = await service.documents.extract_async(
                    project_name=project_name,  # IXP project name
                    tag=settings.uipath_du_project_tag,  # Project version tag from settings
                    file_path=document_path,  # Local file path to process
                    project_type=ProjectType.IXP  # Specify IXP project type
                )
                
                # Process extraction result using data_projection
                if extraction_result and hasattr(extraction_result, 'data_projection'):
                    # Extract all fields from data_projection
                    extracted_fields = {}
                    total_confidence = 0
                    field_count = 0
                    
                    for field_group in extraction_result.data_projection:
                        group_name = field_group.field_group_name
                        
                        for field in field_group.field_values:
                            field_name = field.name
                            field_value = field.value
                            field_confidence = field.confidence
                            
                            # Store field data
                            extracted_fields[field_name] = {
                                "value": field_value,
                                "confidence": field_confidence,
                                "ocr_confidence": field.ocr_confidence,
                                "type": str(field.type),
                                "group": group_name
                            }
                            
                            # Calculate average confidence
                            if field_confidence and field_confidence > 0:
                                total_confidence += field_confidence
                                field_count += 1
                    
                    avg_confidence = total_confidence / field_count if field_count > 0 else 0
                    
                    logger.info(f"📋 Extracted {len(extracted_fields)} fields from document")
                    logger.info(f"📊 Average confidence: {avg_confidence:.2%}")
                    
                    # Determine if high or low confidence
                    if avg_confidence >= 0.8:
                        high_confidence_count += 1
                    else:
                        low_confidence_count += 1
                    
                    result = {
                        "success": True,
                        "document_path": document_path,
                        "filename": os.path.basename(document_path),
                        "extracted_data": extracted_fields,
                        "confidence": avg_confidence,
                        "field_count": len(extracted_fields),
                        "extraction_status": "completed",
                        "uipath_ixp_used": True
                    }
                    
                    logger.info(f"✅ Real UiPath IXP extraction successful: {os.path.basename(document_path)} ({len(extracted_fields)} fields, avg confidence: {avg_confidence:.2%})")
                
                else:
                    # No data_projection or failed extraction
                    logger.warning(f"⚠️ UiPath IXP extraction failed or no data_projection: {os.path.basename(document_path)}")
                    
                    # Fallback analysis based on filename
                    filename = os.path.basename(document_path)
                    if "damage" in filename.lower():
                        extracted_data = {
                            "damage_type": "document analysis needed",
                            "confidence": 0.3,
                            "requires_manual_review": True,
                            "fallback_reason": "UiPath IXP extraction failed"
                        }
                    else:
                        extracted_data = {
                            "document_type": "unknown",
                            "confidence": 0.3,
                            "requires_manual_review": True,
                            "fallback_reason": "UiPath IXP extraction failed"
                        }
                    
                    low_confidence_count += 1
                    
                    result = {
                        "success": False,
                        "document_path": document_path,
                        "extracted_data": extracted_data,
                        "confidence": 0.3,
                        "extraction_status": "failed",
                        "uipath_ixp_used": True,
                        "error": "Low confidence or extraction failed"
                    }
            
            except Exception as extraction_error:
                logger.error(f"❌ UiPath IXP extraction failed for {document_path}: {extraction_error}")
                
                # Fallback: basic file analysis
                filename = os.path.basename(document_path)
                if "damage" in filename.lower():
                    extracted_data = {
                        "damage_type": "file analysis - potential damage evidence",
                        "severity": "unknown - requires manual review",
                        "confidence": 0.2,
                        "fallback_analysis": True,
                        "error": str(extraction_error)
                    }
                else:
                    extracted_data = {
                        "document_type": "unknown document",
                        "confidence": 0.2,
                        "fallback_analysis": True,
                        "error": str(extraction_error)
                    }
                
                low_confidence_count += 1
                
                result = {
                    "success": False,
                    "document_path": document_path,
                    "extracted_data": extracted_data,
                    "confidence": 0.2,
                    "extraction_status": "error",
                    "uipath_ixp_used": False,
                    "error": str(extraction_error)
                }
            
            extracted_docs.append(result)
            logger.info(f"📄 Processed document: {os.path.basename(document_path)}")
        
        # Clean up downloaded files after extraction (only if requested)
        if cleanup_files:
            await _cleanup_files(files_to_cleanup)
            logger.info(f"🗑️ Cleaned up {len(files_to_cleanup)} files")
        else:
            logger.info(f"📁 Keeping {len(files_to_cleanup)} files in downloads folder")
        
        result = {
            "success": True,
            "claim_id": claim_id,
            "processed_count": len(extracted_docs),
            "high_confidence_count": high_confidence_count,
            "low_confidence_count": low_confidence_count,
            "needs_validation": low_confidence_count > 0,
            "documents": extracted_docs,
            "files_cleaned_up": len(files_to_cleanup),
            "uipath_ixp_used": True
        }
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"❌ Real UiPath IXP batch extraction failed: {e}")
        # Still try to clean up files even if extraction failed (only if requested)
        if cleanup_files:
            await _cleanup_files(files_to_cleanup)
        result = {
            "success": False,
            "error": str(e),
            "claim_id": claim_id,
            "processed_count": 0,
            "files_cleaned_up": len(files_to_cleanup),
            "uipath_ixp_used": False
        }
        return json.dumps(result)
