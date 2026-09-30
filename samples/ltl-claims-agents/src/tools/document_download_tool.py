"""
Document Download Tool for downloading documents from UiPath storage buckets.
Uses actual UiPath SDK for real storage operations with proper @tool decorator.
"""

import logging
import json
import os
import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from uipath.tracing import traced

logger = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal attacks.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for filesystem operations
        
    Raises:
        ValueError: If filename is empty or invalid after sanitization
    """
    if not filename or not filename.strip():
        raise ValueError("Filename cannot be empty")
    
    # Remove any path separators and parent directory references
    filename = os.path.basename(filename)
    
    # Remove any remaining dangerous characters
    filename = re.sub(r'[^\w\s\-\.]', '_', filename)
    
    # Ensure filename is not empty after sanitization
    if not filename or filename in ('.', '..'):
        raise ValueError(f"Invalid filename after sanitization: {filename}")
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:250] + ext
    
    return filename

# Import settings for configuration
from ..config.settings import settings

# Global UiPath service instance
_uipath_service = None

@traced(name="get_uipath_storage_service", run_type="setup")
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
            logger.info("✅ UiPath Storage service initialized")
        except ImportError:
            logger.error("❌ UiPath SDK not available")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to initialize UiPath service: {e}")
            raise
    
    return _uipath_service


class DocumentReference(BaseModel):
    """Input schema for document reference."""
    bucket_id: str = Field(description="UiPath storage bucket ID or name")
    file_path: str = Field(description="Path to file within the bucket")
    filename: str = Field(description="Name of the file to download")
    folder_id: str = Field(default=None, description="Optional UiPath folder ID")


class DownloadDocumentsInput(BaseModel):
    """Input schema for downloading multiple documents."""
    claim_id: str = Field(description="Claim ID to organize downloaded files")
    documents: List[Dict[str, Any]] = Field(
        description="List of document references with bucket_id, file_path, filename"
    )
    max_concurrent: int = Field(
        default=3,
        description="Maximum number of concurrent downloads"
    )


def normalize_document_keys(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize document dictionary keys to support both snake_case and camelCase.
    
    Args:
        doc: Document dictionary with mixed key formats
        
    Returns:
        Dictionary with normalized keys
    """
    return {
        "bucket_id": doc.get("bucket_id") or doc.get("bucketId"),
        "file_path": doc.get("file_path") or doc.get("path"),
        "filename": doc.get("filename") or doc.get("fileName") or (
            os.path.basename(doc.get("file_path") or doc.get("path") or "")
        ),
        "folder_id": doc.get("folder_id") or doc.get("folderId"),
    }


def validate_document_input(doc: Dict[str, Any]) -> tuple[bool, str]:
    """Validate document input has required fields.
    
    Args:
        doc: Document dictionary to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    normalized = normalize_document_keys(doc)
    
    if not normalized["bucket_id"]:
        return False, "Missing bucket_id"
    if not normalized["file_path"]:
        return False, "Missing file_path"
    if not normalized["filename"]:
        return False, "Missing filename"
    
    return True, ""


@tool
@traced(name="download_multiple_documents", run_type="tool")
async def download_multiple_documents(
    claim_id: str,
    documents: List[Dict[str, Any]],
    max_concurrent: int = 3
) -> str:
    """Download multiple documents from UiPath storage buckets for claims processing.
    
    IMPORTANT: Use the EXACT 'path' field from the claim input data. Do NOT construct paths yourself.
    
    This tool downloads documents from UiPath storage buckets to a local downloads folder.
    The documents parameter should contain the EXACT document metadata from the claim input,
    including the 'path' field which contains the full bucket path.
    
    Args:
        claim_id: The claim ID to organize downloaded files
        documents: List of document dictionaries from claim input. Each MUST contain:
            - path: The EXACT path from the claim input (e.g., "/claims/xxx/documents/file.pdf")
            - fileName: The filename from the claim input
            - bucketId: The bucket ID from the claim input
            DO NOT construct or modify these paths - use them exactly as provided in the claim data.
        max_concurrent: Maximum number of concurrent downloads (default: 3)
    
    Returns:
        JSON string with download results including success status and local file paths
    
    Example - Use EXACT metadata from claim input:
        If claim input has:
        "shipping_documents": [{
            "bucketId": 99943,
            "path": "/claims/A628BA71/documents/BOL0001.pdf",
            "fileName": "BOL0001.pdf"
        }]
        
        Then call:
        download_multiple_documents(
            claim_id="A628BA71",
            documents=[{
                "bucketId": 99943,
                "path": "/claims/A628BA71/documents/BOL0001.pdf",
                "fileName": "BOL0001.pdf"
            }]
        )
    """
    try:
        logger.info(f"📥 Real UiPath downloading {len(documents)} documents for claim {claim_id}")
        logger.info(f"📋 Document input received: {json.dumps(documents, indent=2)}")
        
        # Create downloads directory
        downloads_dir = os.path.join(os.getcwd(), "downloads")
        os.makedirs(downloads_dir, exist_ok=True)
        
        service = await _get_uipath_service()
        
        downloaded_docs = []
        failed_docs = []
        
        for doc in documents:
            try:
                # Validate and normalize document input
                is_valid, error_msg = validate_document_input(doc)
                if not is_valid:
                    logger.warning(f"Invalid document input: {error_msg} - {doc}")
                    failed_docs.append({**doc, "error": error_msg})
                    continue
                
                # Get normalized values
                normalized = normalize_document_keys(doc)
                bucket_id = normalized["bucket_id"]
                file_path = normalized["file_path"]
                filename = normalized["filename"]
                folder_id = normalized["folder_id"]
                
                logger.info(f"📋 Normalized document: bucket_id={bucket_id}, file_path={file_path}, filename={filename}")
                
                # Sanitize inputs to prevent directory traversal
                safe_claim_id = sanitize_filename(claim_id)
                safe_filename = sanitize_filename(filename)
                
                # Create local path in downloads folder
                local_path = os.path.join(downloads_dir, f"{safe_claim_id}_{safe_filename}")
                
                try:
                    # Use actual UiPath storage download with exact SDK signature
                    logger.info(f"📥 Downloading from UiPath bucket {bucket_id}: {file_path}")
                    
                    # Download file from UiPath storage bucket using exact SDK signature
                    # sdk.buckets.download_async(name: Optional[str]=None, key: Optional[str]=None, 
                    #                            blob_file_path: str, destination_path: str, 
                    #                            folder_key: Optional[str]=None, folder_path: Optional[str]=None)
                    
                    # Always use bucket name instead of ID for better reliability
                    bucket_name = getattr(settings, 'uipath_bucket_name', 'LTL Freight Claim')
                    
                    # Ensure path doesn't have leading slash for UiPath API
                    # UiPath expects paths like "claims/xxx/documents/file.pdf" not "/claims/xxx/documents/file.pdf"
                    clean_path = file_path.lstrip('/')
                    
                    logger.info(f"📥 Downloading from bucket '{bucket_name}': {clean_path}")
                    
                    await service.buckets.download_async(
                        name=bucket_name,  # Use bucket name (more reliable than ID)
                        blob_file_path=clean_path,  # Required: path in bucket (without leading slash)
                        destination_path=local_path,  # Required: local destination
                        folder_path=getattr(settings, 'uipath_folder_path', None)  # Use folder path from settings
                    )
                    
                    # Verify file was actually downloaded
                    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                        downloaded_docs.append({
                            **doc,
                            "local_path": local_path,
                            "download_status": "downloaded",
                            "file_size": os.path.getsize(local_path)
                        })
                        logger.info(f"✅ Real UiPath download successful: {filename} ({os.path.getsize(local_path)} bytes)")
                    else:
                        # File doesn't exist or is empty - this is a real failure
                        error_msg = f"Download completed but file not found or empty: {local_path}"
                        logger.error(f"❌ {error_msg}")
                        failed_docs.append({
                            **doc,
                            "error": error_msg,
                            "download_status": "failed"
                        })
                
                except Exception as download_error:
                    logger.error(f"❌ UiPath download failed for {filename}: {download_error}")
                    
                    # Report actual failure - NO PLACEHOLDERS
                    failed_docs.append({
                        **doc,
                        "error": str(download_error),
                        "download_status": "failed"
                    })
                    
            except Exception as e:
                logger.error(f"❌ Error processing document {doc}: {e}")
                failed_docs.append({**doc, "error": str(e)})
        
        success_rate = len(downloaded_docs) / len(documents) if documents else 0
        
        result = {
            "success": True,
            "claim_id": claim_id,
            "total_documents": len(documents),
            "downloaded_count": len(downloaded_docs),
            "failed_count": len(failed_docs),
            "success_rate": success_rate,
            "documents": downloaded_docs,
            "failed_documents": failed_docs,
            "uipath_storage_used": True
        }
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"❌ Real UiPath document download failed: {e}")
        result = {
            "success": False,
            "error": str(e),
            "claim_id": claim_id,
            "downloaded_count": 0,
            "failed_count": len(documents) if documents else 0,
            "uipath_storage_used": False
        }
        return json.dumps(result)