"""
DocumentExtractor service for downloading files from UiPath buckets,
file validation, format checking, and temporary storage management.
"""

import logging
import os
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import mimetypes
import magic  # python-magic for file type detection

from ..models.document_models import (
    DocumentReference, DocumentMetadata, DocumentValidationResult,
    DocumentProcessingRequest, DocumentProcessingResult, ClaimDocuments,
    DocumentType, DocumentFormat, DocumentStatus
)
from ..config.settings import settings
from .uipath_service import uipath_service, UiPathServiceError

logger = logging.getLogger(__name__)


class DocumentExtractorError(Exception):
    """Custom exception for document extractor errors."""
    pass


class DocumentExtractor:
    """
    Service for downloading files from UiPath buckets with validation,
    format checking, and temporary storage management.
    """
    
    def __init__(self, base_download_dir: Optional[str] = None):
        """
        Initialize DocumentExtractor.
        
        Args:
            base_download_dir: Base directory for downloads (defaults to temp directory)
        """
        self.base_download_dir = Path(base_download_dir) if base_download_dir else Path(tempfile.gettempdir()) / "ltl_claims_documents"
        self.base_download_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported file formats and their MIME types
        self.supported_formats = {
            DocumentFormat.PDF: ["application/pdf"],
            DocumentFormat.IMAGE: [
                "image/jpeg", "image/jpg", "image/png", "image/tiff", 
                "image/bmp", "image/gif", "image/webp"
            ],
            DocumentFormat.TEXT: [
                "text/plain", "text/csv", "application/rtf",
                "application/msword", 
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ]
        }
        
        # Maximum file sizes by format (in bytes)
        self.max_file_sizes = {
            DocumentFormat.PDF: settings.max_document_size_mb * 1024 * 1024,
            DocumentFormat.IMAGE: settings.max_document_size_mb * 1024 * 1024,
            DocumentFormat.TEXT: settings.max_document_size_mb * 1024 * 1024,
            DocumentFormat.UNKNOWN: 10 * 1024 * 1024  # 10MB for unknown formats
        }
        
        logger.info(f"DocumentExtractor initialized with base directory: {self.base_download_dir}")
    
    async def download_document(
        self, 
        document_ref: DocumentReference,
        claim_id: str,
        validate: bool = True
    ) -> DocumentMetadata:
        """
        Download a single document from UiPath bucket with validation.
        
        Args:
            document_ref: Reference to the document in UiPath storage
            claim_id: Associated claim ID for organizing downloads
            validate: Whether to validate the downloaded file
            
        Returns:
            DocumentMetadata with download results
            
        Raises:
            DocumentExtractorError: If download fails
        """
        logger.info(f"📥 Downloading document: {document_ref.filename} for claim {claim_id}")
        
        # Create claim-specific directory
        claim_dir = self.base_download_dir / claim_id
        claim_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique local filename to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = self._sanitize_filename(document_ref.filename)
        local_filename = f"{document_ref.document_type.value}_{timestamp}_{safe_filename}"
        local_path = claim_dir / local_filename
        
        # Initialize metadata
        metadata = DocumentMetadata(
            reference=document_ref,
            local_path=str(local_path),
            download_status=DocumentStatus.DOWNLOADING
        )
        
        try:
            # Download file using UiPath service
            async with uipath_service:
                success = await uipath_service.download_bucket_file(
                    bucket_key=document_ref.bucket_id,
                    blob_file_path=document_ref.file_path,
                    destination_path=str(local_path),
                    folder_key=document_ref.folder_id
                )
            
            if not success or not local_path.exists():
                raise DocumentExtractorError(f"Download failed - file not found at {local_path}")
            
            # Update metadata with download success
            metadata.download_status = DocumentStatus.DOWNLOADED
            metadata.downloaded_at = datetime.now()
            
            # Get actual file size
            file_size = local_path.stat().st_size
            metadata.reference.file_size = file_size
            
            logger.info(f"✅ Downloaded {document_ref.filename} ({file_size} bytes)")
            
            # Validate file if requested
            if validate:
                validation_result = await self.validate_document(str(local_path))
                metadata.is_valid = validation_result.is_valid
                metadata.validation_errors = validation_result.validation_errors
                metadata.file_format = validation_result.file_format
                
                if not validation_result.is_valid:
                    logger.warning(f"⚠️  Document validation failed: {validation_result.validation_errors}")
                else:
                    logger.info(f"✅ Document validation passed: {document_ref.filename}")
            
            return metadata
            
        except UiPathServiceError as e:
            logger.error(f"❌ UiPath service error downloading {document_ref.filename}: {str(e)}")
            metadata.download_status = DocumentStatus.FAILED
            metadata.download_error = f"UiPath service error: {str(e)}"
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Unexpected error downloading {document_ref.filename}: {str(e)}")
            metadata.download_status = DocumentStatus.FAILED
            metadata.download_error = f"Unexpected error: {str(e)}"
            return metadata
    
    async def download_claim_documents(
        self,
        claim_id: str,
        document_refs: List[DocumentReference],
        max_concurrent: int = 3
    ) -> ClaimDocuments:
        """
        Download multiple documents for a claim concurrently.
        
        Args:
            claim_id: Claim ID for organizing downloads
            document_refs: List of document references to download
            max_concurrent: Maximum concurrent downloads
            
        Returns:
            ClaimDocuments with all download results
        """
        logger.info(f"📥 Downloading {len(document_refs)} documents for claim {claim_id}")
        
        claim_docs = ClaimDocuments(claim_id=claim_id)
        
        # Create semaphore to limit concurrent downloads
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def download_with_semaphore(doc_ref: DocumentReference, doc_key: str) -> None:
            async with semaphore:
                try:
                    metadata = await self.download_document(doc_ref, claim_id)
                    
                    # Create processing result
                    processing_result = DocumentProcessingResult(
                        request=DocumentProcessingRequest(
                            claim_id=claim_id,
                            document_reference=doc_ref
                        ),
                        metadata=metadata,
                        processing_status=DocumentStatus.DOWNLOADED if metadata.download_status == DocumentStatus.DOWNLOADED else DocumentStatus.FAILED
                    )
                    
                    claim_docs.add_document(doc_key, processing_result)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to download document {doc_ref.filename}: {str(e)}")
                    
                    # Create failed result
                    failed_metadata = DocumentMetadata(
                        reference=doc_ref,
                        download_status=DocumentStatus.FAILED,
                        download_error=str(e)
                    )
                    
                    processing_result = DocumentProcessingResult(
                        request=DocumentProcessingRequest(
                            claim_id=claim_id,
                            document_reference=doc_ref
                        ),
                        metadata=failed_metadata,
                        processing_status=DocumentStatus.FAILED,
                        processing_errors=[str(e)]
                    )
                    
                    claim_docs.add_document(doc_key, processing_result)
        
        # Create download tasks
        tasks = []
        for i, doc_ref in enumerate(document_refs):
            doc_key = f"{doc_ref.document_type.value}_{i}"
            task = asyncio.create_task(download_with_semaphore(doc_ref, doc_key))
            tasks.append(task)
        
        # Wait for all downloads to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log summary
        summary = claim_docs.get_summary()
        logger.info(f"📊 Download summary for claim {claim_id}: {summary['downloaded_count']}/{summary['total_documents']} successful")
        
        return claim_docs
    
    async def validate_document(self, file_path: str) -> DocumentValidationResult:
        """
        Validate a downloaded document for format, size, and readability.
        
        Args:
            file_path: Path to the downloaded file
            
        Returns:
            DocumentValidationResult with validation details
        """
        logger.debug(f"🔍 Validating document: {file_path}")
        
        file_path_obj = Path(file_path)
        validation_errors = []
        warnings = []
        
        # Check if file exists
        if not file_path_obj.exists():
            validation_errors.append("File does not exist")
            return DocumentValidationResult(
                is_valid=False,
                validation_errors=validation_errors,
                file_size=0,
                file_format=DocumentFormat.UNKNOWN,
                is_readable=False,
                is_corrupted=True
            )
        
        # Get file size
        file_size = file_path_obj.stat().st_size
        
        # Check if file is empty
        if file_size == 0:
            validation_errors.append("File is empty")
        
        # Detect file format
        file_format = await self._detect_file_format(file_path)
        
        # Check file size limits
        max_size = self.max_file_sizes.get(file_format, self.max_file_sizes[DocumentFormat.UNKNOWN])
        if file_size > max_size:
            validation_errors.append(f"File size ({file_size} bytes) exceeds maximum allowed ({max_size} bytes)")
        
        # Check if file is readable
        is_readable = await self._check_file_readability(file_path, file_format)
        if not is_readable:
            validation_errors.append("File is not readable or corrupted")
        
        # Check for corruption based on file format
        is_corrupted = await self._check_file_corruption(file_path, file_format)
        if is_corrupted:
            validation_errors.append("File appears to be corrupted")
        
        # Add warnings for unsupported formats
        if file_format == DocumentFormat.UNKNOWN:
            warnings.append("File format could not be determined")
        
        is_valid = len(validation_errors) == 0
        
        result = DocumentValidationResult(
            is_valid=is_valid,
            validation_errors=validation_errors,
            warnings=warnings,
            file_size=file_size,
            file_format=file_format,
            is_readable=is_readable,
            is_corrupted=is_corrupted
        )
        
        if is_valid:
            logger.debug(f"✅ Document validation passed: {file_path}")
        else:
            logger.warning(f"❌ Document validation failed: {validation_errors}")
        
        return result
    
    async def _detect_file_format(self, file_path: str) -> DocumentFormat:
        """Detect file format using MIME type detection."""
        try:
            # Use python-magic for accurate MIME type detection
            mime_type = magic.from_file(file_path, mime=True)
            
            # Map MIME type to DocumentFormat
            for doc_format, mime_types in self.supported_formats.items():
                if mime_type in mime_types:
                    return doc_format
            
            # Fallback to file extension
            file_ext = Path(file_path).suffix.lower()
            if file_ext in ['.pdf']:
                return DocumentFormat.PDF
            elif file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif', '.webp']:
                return DocumentFormat.IMAGE
            elif file_ext in ['.txt', '.csv', '.rtf', '.doc', '.docx']:
                return DocumentFormat.TEXT
            
            return DocumentFormat.UNKNOWN
            
        except Exception as e:
            logger.warning(f"⚠️  Could not detect file format for {file_path}: {str(e)}")
            return DocumentFormat.UNKNOWN
    
    async def _check_file_readability(self, file_path: str, file_format: DocumentFormat) -> bool:
        """Check if file can be read based on its format."""
        try:
            if file_format == DocumentFormat.PDF:
                # Try to open PDF file
                import PyPDF2
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    # Try to read first page
                    if len(reader.pages) > 0:
                        _ = reader.pages[0].extract_text()
                return True
                
            elif file_format == DocumentFormat.IMAGE:
                # Try to open image file
                from PIL import Image
                with Image.open(file_path) as img:
                    img.verify()
                return True
                
            elif file_format == DocumentFormat.TEXT:
                # Try to read text file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    file.read(1024)  # Read first 1KB
                return True
                
            else:
                # For unknown formats, just try to open the file
                with open(file_path, 'rb') as file:
                    file.read(1024)  # Read first 1KB
                return True
                
        except Exception as e:
            logger.debug(f"File readability check failed for {file_path}: {str(e)}")
            return False
    
    async def _check_file_corruption(self, file_path: str, file_format: DocumentFormat) -> bool:
        """Check if file appears to be corrupted."""
        try:
            if file_format == DocumentFormat.PDF:
                import PyPDF2
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    # Check if PDF has pages and can read metadata
                    return len(reader.pages) == 0
                    
            elif file_format == DocumentFormat.IMAGE:
                from PIL import Image
                with Image.open(file_path) as img:
                    img.verify()
                    # If verify() doesn't raise an exception, file is not corrupted
                return False
                
            else:
                # For other formats, assume not corrupted if readable
                return False
                
        except Exception:
            # If any exception occurs during corruption check, assume corrupted
            return True
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        # Remove or replace unsafe characters
        unsafe_chars = '<>:"/\\|?*'
        safe_filename = filename
        for char in unsafe_chars:
            safe_filename = safe_filename.replace(char, '_')
        
        # Limit filename length
        if len(safe_filename) > 200:
            name, ext = os.path.splitext(safe_filename)
            safe_filename = name[:200-len(ext)] + ext
        
        return safe_filename
    
    async def cleanup_temporary_files(
        self, 
        claim_id: Optional[str] = None,
        max_age_hours: int = 24,
        force_cleanup: bool = False
    ) -> Dict[str, Any]:
        """
        Clean up temporary downloaded files.
        
        Args:
            claim_id: Specific claim ID to clean up (None for all)
            max_age_hours: Maximum age of files to keep
            force_cleanup: Force cleanup regardless of age
            
        Returns:
            Cleanup results summary
        """
        logger.info(f"🧹 Starting cleanup of temporary files (max_age: {max_age_hours}h)")
        
        cleanup_results = {
            "files_removed": 0,
            "directories_removed": 0,
            "total_size_freed": 0,
            "errors": []
        }
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            if claim_id:
                # Clean up specific claim directory
                claim_dir = self.base_download_dir / claim_id
                if claim_dir.exists():
                    await self._cleanup_directory(claim_dir, cutoff_time, force_cleanup, cleanup_results)
            else:
                # Clean up all claim directories
                for claim_dir in self.base_download_dir.iterdir():
                    if claim_dir.is_dir():
                        await self._cleanup_directory(claim_dir, cutoff_time, force_cleanup, cleanup_results)
            
            logger.info(f"🗑️  Cleanup completed: {cleanup_results['files_removed']} files removed, "
                       f"{cleanup_results['total_size_freed']} bytes freed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {str(e)}")
            cleanup_results["errors"].append(str(e))
        
        return cleanup_results
    
    async def _cleanup_directory(
        self, 
        directory: Path, 
        cutoff_time: datetime, 
        force_cleanup: bool,
        results: Dict[str, Any]
    ) -> None:
        """Clean up files in a specific directory."""
        try:
            files_in_dir = 0
            
            for file_path in directory.iterdir():
                if file_path.is_file():
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    if force_cleanup or file_mtime < cutoff_time:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        results["files_removed"] += 1
                        results["total_size_freed"] += file_size
                    else:
                        files_in_dir += 1
            
            # Remove directory if empty
            if files_in_dir == 0 and not any(directory.iterdir()):
                directory.rmdir()
                results["directories_removed"] += 1
                
        except Exception as e:
            logger.warning(f"⚠️  Error cleaning directory {directory}: {str(e)}")
            results["errors"].append(f"Directory {directory}: {str(e)}")
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about current storage usage."""
        try:
            total_size = 0
            total_files = 0
            claim_count = 0
            
            if self.base_download_dir.exists():
                for claim_dir in self.base_download_dir.iterdir():
                    if claim_dir.is_dir():
                        claim_count += 1
                        for file_path in claim_dir.rglob('*'):
                            if file_path.is_file():
                                total_files += 1
                                total_size += file_path.stat().st_size
            
            return {
                "base_directory": str(self.base_download_dir),
                "total_claims": claim_count,
                "total_files": total_files,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "directory_exists": self.base_download_dir.exists()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting storage info: {str(e)}")
            return {
                "error": str(e),
                "base_directory": str(self.base_download_dir)
            }


# Global document extractor instance
document_extractor = DocumentExtractor()