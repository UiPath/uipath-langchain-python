"""
UiPath Storage Bucket service for downloading claim documents.
Handles shipping documents and damage evidence files.
"""

import logging
import os
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

try:
    from ..config.settings import settings
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from config.settings import settings

from .uipath_service import uipath_service, UiPathServiceError

logger = logging.getLogger(__name__)


class StorageServiceError(Exception):
    """Custom exception for storage service errors."""
    pass


class DocumentInfo:
    """Information about a document in storage."""
    
    def __init__(
        self,
        bucket_id: str,
        folder_id: str,
        file_path: str,
        filename: str,
        document_type: str
    ):
        self.bucket_id = bucket_id
        self.folder_id = folder_id
        self.file_path = file_path
        self.filename = filename
        self.document_type = document_type
        self.local_path: Optional[str] = None
        self.download_status: str = "pending"
        self.download_error: Optional[str] = None
        self.file_size: Optional[int] = None
        self.downloaded_at: Optional[datetime] = None
    
    def __repr__(self):
        return f"DocumentInfo(type={self.document_type}, filename={self.filename}, status={self.download_status})"


class UiPathStorageService:
    """Service for handling UiPath Storage Bucket operations."""
    
    def __init__(self, download_directory: str = "downloads"):
        self.download_directory = Path(download_directory)
        self.download_directory.mkdir(exist_ok=True)
        
    async def download_claim_documents(
        self,
        claim_id: str,
        shipping_bucket_id: str,
        damage_bucket_id: str,
        folder_id: str,
        shipping_path: str,
        damage_path: str,
        shipping_filename: str,
        damage_filename: str
    ) -> Dict[str, DocumentInfo]:
        """
        Download all documents for a claim.
        
        Args:
            claim_id: The claim ID for organizing downloads
            shipping_bucket_id: Bucket ID for shipping documents
            damage_bucket_id: Bucket ID for damage evidence
            folder_id: Folder ID in the bucket
            shipping_path: Path to shipping document
            damage_path: Path to damage evidence
            shipping_filename: Shipping document filename
            damage_filename: Damage evidence filename
            
        Returns:
            Dict mapping document type to DocumentInfo
        """
        logger.info(f"📥 Downloading documents for claim: {claim_id}")
        
        # Create claim-specific download directory
        claim_dir = self.download_directory / claim_id
        claim_dir.mkdir(exist_ok=True)
        
        # Prepare document info
        documents = {
            "shipping": DocumentInfo(
                bucket_id=shipping_bucket_id,
                folder_id=folder_id,
                file_path=shipping_path,
                filename=shipping_filename,
                document_type="shipping"
            ),
            "damage_evidence": DocumentInfo(
                bucket_id=damage_bucket_id,
                folder_id=folder_id,
                file_path=damage_path,
                filename=damage_filename,
                document_type="damage_evidence"
            )
        }
        
        # Download documents concurrently
        download_tasks = []
        for doc_type, doc_info in documents.items():
            task = asyncio.create_task(
                self._download_document(doc_info, claim_dir)
            )
            download_tasks.append(task)
        
        # Wait for all downloads to complete
        await asyncio.gather(*download_tasks, return_exceptions=True)
        
        # Log results
        successful = sum(1 for doc in documents.values() if doc.download_status == "completed")
        total = len(documents)
        
        logger.info(f"📊 Download summary: {successful}/{total} documents downloaded successfully")
        
        for doc_type, doc_info in documents.items():
            if doc_info.download_status == "completed":
                logger.info(f"✅ {doc_type}: {doc_info.filename} → {doc_info.local_path}")
            else:
                logger.error(f"❌ {doc_type}: {doc_info.filename} → {doc_info.download_error}")
        
        return documents
    
    async def _download_document(self, doc_info: DocumentInfo, claim_dir: Path) -> None:
        """Download a single document from storage bucket."""
        try:
            logger.debug(f"📄 Downloading {doc_info.document_type}: {doc_info.filename}")
            
            doc_info.download_status = "downloading"
            
            # Determine local file path
            local_filename = f"{doc_info.document_type}_{doc_info.filename}"
            local_path = claim_dir / local_filename
            doc_info.local_path = str(local_path)
            
            async with uipath_service:
                # Download file from bucket
                await uipath_service._client.buckets.download_async(
                    key=doc_info.bucket_id,
                    blob_file_path=doc_info.file_path,
                    destination_path=str(local_path),
                    folder_key=doc_info.folder_id
                )
            
            # Verify download
            if local_path.exists():
                doc_info.file_size = local_path.stat().st_size
                doc_info.download_status = "completed"
                doc_info.downloaded_at = datetime.now()
                logger.debug(f"✅ Downloaded {doc_info.filename} ({doc_info.file_size} bytes)")
            else:
                raise StorageServiceError(f"File not found after download: {local_path}")
                
        except Exception as e:
            doc_info.download_status = "failed"
            doc_info.download_error = str(e)
            logger.error(f"❌ Failed to download {doc_info.filename}: {e}")
    
    async def download_single_document(
        self,
        bucket_id: str,
        file_path: str,
        filename: str,
        destination_dir: Optional[str] = None,
        folder_id: Optional[str] = None
    ) -> DocumentInfo:
        """
        Download a single document from storage bucket.
        
        Args:
            bucket_id: Storage bucket ID
            file_path: Path to file in bucket
            filename: Original filename
            destination_dir: Local destination directory
            folder_id: Optional folder ID
            
        Returns:
            DocumentInfo with download results
        """
        logger.info(f"📥 Downloading single document: {filename}")
        
        # Prepare destination
        if destination_dir:
            dest_dir = Path(destination_dir)
        else:
            dest_dir = self.download_directory
        
        dest_dir.mkdir(exist_ok=True)
        
        # Create document info
        doc_info = DocumentInfo(
            bucket_id=bucket_id,
            folder_id=folder_id or "",
            file_path=file_path,
            filename=filename,
            document_type="single"
        )
        
        # Download the document
        await self._download_document(doc_info, dest_dir)
        
        return doc_info
    
    async def list_bucket_contents(
        self,
        bucket_name: Optional[str] = None,
        bucket_key: Optional[str] = None,
        folder_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List contents of a storage bucket.
        
        Args:
            bucket_name: Bucket name
            bucket_key: Bucket key/ID
            folder_id: Optional folder ID
            
        Returns:
            Bucket information and contents
        """
        logger.info(f"📋 Listing bucket contents: {bucket_name or bucket_key}")
        
        try:
            async with uipath_service:
                # Retrieve bucket information
                bucket_info = await uipath_service._client.buckets.retrieve_async(
                    name=bucket_name,
                    key=bucket_key,
                    folder_key=folder_id
                )
                
                logger.info(f"✅ Retrieved bucket info: {bucket_info}")
                
                return {
                    "success": True,
                    "bucket_info": bucket_info,
                    "bucket_name": bucket_info.name if hasattr(bucket_info, 'name') else None,
                    "bucket_key": bucket_info.key if hasattr(bucket_info, 'key') else None
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to list bucket contents: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def verify_document_access(
        self,
        bucket_id: str,
        file_path: str,
        folder_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify that a document can be accessed without downloading it.
        
        Args:
            bucket_id: Storage bucket ID
            file_path: Path to file in bucket
            folder_id: Optional folder ID
            
        Returns:
            Verification results
        """
        logger.debug(f"🔍 Verifying document access: {file_path}")
        
        try:
            async with uipath_service:
                # Try to get bucket info first
                bucket_info = await uipath_service._client.buckets.retrieve_async(
                    key=bucket_id,
                    folder_key=folder_id
                )
                
                return {
                    "success": True,
                    "accessible": True,
                    "bucket_info": bucket_info,
                    "file_path": file_path
                }
                
        except Exception as e:
            logger.warning(f"⚠️  Document access verification failed: {e}")
            return {
                "success": False,
                "accessible": False,
                "error": str(e),
                "file_path": file_path
            }
    
    def get_download_summary(self, documents: Dict[str, DocumentInfo]) -> Dict[str, Any]:
        """Get a summary of download results."""
        total = len(documents)
        completed = sum(1 for doc in documents.values() if doc.download_status == "completed")
        failed = sum(1 for doc in documents.values() if doc.download_status == "failed")
        
        total_size = sum(
            doc.file_size for doc in documents.values() 
            if doc.file_size is not None
        )
        
        return {
            "total_documents": total,
            "completed": completed,
            "failed": failed,
            "success_rate": completed / total if total > 0 else 0,
            "total_size_bytes": total_size,
            "documents": {
                doc_type: {
                    "filename": doc.filename,
                    "status": doc.download_status,
                    "local_path": doc.local_path,
                    "file_size": doc.file_size,
                    "error": doc.download_error
                }
                for doc_type, doc in documents.items()
            }
        }
    
    def cleanup_downloads(self, claim_id: str, max_age_days: int = 7) -> Dict[str, Any]:
        """
        Clean up old downloaded files for a claim.
        
        Args:
            claim_id: Claim ID to clean up
            max_age_days: Maximum age of files to keep
            
        Returns:
            Cleanup results
        """
        logger.info(f"🧹 Cleaning up downloads for claim: {claim_id}")
        
        claim_dir = self.download_directory / claim_id
        
        if not claim_dir.exists():
            return {"success": True, "message": "No downloads to clean up"}
        
        try:
            import shutil
            from datetime import timedelta
            
            cutoff_time = datetime.now() - timedelta(days=max_age_days)
            files_removed = 0
            total_size_removed = 0
            
            for file_path in claim_dir.iterdir():
                if file_path.is_file():
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_mtime < cutoff_time:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        files_removed += 1
                        total_size_removed += file_size
            
            # Remove directory if empty
            if not any(claim_dir.iterdir()):
                claim_dir.rmdir()
            
            logger.info(f"🗑️  Cleaned up {files_removed} files ({total_size_removed} bytes)")
            
            return {
                "success": True,
                "files_removed": files_removed,
                "size_removed": total_size_removed,
                "directory_removed": not claim_dir.exists()
            }
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Global storage service instance
storage_service = UiPathStorageService()