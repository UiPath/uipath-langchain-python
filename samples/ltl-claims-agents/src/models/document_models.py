"""
Pydantic models for document processing and extraction.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Types of documents that can be processed."""
    SHIPPING_DOCUMENT = "shipping_document"
    DAMAGE_EVIDENCE = "damage_evidence"
    BILL_OF_LADING = "bill_of_lading"
    INVOICE = "invoice"
    PHOTO = "photo"
    REPORT = "report"
    OTHER = "other"


class DocumentFormat(str, Enum):
    """Supported document formats."""
    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"
    UNKNOWN = "unknown"


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    VALIDATED = "validated"


class DocumentReference(BaseModel):
    """Reference to a document in UiPath storage."""
    bucket_id: str = Field(description="UiPath storage bucket ID")
    folder_id: Optional[str] = Field(default=None, description="Folder ID within bucket")
    file_path: str = Field(description="Path to file within bucket")
    filename: str = Field(description="Original filename")
    document_type: DocumentType = Field(description="Type of document")
    content_type: Optional[str] = Field(default=None, description="MIME type")
    file_size: Optional[int] = Field(default=None, description="File size in bytes")
    created_at: Optional[datetime] = Field(default=None, description="Creation timestamp")


class DocumentMetadata(BaseModel):
    """Metadata about a downloaded document."""
    reference: DocumentReference = Field(description="Original document reference")
    local_path: Optional[str] = Field(default=None, description="Local file path after download")
    download_status: DocumentStatus = Field(default=DocumentStatus.PENDING, description="Download status")
    download_error: Optional[str] = Field(default=None, description="Error message if download failed")
    downloaded_at: Optional[datetime] = Field(default=None, description="Download completion timestamp")
    file_format: DocumentFormat = Field(default=DocumentFormat.UNKNOWN, description="Detected file format")
    is_valid: bool = Field(default=False, description="Whether file passed validation")
    validation_errors: List[str] = Field(default_factory=list, description="Validation error messages")


class DocumentExtractionResult(BaseModel):
    """Results from document information extraction."""
    document_path: str = Field(description="Path to the document that was extracted")
    document_type: str = Field(description="Type of document (shipping_document, damage_evidence, etc.)")
    extracted_fields: Dict[str, Any] = Field(default_factory=dict, description="Extracted fields as key-value pairs")
    confidence_scores: Dict[str, float] = Field(default_factory=dict, description="Confidence scores for each extracted field")
    processing_time: float = Field(default=0.0, description="Time taken to process document in seconds")
    extraction_method: str = Field(description="Method used for extraction (uipath_ixp, OCR, etc.)")
    metadata: "DocumentMetadata" = Field(description="Document metadata and processing information")
    
    # Optional legacy fields for backward compatibility
    document_id: Optional[str] = Field(default=None, description="Unique identifier for the document")
    extracted_text: Optional[str] = Field(default=None, description="Raw extracted text")
    confidence_score: Optional[float] = Field(default=None, description="Overall extraction confidence (0-1)")
    extracted_at: Optional[datetime] = Field(default=None, description="Extraction timestamp")
    
    # Structured data fields (legacy)
    damage_descriptions: List[str] = Field(default_factory=list, description="Extracted damage descriptions")
    monetary_amounts: List[float] = Field(default_factory=list, description="Extracted monetary amounts")
    dates: List[datetime] = Field(default_factory=list, description="Extracted dates")
    parties: List[Dict[str, str]] = Field(default_factory=list, description="Extracted party information")
    tracking_numbers: List[str] = Field(default_factory=list, description="Extracted tracking numbers")
    
    # Confidence scores for individual fields (legacy)
    field_confidence: Dict[str, float] = Field(default_factory=dict, description="Confidence scores for specific fields")
    
    # Raw extraction data for debugging
    raw_extraction_data: Optional[Dict[str, Any]] = Field(default=None, description="Raw extraction results")


class DocumentValidationResult(BaseModel):
    """Results from document validation."""
    is_valid: bool = Field(description="Whether document passed validation")
    validation_errors: List[str] = Field(default_factory=list, description="List of validation errors")
    warnings: List[str] = Field(default_factory=list, description="List of validation warnings")
    file_size: int = Field(description="File size in bytes")
    file_format: DocumentFormat = Field(description="Detected file format")
    is_readable: bool = Field(description="Whether file can be read/opened")
    is_corrupted: bool = Field(description="Whether file appears corrupted")
    validated_at: datetime = Field(default_factory=datetime.now, description="Validation timestamp")


class DocumentProcessingRequest(BaseModel):
    """Request for processing a document."""
    claim_id: str = Field(description="Associated claim ID")
    document_reference: DocumentReference = Field(description="Document to process")
    processing_options: Dict[str, Any] = Field(default_factory=dict, description="Processing configuration")
    priority: str = Field(default="medium", description="Processing priority")
    requested_at: datetime = Field(default_factory=datetime.now, description="Request timestamp")


class DocumentProcessingResult(BaseModel):
    """Complete result of document processing."""
    request: DocumentProcessingRequest = Field(description="Original processing request")
    metadata: DocumentMetadata = Field(description="Document metadata and download info")
    validation: Optional[DocumentValidationResult] = Field(default=None, description="Validation results")
    extraction: Optional[DocumentExtractionResult] = Field(default=None, description="Extraction results")
    
    processing_status: DocumentStatus = Field(description="Overall processing status")
    processing_errors: List[str] = Field(default_factory=list, description="Processing error messages")
    processing_warnings: List[str] = Field(default_factory=list, description="Processing warnings")
    
    started_at: datetime = Field(default_factory=datetime.now, description="Processing start time")
    completed_at: Optional[datetime] = Field(default=None, description="Processing completion time")
    processing_duration: Optional[float] = Field(default=None, description="Processing duration in seconds")


class ClaimDocuments(BaseModel):
    """Collection of documents for a claim."""
    claim_id: str = Field(description="Associated claim ID")
    documents: Dict[str, DocumentProcessingResult] = Field(
        default_factory=dict, 
        description="Documents keyed by document type or identifier"
    )
    
    total_documents: int = Field(default=0, description="Total number of documents")
    downloaded_count: int = Field(default=0, description="Number of successfully downloaded documents")
    processed_count: int = Field(default=0, description="Number of successfully processed documents")
    failed_count: int = Field(default=0, description="Number of failed documents")
    
    created_at: datetime = Field(default_factory=datetime.now, description="Collection creation time")
    updated_at: Optional[datetime] = Field(default=None, description="Last update time")
    
    def add_document(self, doc_key: str, result: DocumentProcessingResult) -> None:
        """Add a document processing result to the collection."""
        self.documents[doc_key] = result
        self.total_documents = len(self.documents)
        self._update_counts()
        self.updated_at = datetime.now()
    
    def _update_counts(self) -> None:
        """Update document counts based on current status."""
        self.downloaded_count = sum(
            1 for doc in self.documents.values() 
            if doc.metadata.download_status == DocumentStatus.DOWNLOADED
        )
        self.processed_count = sum(
            1 for doc in self.documents.values() 
            if doc.processing_status == DocumentStatus.PROCESSED
        )
        self.failed_count = sum(
            1 for doc in self.documents.values() 
            if doc.processing_status == DocumentStatus.FAILED
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of document processing status."""
        return {
            "claim_id": self.claim_id,
            "total_documents": self.total_documents,
            "downloaded_count": self.downloaded_count,
            "processed_count": self.processed_count,
            "failed_count": self.failed_count,
            "success_rate": self.processed_count / self.total_documents if self.total_documents > 0 else 0,
            "documents": {
                doc_key: {
                    "filename": doc.metadata.reference.filename,
                    "type": doc.metadata.reference.document_type,
                    "status": doc.processing_status,
                    "download_status": doc.metadata.download_status,
                    "local_path": doc.metadata.local_path,
                    "file_size": doc.metadata.reference.file_size,
                    "has_extraction": doc.extraction is not None,
                    "extraction_confidence": doc.extraction.confidence_score if doc.extraction else 0.0
                }
                for doc_key, doc in self.documents.items()
            }
        }