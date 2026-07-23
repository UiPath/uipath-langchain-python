"""
Pydantic models for claim input data structures.
These are reference models - the agent will autonomously extract and structure data.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field


class ClaimType(str, Enum):
    """Types of claims."""
    DAMAGE = "damage"
    LOSS = "loss"
    SHORTAGE = "shortage"
    DELAY = "delay"
    OTHER = "other"


class SubmissionSource(str, Enum):
    """Sources of claim submission."""
    DAMAGE_PHOTOS_ONLY = "damage-photos-only-test"
    SHIPPING_DOCS_ONLY = "shipping-docs-only-test"
    COMPLETE_WORKFLOW = "complete-workflow-test"
    MANUAL_ENTRY = "manual-entry"
    API_SUBMISSION = "api-submission"


class ProcessingPriority(str, Enum):
    """Processing priority levels."""
    LOW = "Low"
    NORMAL = "Normal"
    HIGH = "High"
    CRITICAL = "Critical"


class FileInfo(BaseModel):
    """Information about uploaded files."""
    bucketId: int = Field(description="Storage bucket ID")
    folderId: int = Field(description="Folder ID within bucket")
    path: str = Field(description="Full path to file")
    fileName: str = Field(description="Original filename")
    size: int = Field(description="File size in bytes")
    type: str = Field(description="MIME type")
    uploadedAt: str = Field(description="Upload timestamp (ISO string)")


class ClaimInputData(BaseModel):
    """
    Reference model for claim input data structure.
    The agent will autonomously extract and populate this structure from raw input.
    """
    # Core claim information
    ObjectClaimId: str = Field(description="Unique claim identifier")
    ClaimType: str = Field(description="Type of claim (damage, loss, etc.)")
    ClaimAmount: Union[str, float] = Field(description="Claimed amount")
    Carrier: str = Field(description="Carrier name")
    
    # Shipment information
    ShipmentID: str = Field(description="Associated shipment ID")
    
    # Customer information
    CustomerName: str = Field(description="Customer full name")
    CustomerEmail: str = Field(description="Customer email address")
    CustomerPhone: str = Field(description="Customer phone number")
    
    # Claim details
    Description: str = Field(description="Claim description")
    SubmissionSource: str = Field(description="Source of submission")
    SubmittedAt: str = Field(description="Submission timestamp")
    
    # Document storage information
    ShippingDocumentsBucketId: Optional[Union[str, int]] = Field(default=None, description="Shipping docs bucket ID")
    DamageEvidenceBucketId: Optional[Union[str, int]] = Field(default=None, description="Damage evidence bucket ID")
    FolderId: Optional[Union[str, int]] = Field(default=None, description="Folder ID")
    
    # File information
    ShippingDocumentsFiles: List[FileInfo] = Field(default_factory=list, description="Shipping document files")
    DamageEvidenceFiles: List[FileInfo] = Field(default_factory=list, description="Damage evidence files")
    
    # File paths (legacy format support)
    ShippingDocumentsPath: Optional[str] = Field(default=None, description="Path to shipping documents")
    DamageEvidencePath: Optional[str] = Field(default=None, description="Path to damage evidence")
    ShippingDocumentsFileName: Optional[str] = Field(default=None, description="Shipping documents filename")
    DamageEvidenceFileName: Optional[str] = Field(default=None, description="Damage evidence filename")
    
    # Processing flags
    RequiresManualReview: Union[str, bool] = Field(default=False, description="Whether manual review is required")
    ProcessingPriority: str = Field(default="Normal", description="Processing priority")
    HasDamageEvidence: Union[str, bool] = Field(default=False, description="Whether damage evidence exists")
    HasShippingDocuments: Union[str, bool] = Field(default=False, description="Whether shipping documents exist")
    
    # Note: The agent will handle parsing and validation autonomously
    
    def to_agent_format(self) -> Dict[str, Any]:
        """Convert to the format expected by the agent."""
        return {
            # Core claim data
            "ObjectClaimId": self.ObjectClaimId,
            "type": self.ClaimType.lower(),
            "amount": self.ClaimAmount,
            "carrier": self.Carrier,
            "description": self.Description,
            
            # Shipment reference
            "ShipmentID": self.ShipmentID,
            "shipmentId": self.ShipmentID,  # Alternative field name
            
            # Customer information
            "CustomerName": self.CustomerName,
            "FullName": self.CustomerName,  # Alternative field name
            "EmailAddress": self.CustomerEmail,
            "Phone": self.CustomerPhone,
            "shipper": self.CustomerName,  # Map to shipper field
            
            # Submission details
            "submissionSource": self.SubmissionSource,
            "submittedDate": self.SubmittedAt,
            
            # Document information
            "Photos": self._convert_files_to_photos(),
            "documents": self._get_all_documents(),
            
            # Processing flags
            "requiresManualReview": self.RequiresManualReview,
            "processingPriority": self.ProcessingPriority,
            "hasDamageEvidence": self.HasDamageEvidence,
            "hasShippingDocuments": self.HasShippingDocuments,
            
            # Storage information
            "bucketId": self.DamageEvidenceBucketId or self.ShippingDocumentsBucketId,
            "folderId": self.FolderId,
            
            # Metadata
            "inputFormat": "structured_claim_data",
            "parsedAt": datetime.now().isoformat()
        }
    
    def _convert_files_to_photos(self) -> List[Dict[str, Any]]:
        """Convert file information to photos format expected by agent."""
        photos = []
        
        # Add damage evidence files
        for file_info in self.DamageEvidenceFiles:
            photos.append({
                "bucket_id": file_info.bucketId,
                "folder_id": file_info.folderId,
                "file_path": file_info.path,
                "path": file_info.path,
                "filename": file_info.fileName,
                "name": file_info.fileName,
                "size": file_info.size,
                "type": file_info.type,
                "document_type": "damage_evidence",
                "uploadedAt": file_info.uploadedAt
            })
        
        # Add shipping document files
        for file_info in self.ShippingDocumentsFiles:
            photos.append({
                "bucket_id": file_info.bucketId,
                "folder_id": file_info.folderId,
                "file_path": file_info.path,
                "path": file_info.path,
                "filename": file_info.fileName,
                "name": file_info.fileName,
                "size": file_info.size,
                "type": file_info.type,
                "document_type": "shipping_documents",
                "uploadedAt": file_info.uploadedAt
            })
        
        # Handle legacy path format if no files but paths exist
        if not photos and (self.DamageEvidencePath or self.ShippingDocumentsPath):
            if self.DamageEvidencePath and self.DamageEvidenceFileName:
                photos.append({
                    "bucket_id": self.DamageEvidenceBucketId,
                    "folder_id": self.FolderId,
                    "file_path": self.DamageEvidencePath,
                    "path": self.DamageEvidencePath,
                    "filename": self.DamageEvidenceFileName,
                    "name": self.DamageEvidenceFileName,
                    "document_type": "damage_evidence"
                })
            
            if self.ShippingDocumentsPath and self.ShippingDocumentsFileName:
                photos.append({
                    "bucket_id": self.ShippingDocumentsBucketId,
                    "folder_id": self.FolderId,
                    "file_path": self.ShippingDocumentsPath,
                    "path": self.ShippingDocumentsPath,
                    "filename": self.ShippingDocumentsFileName,
                    "name": self.ShippingDocumentsFileName,
                    "document_type": "shipping_documents"
                })
        
        return photos
    
    def _get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents in a unified format."""
        documents = []
        
        # Add all files with enhanced metadata
        for file_info in self.DamageEvidenceFiles + self.ShippingDocumentsFiles:
            doc_type = "damage_evidence" if file_info in self.DamageEvidenceFiles else "shipping_documents"
            documents.append({
                "bucketId": file_info.bucketId,
                "folderId": file_info.folderId,
                "path": file_info.path,
                "fileName": file_info.fileName,
                "size": file_info.size,
                "type": file_info.type,
                "uploadedAt": file_info.uploadedAt,
                "documentType": doc_type,
                "category": "evidence" if doc_type == "damage_evidence" else "shipping"
            })
        
        return documents


# Note: The agent handles all parsing autonomously.
# These models serve as reference structures for what the agent might extract.