"""
UiPath Document Understanding service for information extraction.
Focuses specifically on IXP projects and document extraction using UiPath Document Understanding.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

# UiPath SDK imports for Document Understanding only
from uipath import UiPath
from uipath.models.document_understanding import (
    DocumentUnderstandingRequest,
    DocumentUnderstandingResponse,
    ExtractionRequest,
    ClassificationRequest
)

from ..models.document_models import (
    DocumentExtractionResult, DocumentMetadata, DocumentFormat,
    DocumentProcessingResult, DocumentStatus
)
from ..config.settings import settings
from .uipath_service import uipath_service, UiPathServiceError

logger = logging.getLogger(__name__)


class DocumentUnderstandingError(Exception):
    """Custom exception for document understanding errors."""
    pass


class IXPProjectConfig:
    """Configuration for IXP project integration."""
    
    def __init__(
        self,
        project_name: str,
        project_key: str,
        document_types: List[str],
        confidence_threshold: float = 0.7,
        timeout_seconds: int = 300,
        ixp_project_id: Optional[str] = None
    ):
        self.project_name = project_name
        self.project_key = project_key
        self.document_types = document_types
        self.confidence_threshold = confidence_threshold
        self.timeout_seconds = timeout_seconds
        self.ixp_project_id = ixp_project_id


class DocumentUnderstandingService:
    """
    Service for UiPath Document Understanding integration using the UiPath SDK.
    Focuses specifically on IXP projects and structured data extraction from documents.
    """
    
    def __init__(self):
        """Initialize Document Understanding service with UiPath SDK."""
        
        # IXP project configurations
        self.ixp_projects = {
            "claims_general": IXPProjectConfig(
                project_name="LTL Claims General Extraction",
                project_key="ltl-claims-general",
                document_types=["shipping_document", "damage_evidence", "invoice", "report"],
                confidence_threshold=0.7,
                ixp_project_id="ltl-claims-general-ixp"
            ),
            "shipping_documents": IXPProjectConfig(
                project_name="Shipping Documents Extraction", 
                project_key="shipping-docs-extraction",
                document_types=["bill_of_lading", "shipping_document"],
                confidence_threshold=0.8,
                ixp_project_id="shipping-docs-ixp"
            ),
            "damage_evidence": IXPProjectConfig(
                project_name="Damage Evidence Extraction",
                project_key="damage-evidence-extraction", 
                document_types=["damage_evidence", "photo", "report"],
                confidence_threshold=0.6,
                ixp_project_id="damage-evidence-ixp"
            )
        }
        
        # Field mappings for different document types
        self.field_mappings = {
            "shipping_document": {
                "tracking_number": ["TrackingNumber", "TrackingNo", "Tracking", "ShipmentID"],
                "carrier": ["Carrier", "CarrierName", "ShippingCompany"],
                "shipper": ["Shipper", "ShipperName", "FromCompany"],
                "consignee": ["Consignee", "ConsigneeName", "ToCompany"],
                "pro_number": ["PRO", "PRONumber", "ProNumber"],
                "bill_of_lading": ["BOL", "BillOfLading", "BOLNumber"],
                "pickup_date": ["PickupDate", "ShipDate", "PickedUpDate"],
                "delivery_date": ["DeliveryDate", "DeliveredDate", "ExpectedDelivery"],
                "weight": ["Weight", "TotalWeight", "GrossWeight"],
                "pieces": ["Pieces", "PieceCount", "NumberOfPieces"],
                "freight_charges": ["FreightCharges", "Charges", "TotalCharges"]
            },
            "damage_evidence": {
                "damage_type": ["DamageType", "TypeOfDamage", "DamageDescription"],
                "damage_location": ["DamageLocation", "Location", "WhereIsDamage"],
                "damage_extent": ["DamageExtent", "Extent", "SeverityOfDamage"],
                "estimated_cost": ["EstimatedCost", "RepairCost", "DamageCost"],
                "photo_count": ["PhotoCount", "NumberOfPhotos", "Images"],
                "inspection_date": ["InspectionDate", "DateInspected", "ExaminedDate"],
                "inspector_name": ["Inspector", "InspectorName", "ExaminedBy"]
            },
            "invoice": {
                "invoice_number": ["InvoiceNumber", "InvoiceNo", "Invoice"],
                "invoice_date": ["InvoiceDate", "Date", "BillDate"],
                "vendor": ["Vendor", "VendorName", "Supplier"],
                "total_amount": ["TotalAmount", "Total", "Amount"],
                "line_items": ["LineItems", "Items", "Products"],
                "tax_amount": ["Tax", "TaxAmount", "SalesTax"],
                "payment_terms": ["PaymentTerms", "Terms", "PaymentDue"]
            }
        }

    async def extract_document_data(
        self,
        document_path: str,
        document_type: str,
        bucket_name: Optional[str] = None,
        bucket_key: Optional[str] = None,
        folder_key: Optional[str] = None
    ) -> DocumentExtractionResult:
        """
        Extract structured data from documents using UiPath Document Understanding IXP projects.
        
        Args:
            document_path: Path to document in bucket or local file path
            document_type: Type of document (shipping_document, damage_evidence, invoice)
            bucket_name: Storage bucket name (if document is in bucket)
            bucket_key: Storage bucket key (if document is in bucket)
            folder_key: UiPath folder key
            
        Returns:
            DocumentExtractionResult with extracted data and confidence scores
        """
        try:
            logger.info(f"🔍 Extracting data from {document_type} document: {document_path}")
            
            # Get appropriate IXP project configuration
            project_config = self._get_project_config(document_type)
            if not project_config:
                raise DocumentUnderstandingError(f"No IXP project configured for document type: {document_type}")
            
            # Download document from bucket if needed
            local_file_path = await self._prepare_document_file(
                document_path, bucket_name, bucket_key, folder_key
            )
            
            # Extract data using UiPath Document Understanding SDK
            async with uipath_service:
                extraction_response = await uipath_service._client.documents.extract_async(
                    project_name=project_config.project_name,
                    tag="latest",  # Use latest model version
                    file_path=local_file_path
                )
            
            # Process extraction results
            extracted_data = self._process_extraction_response(
                extraction_response, document_type, project_config
            )
            
            # Create result object
            result = DocumentExtractionResult(
                document_path=document_path,
                document_type=document_type,
                extracted_fields=extracted_data["fields"],
                confidence_scores=extracted_data["confidence_scores"],
                processing_time=extracted_data.get("processing_time", 0.0),
                extraction_method="uipath_ixp",
                metadata=DocumentMetadata(
                    file_size=extracted_data.get("file_size", 0),
                    format=DocumentFormat.PDF,  # Assume PDF for now
                    page_count=extracted_data.get("page_count", 1),
                    creation_date=datetime.now(),
                    processing_engine="UiPath Document Understanding"
                )
            )
            
            logger.info(f"✅ Document extraction complete: {len(result.extracted_fields)} fields extracted")
            return result
            
        except Exception as e:
            logger.error(f"❌ Document extraction failed: {e}")
            raise DocumentUnderstandingError(f"Failed to extract document data: {str(e)}")



    async def create_validation_action(
        self,
        extraction_result: DocumentExtractionResult,
        claim_id: str,
        priority: str = "Medium"
    ) -> str:
        """
        Create a validation action in UiPath Action Center for document extraction results.
        
        Args:
            extraction_result: Document extraction results to validate
            claim_id: Related claim ID
            priority: Action priority (Low, Medium, High, Critical)
            
        Returns:
            Action ID of created validation task
        """
        try:
            logger.info(f"📋 Creating validation action for claim: {claim_id}")
            
            # Prepare validation data
            validation_data = {
                "claim_id": claim_id,
                "document_path": extraction_result.document_path,
                "document_type": extraction_result.document_type,
                "extracted_fields": extraction_result.extracted_fields,
                "confidence_scores": extraction_result.confidence_scores,
                "extraction_method": extraction_result.extraction_method,
                "requires_validation": True
            }
            
            # Create validation action using UiPath Actions SDK
            async with uipath_service:
                action = await uipath_service._client.actions.create_async(
                    title=f"Validate Document Extraction - Claim {claim_id}",
                    data=validation_data
                )
            
            action_id = action.key if hasattr(action, 'key') else str(action)
            
            logger.info(f"✅ Validation action created: {action_id}")
            return action_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create validation action: {e}")
            raise DocumentUnderstandingError(f"Failed to create validation action: {str(e)}")



    def _get_project_config(self, document_type: str) -> Optional[IXPProjectConfig]:
        """Get IXP project configuration for document type."""
        # Map document types to project configurations
        type_mapping = {
            "shipping_document": "shipping_documents",
            "bill_of_lading": "shipping_documents", 
            "damage_evidence": "damage_evidence",
            "photo": "damage_evidence",
            "invoice": "claims_general",
            "report": "claims_general"
        }
        
        project_key = type_mapping.get(document_type, "claims_general")
        return self.ixp_projects.get(project_key)

    async def _prepare_document_file(
        self,
        document_path: str,
        bucket_name: Optional[str] = None,
        bucket_key: Optional[str] = None,
        folder_key: Optional[str] = None
    ) -> str:
        """Prepare document file for processing (download from bucket if needed)."""
        
        # If it's already a local file path, return as-is
        if not bucket_name and not bucket_key:
            return document_path
        
        # Download from UiPath Storage Bucket using SDK
        try:
            local_path = f"/tmp/{Path(document_path).name}"
            
            async with uipath_service:
                await uipath_service._client.buckets.download_async(
                    name=bucket_name,
                    key=bucket_key,
                    blob_file_path=document_path,
                    destination_path=local_path,
                    folder_key=folder_key
                )
            
            return local_path
            
        except Exception as e:
            logger.error(f"❌ Failed to download document from bucket: {e}")
            raise DocumentUnderstandingError(f"Failed to download document: {str(e)}")

    def _process_extraction_response(
        self,
        extraction_response,
        document_type: str,
        project_config: IXPProjectConfig
    ) -> Dict[str, Any]:
        """Process UiPath Document Understanding extraction response."""
        
        extracted_fields = {}
        confidence_scores = {}
        
        # Process extraction results based on response structure
        if hasattr(extraction_response, 'predictions'):
            for prediction in extraction_response.predictions:
                field_name = getattr(prediction, 'field_name', 'unknown')
                field_value = getattr(prediction, 'value', '')
                confidence = getattr(prediction, 'confidence', 0.0)
                
                # Map to standardized field names
                mapped_field = self._map_field_name(field_name, document_type)
                if mapped_field:
                    extracted_fields[mapped_field] = field_value
                    confidence_scores[mapped_field] = confidence
        
        return {
            "fields": extracted_fields,
            "confidence_scores": confidence_scores,
            "processing_time": getattr(extraction_response, 'processing_time', 0.0),
            "page_count": getattr(extraction_response, 'page_count', 1)
        }

    def _map_field_name(self, field_name: str, document_type: str) -> Optional[str]:
        """Map extracted field name to standardized field name."""
        field_mappings = self.field_mappings.get(document_type, {})
        
        for standard_field, possible_names in field_mappings.items():
            if field_name in possible_names:
                return standard_field
        
        # Return original field name if no mapping found
        return field_name.lower().replace(' ', '_')

    async def classify_document(
        self,
        document_path: str,
        bucket_name: Optional[str] = None,
        bucket_key: Optional[str] = None,
        folder_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Classify document type using UiPath Document Understanding.
        
        Args:
            document_path: Path to document
            bucket_name: Storage bucket name (if document is in bucket)
            bucket_key: Storage bucket key (if document is in bucket)
            folder_key: UiPath folder key
            
        Returns:
            Classification results with document type and confidence
        """
        try:
            logger.info(f"📋 Classifying document: {document_path}")
            
            # Download document from bucket if needed
            local_file_path = await self._prepare_document_file(
                document_path, bucket_name, bucket_key, folder_key
            )
            
            # Use a general classification project
            async with uipath_service:
                # Note: This would use a classification-specific IXP project
                classification_response = await uipath_service._client.documents.extract_async(
                    project_name="LTL Claims Document Classifier",
                    tag="latest",
                    file_path=local_file_path
                )
            
            # Process classification results
            document_type = "unknown"
            confidence = 0.0
            
            if hasattr(classification_response, 'document_type'):
                document_type = getattr(classification_response, 'document_type', 'unknown')
                confidence = getattr(classification_response, 'confidence', 0.0)
            
            result = {
                "document_type": document_type,
                "confidence": confidence,
                "document_path": document_path,
                "classification_method": "uipath_ixp_classifier"
            }
            
            logger.info(f"✅ Document classified as: {document_type} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Document classification failed: {e}")
            return {
                "document_type": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }

    async def extract_multiple_documents(
        self,
        document_paths: List[str],
        document_types: Optional[List[str]] = None,
        bucket_name: Optional[str] = None,
        bucket_key: Optional[str] = None,
        folder_key: Optional[str] = None
    ) -> List[DocumentExtractionResult]:
        """
        Extract data from multiple documents in batch.
        
        Args:
            document_paths: List of document paths
            document_types: Optional list of document types (same order as paths)
            bucket_name: Storage bucket name
            bucket_key: Storage bucket key
            folder_key: UiPath folder key
            
        Returns:
            List of extraction results
        """
        try:
            logger.info(f"📄 Batch extracting {len(document_paths)} documents")
            
            results = []
            
            for i, document_path in enumerate(document_paths):
                try:
                    # Get document type for this document
                    doc_type = "unknown"
                    if document_types and i < len(document_types):
                        doc_type = document_types[i]
                    else:
                        # Auto-classify if type not provided
                        classification = await self.classify_document(
                            document_path, bucket_name, bucket_key, folder_key
                        )
                        doc_type = classification.get("document_type", "unknown")
                    
                    # Extract data
                    extraction_result = await self.extract_document_data(
                        document_path=document_path,
                        document_type=doc_type,
                        bucket_name=bucket_name,
                        bucket_key=bucket_key,
                        folder_key=folder_key
                    )
                    
                    results.append(extraction_result)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to extract document {document_path}: {e}")
                    # Create error result
                    error_result = DocumentExtractionResult(
                        document_path=document_path,
                        document_type="error",
                        extracted_fields={},
                        confidence_scores={},
                        processing_time=0.0,
                        extraction_method="uipath_ixp",
                        metadata=DocumentMetadata(
                            file_size=0,
                            format=DocumentFormat.UNKNOWN,
                            page_count=0,
                            creation_date=datetime.now(),
                            processing_engine="UiPath Document Understanding"
                        )
                    )
                    results.append(error_result)
            
            logger.info(f"✅ Batch extraction complete: {len(results)} documents processed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch extraction failed: {e}")
            return []

    async def get_extraction_confidence_summary(
        self,
        extraction_result: DocumentExtractionResult
    ) -> Dict[str, Any]:
        """
        Get confidence summary and recommendations for extraction results.
        
        Args:
            extraction_result: Document extraction results
            
        Returns:
            Confidence summary with recommendations
        """
        try:
            confidence_scores = extraction_result.confidence_scores
            
            if not confidence_scores:
                return {
                    "overall_confidence": 0.0,
                    "confidence_level": "very_low",
                    "recommendation": "manual_review_required",
                    "low_confidence_fields": [],
                    "high_confidence_fields": []
                }
            
            # Calculate overall confidence
            overall_confidence = sum(confidence_scores.values()) / len(confidence_scores)
            
            # Categorize fields by confidence
            low_confidence_fields = [
                field for field, score in confidence_scores.items() 
                if score < 0.7
            ]
            
            high_confidence_fields = [
                field for field, score in confidence_scores.items() 
                if score >= 0.8
            ]
            
            # Determine confidence level
            if overall_confidence >= 0.9:
                confidence_level = "very_high"
                recommendation = "auto_approve"
            elif overall_confidence >= 0.7:
                confidence_level = "high"
                recommendation = "auto_approve_with_audit"
            elif overall_confidence >= 0.5:
                confidence_level = "medium"
                recommendation = "validation_required"
            elif overall_confidence >= 0.3:
                confidence_level = "low"
                recommendation = "manual_review_required"
            else:
                confidence_level = "very_low"
                recommendation = "manual_processing_required"
            
            summary = {
                "overall_confidence": overall_confidence,
                "confidence_level": confidence_level,
                "recommendation": recommendation,
                "low_confidence_fields": low_confidence_fields,
                "high_confidence_fields": high_confidence_fields,
                "field_count": len(confidence_scores),
                "extraction_method": extraction_result.extraction_method,
                "document_type": extraction_result.document_type
            }
            
            logger.info(f"📊 Confidence summary: {confidence_level} ({overall_confidence:.2f})")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to generate confidence summary: {e}")
            return {
                "overall_confidence": 0.0,
                "confidence_level": "error",
                "recommendation": "manual_processing_required",
                "error": str(e)
            }




# Global document understanding service instance
document_understanding_service = DocumentUnderstandingService()