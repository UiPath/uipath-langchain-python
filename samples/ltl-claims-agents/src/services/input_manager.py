"""
Input Manager for LTL Claims Agent.
Provides abstraction layer for different input sources (queue vs file).
"""

import logging
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DocumentReference:
    """Reference to a document in storage."""
    bucket_id: str
    folder_id: str
    path: str
    file_name: str
    size: Optional[int] = None
    type: Optional[str] = None
    uploaded_at: Optional[str] = None


@dataclass
class ClaimInput:
    """Unified claim input structure."""
    claim_id: str
    claim_type: str
    claim_amount: float
    carrier: str
    shipment_id: str
    customer_name: str
    customer_email: str
    customer_phone: str
    description: str
    submission_source: str
    submitted_at: str
    shipping_documents: List[DocumentReference]
    damage_evidence: List[DocumentReference]
    requires_manual_review: bool
    processing_priority: str
    
    # Queue-specific fields (optional)
    transaction_key: Optional[str] = None
    queue_item_id: Optional[str] = None
    
    # Raw data for additional processing
    raw_data: Optional[Dict[str, Any]] = None


class InputSource(ABC):
    """Abstract base class for input sources."""
    
    @abstractmethod
    async def get_next_claim(self) -> Optional[ClaimInput]:
        """
        Retrieve next claim for processing.
        
        Returns:
            ClaimInput object if available, None if no claims to process
        """
        pass
    
    @abstractmethod
    async def update_status(self, claim_id: str, status: str, details: Dict[str, Any]) -> None:
        """
        Update processing status for a claim.
        
        Args:
            claim_id: Unique claim identifier
            status: Status message or state
            details: Additional status details
        """
        pass



class QueueInputSource(InputSource):
    """Input source that retrieves claims from UiPath Queue."""
    
    def __init__(self, uipath_service, queue_name: str):
        """
        Initialize queue input source.
        
        Args:
            uipath_service: UiPath service instance for SDK operations
            queue_name: Name of the queue to retrieve items from
        """
        self.uipath_service = uipath_service
        self.queue_name = queue_name
        self.current_transaction_key: Optional[str] = None
        logger.info(f"Initialized QueueInputSource for queue: {queue_name}")
    
    async def get_next_claim(self) -> Optional[ClaimInput]:
        """
        Retrieve next claim from UiPath Queue using list method.
        
        Returns:
            ClaimInput object if queue item available, None otherwise
        """
        try:
            logger.info(f"Retrieving next queue item from: {self.queue_name}")
            
            # Get UiPath SDK client
            if not self.uipath_service._client:
                await self.uipath_service.authenticate()
            
            sdk = self.uipath_service._client
            
            # List queue items to find New items
            # Use direct API call with OAuth token from .uipath/.auth.json
            from ..config.settings import settings
            import httpx
            import json as json_lib
            import os
            
            # Try to get OAuth token from .uipath/.auth.json
            try:
                auth_file_path = os.path.join(os.getcwd(), ".uipath", ".auth.json")
                with open(auth_file_path, "r") as f:
                    auth_data = json_lib.load(f)
                    access_token = auth_data.get("access_token")
                    logger.debug("Using OAuth token from .uipath/.auth.json")
            except Exception as e:
                # Fallback to PAT
                access_token = settings.uipath_access_token
                logger.debug(f"Using PAT from settings (OAuth file not found: {e})")
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "X-UIPATH-OrganizationUnitId": str(settings.uipath_folder_id)
            }
            
            url = f"{settings.effective_base_url}/orchestrator_/odata/QueueItems"
            params = {
                "$filter": "Status eq 'New'",
                "$expand": "QueueDefinition"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                items_data = response.json()
            
            if not items_data:
                logger.info("No queue items data returned")
                return None
            
            # Look for items in our queue that are in "New" status
            queue_items = items_data.get("value", []) if isinstance(items_data, dict) else []
            
            logger.info(f"Found {len(queue_items)} total queue items")
            
            for item in queue_items:
                # Try different ways to get queue name
                queue_def = item.get("QueueDefinition", {})
                queue_name = queue_def.get("Name", "") if isinstance(queue_def, dict) else ""
                
                status = item.get("Status", "")
                queue_item_id = item.get("Id")
                
                # If status is New, process it (queue name might be empty in list response)
                if status == "New":
                    logger.info(f"Found New queue item: {queue_item_id}")
                    
                    # Store for status updates
                    self.current_transaction_key = str(queue_item_id)
                    
                    # Extract SpecificContent (the claim data payload)
                    specific_content = item.get("SpecificContent", {})
                    
                    if not specific_content:
                        logger.warning("Queue item has no SpecificContent")
                        continue
                    
                    # Parse JSON strings in SpecificContent
                    import json
                    for key in ["ShippingDocumentsFiles", "DamageEvidenceFiles"]:
                        if key in specific_content and isinstance(specific_content[key], str):
                            try:
                                specific_content[key] = json.loads(specific_content[key])
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse {key} as JSON")
                                specific_content[key] = []
                    
                    logger.info(f"Retrieved queue item: {queue_item_id}")
                    
                    # Parse claim data from SpecificContent
                    claim_input = self._parse_queue_item(specific_content, self.current_transaction_key, queue_item_id)
                    
                    return claim_input
            
            logger.info(f"No New items found in queue '{self.queue_name}'")
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve queue item: {e}")
            # Don't raise - return None to indicate no items available
            return None
    
    def _parse_queue_item(self, specific_content: Dict[str, Any], transaction_key: str, queue_item_id: str) -> ClaimInput:
        """
        Parse queue item SpecificContent into ClaimInput structure.
        
        Args:
            specific_content: SpecificContent from queue item
            transaction_key: Transaction key for status updates
            queue_item_id: Queue item ID
            
        Returns:
            ClaimInput object
        """
        # Extract core claim information
        claim_id = specific_content.get("ObjectClaimId", specific_content.get("ClaimId", ""))
        claim_type = specific_content.get("ClaimType", "")
        
        # Parse claim amount (handle both string and numeric)
        claim_amount_raw = specific_content.get("ClaimAmount", 0)
        try:
            claim_amount = float(claim_amount_raw) if claim_amount_raw else 0.0
        except (ValueError, TypeError):
            claim_amount = 0.0
            logger.warning(f"Could not parse claim amount: {claim_amount_raw}")
        
        carrier = specific_content.get("Carrier", "")
        shipment_id = specific_content.get("ShipmentID", specific_content.get("ShipmentId", ""))
        
        # Extract customer information
        customer_name = specific_content.get("CustomerName", "")
        customer_email = specific_content.get("CustomerEmail", "")
        customer_phone = specific_content.get("CustomerPhone", "")
        
        # Extract claim details
        description = specific_content.get("Description", "")
        submission_source = specific_content.get("SubmissionSource", "queue")
        submitted_at = specific_content.get("SubmittedAt", datetime.utcnow().isoformat())
        
        # Extract processing flags
        requires_manual_review = self._parse_bool(specific_content.get("RequiresManualReview", False))
        processing_priority = specific_content.get("ProcessingPriority", "Normal")
        
        # Parse document references
        shipping_documents = self._parse_documents(specific_content, "ShippingDocuments")
        damage_evidence = self._parse_documents(specific_content, "DamageEvidence")
        
        return ClaimInput(
            claim_id=claim_id,
            claim_type=claim_type,
            claim_amount=claim_amount,
            carrier=carrier,
            shipment_id=shipment_id,
            customer_name=customer_name,
            customer_email=customer_email,
            customer_phone=customer_phone,
            description=description,
            submission_source=submission_source,
            submitted_at=submitted_at,
            shipping_documents=shipping_documents,
            damage_evidence=damage_evidence,
            requires_manual_review=requires_manual_review,
            processing_priority=processing_priority,
            transaction_key=transaction_key,
            queue_item_id=queue_item_id,
            raw_data=specific_content
        )
    
    def _parse_documents(self, data: Dict[str, Any], doc_type: str) -> List[DocumentReference]:
        """
        Parse document references from queue item data.
        
        Args:
            data: Queue item SpecificContent
            doc_type: Document type key (e.g., "ShippingDocuments", "DamageEvidence")
            
        Returns:
            List of DocumentReference objects
        """
        documents = []
        
        # Check for files array
        files_key = f"{doc_type}Files"
        files = data.get(files_key, [])
        
        for file_info in files:
            if isinstance(file_info, dict):
                documents.append(DocumentReference(
                    bucket_id=str(file_info.get("bucketId", "")),
                    folder_id=str(file_info.get("folderId", "")),
                    path=file_info.get("path", ""),
                    file_name=file_info.get("fileName", ""),
                    size=file_info.get("size"),
                    type=file_info.get("type"),
                    uploaded_at=file_info.get("uploadedAt")
                ))
        
        # Handle legacy format with single path/filename
        if not documents:
            path_key = f"{doc_type}Path"
            filename_key = f"{doc_type}FileName"
            bucket_key = f"{doc_type}BucketId"
            
            path = data.get(path_key)
            filename = data.get(filename_key)
            bucket_id = data.get(bucket_key, data.get("BucketId", ""))
            folder_id = data.get("FolderId", "")
            
            if path and filename:
                documents.append(DocumentReference(
                    bucket_id=str(bucket_id),
                    folder_id=str(folder_id),
                    path=path,
                    file_name=filename
                ))
        
        return documents
    
    def _parse_bool(self, value: Any) -> bool:
        """Parse boolean value from various formats."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "yes", "1")
        return bool(value)
    
    async def update_status(self, claim_id: str, status: str, details: Dict[str, Any]) -> None:
        """
        Update queue transaction progress.
        
        Args:
            claim_id: Claim identifier
            status: Status message
            details: Additional details (may include result data)
        """
        if not self.current_transaction_key:
            logger.warning(f"No transaction key available for claim {claim_id}")
            return
        
        try:
            # Get UiPath SDK client
            if not self.uipath_service._client:
                await self.uipath_service.authenticate()
            
            sdk = self.uipath_service._client
            
            # Check if this is a completion or failure
            if details.get("complete") or details.get("success") is not None:
                # Complete or fail the transaction
                result_status = "Successful" if details.get("success", True) else "Failed"
                
                await sdk.queues.complete_transaction_item_async(
                    transaction_key=self.current_transaction_key,
                    result={
                        "Status": result_status,
                        "OutputData": details.get("output_data", {}),
                        "ErrorMessage": details.get("error_message"),
                        "CompletedAt": datetime.utcnow().isoformat()
                    }
                )
                
                logger.info(f"Completed transaction {self.current_transaction_key} with status: {result_status}")
            else:
                # Update progress
                await sdk.queues.update_progress_of_transaction_item_async(
                    transaction_key=self.current_transaction_key,
                    progress=status
                )
                
                logger.info(f"Updated transaction progress: {status}")
                
        except Exception as e:
            logger.error(f"Failed to update queue status: {e}")
            # Don't raise - status update failures shouldn't stop processing



class FileInputSource(InputSource):
    """Input source that reads claims from JSON file."""
    
    def __init__(self, file_path: str):
        """
        Initialize file input source.
        
        Args:
            file_path: Path to JSON file containing claim data
        """
        self.file_path = file_path
        self.processed = False
        logger.info(f"Initialized FileInputSource with file: {file_path}")
    
    async def get_next_claim(self) -> Optional[ClaimInput]:
        """
        Read claim from JSON file.
        
        Returns:
            ClaimInput object if file exists and not yet processed, None otherwise
        """
        # Only process file once
        if self.processed:
            logger.info("File already processed")
            return None
        
        try:
            # Check if file exists
            if not os.path.exists(self.file_path):
                logger.error(f"Input file not found: {self.file_path}")
                self.processed = True
                return None
            
            logger.info(f"Reading claim data from file: {self.file_path}")
            
            # Read and parse JSON file
            with open(self.file_path, 'r', encoding='utf-8') as f:
                claim_data = json.load(f)
            
            # Validate that JSON structure matches expected schema
            if not self._validate_structure(claim_data):
                logger.error("Invalid JSON structure - does not match queue SpecificContent schema")
                self.processed = True
                return None
            
            # Parse claim data
            claim_input = self._parse_file_data(claim_data)
            
            # Mark as processed
            self.processed = True
            
            logger.info(f"Successfully loaded claim from file: {claim_input.claim_id}")
            return claim_input
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file: {e}")
            self.processed = True
            return None
        except Exception as e:
            logger.error(f"Failed to read claim from file: {e}")
            self.processed = True
            return None
    
    def _validate_structure(self, data: Dict[str, Any]) -> bool:
        """
        Validate that JSON structure matches queue SpecificContent schema.
        
        Args:
            data: Parsed JSON data
            
        Returns:
            True if valid, False otherwise
        """
        # Check for required fields that should be in SpecificContent
        required_fields = ["ObjectClaimId", "ClaimType", "ClaimAmount", "Carrier"]
        
        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing required field: {field}")
                # Don't fail - just warn, as we can handle missing fields
        
        # Structure is valid if it's a dictionary with at least some claim data
        return isinstance(data, dict) and len(data) > 0
    
    def _parse_file_data(self, data: Dict[str, Any]) -> ClaimInput:
        """
        Parse file data into ClaimInput structure.
        
        Args:
            data: Parsed JSON data
            
        Returns:
            ClaimInput object
        """
        # Extract core claim information
        claim_id = data.get("ObjectClaimId", data.get("ClaimId", f"FILE-{datetime.utcnow().timestamp()}"))
        claim_type = data.get("ClaimType", "")
        
        # Parse claim amount (handle both string and numeric)
        claim_amount_raw = data.get("ClaimAmount", 0)
        try:
            claim_amount = float(claim_amount_raw) if claim_amount_raw else 0.0
        except (ValueError, TypeError):
            claim_amount = 0.0
            logger.warning(f"Could not parse claim amount: {claim_amount_raw}")
        
        carrier = data.get("Carrier", "")
        shipment_id = data.get("ShipmentID", data.get("ShipmentId", ""))
        
        # Extract customer information
        customer_name = data.get("CustomerName", "")
        customer_email = data.get("CustomerEmail", "")
        customer_phone = data.get("CustomerPhone", "")
        
        # Extract claim details
        description = data.get("Description", "")
        submission_source = data.get("SubmissionSource", "file")
        submitted_at = data.get("SubmittedAt", datetime.utcnow().isoformat())
        
        # Extract processing flags
        requires_manual_review = self._parse_bool(data.get("RequiresManualReview", False))
        processing_priority = data.get("ProcessingPriority", "Normal")
        
        # Parse document references
        shipping_documents = self._parse_documents(data, "ShippingDocuments")
        damage_evidence = self._parse_documents(data, "DamageEvidence")
        
        return ClaimInput(
            claim_id=claim_id,
            claim_type=claim_type,
            claim_amount=claim_amount,
            carrier=carrier,
            shipment_id=shipment_id,
            customer_name=customer_name,
            customer_email=customer_email,
            customer_phone=customer_phone,
            description=description,
            submission_source=submission_source,
            submitted_at=submitted_at,
            shipping_documents=shipping_documents,
            damage_evidence=damage_evidence,
            requires_manual_review=requires_manual_review,
            processing_priority=processing_priority,
            transaction_key=None,  # No transaction for file input
            queue_item_id=None,  # No queue item for file input
            raw_data=data
        )
    
    def _parse_documents(self, data: Dict[str, Any], doc_type: str) -> List[DocumentReference]:
        """
        Parse document references from file data.
        
        Args:
            data: Parsed JSON data
            doc_type: Document type key (e.g., "ShippingDocuments", "DamageEvidence")
            
        Returns:
            List of DocumentReference objects
        """
        documents = []
        
        # Check for files array
        files_key = f"{doc_type}Files"
        files = data.get(files_key, [])
        
        for file_info in files:
            if isinstance(file_info, dict):
                documents.append(DocumentReference(
                    bucket_id=str(file_info.get("bucketId", "")),
                    folder_id=str(file_info.get("folderId", "")),
                    path=file_info.get("path", ""),
                    file_name=file_info.get("fileName", ""),
                    size=file_info.get("size"),
                    type=file_info.get("type"),
                    uploaded_at=file_info.get("uploadedAt")
                ))
        
        # Handle legacy format with single path/filename
        if not documents:
            path_key = f"{doc_type}Path"
            filename_key = f"{doc_type}FileName"
            bucket_key = f"{doc_type}BucketId"
            
            path = data.get(path_key)
            filename = data.get(filename_key)
            bucket_id = data.get(bucket_key, data.get("BucketId", ""))
            folder_id = data.get("FolderId", "")
            
            if path and filename:
                documents.append(DocumentReference(
                    bucket_id=str(bucket_id),
                    folder_id=str(folder_id),
                    path=path,
                    file_name=filename
                ))
        
        return documents
    
    def _parse_bool(self, value: Any) -> bool:
        """Parse boolean value from various formats."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "yes", "1")
        return bool(value)
    
    async def update_status(self, claim_id: str, status: str, details: Dict[str, Any]) -> None:
        """
        Log status updates (no queue to update for file input).
        
        Args:
            claim_id: Claim identifier
            status: Status message
            details: Additional details
        """
        # For file input, just log the status
        logger.info(f"[File Input] Claim {claim_id} status: {status}")
        
        if details:
            logger.debug(f"[File Input] Status details: {json.dumps(details, indent=2)}")



class InputManager:
    """
    Factory class for creating appropriate input sources.
    Provides unified interface to agent for claim retrieval.
    """
    
    def __init__(self, settings, uipath_service=None):
        """
        Initialize InputManager with configuration.
        
        Args:
            settings: Settings object with configuration
            uipath_service: UiPath service instance (required for queue mode)
        """
        self.settings = settings
        self.uipath_service = uipath_service
        self.input_source = self._create_input_source()
    
    def _create_input_source(self) -> InputSource:
        """
        Factory method to create appropriate input source based on configuration.
        
        Returns:
            InputSource instance (QueueInputSource or FileInputSource)
        """
        if self.settings.use_queue_input:
            # Queue mode - requires UiPath service
            if not self.uipath_service:
                raise ValueError("UiPath service required for queue input mode")
            
            logger.info("🔄 Input Mode: QUEUE")
            logger.info(f"📋 Queue Name: {self.settings.effective_queue_name}")
            
            return QueueInputSource(
                uipath_service=self.uipath_service,
                queue_name=self.settings.effective_queue_name
            )
        else:
            # File mode
            logger.info("📁 Input Mode: FILE")
            logger.info(f"📄 Input File: {self.settings.input_file_path}")
            
            return FileInputSource(
                file_path=self.settings.input_file_path
            )
    
    async def get_next_claim(self) -> Optional[ClaimInput]:
        """
        Get next claim from configured input source.
        
        Returns:
            ClaimInput object if available, None otherwise
        """
        return await self.input_source.get_next_claim()
    
    async def update_status(self, claim_id: str, status: str, details: Dict[str, Any] = None) -> None:
        """
        Update claim processing status.
        
        Args:
            claim_id: Claim identifier
            status: Status message
            details: Additional status details
        """
        await self.input_source.update_status(claim_id, status, details or {})
    
    def get_input_mode(self) -> str:
        """
        Get current input mode.
        
        Returns:
            "queue" or "file"
        """
        return "queue" if self.settings.use_queue_input else "file"
    
    def is_queue_mode(self) -> bool:
        """Check if running in queue mode."""
        return self.settings.use_queue_input
    
    def is_file_mode(self) -> bool:
        """Check if running in file mode."""
        return not self.settings.use_queue_input
