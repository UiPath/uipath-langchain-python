"""
Input validation and normalization for LTL Claims Agent.

Provides validation and normalization of input data from various sources
(UiPath queues, files, etc.) to ensure consistent data format throughout
the agent processing pipeline.

Usage Example:
    from src.utils.validators import InputValidator, ValidationError
    
    # Raw data from UiPath queue
    raw_data = {
        "ObjectClaimId": "CLM-12345",
        "ClaimType": "Damage",
        "ClaimAmount": "1500.50",
        "Carrier": "XYZ Freight"
    }
    
    try:
        # Validate and normalize
        normalized = InputValidator.validate_and_normalize(raw_data)
        
        # Use normalized data
        claim_id = normalized["claim_id"]  # "CLM-12345"
        claim_type = normalized["claim_type"]  # "Damage"
        claim_amount = normalized["claim_amount"]  # 1500.5 (float)
        priority = normalized["processing_priority"]  # "Normal" (default)
        
    except ValidationError as e:
        # Handle validation errors
        print(f"Validation failed: {e.message}")
        print(f"Missing fields: {e.missing_fields}")
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from .errors import InputError

logger = logging.getLogger(__name__)


class ValidationError(InputError):
    """
    Exception raised when input validation fails.
    
    Used when:
    - Required fields are missing
    - Field values are invalid
    - Data format is incorrect
    
    Example:
        raise ValidationError(
            "Missing required fields: claim_id, claim_type",
            context={"source": "queue"},
            details={"missing_fields": ["claim_id", "claim_type"]},
            input_source="queue"
        )
    """
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        missing_fields: Optional[List[str]] = None,
        input_source: Optional[str] = None
    ):
        """
        Initialize ValidationError with validation-specific information.
        
        Args:
            message: Human-readable error message
            context: Additional context about the validation error
            details: Detailed error information
            missing_fields: List of missing required fields
            input_source: Source of the input (e.g., "queue", "file")
        """
        super().__init__(message, context, details, input_source=input_source)
        self.missing_fields = missing_fields or []
        if missing_fields:
            self.details["missing_fields"] = missing_fields


class InputValidator:
    """
    Validates and normalizes input data for claims processing.
    
    Handles:
    - Field validation (required fields, data types)
    - Field name mapping (UiPath queue format to standard format)
    - Default value application
    - Data normalization (snake_case conversion, type coercion)
    """
    
    # Standard field names (constants for type safety)
    FIELD_CLAIM_ID = "claim_id"
    FIELD_CLAIM_TYPE = "claim_type"
    FIELD_CLAIM_AMOUNT = "claim_amount"
    FIELD_SHIPMENT_ID = "shipment_id"
    FIELD_CARRIER = "carrier"
    FIELD_CUSTOMER_NAME = "customer_name"
    FIELD_CUSTOMER_EMAIL = "customer_email"
    FIELD_CUSTOMER_PHONE = "customer_phone"
    FIELD_DESCRIPTION = "description"
    FIELD_SUBMISSION_SOURCE = "submission_source"
    FIELD_SUBMITTED_AT = "submitted_at"
    FIELD_REQUIRES_MANUAL_REVIEW = "requires_manual_review"
    FIELD_PROCESSING_PRIORITY = "processing_priority"
    FIELD_SHIPPING_DOCUMENTS = "shipping_documents"
    FIELD_DAMAGE_EVIDENCE = "damage_evidence"
    FIELD_TRANSACTION_KEY = "transaction_key"
    FIELD_QUEUE_ITEM_ID = "queue_item_id"
    
    # Required fields for claim processing
    REQUIRED_FIELDS = [FIELD_CLAIM_ID, FIELD_CLAIM_TYPE, FIELD_CLAIM_AMOUNT]
    
    # Valid claim types (add more as needed)
    VALID_CLAIM_TYPES = {"damage", "loss", "shortage", "delay", "other"}
    
    # Field mappings from UiPath queue format to standard format
    FIELD_MAPPINGS = {
        # Core claim fields
        "ObjectClaimId": FIELD_CLAIM_ID,
        "ClaimId": FIELD_CLAIM_ID,
        "ClaimType": FIELD_CLAIM_TYPE,
        "ClaimAmount": FIELD_CLAIM_AMOUNT,
        
        # Shipment fields
        "ShipmentID": FIELD_SHIPMENT_ID,
        "ShipmentId": FIELD_SHIPMENT_ID,
        
        # Carrier fields
        "Carrier": FIELD_CARRIER,
        "CarrierName": FIELD_CARRIER,
        
        # Customer fields
        "CustomerName": FIELD_CUSTOMER_NAME,
        "CustomerEmail": FIELD_CUSTOMER_EMAIL,
        "CustomerPhone": FIELD_CUSTOMER_PHONE,
        
        # Claim details
        "Description": FIELD_DESCRIPTION,
        "ClaimDescription": FIELD_DESCRIPTION,
        
        # Submission info
        "SubmissionSource": FIELD_SUBMISSION_SOURCE,
        "SubmittedAt": FIELD_SUBMITTED_AT,
        "SubmissionDate": FIELD_SUBMITTED_AT,
        
        # Processing flags
        "RequiresManualReview": FIELD_REQUIRES_MANUAL_REVIEW,
        "ProcessingPriority": FIELD_PROCESSING_PRIORITY,
        "Priority": FIELD_PROCESSING_PRIORITY,
        
        # Document references
        "ShippingDocumentsFiles": FIELD_SHIPPING_DOCUMENTS,
        "DamageEvidenceFiles": FIELD_DAMAGE_EVIDENCE,
        
        # Queue-specific fields
        "TransactionKey": FIELD_TRANSACTION_KEY,
        "QueueItemId": FIELD_QUEUE_ITEM_ID,
    }
    
    # Default values for optional fields
    DEFAULT_VALUES = {
        FIELD_SUBMISSION_SOURCE: "unknown",
        FIELD_SUBMITTED_AT: None,  # Will be set to current time if None
        FIELD_REQUIRES_MANUAL_REVIEW: False,
        FIELD_PROCESSING_PRIORITY: "Normal",
        FIELD_CUSTOMER_NAME: "",
        FIELD_CUSTOMER_EMAIL: "",
        FIELD_CUSTOMER_PHONE: "",
        FIELD_DESCRIPTION: "",
        FIELD_CARRIER: "",
        FIELD_SHIPMENT_ID: "",
        FIELD_SHIPPING_DOCUMENTS: [],
        FIELD_DAMAGE_EVIDENCE: [],
        FIELD_TRANSACTION_KEY: None,
        FIELD_QUEUE_ITEM_ID: None,
    }
    
    @staticmethod
    def validate_and_normalize(raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize input data.
        
        This is the main entry point for input validation. It performs:
        1. Field name mapping (UiPath format to standard format)
        2. Required field validation
        3. Default value application
        4. Data type normalization
        
        Args:
            raw_data: Raw input data from queue or file
            
        Returns:
            Normalized and validated data dictionary
            
        Raises:
            ValidationError: If required fields are missing or validation fails
        """
        logger.debug(f"Validating and normalizing input data with {len(raw_data)} fields")
        
        # Step 1: Apply field mappings
        normalized = InputValidator._apply_mappings(raw_data)
        
        # Step 2: Check required fields
        missing_fields = InputValidator._check_required_fields(normalized)
        if missing_fields:
            error_msg = f"Missing required fields: {', '.join(missing_fields)}"
            logger.error(error_msg)
            raise ValidationError(
                error_msg,
                context={"validation_step": "required_fields"},
                details={
                    "missing_fields": missing_fields,
                    "received_fields": list(normalized.keys())
                },
                missing_fields=missing_fields
            )
        
        # Step 3: Apply default values
        normalized = InputValidator._apply_defaults(normalized)
        
        # Step 4: Normalize data types
        normalized = InputValidator._normalize_types(normalized)
        
        logger.info(f"Successfully validated and normalized input for claim: {normalized.get(InputValidator.FIELD_CLAIM_ID, 'UNKNOWN')}")
        
        return normalized
    
    @staticmethod
    def _apply_mappings(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map UiPath queue field names to standard snake_case format.
        
        Converts field names like "ObjectClaimId" to "claim_id" while
        preserving fields that don't have mappings.
        
        Args:
            data: Raw input data with UiPath field names
            
        Returns:
            Dictionary with standardized field names
        """
        result = {}
        
        for key, value in data.items():
            # Check if we have a mapping for this field
            standard_key = InputValidator.FIELD_MAPPINGS.get(key, key)
            
            # If the standard key already exists, don't overwrite it
            # (prefer already-standard field names)
            if standard_key not in result:
                result[standard_key] = value
            else:
                # If both formats exist, log a warning
                logger.debug(f"Field '{key}' maps to '{standard_key}' which already exists, skipping")
        
        logger.debug(f"Mapped {len(data)} fields to {len(result)} standardized fields")
        
        return result
    
    @staticmethod
    def _check_required_fields(data: Dict[str, Any]) -> List[str]:
        """
        Check for required fields in the data.
        
        Args:
            data: Normalized data dictionary
            
        Returns:
            List of missing required field names (empty if all present)
        """
        missing = []
        
        for field in InputValidator.REQUIRED_FIELDS:
            if field not in data or data[field] is None or data[field] == "":
                missing.append(field)
        
        if missing:
            logger.warning(f"Missing required fields: {missing}")
        
        return missing
    
    @staticmethod
    def _apply_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply default values for optional fields that are missing.
        
        Args:
            data: Normalized data dictionary
            
        Returns:
            Dictionary with default values applied
        """
        result = data.copy()
        
        for field, default_value in InputValidator.DEFAULT_VALUES.items():
            if field not in result or result[field] is None:
                # Special handling for submitted_at - use current time if not provided
                if field == InputValidator.FIELD_SUBMITTED_AT and default_value is None:
                    result[field] = datetime.now(timezone.utc).isoformat()
                else:
                    result[field] = default_value
                    
                logger.debug(f"Applied default value for '{field}': {result[field]}")
        
        return result
    
    @staticmethod
    def _normalize_types(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize data types for known fields.
        
        Ensures:
        - claim_amount is a float
        - requires_manual_review is a boolean
        - Lists are properly formatted
        
        Args:
            data: Data dictionary with default values applied
            
        Returns:
            Dictionary with normalized data types
        """
        result = data.copy()
        
        # Normalize claim_amount to float
        if InputValidator.FIELD_CLAIM_AMOUNT in result:
            try:
                result[InputValidator.FIELD_CLAIM_AMOUNT] = float(result[InputValidator.FIELD_CLAIM_AMOUNT])
            except (ValueError, TypeError) as e:
                error_msg = f"Invalid claim_amount value: {result[InputValidator.FIELD_CLAIM_AMOUNT]} - must be numeric"
                logger.error(error_msg)
                raise ValidationError(
                    error_msg,
                    context={"validation_step": "type_normalization", "field": InputValidator.FIELD_CLAIM_AMOUNT},
                    details={"invalid_value": str(result[InputValidator.FIELD_CLAIM_AMOUNT]), "error": str(e)}
                )
        
        # Validate claim_type against allowed values
        if InputValidator.FIELD_CLAIM_TYPE in result:
            claim_type = str(result[InputValidator.FIELD_CLAIM_TYPE]).lower().strip()
            if claim_type not in InputValidator.VALID_CLAIM_TYPES:
                logger.warning(
                    f"Claim type '{claim_type}' not in valid types {InputValidator.VALID_CLAIM_TYPES}, "
                    f"but allowing it to proceed"
                )
            result[InputValidator.FIELD_CLAIM_TYPE] = claim_type
        
        # Normalize requires_manual_review to boolean
        if InputValidator.FIELD_REQUIRES_MANUAL_REVIEW in result:
            result[InputValidator.FIELD_REQUIRES_MANUAL_REVIEW] = InputValidator._parse_bool(
                result[InputValidator.FIELD_REQUIRES_MANUAL_REVIEW]
            )
        
        # Ensure document lists are lists
        for doc_field in [InputValidator.FIELD_SHIPPING_DOCUMENTS, InputValidator.FIELD_DAMAGE_EVIDENCE]:
            if doc_field in result and not isinstance(result[doc_field], list):
                logger.warning(f"Field '{doc_field}' is not a list, converting")
                result[doc_field] = []
        
        return result
    
    @staticmethod
    def get_validation_summary(raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a summary of validation status without raising errors.
        
        Useful for pre-validation checks or reporting.
        
        Args:
            raw_data: Raw input data to analyze
            
        Returns:
            Dictionary with validation summary including:
            - has_required_fields: bool
            - missing_fields: List[str]
            - mapped_fields: int
            - unmapped_fields: List[str]
        """
        normalized = InputValidator._apply_mappings(raw_data)
        missing = InputValidator._check_required_fields(normalized)
        
        unmapped = [
            key for key in raw_data.keys() 
            if key not in InputValidator.FIELD_MAPPINGS and key not in normalized
        ]
        
        return {
            "has_required_fields": len(missing) == 0,
            "missing_fields": missing,
            "mapped_fields": len(normalized),
            "unmapped_fields": unmapped,
            "total_input_fields": len(raw_data)
        }
    
    @staticmethod
    def _parse_bool(value: Any) -> bool:
        """
        Parse boolean value from various formats.
        
        Handles:
        - Boolean values (True/False)
        - String values ("true", "yes", "1", etc.)
        - Numeric values (0/1)
        
        Args:
            value: Value to parse as boolean
            
        Returns:
            Boolean value
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "yes", "1", "y")
        if isinstance(value, (int, float)):
            return bool(value)
        return False


__all__ = [
    "ValidationError",
    "InputValidator"
]
