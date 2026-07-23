"""Constants for LTL Claims Processing Agent."""

from typing import Final


class ThresholdConstants:
    """Thresholds for decision making and escalation."""
    
    CONFIDENCE_THRESHOLD: Final[float] = 0.7
    """Minimum confidence threshold for automated decisions."""
    
    EXTRACTION_CONFIDENCE_THRESHOLD: Final[float] = 0.8
    """Minimum confidence threshold for document extraction."""
    
    DEFAULT_RISK_SCORE: Final[float] = 0.5
    """Default risk score when assessment fails."""
    
    HIGH_RISK_THRESHOLD: Final[float] = 0.7
    """Threshold above which risk is considered high."""
    
    LOW_RISK_THRESHOLD: Final[float] = 0.3
    """Threshold below which risk is considered low."""


class DecisionConstants:
    """Decision outcome constants."""
    
    APPROVED: Final[str] = "approved"
    DENIED: Final[str] = "denied"
    PENDING: Final[str] = "pending"
    
    VALID_DECISIONS: Final[tuple] = (APPROVED, DENIED, PENDING)


class RiskLevelConstants:
    """Risk level categorization constants."""
    
    LOW: Final[str] = "low"
    MEDIUM: Final[str] = "medium"
    HIGH: Final[str] = "high"
    
    VALID_LEVELS: Final[tuple] = (LOW, MEDIUM, HIGH)


class PriorityConstants:
    """Processing priority constants."""
    
    LOW: Final[str] = "Low"
    NORMAL: Final[str] = "Normal"
    HIGH: Final[str] = "High"
    CRITICAL: Final[str] = "Critical"
    
    VALID_PRIORITIES: Final[tuple] = (LOW, NORMAL, HIGH, CRITICAL)


class ClaimTypeConstants:
    """Valid claim type constants."""
    
    DAMAGE: Final[str] = "damage"
    LOSS: Final[str] = "loss"
    SHORTAGE: Final[str] = "shortage"
    DELAY: Final[str] = "delay"
    OTHER: Final[str] = "other"
    
    VALID_TYPES: Final[tuple] = (DAMAGE, LOSS, SHORTAGE, DELAY, OTHER)


class FieldMappingConstants:
    """Field name mappings between UiPath queue format and standard format."""
    
    QUEUE_TO_STANDARD: Final[dict] = {
        'ObjectClaimId': 'claim_id',
        'ClaimType': 'claim_type',
        'ClaimAmount': 'claim_amount',
        'ShipmentID': 'shipment_id',
        'Carrier': 'carrier',
        'CustomerName': 'customer_name',
        'CustomerEmail': 'customer_email',
        'CustomerPhone': 'customer_phone',
        'Description': 'description',
        'SubmissionSource': 'submission_source',
        'SubmittedAt': 'submitted_at',
        'ShippingDocumentsFiles': 'shipping_documents',
        'DamageEvidenceFiles': 'damage_evidence',
        'TransactionKey': 'transaction_key',
        'ProcessingPriority': 'processing_priority',
    }
    
    STANDARD_TO_QUEUE: Final[dict] = {v: k for k, v in QUEUE_TO_STANDARD.items()}


class ValidationConstants:
    """Validation limits and constraints."""
    
    MAX_CLAIM_AMOUNT: Final[float] = 1_000_000.0
    """Maximum allowed claim amount in USD."""
    
    MIN_CLAIM_AMOUNT: Final[float] = 0.0
    """Minimum allowed claim amount in USD."""
    
    MAX_DESCRIPTION_LENGTH: Final[int] = 5000
    """Maximum length for claim description."""
    
    MAX_DOCUMENTS_PER_CLAIM: Final[int] = 50
    """Maximum number of documents per claim."""


class RetryConstants:
    """Retry configuration constants."""
    
    MAX_RETRY_ATTEMPTS: Final[int] = 3
    """Maximum number of retry attempts for transient failures."""
    
    INITIAL_RETRY_DELAY: Final[float] = 1.0
    """Initial delay in seconds before first retry."""
    
    MAX_RETRY_DELAY: Final[float] = 10.0
    """Maximum delay in seconds between retries."""
    
    EXPONENTIAL_BASE: Final[float] = 2.0
    """Exponential backoff base multiplier."""
