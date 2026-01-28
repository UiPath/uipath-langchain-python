"""Enums for UiPath guardrails."""

from enum import Enum

from uipath.core.guardrails import GuardrailScope as CoreGuardrailScope

# Re-export GuardrailScope from core for convenience
GuardrailScope = CoreGuardrailScope


class PIIDetectionEntityType(str, Enum):
    """PII detection entity types supported by UiPath guardrails.

    These entities match the available options from the UiPath guardrails service backend.
    The enum values correspond to the exact strings expected by the backend API.
    """

    PERSON = "Person"
    ADDRESS = "Address"
    DATE = "Date"
    PHONE_NUMBER = "PhoneNumber"
    EUGPS_COORDINATES = "EugpsCoordinates"
    EMAIL = "Email"
    CREDIT_CARD_NUMBER = "CreditCardNumber"
    INTERNATIONAL_BANKING_ACCOUNT_NUMBER = "InternationalBankingAccountNumber"
    SWIFT_CODE = "SwiftCode"
    ABA_ROUTING_NUMBER = "ABARoutingNumber"
    US_DRIVERS_LICENSE_NUMBER = "USDriversLicenseNumber"
    UK_DRIVERS_LICENSE_NUMBER = "UKDriversLicenseNumber"
    US_INDIVIDUAL_TAXPAYER_IDENTIFICATION = "USIndividualTaxpayerIdentification"
    UK_UNIQUE_TAXPAYER_NUMBER = "UKUniqueTaxpayerNumber"
    US_BANK_ACCOUNT_NUMBER = "USBankAccountNumber"
    US_SOCIAL_SECURITY_NUMBER = "USSocialSecurityNumber"
    USUK_PASSPORT_NUMBER = "UsukPassportNumber"
    URL = "URL"
    IP_ADDRESS = "IPAddress"


class GuardrailExecutionStage(str, Enum):
    """Execution stage for deterministic guardrails."""

    PRE = "pre"  # Pre-execution only
    POST = "post"  # Post-execution only
    PRE_AND_POST = "pre&post"  # Both pre and post execution
