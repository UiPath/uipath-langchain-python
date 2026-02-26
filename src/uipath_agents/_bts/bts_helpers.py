"""BTS ID generation and parsing utilities."""

import uuid


def generate_guid() -> str:
    """Generate a GUID without hyphens (32 hex characters)."""
    return uuid.uuid4().hex


def generate_operation_id(transaction_id: str) -> str:
    """Generate a composite operation ID: {transaction_id}-{element_guid}."""
    return f"{transaction_id}-{generate_guid()}"


def extract_transaction_id(operation_id: str) -> str:
    """Extract the transaction ID from a composite operation ID."""
    return operation_id.split("-")[0]
