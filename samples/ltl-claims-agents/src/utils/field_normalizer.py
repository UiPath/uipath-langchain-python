"""Field normalization utilities for consistent data transformation."""

import logging
from typing import Dict, Any

from ..config.constants import FieldMappingConstants

logger = logging.getLogger(__name__)


class FieldNormalizer:
    """Utility class for normalizing field names between different formats."""
    
    @staticmethod
    def queue_to_standard(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert UiPath queue format (PascalCase) to standard format (snake_case).
        
        Args:
            data: Dictionary with PascalCase keys
            
        Returns:
            Dictionary with snake_case keys
        """
        normalized = dict(data)
        
        for queue_field, standard_field in FieldMappingConstants.QUEUE_TO_STANDARD.items():
            if queue_field in normalized and standard_field not in normalized:
                normalized[standard_field] = normalized[queue_field]
        
        return normalized
    
    @staticmethod
    def standard_to_queue(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format (snake_case) to UiPath queue format (PascalCase).
        
        Args:
            data: Dictionary with snake_case keys
            
        Returns:
            Dictionary with PascalCase keys
        """
        normalized = {}
        
        # Convert snake_case to PascalCase using mapping
        for snake_key, pascal_key in FieldMappingConstants.STANDARD_TO_QUEUE.items():
            if snake_key in data:
                normalized[pascal_key] = data[snake_key]
        
        # Preserve any existing PascalCase keys not in mapping
        for key, value in data.items():
            if key not in FieldMappingConstants.QUEUE_TO_STANDARD.values():
                normalized[key] = value
        
        return normalized
    
    @staticmethod
    def safe_float(value: Any, default: float = 0.0) -> float:
        """
        Safely convert value to float with error handling.
        
        Args:
            value: Value to convert
            default: Default value if conversion fails
            
        Returns:
            Float value or default
        """
        try:
            return float(value) if value not in (None, "", []) else default
        except (ValueError, TypeError):
            logger.warning(f"Could not convert {value} to float, using {default}")
            return default
    
    @staticmethod
    def safe_int(value: Any, default: int = 0) -> int:
        """
        Safely convert value to int with error handling.
        
        Args:
            value: Value to convert
            default: Default value if conversion fails
            
        Returns:
            Int value or default
        """
        try:
            return int(value) if value not in (None, "", []) else default
        except (ValueError, TypeError):
            logger.warning(f"Could not convert {value} to int, using {default}")
            return default
