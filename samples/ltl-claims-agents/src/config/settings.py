"""Configuration management for LTL Claims Agent System."""

import os
import logging
from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

from .errors import ConfigurationError

logger = logging.getLogger(__name__)

# Load .env file on module import
env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info(f"Loaded environment variables from {env_path}")
else:
    logger.warning(f"No .env file found at {env_path}, using environment variables only")


class Settings(BaseSettings):
    """Application settings with environment variable support and validation."""
    
    # ============================================================================
    # UiPath Connection Configuration
    # ============================================================================
    # Support both token and client credential authentication
    uipath_base_url: str = Field("", env="UIPATH_BASE_URL")
    uipath_url: Optional[str] = Field(None, env="UIPATH_URL")  # Alternative URL field
    uipath_tenant: str = Field("", env="UIPATH_TENANT")
    uipath_tenant_id: Optional[str] = Field(None, env="UIPATH_TENANT_ID")  # Alternative tenant field
    uipath_organization: str = Field("", env="UIPATH_ORGANIZATION")
    uipath_organization_id: Optional[str] = Field(None, env="UIPATH_ORGANIZATION_ID")  # Alternative org field
    uipath_client_id: str = Field("", env="UIPATH_CLIENT_ID")
    uipath_client_secret: str = Field("", env="UIPATH_CLIENT_SECRET")
    uipath_access_token: Optional[str] = Field(None, env="UIPATH_ACCESS_TOKEN")  # Token-based auth
    uipath_pat_access_token: Optional[str] = Field(None, env="UIPATH_PAT_ACCESS_TOKEN")  # PAT token auth
    uipath_scope: str = Field("OR.Default", env="UIPATH_SCOPE")
    
    @property
    def effective_base_url(self) -> str:
        """Get the effective base URL, preferring UIPATH_BASE_URL over UIPATH_URL."""
        return self._get_effective_value('uipath_base_url', 'uipath_url')
    
    @property
    def effective_tenant(self) -> str:
        """Get the effective tenant, preferring UIPATH_TENANT over UIPATH_TENANT_ID."""
        return self._get_effective_value('uipath_tenant', 'uipath_tenant_id')
    
    @property
    def effective_organization(self) -> str:
        """Get the effective organization, preferring UIPATH_ORGANIZATION over UIPATH_ORGANIZATION_ID."""
        return self._get_effective_value('uipath_organization', 'uipath_organization_id')
    
    # UiPath Folder Configuration
    uipath_folder_id: str = Field("2360549", env="UIPATH_FOLDER_ID")
    uipath_folder_path: str = Field("Agents", env="UIPATH_FOLDER_PATH")
    
    # Storage Bucket Configuration
    uipath_bucket_id: str = Field("99943", env="UIPATH_BUCKET_ID")
    uipath_bucket_name: str = Field("LTL Freight Claim", env="UIPATH_BUCKET_NAME")
    
    # Data Fabric Entity Names (use IDs for API calls, names for reference)
    uipath_claims_entity: str = Field("73db44d1-08ad-f011-8e61-000d3a331eb3", env="UIPATH_CLAIMS_ENTITY")
    uipath_claims_entity_name: str = Field("LTLClaims", env="UIPATH_CLAIMS_ENTITY_NAME")
    uipath_shipments_entity: str = Field("9aea7964-7bad-f011-8e61-000d3a331eb3", env="UIPATH_SHIPMENTS_ENTITY")
    uipath_shipments_entity_name: str = Field("LTLShipments", env="UIPATH_SHIPMENTS_ENTITY_NAME")
    uipath_processing_history_entity: str = Field("1f197e60-09ad-f011-8e61-000d3a331eb3", env="UIPATH_PROCESSING_HISTORY_ENTITY")
    uipath_processing_history_entity_name: str = Field("LTLProcessingHistory", env="UIPATH_PROCESSING_HISTORY_ENTITY_NAME")
    
    # ============================================================================
    # Queue Configuration
    # ============================================================================
    # Support both QUEUE_NAME and UIPATH_QUEUE_NAME for backward compatibility
    queue_name: str = Field("LTL Claims Processing", env="QUEUE_NAME")
    uipath_queue_name: Optional[str] = Field(None, env="UIPATH_QUEUE_NAME")
    use_queue_input: bool = Field(True, env="USE_QUEUE_INPUT")
    input_file_path: str = Field("./claim_input.json", env="INPUT_FILE_PATH")
    queue_polling_interval: int = Field(30, env="QUEUE_POLLING_INTERVAL")
    
    @property
    def effective_queue_name(self) -> str:
        """Get the effective queue name, preferring QUEUE_NAME over UIPATH_QUEUE_NAME."""
        return self._get_effective_value('queue_name', 'uipath_queue_name') or "LTL Claims Processing"
    
    # ============================================================================
    # Processing Configuration
    # ============================================================================
    max_recursion_depth: int = Field(20, env="MAX_RECURSION_DEPTH")
    confidence_threshold: float = Field(0.7, env="CONFIDENCE_THRESHOLD")
    processing_timeout: int = Field(300, env="PROCESSING_TIMEOUT")  # seconds
    
    # ============================================================================
    # Memory Configuration
    # ============================================================================
    enable_long_term_memory: bool = Field(False, env="ENABLE_LONG_TERM_MEMORY")
    memory_store_type: str = Field("postgres", env="MEMORY_STORE_TYPE")  # postgres, redis, sqlite
    memory_connection_string: str = Field("", env="MEMORY_CONNECTION_STRING")
    
    # Action Center Configuration
    # ============================================================================
    enable_action_center: bool = Field(False, env="ENABLE_ACTION_CENTER")
    
    # ============================================================================
    # Action Center Configuration
    # ============================================================================
    action_center_app_name: str = Field("ClaimsTrackingApp", env="ACTION_CENTER_APP_NAME")
    action_center_folder_path: str = Field("Agents", env="ACTION_CENTER_FOLDER_PATH")
    action_center_assignee: str = Field("Claims_Reviewers", env="ACTION_CENTER_ASSIGNEE")
    
    # Context Grounding Configuration
    # ============================================================================
    context_grounding_index_name: str = Field("LTL Claims Processing", env="CONTEXT_GROUNDING_INDEX_NAME")
    enable_context_grounding: bool = Field(False, env="ENABLE_CONTEXT_GROUNDING")
    
    # ============================================================================
    # Timeout Configuration
    # ============================================================================
    api_timeout: int = Field(30, env="API_TIMEOUT")  # seconds
    document_extraction_timeout: int = Field(120, env="DOCUMENT_EXTRACTION_TIMEOUT")  # seconds
    
    # ============================================================================
    # Logging Configuration
    # ============================================================================
    log_level: str = Field("INFO", env="LOG_LEVEL")
    enable_debug_logging: bool = Field(False, env="ENABLE_DEBUG_LOGGING")
    log_file_path: str = Field("./logs/agent.log", env="LOG_FILE_PATH")
    log_format: str = Field("json", env="LOG_FORMAT")  # json or text
    
    # ============================================================================
    # Document Understanding Configuration
    # ============================================================================
    uipath_du_project_name: str = Field("LTL Claims Processing", env="UIPATH_DU_PROJECT_NAME")
    uipath_du_project_tag: str = Field("staging", env="UIPATH_DU_PROJECT_TAG")
    max_document_size_mb: int = Field(50, env="MAX_DOCUMENT_SIZE_MB")
    
    # ============================================================================
    # Context Grounding Configuration
    # ============================================================================
    context_grounding_index: str = Field("ltl-claims-policies", env="CONTEXT_GROUNDING_INDEX")
    
    # ============================================================================
    # MCP Configuration
    # ============================================================================
    stripe_mcp_endpoint: Optional[str] = Field(None, env="STRIPE_MCP_ENDPOINT")
    external_api_mcp_endpoint: Optional[str] = Field(None, env="EXTERNAL_API_MCP_ENDPOINT")
    
    # ============================================================================
    # Agent Configuration
    # ============================================================================
    max_concurrent_claims: int = Field(5, env="MAX_CONCURRENT_CLAIMS")
    risk_threshold_high: float = Field(0.8, env="RISK_THRESHOLD_HIGH")
    risk_threshold_medium: float = Field(0.5, env="RISK_THRESHOLD_MEDIUM")
    
    # ============================================================================
    # Notification Configuration
    # ============================================================================
    # Email settings (SendGrid)
    email_service: str = Field("sendgrid", env="EMAIL_SERVICE")
    sendgrid_api_key: str = Field("", env="SENDGRID_API_KEY")
    email_from_address: str = Field("noreply@ltlclaims.com", env="EMAIL_FROM_ADDRESS")
    email_from_name: str = Field("LTL Claims Processing", env="EMAIL_FROM_NAME")
    
    # Notification delivery settings
    notification_retry_max: int = Field(3, env="NOTIFICATION_RETRY_MAX")
    notification_retry_delay: int = Field(300, env="NOTIFICATION_RETRY_DELAY")  # seconds
    notification_batch_size: int = Field(10, env="NOTIFICATION_BATCH_SIZE")
    
    # ============================================================================
    # Tracing Configuration
    # ============================================================================
    enable_tracing: bool = Field(True, env="ENABLE_TRACING")
    trace_output_dir: str = Field("logs/traces", env="TRACE_OUTPUT_DIR")
    trace_level: str = Field("INFO", env="TRACE_LEVEL")  # DEBUG, INFO, WARNING, ERROR
    trace_include_inputs: bool = Field(True, env="TRACE_INCLUDE_INPUTS")
    trace_include_outputs: bool = Field(True, env="TRACE_INCLUDE_OUTPUTS")
    trace_max_string_length: int = Field(1000, env="TRACE_MAX_STRING_LENGTH")
    
    # ============================================================================
    # Development/Testing
    # ============================================================================
    debug_mode: bool = Field(False, env="DEBUG_MODE")
    test_mode: bool = Field(False, env="TEST_MODE")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    def _get_effective_value(self, *fields: str) -> str:
        """Get the first non-empty value from the provided fields."""
        for field in fields:
            value = getattr(self, field, None)
            if value and str(value).strip():
                return str(value)
        return ""
    
    def get_auth_method(self) -> str:
        """
        Determine which authentication method is being used.
        
        Returns:
            str: One of 'pat_token', 'access_token', 'client_credentials', or 'none'
        """
        if self.uipath_pat_access_token and self.uipath_pat_access_token.strip():
            return 'pat_token'
        elif self.uipath_access_token and self.uipath_access_token.strip():
            return 'access_token'
        elif (self.uipath_client_id and self.uipath_client_id.strip() and
              self.uipath_client_secret and self.uipath_client_secret.strip()):
            return 'client_credentials'
        else:
            return 'none'
    
    def get_config_summary(self) -> dict:
        """
        Get a safe summary of configuration without exposing secrets.
        
        Returns:
            dict: Configuration summary with sensitive values masked
        """
        return {
            "uipath": {
                "base_url": self.effective_base_url,
                "tenant": self.effective_tenant or "[not set]",
                "organization": self.effective_organization or "[not set]",
                "auth_method": self.get_auth_method(),
                "folder_path": self.uipath_folder_path,
                "bucket_name": self.uipath_bucket_name,
            },
            "input": {
                "use_queue": self.use_queue_input,
                "queue_name": self.effective_queue_name if self.use_queue_input else "[not applicable]",
                "file_path": self.input_file_path if not self.use_queue_input else "[not applicable]",
            },
            "processing": {
                "max_recursion_depth": self.max_recursion_depth,
                "confidence_threshold": self.confidence_threshold,
                "processing_timeout": self.processing_timeout,
            },
            "features": {
                "long_term_memory": self.enable_long_term_memory,
                "action_center": self.enable_action_center,
                "context_grounding": self.enable_context_grounding,
            },
            "logging": {
                "level": self.log_level,
                "format": self.log_format,
                "debug_mode": self.debug_mode,
            }
        }
    
    def configure_logging(self) -> None:
        """
        Configure Python logging based on settings.
        
        Sets up logging level, format, and file output based on configuration.
        Should be called early in application startup.
        """
        import logging
        from pathlib import Path
        
        # Set log level
        log_level = getattr(logging, self.log_level.upper(), logging.INFO)
        
        # Create logs directory if needed
        if self.log_file_path:
            log_path = Path(self.log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' if self.log_format == 'text' 
                   else '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_file_path) if self.log_file_path else logging.NullHandler()
            ]
        )
        
        # Enable debug logging if configured
        if self.enable_debug_logging or self.debug_mode:
            logging.getLogger().setLevel(logging.DEBUG)
    
    @field_validator('uipath_bucket_name')
    @classmethod
    def validate_bucket_name(cls, v: str) -> str:
        """Validate that bucket name is provided."""
        if not v or v.strip() == "":
            raise ConfigurationError(
                "UIPATH_BUCKET_NAME is required and cannot be empty",
                context={"validation_phase": "bucket_configuration"},
                details={"field": "uipath_bucket_name", "value": v},
                missing_fields=["UIPATH_BUCKET_NAME"]
            )
        return v
    
    @field_validator('uipath_claims_entity_name')
    @classmethod
    def validate_claims_entity_name(cls, v: str) -> str:
        """Validate that claims entity name is provided."""
        if not v or v.strip() == "":
            raise ConfigurationError(
                "UIPATH_CLAIMS_ENTITY_NAME is required and cannot be empty",
                context={"validation_phase": "entity_configuration"},
                details={"field": "uipath_claims_entity_name", "value": v},
                missing_fields=["UIPATH_CLAIMS_ENTITY_NAME"]
            )
        return v
    
    @field_validator('max_recursion_depth')
    @classmethod
    def validate_max_recursion_depth(cls, v: int) -> int:
        """Validate that max recursion depth is within acceptable range."""
        if v < 1 or v > 100:
            raise ConfigurationError(
                f"MAX_RECURSION_DEPTH must be between 1 and 100, got {v}",
                context={"validation_phase": "processing_configuration"},
                details={"field": "max_recursion_depth", "value": v, "min": 1, "max": 100}
            )
        return v
    
    @field_validator('confidence_threshold')
    @classmethod
    def validate_confidence_threshold(cls, v: float) -> float:
        """Validate that confidence threshold is between 0 and 1."""
        if v < 0.0 or v > 1.0:
            raise ConfigurationError(
                f"CONFIDENCE_THRESHOLD must be between 0.0 and 1.0, got {v}",
                context={"validation_phase": "processing_configuration"},
                details={"field": "confidence_threshold", "value": v, "min": 0.0, "max": 1.0}
            )
        return v
    
    @field_validator('processing_timeout', 'api_timeout', 'document_extraction_timeout')
    @classmethod
    def validate_positive_timeout(cls, v: int, info) -> int:
        """Validate that timeout values are positive."""
        if v < 1:
            field_name = info.field_name.upper()
            raise ConfigurationError(
                f"{field_name} must be a positive integer (seconds), got {v}",
                context={"validation_phase": "timeout_configuration"},
                details={"field": info.field_name, "value": v, "min": 1, "unit": "seconds"},
                missing_fields=[field_name]
            )
        return v
    
    @field_validator('log_level', 'trace_level')
    @classmethod
    def validate_log_level(cls, v: str, info) -> str:
        """Validate that log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            field_name = info.field_name.upper()
            raise ConfigurationError(
                f"{field_name} must be one of {valid_levels}, got {v}",
                context={"validation_phase": "logging_configuration"},
                details={"field": info.field_name, "value": v, "valid_values": valid_levels}
            )
        return v.upper()
    
    @field_validator('trace_max_string_length')
    @classmethod
    def validate_trace_max_string_length(cls, v: int) -> int:
        """Validate that trace max string length is reasonable."""
        if v < 100:
            raise ConfigurationError(
                f"TRACE_MAX_STRING_LENGTH must be at least 100, got {v}",
                context={"validation_phase": "tracing_configuration"},
                details={"field": "trace_max_string_length", "value": v, "min": 100}
            )
        return v
    
    @model_validator(mode='after')
    def validate_authentication(self) -> 'Settings':
        """Validate that proper authentication credentials are provided."""
        # Check available authentication methods
        has_pat = bool(self.uipath_pat_access_token and self.uipath_pat_access_token.strip())
        has_access_token = bool(self.uipath_access_token and self.uipath_access_token.strip())
        has_client_id = bool(self.uipath_client_id and self.uipath_client_id.strip())
        has_client_secret = bool(self.uipath_client_secret and self.uipath_client_secret.strip())
        
        # Determine which auth method is being used
        using_token_auth = has_pat or has_access_token
        using_client_creds = has_client_id and has_client_secret
        
        # Must have at least one complete authentication method
        if not (using_token_auth or using_client_creds):
            missing = []
            if not has_pat:
                missing.append("UIPATH_PAT_ACCESS_TOKEN")
            if not has_access_token:
                missing.append("UIPATH_ACCESS_TOKEN")
            if not has_client_id or not has_client_secret:
                missing.extend(["UIPATH_CLIENT_ID", "UIPATH_CLIENT_SECRET"])
            
            raise ConfigurationError(
                "Authentication credentials required: provide either "
                "UIPATH_PAT_ACCESS_TOKEN, UIPATH_ACCESS_TOKEN, or "
                "both UIPATH_CLIENT_ID and UIPATH_CLIENT_SECRET",
                context={"validation_phase": "authentication"},
                missing_fields=missing
            )
        
        # If using client credentials (and no token), require tenant and organization
        if using_client_creds and not using_token_auth:
            if not self.effective_tenant:
                raise ConfigurationError(
                    "UIPATH_TENANT or UIPATH_TENANT_ID is required when using client credentials",
                    context={"validation_phase": "authentication", "auth_method": "client_credentials"},
                    missing_fields=["UIPATH_TENANT", "UIPATH_TENANT_ID"]
                )
            if not self.effective_organization:
                raise ConfigurationError(
                    "UIPATH_ORGANIZATION or UIPATH_ORGANIZATION_ID is required when using client credentials",
                    context={"validation_phase": "authentication", "auth_method": "client_credentials"},
                    missing_fields=["UIPATH_ORGANIZATION", "UIPATH_ORGANIZATION_ID"]
                )
        
        return self
    
    @model_validator(mode='after')
    def validate_base_url(self) -> 'Settings':
        """Validate that base URL is provided."""
        if not self.effective_base_url:
            raise ConfigurationError(
                "UIPATH_BASE_URL or UIPATH_URL is required",
                context={"validation_phase": "connection_configuration"},
                missing_fields=["UIPATH_BASE_URL", "UIPATH_URL"]
            )
        return self
    
    @model_validator(mode='after')
    def validate_input_source(self) -> 'Settings':
        """Validate input source configuration."""
        if self.use_queue_input:
            if not self.effective_queue_name:
                raise ConfigurationError(
                    "QUEUE_NAME or UIPATH_QUEUE_NAME is required when USE_QUEUE_INPUT is true",
                    context={"validation_phase": "input_configuration", "use_queue_input": True},
                    missing_fields=["QUEUE_NAME", "UIPATH_QUEUE_NAME"]
                )
        else:
            if not self.input_file_path:
                raise ConfigurationError(
                    "INPUT_FILE_PATH is required when USE_QUEUE_INPUT is false",
                    context={"validation_phase": "input_configuration", "use_queue_input": False},
                    missing_fields=["INPUT_FILE_PATH"]
                )
        return self
    
    @model_validator(mode='after')
    def validate_memory_config(self) -> 'Settings':
        """Validate memory configuration if enabled."""
        if self.enable_long_term_memory:
            if not self.memory_connection_string:
                raise ConfigurationError(
                    "MEMORY_CONNECTION_STRING is required when ENABLE_LONG_TERM_MEMORY is true",
                    context={"validation_phase": "memory_configuration", "enable_long_term_memory": True},
                    missing_fields=["MEMORY_CONNECTION_STRING"]
                )
            
            valid_store_types = ["postgres", "redis", "sqlite"]
            if self.memory_store_type not in valid_store_types:
                raise ConfigurationError(
                    f"MEMORY_STORE_TYPE must be one of {valid_store_types}, got {self.memory_store_type}",
                    context={"validation_phase": "memory_configuration"},
                    details={"field": "memory_store_type", "value": self.memory_store_type, 
                            "valid_values": valid_store_types}
                )
        return self
    



# Global settings instance - initialized on module import
# This will raise ConfigurationError if validation fails
try:
    settings = Settings()
except Exception as e:
    # Re-raise as ConfigurationError if it's not already
    if not isinstance(e, ConfigurationError):
        raise ConfigurationError(f"Failed to initialize settings: {str(e)}") from e
    raise