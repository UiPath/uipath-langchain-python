"""
Configuration for orchestrator agent.
"""

from dataclasses import dataclass
from typing import Set


# Global constants
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_TIMEOUT_SECONDS = 30
MAX_RETRY_ATTEMPTS = 2


@dataclass
class OrchestratorConfig:
    """Configuration for orchestrator agent."""
    
    # Model configuration
    model_name: str = "gpt-4o-2024-08-06"
    temperature: float = 0.1
    max_tokens: int = 4000
    timeout: int = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = MAX_RETRY_ATTEMPTS
    
    # Confidence thresholds
    low_confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    escalation_threshold: float = 0.6
    
    # Confidence penalties
    high_risk_penalty: float = 0.2
    policy_violation_penalty: float = 0.15
    error_penalty: float = 0.1
    low_confidence_field_penalty: float = 0.05
    
    # Retry configuration
    max_step_retries: int = 2
    
    # Step names (for consistency)
    STEP_VALIDATE_DATA: str = 'validate_data'
    STEP_DOWNLOAD_DOCUMENTS: str = 'download_documents'
    STEP_EXTRACT_DATA: str = 'extract_data'
    STEP_ASSESS_RISK: str = 'assess_risk'
    STEP_VALIDATE_POLICY: str = 'validate_policy'
    STEP_MAKE_DECISION: str = 'make_decision'
    STEP_UPDATE_SYSTEMS: str = 'update_systems'
    
    @property
    def critical_steps(self) -> Set[str]:
        """Get set of critical steps that require escalation on failure."""
        return {
            self.STEP_VALIDATE_DATA,
            self.STEP_MAKE_DECISION,
            self.STEP_UPDATE_SYSTEMS
        }
    
    def get_default_plan_steps(self) -> list[str]:
        """Get default plan steps for fallback scenarios."""
        return [
            "Validate claim and shipment data in Data Fabric",
            "Download and process documents if available",
            "Assess risk factors and calculate risk score",
            "Validate against policies and carrier liability",
            "Make final decision based on all gathered information",
            "Update queue status and Data Fabric with results"
        ]


@dataclass
class BaseAgentConfig:
    """Base configuration shared by all agents."""
    
    # Common timeout settings
    default_timeout: int = DEFAULT_TIMEOUT_SECONDS
    
    # Common confidence thresholds
    low_confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD


@dataclass
class DocumentProcessorConfig(BaseAgentConfig):
    """Configuration for document processor agent."""
    
    # Document processing settings
    max_concurrent_downloads: int = 3
    cleanup_after_extraction: bool = False
    
    # IXP configuration
    ixp_project_name: str = "LTL Claims Processing"  # Default project name
    ixp_project_tag: str = "staging"  # Default tag
    
    # Timeout settings (override base)
    download_timeout: int = 30
    extraction_timeout: int = 60


@dataclass
class RiskAssessorConfig(BaseAgentConfig):
    """Configuration for risk assessor agent."""
    
    # Risk thresholds
    high_amount_threshold: float = 10000.0
    
    # High-risk claim types
    high_risk_claim_types: tuple = ("loss", "theft", "stolen")
    
    # Timeout settings (override base)
    assessment_timeout: int = 30


@dataclass
class ComplianceValidatorConfig(BaseAgentConfig):
    """Configuration for compliance validator agent."""
    
    # Policy limits
    default_max_claim_amount: float = 50000.0
    
    # Timeout settings (override base)
    validation_timeout: int = 30
