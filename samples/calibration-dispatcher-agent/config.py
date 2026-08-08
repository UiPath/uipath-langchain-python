"""
Calibration Dispatcher Agent - Configuration

This file contains all environment-specific configurations including:
- UiPath Data Fabric entity IDs
- Folder paths and index names
- LLM model selection
- API keys and service endpoints
- MCP server configuration
- Business logic parameters

Adjust these values according to your UiPath environment.
"""

import os
from typing import Dict, Tuple

# =============================================================================
# UIPATH PLATFORM CONFIGURATION
# =============================================================================

# Folder path in UiPath Orchestrator (where processes and entities are deployed)
UIPATH_FOLDER_PATH = os.getenv("UIPATH_FOLDER_PATH", "Calibration Services")

# Context Grounding index name (created from Storage Bucket containing calibration policies)
CONTEXT_GROUNDING_INDEX_NAME = os.getenv(
    "CONTEXT_GROUNDING_INDEX_NAME",
    "Calibration Procedures"
)

# Number of policy documents to retrieve for RAG
CONTEXT_GROUNDING_NUM_RESULTS = int(os.getenv("CONTEXT_GROUNDING_NUM_RESULTS", "3"))

# =============================================================================
# DATA FABRIC ENTITY IDS
# =============================================================================
# These IDs must match your Data Fabric entities in UiPath Orchestrator.
# You can find them in Data Service > Entities > Entity Details

EQUIPMENT_ENTITY_ID = os.getenv(
    "EQUIPMENT_ENTITY_ID",
    "00000000-0000-0000-0000-000000000001"  # Replace with your Equipment entity ID
)

CLINICS_ENTITY_ID = os.getenv(
    "CLINICS_ENTITY_ID", 
    "00000000-0000-0000-0000-000000000002"  # Replace with your Clinics entity ID
)

TECHNICIANS_ENTITY_ID = os.getenv(
    "TECHNICIANS_ENTITY_ID",
    "00000000-0000-0000-0000-000000000003"  # Replace with your Technicians entity ID
)

# =============================================================================
# LLM CONFIGURATION
# =============================================================================

# Model selection for UiPath LLM Gateway
# Options: "gpt-4o-2024-11-20", "gpt-4o-mini", "claude-sonnet-4-5-20250929", etc.
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-2024-11-20")

# Temperature setting for LLM responses (0.0 = deterministic, 1.0 = creative)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# =============================================================================
# GOOGLE MAPS API CONFIGURATION
# =============================================================================

# Google Maps API key for route optimization
# Can be set via environment variable or UiPath Asset
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

# UiPath Asset name for Google Maps API key (fallback if env var not set)
GOOGLE_MAPS_ASSET_NAME = os.getenv("GOOGLE_MAPS_ASSET_NAME", "GoogleMapsApiKey")

# =============================================================================
# MCP SERVER CONFIGURATION
# =============================================================================

# Enable/disable MCP integration (set to "false" to use classic RPA invocation)
USE_MCP = os.getenv("USE_MCP", "true").lower() == "true"

# MCP server URL from UiPath Orchestrator (MCP Servers page)
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "")

# MCP tool input argument names (must match your RPA workflow parameter names)
MCP_ARG_EMAIL = os.getenv("RPA_ARG_NAME_EMAIL", "in_RouteData")
MCP_ARG_SLACK = os.getenv("RPA_ARG_NAME_SLACK", "in_MessageData")
MCP_ARG_ENTITY = os.getenv("RPA_ARG_NAME_ENTITY", "in_ServiceOrderData")

# =============================================================================
# ACTION CENTER CONFIGURATION
# =============================================================================

# Action Center form field names (must match your UiPath Apps form design)
APP_FIELD_SELECTED_OUTCOME = os.getenv("APP_FIELD_SELECTED_OUTCOME", "SelectedOutcome")
APP_FIELD_MANAGER_COMMENTS = os.getenv("APP_FIELD_MANAGER_COMMENTS", "ManagerComments")

# Email address for approval notifications
APPROVER_EMAIL = os.getenv("APPROVER_EMAIL", "manager@example.com")

# Maximum revision iterations per route before automatic rejection
MAX_REVISION_ITERATIONS = int(os.getenv("MAX_REVISION_ITERATIONS", "3"))

# =============================================================================
# BUSINESS LOGIC PARAMETERS
# =============================================================================

# City coordinates for distance calculations (lat, lng)
CITY_COORDS: Dict[str, Tuple[float, float]] = {
    "Warsaw": (52.2297, 21.0122),
    "Poznan": (52.4064, 16.9252),
    "Wroclaw": (51.1079, 17.0385),
    "Szczecin": (53.4285, 14.5528),
    "Krakow": (50.0647, 19.9450),
    "Gdansk": (54.3520, 18.6466),
}

# Device type to technician specialization mapping
DEVICE_TO_SPECIALIZATION: Dict[str, set] = {
    "Audiometer": {"Audiometry", "All"},
    "Tympanometer": {"Tympanometry", "All"},
}

# Standard service time per device type (hours)
SERVICE_TIME_AUDIOMETER = float(os.getenv("SERVICE_TIME_AUDIOMETER", "2.0"))
SERVICE_TIME_TYMPANOMETER = float(os.getenv("SERVICE_TIME_TYMPANOMETER", "1.5"))

# Default routing constraints (can be overridden by manager notes)
DEFAULT_MAX_VISITS_PER_ROUTE = int(os.getenv("DEFAULT_MAX_VISITS_PER_ROUTE", "4"))
DEFAULT_MAX_DISTANCE_KM = float(os.getenv("DEFAULT_MAX_DISTANCE_KM", "200.0"))
DEFAULT_MAX_WORK_HOURS = float(os.getenv("DEFAULT_MAX_WORK_HOURS", "8.0"))

# Override constraints for OVERDUE devices (emergency mode)
OVERDUE_MAX_VISITS_PER_ROUTE = int(os.getenv("OVERDUE_MAX_VISITS_PER_ROUTE", "5"))
OVERDUE_MAX_DISTANCE_KM = float(os.getenv("OVERDUE_MAX_DISTANCE_KM", "300.0"))
OVERDUE_MAX_WORK_HOURS = float(os.getenv("OVERDUE_MAX_WORK_HOURS", "12.0"))

# Cost parameters for route optimization
COST_PER_KM = float(os.getenv("COST_PER_KM", "0.50"))  # EUR per kilometer
TECHNICIAN_HOURLY_RATE = float(os.getenv("TECHNICIAN_HOURLY_RATE", "45.0"))  # EUR per hour

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Log format
LOG_FORMAT = os.getenv(
    "LOG_FORMAT",
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

# =============================================================================
# MOCK DATA CONFIGURATION (for testing without full UiPath setup)
# =============================================================================

# Enable mock mode for local testing (relaxes config validation, requires Data Fabric with imported data)
USE_MOCK_DATA = os.getenv("USE_MOCK_DATA", "true").lower() == "true"

# Auto-approve all routes in local testing (skips Action Center, auto-approves routes)
AUTO_APPROVE_IN_LOCAL = os.getenv("AUTO_APPROVE_IN_LOCAL", "true").lower() == "true"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def validate_config() -> bool:
    """
    Validate critical configuration values.
    Returns True if configuration is valid, False otherwise.
    """
    errors = []
    
    if not USE_MOCK_DATA:
        if EQUIPMENT_ENTITY_ID.startswith("00000000"):
            errors.append("EQUIPMENT_ENTITY_ID must be set to your actual Data Fabric entity ID")
        
        if CLINICS_ENTITY_ID.startswith("00000000"):
            errors.append("CLINICS_ENTITY_ID must be set to your actual Data Fabric entity ID")
        
        if TECHNICIANS_ENTITY_ID.startswith("00000000"):
            errors.append("TECHNICIANS_ENTITY_ID must be set to your actual Data Fabric entity ID")
    
    if USE_MCP and not MCP_SERVER_URL:
        errors.append("MCP_SERVER_URL must be set when USE_MCP=true")
    
    if errors:
        print("\n⚠️  Configuration Errors:")
        for error in errors:
            print(f"   - {error}")
        print("\n💡 Please update config.py or set environment variables.\n")
        return False
    
    return True


def print_config_summary():
    """Print a summary of current configuration (useful for debugging)."""
    print("=" * 70)
    print("CALIBRATION DISPATCHER AGENT - CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"Folder Path:           {UIPATH_FOLDER_PATH}")
    print(f"Context Grounding:     {CONTEXT_GROUNDING_INDEX_NAME}")
    print(f"LLM Model:             {LLM_MODEL}")
    print(f"Google Maps:           {'Enabled' if GOOGLE_MAPS_API_KEY else 'Disabled'}")
    print(f"MCP Integration:       {'Enabled' if USE_MCP else 'Disabled'}")
    print(f"Mock Data Mode:        {'Enabled' if USE_MOCK_DATA else 'Disabled'}")
    print(f"Max Visits/Route:      {DEFAULT_MAX_VISITS_PER_ROUTE}")
    print(f"Max Distance (km):     {DEFAULT_MAX_DISTANCE_KM}")
    print(f"Max Work Hours:        {DEFAULT_MAX_WORK_HOURS}")
    print("=" * 70)
    print()
