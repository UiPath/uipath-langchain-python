"""Authentication helper utilities for UiPath SDK."""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def get_access_token(fallback_pat: Optional[str] = None) -> str:
    """
    Get access token from OAuth file or fallback to PAT.
    
    Tries to read OAuth token from .uipath/.auth.json first,
    then falls back to provided PAT if OAuth file is unavailable.
    
    Args:
        fallback_pat: Personal Access Token to use if OAuth token unavailable
        
    Returns:
        Access token string
        
    Raises:
        ValueError: If neither OAuth token nor PAT is available
    """
    # Try OAuth token from .uipath/.auth.json
    auth_file_path = os.path.join(os.getcwd(), ".uipath", ".auth.json")
    
    try:
        with open(auth_file_path, "r") as f:
            auth_data = json.load(f)
            access_token = auth_data.get("access_token")
            
        if access_token:
            logger.debug("Using OAuth token from .uipath/.auth.json")
            return access_token
        else:
            logger.warning("OAuth file exists but contains no access_token")
            
    except FileNotFoundError:
        logger.debug(f"OAuth file not found at {auth_file_path}")
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse OAuth file: {e}")
    except Exception as e:
        logger.warning(f"Error reading OAuth file: {e}")
    
    # Fallback to PAT
    if fallback_pat:
        logger.debug("Using PAT from settings")
        return fallback_pat
    
    raise ValueError(
        "No access token available. OAuth file not found and no PAT provided."
    )
