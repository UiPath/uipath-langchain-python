#!/usr/bin/env python3
"""
MCP Server for UiPath Process Orchestration - Stateless Implementation
"""
from typing import Dict, Any, Optional
import httpx
import os
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for PIMS settings."""
    def __init__(self):
        self.environment = os.getenv("UIPATH_ENVIRONMENT", "alpha")
        self.org_id = os.getenv("UIPATH_ORG_ID", "")
        self.tenant_name = os.getenv("UIPATH_TENANT", "")
        self.folder_key = os.getenv("FOLDER_KEY", "")
        # Support both AUTH_TOKEN and UIPATH_ACCESS_TOKEN
        self.auth_token = os.getenv("AUTH_TOKEN", "") or os.getenv("UIPATH_ACCESS_TOKEN", "")
        
    def set_config(self, org_id: str = None, tenant_name: str = None, environment: str = None, folder_key: str = None, auth_token: str = None):
        """Set configuration values."""
        if org_id:
            self.org_id = org_id
        if tenant_name:
            self.tenant_name = tenant_name
        if environment:
            self.environment = environment
        if folder_key:
            self.folder_key = folder_key
        if auth_token:
            self.auth_token = auth_token
    
    @property
    def pims_url(self) -> str:
        """Construct the PIMS URL from components."""
        if not self.org_id or not self.tenant_name:
            return ""
        return f"https://{self.environment}.uipath.com/{self.org_id}/{self.tenant_name}/pims_/api"
        
    def validate(self):
        """Validate that required configuration is present."""
        if not self.org_id:
            raise ValueError("UIPATH_ORG_ID is required (set via environment variable or configure_server tool)")
        if not self.tenant_name:
            raise ValueError("UIPATH_TENANT is required (set via environment variable or configure_server tool)")
        if not self.folder_key:
            raise ValueError("FOLDER_KEY is required (set via environment variable or configure_server tool)")
        if not self.auth_token:
            raise ValueError("AUTH_TOKEN or UIPATH_ACCESS_TOKEN is required (set via environment variable or configure_server tool)")
        return True

# Global configuration instance
config = Config()

def get_auth_headers(
    folder_key: str = None,
    auth_token: str = "",
    request_headers: Optional[Dict[str, str]] = None,
    content_type: bool = True
) -> Dict[str, str]:
    """Build headers with authentication token.
    
    Args:
        folder_key: UiPath folder key (uses config if not provided)
        auth_token: Direct auth token (uses config if not provided)
        request_headers: MCP connection headers (if available)
        content_type: Whether to include Content-Type header
    """
    # Debug: Show what headers we received from MCP
    print(f"=== DEBUG: MCP Headers Analysis ===")
    print(f"request_headers type: {type(request_headers)}")
    print(f"request_headers is None: {request_headers is None}")
    if request_headers:
        print(f"request_headers keys: {list(request_headers.keys())}")
        print(f"request_headers content:")
        for key, value in request_headers.items():
            if key.lower() in ['authorization', 'auth', 'token']:
                # Show auth-related headers with partial masking
                if len(value) > 20:
                    masked_value = f"{value[:10]}...{value[-10:]}"
                else:
                    masked_value = f"{value[:5]}..."
                print(f"  {key}: '{masked_value}' (masked)")
            else:
                print(f"  {key}: '{value}'")
    else:
        print("  No request_headers provided")
    print(f"auth_token parameter: '{auth_token[:10]}...' (truncated)" if auth_token else "auth_token parameter: (empty)")
    print(f"config.auth_token: '{config.auth_token[:10]}...' (truncated)" if config.auth_token else "config.auth_token: (empty)")
    print("=== END DEBUG ===")
    
    # Use config folder_key if not provided
    if folder_key is None:
        folder_key = config.folder_key
        print(f"Using config folder_key: '{folder_key}'")
    
    # Validate that we have a folder_key
    if not folder_key:
        raise ValueError("folder_key is required but not provided and not configured in server")
    
    # Priority order for auth token:
    # 1. Explicitly provided auth_token parameter
    # 2. Token from MCP request headers
    # 3. Config auth_token (from environment variables)
    
    final_auth_token = ""
    token_source = ""
    
    if auth_token:
        final_auth_token = auth_token
        token_source = "provided parameter"
        print(f"[OK] Using provided auth_token: '{auth_token[:10]}...' (truncated)")
    elif request_headers and "Authorization" in request_headers:
        auth_header = request_headers["Authorization"]
        if auth_header.startswith("Bearer "):
            final_auth_token = auth_header.split(" ")[1]
            token_source = "MCP headers (Bearer format)"
            print(f"[OK] Using token from MCP request headers (Bearer): '{final_auth_token[:10]}...' (truncated)")
        else:
            final_auth_token = auth_header
            token_source = "MCP headers (raw format)"
            print(f"[OK] Using token from MCP request headers (raw): '{final_auth_token[:10]}...' (truncated)")
    elif request_headers:
        # Check for other possible auth header names
        auth_candidates = ['authorization', 'auth', 'token', 'access-token', 'access_token']
        for candidate in auth_candidates:
            for header_key, header_value in request_headers.items():
                if header_key.lower() == candidate.lower():
                    final_auth_token = header_value
                    if final_auth_token.startswith("Bearer "):
                        final_auth_token = final_auth_token.split(" ")[1]
                    token_source = f"MCP headers ({header_key})"
                    print(f"[OK] Found token in MCP header '{header_key}': '{final_auth_token[:10]}...' (truncated)")
                    break
            if final_auth_token:
                break
    
    if not final_auth_token and config.auth_token:
        final_auth_token = config.auth_token
        token_source = "config/environment"
        print(f"[OK] Using config auth_token: '{config.auth_token[:10]}...' (truncated)")

    if not final_auth_token:
        print("[ERROR] No auth token found in any source!")
        raise ValueError("No auth token available from parameters, MCP headers, or config")

    print(f"Final token source: {token_source}")

    headers = {
        "Accept": "application/json",
        "x-uipath-folderkey": folder_key,
        "Authorization": f"Bearer {final_auth_token}"
    }
    
    if content_type:
        headers["Content-Type"] = "application/json"
    
    print(f"Final headers being sent:")
    for key, value in headers.items():
        if key.lower() == 'authorization':
            print(f"  {key}: Bearer {value.split(' ')[1][:10] if len(value.split(' ')) > 1 else 'invalid'}... (truncated)")
        else:
            print(f"  {key}: {value}")
        
    return headers

# Initialize MCP Server
mcp = FastMCP("UiPath Process Instance Management Server")

@mcp.tool()
async def configure_server(
    org_id: str,
    tenant_name: str,
    environment: str,
    auth_token: str = None,
    folder_key: str = None,
    *,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Configure the server for subsequent operations.

    Args:
        org_id: UiPath organization ID
        tenant_name: UiPath tenant name
        environment: UiPath environment (e.g., "alpha", "beta", "production")
        folder_key: UiPath folder key for authorization context (optional if provided in env)
        auth_token: Authentication token for API access (optional if provided in headers or env)
        headers: Request headers (automatically provided by MCP, can contain Authorization token)

    Returns:
        Dictionary containing the configuration result
    """
  
    
    # Auth token priority: 1. Function parameter, 2. MCP headers, 3. Environment variables
    final_auth_token = None
    token_source = "none"
    
    # 1. Check function parameter
    if auth_token:
        final_auth_token = auth_token
        token_source = "function parameter"
        print(f"[OK] Using auth token from function parameter: {auth_token[:10]}...")
    
    # 2. Check MCP headers
    elif headers:
        print(f"Checking MCP headers for auth token...")
        print(f"Available headers: {list(headers.keys())}")
        
        # Check Authorization header first
        if "Authorization" in headers:
            auth_header = headers["Authorization"]
            if auth_header.startswith("Bearer "):
                final_auth_token = auth_header.split(" ")[1]
                token_source = "MCP Authorization header (Bearer)"
                print(f"[OK] Using auth token from MCP Authorization header: {final_auth_token[:10]}...")
            else:
                final_auth_token = auth_header
                token_source = "MCP Authorization header (raw)"
                print(f"[OK] Using auth token from MCP Authorization header (raw): {final_auth_token[:10]}...")
        
        # Check other possible auth header names
        else:
            auth_candidates = ['authorization', 'auth', 'token', 'access-token', 'access_token']
            for candidate in auth_candidates:
                for header_key, header_value in headers.items():
                    if header_key.lower() == candidate.lower():
                        final_auth_token = header_value
                        if final_auth_token.startswith("Bearer "):
                            final_auth_token = final_auth_token.split(" ")[1]
                        token_source = f"MCP header ({header_key})"
                        print(f"[OK] Using auth token from MCP header '{header_key}': {final_auth_token[:10]}...")
                        break
                if final_auth_token:
                    break
    
    # 3. Check environment variables
    if not final_auth_token:
        env_auth_token = os.getenv("AUTH_TOKEN", "") or os.getenv("UIPATH_ACCESS_TOKEN", "")
        if env_auth_token:
            final_auth_token = env_auth_token
            token_source = "environment variable"
            print(f"[OK] Using auth token from environment variable: {env_auth_token[:10]}...")
    

    # Validate that we have an auth token from some source
    if not final_auth_token:
        error_msg = "Authentication token must be provided through one of these methods:\n"
        error_msg += "1. Pass auth_token parameter to this function\n"
        error_msg += "2. Include Authorization header in MCP connection\n"
        error_msg += "3. Set AUTH_TOKEN or UIPATH_ACCESS_TOKEN environment variable\n"
        error_msg += "4. Create a .env file with AUTH_TOKEN or UIPATH_ACCESS_TOKEN"
        raise ValueError(error_msg)

    print(f"Final auth token source: {token_source}")
    
    # Use environment folder_key if not provided
    if not folder_key:
        folder_key = os.getenv("FOLDER_KEY", "")
        if folder_key:
            print(f"Using folder_key from environment variable: {folder_key}")
        else:
            print("[WARNING] No folder_key provided and not found in environment variables")

    print(f"Configuring server with org_id: '{org_id}', tenant_name: '{tenant_name}', environment: '{environment}', folder_key: '{folder_key}', auth_token: '{final_auth_token[:10]}...' (truncated)")
    config.set_config(org_id, tenant_name, environment, folder_key, final_auth_token)

    try:
        config.validate()  # Will raise ValueError if any required config is missing
        status = "success"
        message = "Server configured successfully"
    except ValueError as e:
        status = "error"
        message = str(e)
        raise  # Re-raise to maintain existing error handling

    print("=== Configuring Server ===")
    print(f"Org ID: {org_id}")
    print(f"Tenant: {tenant_name}")
    print(f"Environment: {environment}")
    print(f"Folder Key: {folder_key if folder_key else 'NOT PROVIDED'}")
    print(f"Auth Token: {f'PROVIDEDToken:{final_auth_token[:10]}' if auth_token else 'NOT PROVIDED'}")

    return {
        "status": status,
        "message": message,
        "pims_url": config.pims_url,
        "org_id": config.org_id,
        "tenant_name": config.tenant_name,
        "environment": config.environment,
        "folder_key": config.folder_key,
        "auth_token_configured": config.auth_token[:10] if config.auth_token else "NOT SET",
        "auth_token_source": token_source,
        "auth_token_full": final_auth_token[:10] if final_auth_token else "NOT PROVIDED",
        "auth_token_length": len(final_auth_token) if final_auth_token else 0
    }

@mcp.tool()
async def debug_headers(
    *,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Debug tool to inspect MCP headers and configuration.
    
    Returns:
        Dictionary containing header and configuration analysis
    """
    result = {
        "mcp_headers_received": {},
        "environment_variables": {},
        "config_status": {},
        "auth_analysis": {}
    }
    
    # Analyze MCP headers
    if headers:
        result["mcp_headers_received"] = {
            "count": len(headers),
            "keys": list(headers.keys()),
            "headers": {}
        }
        for key, value in headers.items():
            if key.lower() in ['authorization', 'auth', 'token', 'access-token', 'access_token']:
                # Mask sensitive values
                if len(value) > 20:
                    masked_value = f"{value[:10]}...{value[-10:]}"
                else:
                    masked_value = f"{value[:5]}..."
                result["mcp_headers_received"]["headers"][key] = f"{masked_value} (masked)"
            else:
                result["mcp_headers_received"]["headers"][key] = value
    else:
        result["mcp_headers_received"]["message"] = "No headers received from MCP"
    
    # Check environment variables
    result["environment_variables"] = {
        "UIPATH_ACCESS_TOKEN": f"Set {len(os.getenv('UIPATH_ACCESS_TOKEN',''))}" if os.getenv("UIPATH_ACCESS_TOKEN") else "Not set",
        "AUTH_TOKEN": "Set" if os.getenv("AUTH_TOKEN") else "Not set",
        "UIPATH_ORG_ID": "Set" if os.getenv("UIPATH_ORG_ID") else "Not set",
        "UIPATH_TENANT": "Set" if os.getenv("UIPATH_TENANT") else "Not set",
        "FOLDER_KEY": "Set" if os.getenv("FOLDER_KEY") else "Not set",
        "UIPATH_ENVIRONMENT": os.getenv("UIPATH_ENVIRONMENT", "alpha")
    }
    
    # Check config status
    result["config_status"] = {
        "org_id": config.org_id if config.org_id else "Not configured",
        "tenant_name": config.tenant_name if config.tenant_name else "Not configured",
        "environment": config.environment,
        "folder_key": config.folder_key if config.folder_key else "Not configured",
        "auth_token_configured": bool(config.auth_token),
        "pims_url": config.pims_url if config.pims_url else "Cannot construct (missing org_id or tenant)"
    }
    
    # Auth analysis
    auth_sources = []
    if headers and "Authorization" in headers:
        auth_sources.append("MCP Authorization header")
    if headers:
        other_auth_headers = [k for k in headers.keys() if k.lower() in ['auth', 'token', 'access-token', 'access_token']]
        if other_auth_headers:
            auth_sources.extend([f"MCP {h} header" for h in other_auth_headers])
    if config.auth_token:
        auth_sources.append("Environment variable/config")
    
    result["auth_analysis"] = {
        "available_sources": auth_sources,
        "recommended_action": "All required config present" if len(auth_sources) > 0 and config.org_id and config.tenant_name and config.folder_key else "Missing required configuration"
    }
    
    return result

@mcp.tool()
async def get_server_config() -> Dict[str, Any]:
    """Get the current server configuration.
    
    Returns:
        Dictionary containing the current configuration
    """
    return {
        "pims_url": config.pims_url,
        "org_id": config.org_id,
        "tenant_name": config.tenant_name,
        "environment": config.environment,
        "folder_key": config.folder_key,
        "auth_token_configured": bool(config.auth_token),
        "configured": bool(config.org_id and config.tenant_name and config.folder_key and config.auth_token)
    }

@mcp.tool()
async def get_instance(
    instance_id: str,
    *,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Get a UiPath process instance details.

    Args:
        instance_id: Process instance ID to retrieve

    Returns:
        Dictionary containing the process instance details
    """
    config.validate()
    url = f"{config.pims_url}/v1/instances/{instance_id}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers=get_auth_headers(None, "", headers, content_type=False),
                timeout=30.0
            )
            
            if response.status_code == 200:
                try:
                    result_data = response.json()
                except Exception:
                    result_data = {"raw_response": response.text}
                    
                return {
                    "status": "success",
                    "message": f"Successfully retrieved instance {instance_id}",
                    "instanceId": instance_id,
                    "data": result_data
                }
            else:
                error_message = f"Failed to get instance {instance_id}. Status: {response.status_code}"
                if response.text:
                    error_message += f", Response: {response.text}"
                raise ValueError(error_message)
                
    except Exception as e:
        raise ValueError(f"Error getting instance: {str(e)}")

@mcp.tool()
async def pause_instance(
    instance_id: str,
    comment: str = "",
    *,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Pause a UiPath process instance.
    
    Args:
        instance_id: Process instance ID to pause
        comment: Optional comment for pausing the instance
        
    Returns:
        Dictionary containing the pause operation result
    """
    config.validate()
    url = f"{config.pims_url}/v1/instances/{instance_id}/pause"
    pause_request = {"comment": comment}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=get_auth_headers(None, "", headers),
                json=pause_request,
                timeout=30.0
            )
            
            if response.status_code == 200:
                try:
                    result_data = response.json()
                except Exception:
                    result_data = {"raw_response": response.text}
                
                return {
                    "status": "success",
                    "message": f"Successfully paused instance {instance_id}",
                    "instanceId": instance_id,
                    "comment": comment,
                    "data": result_data
                }
            else:
                error_message = f"Failed to pause instance {instance_id}. Status: {response.status_code}"
                if response.text:
                    error_message += f", Response: {response.text}"
                raise ValueError(error_message)
                
    except Exception as e:
        raise ValueError(f"Error pausing instance: {str(e)}")


@mcp.tool()
async def update_instance_variables(
    instance_id: str,
    variables: dict,
    parent_element_id: str = None,
    *,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Update global variables and element inputs for a UiPath process instance.

    Args:
        instance_id: Process instance ID to update variables for
        variables: Dictionary representing the PatchVariablesRequest body
        parent_element_id: Optional parent element ID for scoping the update

    Returns:
        Dictionary containing the update operation result
    """
    config.validate()
    url = f"{config.pims_url}/v1/instances/{instance_id}/variables"
    params = {}
    if parent_element_id:
        params["parentElementId"] = parent_element_id

    try:
        async with httpx.AsyncClient() as client:
            response = await client.patch(
                url,
                headers=get_auth_headers(None, "", headers),
                json=variables,
                params=params,
                timeout=30.0
            )

            if response.status_code == 204:
                return {
                    "status": "success",
                    "message": f"Successfully updated variables for instance {instance_id}",
                    "instanceId": instance_id,
                    "data": None
                }
            else:
                error_message = f"Failed to update variables for instance {instance_id}. Status: {response.status_code}"
                if response.text:
                    error_message += f", Response: {response.text}"
                raise ValueError(error_message)

    except Exception as e:
        raise ValueError(f"Error updating instance variables: {str(e)}")

@mcp.tool()
async def resume_instance(
    instance_id: str,
    comment: str = "",
    *,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Resume a UiPath process instance.

    Args:
        instance_id: Process instance ID to resume
        comment: Optional comment for resuming the instance

    Returns:
        Dictionary containing the resume operation result
    """
    config.validate()
    url = f"{config.pims_url}/v1/instances/{instance_id}/resume"
    resume_request = {"comment": comment}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=get_auth_headers(None, "", headers),
                json=resume_request,
                timeout=30.0
            )

            if response.status_code == 200:
                try:
                    result_data = response.json()
                except Exception:
                    result_data = {"raw_response": response.text}

                return {
                    "status": "success",
                    "message": f"Successfully resumed instance {instance_id}",
                    "instanceId": instance_id,
                    "comment": comment,
                    "data": result_data
                }
            else:
                error_message = f"Failed to resume instance {instance_id}. Status: {response.status_code}"
                if response.text:
                    error_message += f", Response: {response.text}"
                raise ValueError(error_message)
                
    except Exception as e:
        raise ValueError(f"Error resuming instance: {str(e)}")

@mcp.tool()
async def retry_instance(
    instance_id: str,
    comment: str = "",
    *,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Retry a UiPath process instance.

    Args:
        instance_id: Process instance ID to retry
        comment: Optional comment for retrying the instance
        headers: Request headers (automatically provided by MCP when published)

    Returns:
        Dictionary containing the retry operation result
    """
    config.validate()
    url = f"{config.pims_url}/v1/instances/{instance_id}/retry"
    retry_request = {"comment": comment}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=get_auth_headers(None, "", headers),
                json=retry_request,
                timeout=30.0
            )

            if response.status_code == 200:
                try:
                    result_data = response.json()
                except Exception:
                    result_data = {"raw_response": response.text}

                return {
                    "status": "success",
                    "message": f"Successfully retried instance {instance_id}",
                    "instanceId": instance_id,
                    "comment": comment,
                    "data": result_data
                }
            else:
                error_message = f"Failed to retry instance {instance_id}. Status: {response.status_code}"
                if response.text:
                    error_message += f", Response: {response.text}"
                raise ValueError(error_message)

    except Exception as e:
        raise ValueError(f"Error retrying instance: {str(e)}")


@mcp.tool()
async def goto_transitions(
    instance_id: str,
    source_element_id: str,
    target_element_id: str,
    *,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Apply GoTo transition on a UiPath process instance.

    Args:
        instance_id: Process instance ID to apply GoTo transition
        source_element_id: ID of the source element (e.g., "Event_lVLnFF")
        target_element_id: ID of the target element (e.g., "Activity_zGJQzH")

    Returns:
        Dictionary containing the GoTo operation result
    """
    config.validate()
    url = f"{config.pims_url}/v1/instances/{instance_id}/goto"
    
    # Create single transition object
    transition = {
        "sourceElementId": source_element_id,
        "targetElementId": target_element_id
    }
    
    goto_request = {
        "comment": "",
        "transitions": [transition]  # Wrap in list as required by API
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=get_auth_headers(None, "", headers),
                json=goto_request,
                timeout=30.0
            )

            if response.status_code == 200:
                try:
                    result_data = response.json()
                except Exception:
                    result_data = {"raw_response": response.text}

                return {
                    "status": "success",
                    "message": f"Successfully applied GoTo transition for instance {instance_id}",
                    "instanceId": instance_id,
                    "transition": transition,
                    "data": result_data
                }
            else:
                error_message = f"Failed to apply GoTo transition for instance {instance_id}. Status: {response.status_code}"
                if response.text:
                    error_message += f", Response: {response.text}"
                raise ValueError(error_message)

    except Exception as e:
        raise ValueError(f"Error applying GoTo transition: {str(e)}")

@mcp.tool()
async def cancel_instance(
    instance_id: str,
    comment: str = "",
    *,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Cancel a UiPath process instance.

    Args:
        instance_id: Process instance ID to cancel
        comment: Optional comment for canceling the instance
        headers: Request headers (automatically provided by MCP when published)

    Returns:
        Dictionary containing the cancel operation result
    """
    config.validate()
    url = f"{config.pims_url}/v1/instances/{instance_id}/cancel"
    cancel_request = {"comment": comment}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=get_auth_headers(None, "", headers),
                json=cancel_request,
                timeout=30.0
            )

            if response.status_code == 200:
                try:
                    result_data = response.json()
                except Exception:
                    result_data = {"raw_response": response.text}

                return {
                    "status": "success",
                    "message": f"Successfully cancelled instance {instance_id}",
                    "instanceId": instance_id,
                    "comment": comment,
                    "data": result_data
                }
            else:
                error_message = f"Failed to cancel instance {instance_id}. Status: {response.status_code}"
                if response.text:
                    error_message += f", Response: {response.text}"
                raise ValueError(error_message)

    except Exception as e:
        raise ValueError(f"Error cancelling instance: {str(e)}")

@mcp.tool()
async def get_all_instances(
    package_id: str = None,
    package_version: str = None,
    process_key: str = None,
    error_code: str = "",
    *,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Get all UiPath process instances with optional filtering.

    Args:
        package_id: Optional package ID filter
        package_version: Optional package version filter (requires package_id)
        process_key: Optional process key filter
        error_code: Optional error code filter
        headers: Request headers (automatically provided by MCP when published)

    Returns:
        Dictionary containing the instances list
    """
    config.validate()
    url = f"{config.pims_url}/v1/instances"
    
    # Build query parameters with default pagination for MCP usage
    params = {
        "pageSize": 100  # Get more results in one call for MCP
    }
    if package_id:
        params["packageId"] = package_id
    if package_version:
        params["packageVersion"] = package_version
    if process_key:
        params["processKey"] = process_key
    if error_code:
        params["errorCode"] = error_code

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers=get_auth_headers(None, "", headers, content_type=False),
                params=params,
                timeout=30.0
            )

            if response.status_code == 200:
                try:
                    result_data = response.json()
                except Exception:
                    result_data = {"raw_response": response.text}

                return {
                    "status": "success",
                    "message": f"Successfully retrieved instances",
                    "filters": {k: v for k, v in params.items() if k != "pageSize"},
                    "count": len(result_data.get("instances", [])),
                    "data": result_data
                }
            else:
                error_message = f"Failed to get instances. Status: {response.status_code}"
                if response.text:
                    error_message += f", Response: {response.text}"
                raise ValueError(error_message)

    except Exception as e:
        raise ValueError(f"Error getting instances: {str(e)}")

@mcp.tool()
async def migrate_instance(
    instance_id: str,
    new_version: str,
    comment: str = "",
    *,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Migrate a UiPath process instance to a new version.

    Args:
        instance_id: Process instance ID to migrate
        new_version: New version to migrate the instance to
        comment: Optional comment for the migration operation
        headers: Request headers (automatically provided by MCP when published)

    Returns:
        Dictionary containing the migration operation result
    """
    config.validate()
    url = f"{config.pims_url}/v1/instances/{instance_id}/upgrade"
    upgrade_request = {
        "newVersion": new_version,
        "comment": comment
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=get_auth_headers(None, "", headers),
                json=upgrade_request,
                timeout=30.0
            )

            if response.status_code == 200:
                try:
                    result_data = response.json()
                except Exception:
                    result_data = {"raw_response": response.text}

                return {
                    "status": "success",
                    "message": f"Successfully migrated instance {instance_id} to version {new_version}",
                    "instanceId": instance_id,
                    "newVersion": new_version,
                    "comment": comment,
                    "data": result_data
                }
            else:
                error_message = f"Failed to migrate instance {instance_id}. Status: {response.status_code}"
                if response.text:
                    error_message += f", Response: {response.text}"
                raise ValueError(error_message)

    except Exception as e:
        raise ValueError(f"Error migrating instance: {str(e)}")

@mcp.tool()
async def get_incident_summary(
    *,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Get incident summary for the current tenant/folder.

    Returns:
        Dictionary containing the incident summary
    """
    config.validate()
    url = f"{config.pims_url}/v1/incidents/summary"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers=get_auth_headers(None, "", headers, content_type=False),
                timeout=30.0
            )

            if response.status_code == 200:
                try:
                    result_data = response.json()
                except Exception:
                    result_data = {"raw_response": response.text}

                return {
                    "status": "success",
                    "message": "Successfully retrieved incident summary",
                    "count": len(result_data) if isinstance(result_data, list) else 0,
                    "data": result_data
                }
            else:
                error_message = f"Failed to get incident summary. Status: {response.status_code}"
                if response.text:
                    error_message += f", Response: {response.text}"
                raise ValueError(error_message)

    except Exception as e:
        raise ValueError(f"Error getting incident summary: {str(e)}")

@mcp.tool()
async def get_incidents_by_instance_id(
    instance_id: str,
    *,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Get incidents for a specific UiPath process instance.

    Args:
        instance_id: Process instance ID to get incidents for

    Returns:
        Dictionary containing the incidents for the instance
    """
    config.validate()
    url = f"{config.pims_url}/v1/debug-instances/{instance_id}/incidents"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers=get_auth_headers(None, "", headers, content_type=False),
                timeout=30.0
            )

            if response.status_code == 200:
                try:
                    result_data = response.json()
                except Exception:
                    result_data = {"raw_response": response.text}

                return {
                    "status": "success",
                    "message": f"Successfully retrieved incidents for instance {instance_id}",
                    "instanceId": instance_id,
                    "count": len(result_data) if isinstance(result_data, list) else 0,
                    "data": result_data
                }
            else:
                error_message = f"Failed to get incidents for instance {instance_id}. Status: {response.status_code}"
                if response.text:
                    error_message += f", Response: {response.text}"
                raise ValueError(error_message)

    except Exception as e:
        raise ValueError(f"Error getting incidents for instance: {str(e)}")

@mcp.tool()
async def get_spans_by_instance_id(
    instance_id: str,
    *,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Get spans for a specific UiPath process instance.

    Args:
        instance_id: Process instance ID to get spans for
        headers: Request headers (automatically provided by MCP when published)

    Returns:
        Dictionary containing the spans for the instance
    """
    config.validate()
    url = f"{config.pims_url}/v1/spans/{instance_id}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers=get_auth_headers(None, "", headers, content_type=False),
                timeout=30.0
            )

            if response.status_code == 200:
                try:
                    result_data = response.json()
                except Exception:
                    result_data = {"raw_response": response.text}

                return {
                    "status": "success",
                    "message": f"Successfully retrieved spans for instance {instance_id}",
                    "instanceId": instance_id,
                    "count": len(result_data) if isinstance(result_data, list) else 0,
                    "data": result_data
                }
            else:
                error_message = f"Failed to get spans for instance {instance_id}. Status: {response.status_code}"
                if response.text:
                    error_message += f", Response: {response.text}"
                raise ValueError(error_message)

    except Exception as e:
        raise ValueError(f"Error getting spans for instance: {str(e)}")

if __name__ == "__main__":
    print("Starting UiPath Process Instance Management Server...")
    mcp.run()
