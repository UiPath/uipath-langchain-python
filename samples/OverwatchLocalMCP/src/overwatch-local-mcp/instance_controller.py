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
async def get_spans_by_process_key(
    process_key: str,
    max_instances: int = 3,  # Reduced default to prevent timeouts
    *,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Get raw spans for multiple UiPath process instances from the same process key.

    This tool retrieves execution spans from multiple instances of the same process and returns them as a simple list.
    Each span includes the instance ID and process key for context.

    Args:
        process_key: Process key to get instances for
        max_instances: Maximum number of instances to retrieve spans for (default: 10)
        headers: Request headers (automatically provided by MCP when published)

    Returns:
        Dictionary containing:
        - status: Success/error status
        - message: Description of the operation
        - processKey: The process key that was queried
        - instanceCount: Number of instances processed
        - totalSpansCount: Total number of spans retrieved
        - maxInstancesRequested: Maximum instances requested
        - data.spans: List of all raw spans with instanceId and processKey added to each span
    """
    config.validate()
    
    # Use the existing get_all_instances function
    instances_result = await get_all_instances(
        process_key=process_key,
        headers=headers
    )
    
    if instances_result.get("status") != "success":
        return {
            "status": "error",
            "message": f"Failed to get instances for process key {process_key}",
            "error": instances_result.get("message", "Unknown error")
        }

    instances_data = instances_result.get("data", {})
    instances = instances_data.get("instances", [])
    
    # Debug: Print instance information
    print(f"=== DEBUG: get_spans_by_process_key ===")
    print(f"Process Key: {process_key}")
    print(f"Max Instances Requested: {max_instances}")
    print(f"Total Instances Found: {len(instances)}")
    
    if instances:
        print(f"Instance IDs found:")
        for i, instance in enumerate(instances[:max_instances]):
            # Try both "id" and "instanceId" fields
            instance_id = instance.get("id") or instance.get("instanceId", "NO_ID")
            status = instance.get("status", "UNKNOWN")
            latest_run_status = instance.get("latestRunStatus", "UNKNOWN")
            print(f"  {i+1}. Instance ID: {instance_id} (Status: {status}, Latest Run: {latest_run_status})")
    else:
        print("No instances found for this process key")
    print("=== END DEBUG ===")
    
    if not instances:
        return {
            "status": "success",
            "message": f"No instances found for process key {process_key}",
            "processKey": process_key,
            "instanceCount": 0,
            "totalSpansCount": 0,
            "data": {
                "instances": [],
                "combinedSpans": [],
                "spanAnalysis": {
                    "spanTypes": {},
                    "statusDistribution": {},
                    "operationTypes": {}
                }
            }
        }

    # Get spans for each instance
    all_spans = []
    
    async with httpx.AsyncClient(timeout=20.0) as client:
        for i, instance in enumerate(instances[:max_instances]):  # Limit to max_instances
            print(f"  Processing instance {i+1}/{min(len(instances), max_instances)}...")
            # Try both "id" and "instanceId" fields
            instance_id = instance.get("id") or instance.get("instanceId")
            if not instance_id:
                print(f"  Skipping instance {i+1}: No instance ID found (tried 'id' and 'instanceId')")
                print(f"    Available fields: {list(instance.keys())}")
                continue
            
            print(f"  Processing instance {i+1}/{min(len(instances), max_instances)}: {instance_id}")
            
            # Get spans for this instance
            spans_url = f"{config.pims_url}/v1/spans/{instance_id}"
            print(f"    Requesting spans from: {spans_url}")
            
            try:
                spans_response = await client.get(
                    spans_url,
                    headers=get_auth_headers(None, "", headers, content_type=False),
                    timeout=15.0  # Reduced timeout
                )
            except Exception as e:
                print(f"    Error getting spans for instance {instance_id}: {str(e)}")
                continue
            
            instance_spans = []
            print(f"    Response status: {spans_response.status_code}")
            
            if spans_response.status_code == 200:
                try:
                    spans_data = spans_response.json()
                    if isinstance(spans_data, list):
                        instance_spans = spans_data
                        print(f"    Successfully retrieved {len(instance_spans)} spans")
                        
                        # Add instance context to each span
                        for span in instance_spans:
                            span["instanceId"] = instance_id
                            span["processKey"] = process_key
                        
                        all_spans.extend(instance_spans)
                    else:
                        instance_spans = {"raw_response": spans_response.text}
                        print(f"    Unexpected response format: {type(spans_data)}")
                except Exception as e:
                    instance_spans = {"raw_response": spans_response.text}
                    print(f"    Error parsing response: {str(e)}")
            else:
                print(f"    Failed to get spans: HTTP {spans_response.status_code}")
                print(f"    Response text: {spans_response.text[:200]}...")
            
            # Add instance context to each span and collect all spans
            if isinstance(instance_spans, list):
                for span in instance_spans:
                    span["instanceId"] = instance_id
                    span["processKey"] = process_key
                all_spans.extend(instance_spans)

    return {
        "status": "success",
        "message": f"Successfully retrieved spans for {len(instances[:max_instances])} instances of process key {process_key}",
        "processKey": process_key,
        "instanceCount": len(instances[:max_instances]),
        "totalSpansCount": len(all_spans),
        "maxInstancesRequested": max_instances,
        "data": {
            "spans": all_spans
        }
    }


def organize_spans_hierarchically(spans: list) -> dict:
    """Organize spans into a hierarchical structure based on parent-child relationships."""
    if not spans:
        return {}
    
    # Create a map of spans by ID
    spans_by_id = {span["id"]: span for span in spans}
    
    # Find root spans (no parent or parent not in the list)
    root_spans = []
    child_spans = {}
    
    for span in spans:
        parent_id = span.get("parentId")
        if not parent_id or parent_id not in spans_by_id:
            root_spans.append(span)
        else:
            if parent_id not in child_spans:
                child_spans[parent_id] = []
            child_spans[parent_id].append(span)
    
    # Build hierarchical structure
    def build_tree(span):
        span_id = span["id"]
        children = child_spans.get(span_id, [])
        return {
            "span": span,
            "children": [build_tree(child) for child in children]
        }
    
    return {
        "rootSpans": [build_tree(span) for span in root_spans],
        "spanTypes": categorize_spans_by_type(spans),
        "timeline": create_timeline(spans)
    }


def categorize_spans_by_type(spans: list) -> dict:
    """Categorize spans by their type."""
    categories = {}
    for span in spans:
        span_type = span.get("spanType", "Unknown")
        if span_type not in categories:
            categories[span_type] = []
        categories[span_type].append(span)
    return categories


def create_timeline(spans: list) -> list:
    """Create a chronological timeline of spans."""
    timeline_spans = []
    for span in spans:
        timeline_spans.append({
            "id": span["id"],
            "name": span["name"],
            "startTime": span.get("startTime"),
            "endTime": span.get("endTime"),
            "spanType": span.get("spanType"),
            "status": span.get("status"),
            "attributes": span.get("attributes", {})
        })
    
    # Sort by start time
    timeline_spans.sort(key=lambda x: x.get("startTime", ""))
    return timeline_spans


def create_span_summary(spans: list) -> dict:
    """Create a comprehensive summary of spans for an instance.
    
    Args:
        spans: List of span objects from UiPath PIMS API
        
    Returns:
        Dictionary containing detailed span analysis and metrics
    """
    if not spans:
        return {}
    
    summary = {
        # Basic counts
        "totalSpans": len(spans),
        "spanTypes": {},
        "statusCounts": {},
        
        # Timing and performance
        "duration": None,
        "executionMetrics": {
            "totalExecutionTime": None,
            "averageSpanDuration": None,
            "longestSpan": None,
            "shortestSpan": None
        },
        
        # Process health indicators
        "healthIndicators": {
            "hasFailures": False,
            "hasRetries": False,
            "hasActionCenterTasks": False,
            "hasSystemErrors": False,
            "hasUserInteractions": False,
            "failureRate": 0.0
        },
        
        # Business logic insights
        "businessMetrics": {
            "userInteractions": [],
            "externalSystemCalls": [],
            "dataProcessingSteps": [],
            "decisionPoints": []
        },
        
        # Error analysis
        "errorAnalysis": {
            "failedElements": [],
            "errorTypes": {},
            "retryAttempts": [],
            "systemErrors": []
        },
        
        # Performance bottlenecks
        "performanceBottlenecks": {
            "slowestElements": [],
            "longestOperations": [],
            "resourceIntensiveSpans": []
        },
        
        # Process flow analysis
        "flowAnalysis": {
            "startElements": [],
            "endElements": [],
            "decisionBranches": [],
            "parallelExecutions": []
        }
    }
    
    # Find start and end times
    start_times = [span.get("startTime") for span in spans if span.get("startTime")]
    end_times = [span.get("endTime") for span in spans if span.get("endTime")]
    
    if start_times and end_times:
        earliest_start = min(start_times)
        latest_end = max(end_times)
        summary["duration"] = {
            "start": earliest_start,
            "end": latest_end
        }
    
    # Analyze spans for detailed insights
    span_durations = []
    failed_spans = []
    
    for span in spans:
        # Basic counting
        span_type = span.get("spanType", "Unknown")
        summary["spanTypes"][span_type] = summary["spanTypes"].get(span_type, 0) + 1
        
        status = span.get("status")
        if status is not None:
            status_key = str(status)
            summary["statusCounts"][status_key] = summary["statusCounts"].get(status_key, 0) + 1
        
        # Calculate span duration
        start_time = span.get("startTime")
        end_time = span.get("endTime")
        if start_time and end_time:
            try:
                from datetime import datetime
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                duration_seconds = (end_dt - start_dt).total_seconds()
                span_durations.append(duration_seconds)
                
                # Track longest operations
                if duration_seconds > 5:  # Spans longer than 5 seconds
                    summary["performanceBottlenecks"]["longestOperations"].append({
                        "spanId": span.get("id"),
                        "name": span.get("name"),
                        "duration": duration_seconds,
                        "spanType": span_type,
                        "elementId": span.get("attributes", {}).get("elementId")
                    })
            except Exception:
                pass
        
        # Analyze attributes for business insights
        attributes = span.get("attributes", {})
        
        # Check for failures
        if span.get("status") == 2:  # Failed status
            summary["healthIndicators"]["hasFailures"] = True
            failed_spans.append({
                "spanId": span.get("id"),
                "name": span.get("name"),
                "elementId": attributes.get("elementId"),
                "spanType": span_type,
                "timestamp": start_time
            })
            
            # Categorize failures
            error_type = attributes.get("errorType", "Unknown")
            summary["errorAnalysis"]["errorTypes"][error_type] = summary["errorAnalysis"]["errorTypes"].get(error_type, 0) + 1
        
        # Check for retries
        if attributes.get("operationType") == "Retry":
            summary["healthIndicators"]["hasRetries"] = True
            summary["errorAnalysis"]["retryAttempts"].append({
                "spanId": span.get("id"),
                "timestamp": start_time,
                "comment": attributes.get("comment", ""),
                "elementId": attributes.get("elementId")
            })
        
        # Check for Action Center tasks
        if attributes.get("actionCenterTaskLink"):
            summary["healthIndicators"]["hasActionCenterTasks"] = True
            summary["businessMetrics"]["userInteractions"].append({
                "spanId": span.get("id"),
                "taskLink": attributes.get("actionCenterTaskLink"),
                "timestamp": start_time,
                "elementId": attributes.get("elementId")
            })
        
        # Check for system errors
        if attributes.get("errorType") == "System":
            summary["healthIndicators"]["hasSystemErrors"] = True
            summary["errorAnalysis"]["systemErrors"].append({
                "spanId": span.get("id"),
                "name": span.get("name"),
                "timestamp": start_time,
                "elementId": attributes.get("elementId")
            })
        
        # Check for user interactions
        if attributes.get("userInteraction") or attributes.get("actionCenterTaskLink"):
            summary["healthIndicators"]["hasUserInteractions"] = True
        
        # Track external system calls
        if attributes.get("externalSystemCall") or "http" in span.get("name", "").lower():
            summary["businessMetrics"]["externalSystemCalls"].append({
                "spanId": span.get("id"),
                "name": span.get("name"),
                "timestamp": start_time,
                "elementId": attributes.get("elementId")
            })
        
        # Track data processing steps
        if "data" in span.get("name", "").lower() or "process" in span.get("name", "").lower():
            summary["businessMetrics"]["dataProcessingSteps"].append({
                "spanId": span.get("id"),
                "name": span.get("name"),
                "timestamp": start_time,
                "elementId": attributes.get("elementId")
            })
        
        # Track decision points
        if "decision" in span.get("name", "").lower() or "if" in span.get("name", "").lower():
            summary["businessMetrics"]["decisionPoints"].append({
                "spanId": span.get("id"),
                "name": span.get("name"),
                "timestamp": start_time,
                "elementId": attributes.get("elementId")
            })
        
        # Track start/end elements
        if "start" in span.get("name", "").lower():
            summary["flowAnalysis"]["startElements"].append({
                "spanId": span.get("id"),
                "name": span.get("name"),
                "timestamp": start_time
            })
        
        if "end" in span.get("name", "").lower():
            summary["flowAnalysis"]["endElements"].append({
                "spanId": span.get("id"),
                "name": span.get("name"),
                "timestamp": start_time
            })
    
    # Calculate execution metrics
    if span_durations:
        summary["executionMetrics"]["totalExecutionTime"] = sum(span_durations)
        summary["executionMetrics"]["averageSpanDuration"] = sum(span_durations) / len(span_durations)
        summary["executionMetrics"]["longestSpan"] = max(span_durations)
        summary["executionMetrics"]["shortestSpan"] = min(span_durations)
    
    # Calculate failure rate
    if summary["totalSpans"] > 0:
        summary["healthIndicators"]["failureRate"] = len(failed_spans) / summary["totalSpans"]
    
    # Sort performance bottlenecks by duration
    summary["performanceBottlenecks"]["longestOperations"].sort(key=lambda x: x["duration"], reverse=True)
    summary["performanceBottlenecks"]["longestOperations"] = summary["performanceBottlenecks"]["longestOperations"][:5]  # Top 5
    
    # Add failed elements to error analysis
    summary["errorAnalysis"]["failedElements"] = failed_spans
    
    return summary


def analyze_spans_for_patterns(spans: list, analysis: dict, instance_id: str):
    """Analyze spans for common patterns and failures across instances.
    
    Args:
        spans: List of span objects from UiPath PIMS API
        analysis: Dictionary to accumulate analysis across instances
        instance_id: ID of the current instance being analyzed
    """
    # Initialize analysis structure if not present
    if "spanTypes" not in analysis:
        analysis["spanTypes"] = {}
    if "statusDistribution" not in analysis:
        analysis["statusDistribution"] = {}
    if "operationTypes" not in analysis:
        analysis["operationTypes"] = {}
    if "elementFailures" not in analysis:
        analysis["elementFailures"] = []
    if "retryPatterns" not in analysis:
        analysis["retryPatterns"] = []
    if "performancePatterns" not in analysis:
        analysis["performancePatterns"] = {
            "slowElements": {},
            "commonBottlenecks": [],
            "executionTimes": []
        }
    if "businessPatterns" not in analysis:
        analysis["businessPatterns"] = {
            "userInteractionPoints": [],
            "externalSystemDependencies": [],
            "dataProcessingSteps": [],
            "decisionPoints": []
        }
    if "errorPatterns" not in analysis:
        analysis["errorPatterns"] = {
            "commonErrorTypes": {},
            "failureHotspots": {},
            "systemErrorFrequency": 0
        }
    
    instance_analysis = {
        "instanceId": instance_id,
        "totalSpans": len(spans),
        "executionTime": None,
        "failureCount": 0,
        "retryCount": 0,
        "userInteractions": 0,
        "externalCalls": 0
    }
    
    # Calculate instance execution time
    start_times = [span.get("startTime") for span in spans if span.get("startTime")]
    end_times = [span.get("endTime") for span in spans if span.get("endTime")]
    
    if start_times and end_times:
        try:
            from datetime import datetime
            earliest_start = min(start_times)
            latest_end = max(end_times)
            start_dt = datetime.fromisoformat(earliest_start.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(latest_end.replace('Z', '+00:00'))
            instance_analysis["executionTime"] = (end_dt - start_dt).total_seconds()
            analysis["performancePatterns"]["executionTimes"].append(instance_analysis["executionTime"])
        except Exception:
            pass
    
    for span in spans:
        # Basic counting
        span_type = span.get("spanType", "Unknown")
        analysis["spanTypes"][span_type] = analysis["spanTypes"].get(span_type, 0) + 1
        
        status = span.get("status")
        if status is not None:
            status_key = str(status)
            analysis["statusDistribution"][status_key] = analysis["statusDistribution"].get(status_key, 0) + 1
        
        # Track operation types
        attributes = span.get("attributes", {})
        operation_type = attributes.get("operationType")
        if operation_type:
            analysis["operationTypes"][operation_type] = analysis["operationTypes"].get(operation_type, 0) + 1
        
        # Track element failures with more context
        if span.get("status") == 2:  # Failed
            instance_analysis["failureCount"] += 1
            element_id = attributes.get("elementId")
            element_name = span.get("name")
            
            if element_id:
                # Track failure hotspots
                failure_key = f"{element_id}:{element_name}"
                analysis["errorPatterns"]["failureHotspots"][failure_key] = analysis["errorPatterns"]["failureHotspots"].get(failure_key, 0) + 1
                
                analysis["elementFailures"].append({
                    "instanceId": instance_id,
                    "elementId": element_id,
                    "elementName": element_name,
                    "spanType": span_type,
                    "errorType": attributes.get("errorType", "Unknown"),
                    "timestamp": span.get("startTime")
                })
        
        # Track retry patterns with more detail
        if attributes.get("operationType") == "Retry":
            instance_analysis["retryCount"] += 1
            analysis["retryPatterns"].append({
                "instanceId": instance_id,
                "timestamp": span.get("startTime"),
                "comment": attributes.get("comment", ""),
                "elementId": attributes.get("elementId"),
                "elementName": span.get("name"),
                "retryCount": attributes.get("retryCount", 1)
            })
        
        # Track performance patterns
        start_time = span.get("startTime")
        end_time = span.get("endTime")
        if start_time and end_time:
            try:
                from datetime import datetime
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                duration = (end_dt - start_dt).total_seconds()
                
                # Track slow elements
                element_id = attributes.get("elementId")
                if element_id and duration > 2:  # Elements taking more than 2 seconds
                    if element_id not in analysis["performancePatterns"]["slowElements"]:
                        analysis["performancePatterns"]["slowElements"][element_id] = {
                            "elementName": span.get("name"),
                            "totalDuration": 0,
                            "count": 0,
                            "averageDuration": 0,
                            "maxDuration": 0
                        }
                    
                    slow_element = analysis["performancePatterns"]["slowElements"][element_id]
                    slow_element["totalDuration"] += duration
                    slow_element["count"] += 1
                    slow_element["maxDuration"] = max(slow_element["maxDuration"], duration)
                    slow_element["averageDuration"] = slow_element["totalDuration"] / slow_element["count"]
            except Exception:
                pass
        
        # Track business patterns
        if attributes.get("actionCenterTaskLink"):
            instance_analysis["userInteractions"] += 1
            analysis["businessPatterns"]["userInteractionPoints"].append({
                "instanceId": instance_id,
                "elementId": attributes.get("elementId"),
                "elementName": span.get("name"),
                "taskLink": attributes.get("actionCenterTaskLink"),
                "timestamp": start_time
            })
        
        # Track external system calls
        if attributes.get("externalSystemCall") or "http" in span.get("name", "").lower():
            instance_analysis["externalCalls"] += 1
            analysis["businessPatterns"]["externalSystemDependencies"].append({
                "instanceId": instance_id,
                "elementId": attributes.get("elementId"),
                "elementName": span.get("name"),
                "systemType": attributes.get("systemType", "Unknown"),
                "timestamp": start_time
            })
        
        # Track data processing steps
        if "data" in span.get("name", "").lower() or "process" in span.get("name", "").lower():
            analysis["businessPatterns"]["dataProcessingSteps"].append({
                "instanceId": instance_id,
                "elementId": attributes.get("elementId"),
                "elementName": span.get("name"),
                "timestamp": start_time
            })
        
        # Track decision points
        if "decision" in span.get("name", "").lower() or "if" in span.get("name", "").lower():
            analysis["businessPatterns"]["decisionPoints"].append({
                "instanceId": instance_id,
                "elementId": attributes.get("elementId"),
                "elementName": span.get("name"),
                "timestamp": start_time
            })
        
        # Track system errors
        if attributes.get("errorType") == "System":
            analysis["errorPatterns"]["systemErrorFrequency"] += 1
    
    # Add instance analysis to performance patterns
    analysis["performancePatterns"]["instanceAnalyses"] = analysis["performancePatterns"].get("instanceAnalyses", [])
    analysis["performancePatterns"]["instanceAnalyses"].append(instance_analysis)
    
    # Calculate common bottlenecks (elements that appear in multiple slow instances)
    slow_elements = analysis["performancePatterns"]["slowElements"]
    for element_id, element_data in slow_elements.items():
        if element_data["count"] > 1:  # Element appears in multiple instances
            analysis["performancePatterns"]["commonBottlenecks"].append({
                "elementId": element_id,
                "elementName": element_data["elementName"],
                "occurrenceCount": element_data["count"],
                "averageDuration": element_data["averageDuration"],
                "maxDuration": element_data["maxDuration"]
            })
    
    # Sort common bottlenecks by average duration
    analysis["performancePatterns"]["commonBottlenecks"].sort(key=lambda x: x["averageDuration"], reverse=True)
    
    # Calculate error type frequencies
    for failure in analysis["elementFailures"]:
        error_type = failure.get("errorType", "Unknown")
        analysis["errorPatterns"]["commonErrorTypes"][error_type] = analysis["errorPatterns"]["commonErrorTypes"].get(error_type, 0) + 1


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
