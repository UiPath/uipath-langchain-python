#!/usr/bin/env python3
"""
UiPath Task Management MCP Server

This MCP server provides tools for managing UiPath HITL (Human-in-the-Loop) tasks,
including task assignment, completion, and user management.
"""

import os
import json
import asyncio
import ssl
from typing import Dict, Any, List, Optional
from urllib.parse import urlencode
import aiohttp
from mcp.server import FastMCP
from mcp.server.models import InitializationOptions
import pydantic

# Initialize MCP Server
mcp = FastMCP("UiPath Task Management Server")

# SSL context to bypass certificate verification
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

class TaskAssignmentPayload(pydantic.BaseModel):
    """Payload for task assignment."""
    TaskId: int
    UserId: Optional[int] = None
    AssignmentCriteria: str
    AssigneeNamesOrEmails: Optional[List[str]] = None

class TaskCompletionPayload(pydantic.BaseModel):
    """Payload for task completion."""
    taskId: int
    data: Dict[str, Any]
    action: str

@mcp.tool()
async def get_tasks(
    top: int = 10,
    *,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Get tasks.
    
    Args:
        top: Maximum number of tasks to return (default: 10)
        headers: Request headers (automatically provided by MCP)
    
    Returns:
        Dictionary containing the tasks
    """
    try:
        # Get environment variables
        base_url = os.getenv("UIPATH_BASE_URL", "alpha.uipath.com")
        org_id = os.getenv("UIPATH_ORG_ID", "")
        tenant_id = os.getenv("UIPATH_TENANT", "DefaultTenant")
        auth_token = os.getenv("UIPATH_ACCESS_TOKEN", "")
        
        if not auth_token:
            raise ValueError("UIPATH_ACCESS_TOKEN not found in environment variables")
        
        url = f"https://{base_url}/{org_id}/{tenant_id}/orchestrator_/odata/Tasks/UiPath.Server.Configuration.OData.GetTasksAcrossFoldersForAdmin()"
        
        query_params = {
            "$top": top,
            "$orderby": "CreationTime desc"
        }
        
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
            async with session.get(
                f"{url}?{urlencode(query_params)}",
                headers={
                    'Authorization': f"Bearer {auth_token}",
                    'serviceurl': f"https://{base_url}/{org_id}/{tenant_id}/orchestrator_"
                }
            ) as response:
                if response.status != 200:
                    raise Exception(f"HTTP error! Status: {response.status}")
                
                data = await response.json()
                return {"success": True, "data": data}
                
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
async def add_note_to_task(
    task_id: int,
    note: str,
    *,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Add a note to a specific task.
    
    Args:
        task_id: The ID of the task
        note: The note text to add
        headers: Request headers (automatically provided by MCP)
    
    Returns:
        Dictionary containing the result of adding the note
    """
    try:
        base_url = os.getenv("UIPATH_BASE_URL", "alpha.uipath.com")
        org_id = os.getenv("UIPATH_ORG_ID", "")
        tenant_id = os.getenv("UIPATH_TENANT", "DefaultTenant")
        auth_token = os.getenv("UIPATH_ACCESS_TOKEN", "")
        
        if not auth_token:
            raise ValueError("UIPATH_ACCESS_TOKEN not found in environment variables")
        
        url = f"https://{base_url}/{org_id}/{tenant_id}/orchestrator_/odata/TaskNotes/UiPath.Server.Configuration.OData.CreateTaskNote"
        
        payload = {
            "taskId": task_id,
            "text": note
        }
        
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
            async with session.post(
                url,
                headers={
                    'accept': 'application/json',
                    'Content-Type': 'application/json',
                    'Authorization': f"Bearer {auth_token}",
                    'serviceurl': f"https://{base_url}/{org_id}/{tenant_id}/orchestrator_"
                },
                json=payload
            ) as response:
                if response.status != 200:
                    raise Exception(f"HTTP error! Status: {response.status}")
                
                data = await response.json()
                return {"success": True, "data": data}
                
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
async def assign_task(
    task_id: int,
    assignment_criteria: str,
    assignees: str,
    reassign: bool = False,
    *,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Assign a task to users.
    
    Args:
        task_id: The ID of the task to assign
        assignment_criteria: "SingleUser" or other criteria
        assignees: User ID (for SingleUser) or comma-separated list of emails/names
        reassign: Whether this is a reassignment (default: False)
        headers: Request headers (automatically provided by MCP)
    
    Returns:
        Dictionary containing the assignment result
    """
    try:
        base_url = os.getenv("UIPATH_BASE_URL", "alpha.uipath.com")
        org_id = os.getenv("UIPATH_ORG_ID", "")
        tenant_id = os.getenv("UIPATH_TENANT", "DefaultTenant")
        auth_token = os.getenv("UIPATH_ACCESS_TOKEN", "")
        
        if not auth_token:
            raise ValueError("UIPATH_ACCESS_TOKEN not found in environment variables")
        
        if not reassign:
            url = f"https://{base_url}/{org_id}/{tenant_id}/orchestrator_/odata/Tasks/UiPath.Server.Configuration.OData.AssignTasks"
        else:
            url = f"https://{base_url}/{org_id}/{tenant_id}/orchestrator_/odata/Tasks/UiPath.Server.Configuration.OData.ReassignTasks"
        
        # Parse assignees
        if assignment_criteria == "SingleUser":
            try:
                user_id = int(assignees)
                task_assignment = {
                    "TaskId": task_id,
                    "UserId": user_id,
                    "AssignmentCriteria": "SingleUser"
                }
            except ValueError:
                raise ValueError("For SingleUser assignment, assignees must be a numeric user ID")
        else:
            assignee_list = [email.strip() for email in assignees.split(",")]
            task_assignment = {
                "TaskId": task_id,
                "AssignmentCriteria": assignment_criteria,
                "AssigneeNamesOrEmails": assignee_list
            }
        
        payload = {
            "taskAssignments": [task_assignment]
        }
        
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
            async with session.post(
                url,
                headers={
                    'accept': 'application/json',
                    'Content-Type': 'application/json',
                    'Authorization': f"Bearer {auth_token}",
                    'serviceurl': f"https://{base_url}/{org_id}/{tenant_id}/orchestrator_"
                },
                json=payload
            ) as response:
                if response.status != 200:
                    raise Exception(f"HTTP error! Status: {response.status}")
                
                data = await response.json()
                return {"success": True, "data": data}
                
    except Exception as e:
        return {"success": False, "error": str(e)}

# @mcp.tool()
# async def complete_task(
#     task_id: int,
#     action: str,
#     app_type: str = "GenericTask",
#     task_data: Optional[str] = None,
#     *,
#     headers: Optional[Dict[str, str]] = None
# ) -> Dict[str, Any]:
#     """Complete a task with specified action and data.
    
#     Args:
#         task_id: The ID of the task to complete
#         action: The action to take (e.g., "Approve", "Reject")
#         app_type: Type of task ("FormTask", "AppTask", or "GenericTask")
#         task_data: JSON string containing task data (optional)
#         headers: Request headers (automatically provided by MCP)
    
#     Returns:
#         Dictionary containing the completion result
#     """
#     try:
#         base_url = os.getenv("UIPATH_BASE_URL", "alpha.uipath.com")
#         org_id = os.getenv("UIPATH_ORG_ID", "")
#         tenant_id = os.getenv("UIPATH_TENANT", "DefaultTenant")
#         auth_token = os.getenv("UIPATH_ACCESS_TOKEN", "")
        
#         if not auth_token:
#             raise ValueError("UIPATH_ACCESS_TOKEN not found in environment variables")
        
#         # Determine URL based on app type
#         if app_type == "FormTask":
#             url = f"https://{base_url}/{org_id}/{tenant_id}/bupproxyservice_/orchestrator/forms/TaskForms/CompleteTask"
#         elif app_type == "AppTask":
#             url = f"https://{base_url}/{org_id}/{tenant_id}/bupproxyservice_/orchestrator/tasks/AppTasks/CompleteAppTask"
#         else:
#             url = f"https://{base_url}/{org_id}/{tenant_id}/bupproxyservice_/orchestrator/Tasks/GenericTasks/CompleteTask"
        
#         # Parse task data
#         data_dict = {}
#         if task_data:
#             try:
#                 data_dict = json.loads(task_data)
#             except json.JSONDecodeError:
#                 raise ValueError("Invalid JSON in task_data")
        
#         payload = {
#             "taskId": task_id,
#             "data": data_dict,
#             "action": action
#         }
        
#         async with aiohttp.ClientSession() as session:
#             async with session.post(
#                 url,
#                 headers={
#                     'accept': 'application/json',
#                     'Content-Type': 'application/json',
#                     'Authorization': f"Bearer {auth_token}",
#                     'serviceurl': f"https://{base_url}/{org_id}/{tenant_id}/orchestrator_"
#                 },
#                 json=payload
#             ) as response:
#                 if response.status != 200:
#                     raise Exception(f"HTTP error! Status: {response.status}")
                
#                 data = await response.json()
#                 return {"success": True, "data": data}
                
#     except Exception as e:
#         return {"success": False, "error": str(e)}

@mcp.tool()
async def get_unassigned_tasks(
    top: int = 10,
    *,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Get unassigned tasks.
    
    Args:
        top: Maximum number of tasks to return (default: 10)
        headers: Request headers (automatically provided by MCP)
    
    Returns:
        Dictionary containing unassigned tasks
    """
    try:
        base_url = os.getenv("UIPATH_BASE_URL", "alpha.uipath.com")
        org_id = os.getenv("UIPATH_ORG_ID", "")
        tenant_id = os.getenv("UIPATH_TENANT", "DefaultTenant")
        auth_token = os.getenv("UIPATH_ACCESS_TOKEN", "")
        
        if not auth_token:
            raise ValueError("UIPATH_ACCESS_TOKEN not found in environment variables")
        
        url = f"https://{base_url}/{org_id}/{tenant_id}/orchestrator_/odata/Tasks/UiPath.Server.Configuration.OData.GetTasksAcrossFoldersForAdmin()"
        
        query_params = {
            "$top": top,
            "$orderby": "CreationTime desc"
        }
        
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
            async with session.get(
                f"{url}?{urlencode(query_params)}",
                headers={
                    'Authorization': f"Bearer {auth_token}",
                    'serviceurl': f"https://{base_url}/{org_id}/{tenant_id}/orchestrator_"
                }
            ) as response:
                if response.status != 200:
                    raise Exception(f"HTTP error! Status: {response.status}")
                
                data = await response.json()
                return {"success": True, "data": data}
                
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
async def get_task_data(
    task_id: int,
    *,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Get data for a specific task.
    
    Args:
        task_id: The ID of the task
        headers: Request headers (automatically provided by MCP)
    
    Returns:
        Dictionary containing the task data
    """
    try:
        base_url = os.getenv("UIPATH_BASE_URL", "alpha.uipath.com")
        org_id = os.getenv("UIPATH_ORG_ID", "")
        tenant_id = os.getenv("UIPATH_TENANT", "DefaultTenant")
        auth_token = os.getenv("UIPATH_ACCESS_TOKEN", "")
        
        if not auth_token:
            raise ValueError("UIPATH_ACCESS_TOKEN not found in environment variables")
        
        url = f"https://{base_url}/{org_id}/{tenant_id}/orchestrator_/odata/Tasks({task_id})"
        
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
            async with session.get(
                url,
                headers={
                    'Authorization': f"Bearer {auth_token}",
                    'serviceurl': f"https://{base_url}/{org_id}/{tenant_id}/orchestrator_"
                }
            ) as response:
                if response.status != 200:
                    raise Exception(f"HTTP error! Status: {response.status}")
                
                data = await response.json()
                return {"success": True, "data": data}
                
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run()) 