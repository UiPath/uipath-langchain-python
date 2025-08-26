import httpx
import logging
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, PrivateAttr

logger = logging.getLogger(__name__)

class OutlookClient(BaseModel):
    """Microsoft Graph API client for Outlook operations"""
    access_token: str
    base_url: str = "https://graph.microsoft.com/v1.0"
    user: str = "me"
    _client: Optional[httpx.AsyncClient] = PrivateAttr(default=None)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the httpx client"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "User-Agent": "EmailAgent/1.0"
                }
            )
        return self._client
    
    async def close(self):
        """Close the httpx client"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def _get(self, path: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make GET request to Microsoft Graph API"""
        client = await self._get_client()
        url = f"{self.base_url}/{self.user}/{path}"
        
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    async def _post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make POST request to Microsoft Graph API"""
        client = await self._get_client()
        url = f"{self.base_url}/{self.user}/{path}"
        
        headers = {"Content-Type": "application/json"}
        response = await client.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()
    
    async def _delete(self, path: str) -> bool:
        """Make DELETE request to Microsoft Graph API"""
        client = await self._get_client()
        url = f"{self.base_url}/{self.user}/{path}"
        
        response = await client.delete(url)
        return response.status_code == 204  # No Content = Success

    async def get_messages(self, max_emails: int, folder: str) -> List[Dict[str, Any]]:
        """Fetch emails from specified folder"""
        try:
            path = f"mailFolders/{folder}/messages?$top={max_emails}"
            response = await self._get(path)

            return response.get('value', [])
        except Exception as e:
            logger.error(f"Error fetching messages: {e}")
            return []
    
    async def get_mail_folders(self, include_subfolders: bool = True) -> Dict[str, str]:
        """Get all mail folders and return name -> ID mapping (excluding deleted items)"""
        try:
            all_folders = {}
            
            response = await self._get("mailFolders", params={
                "$select": "id,displayName,parentFolderId",
                "$top": 250
            })
            
            folders = response.get("value", [])
            
            for folder in folders:
                folder_name = folder["displayName"]
                folder_id = folder["id"]
                
                # Skip deleted items folder
                if folder_name.lower() in ['deleted items', 'trash', 'bin']:
                    logger.info(f"Skipping deleted folder: {folder_name}")
                    continue
                    
                all_folders[folder_name] = folder_id
                
                # Get subfolders for non-deleted folders only
                if include_subfolders:
                    subfolder_response = await self._get(f"mailFolders/{folder_id}/childFolders", params={
                        "$select": "id,displayName",
                        "$top": 100
                    })
                    subfolders = subfolder_response.get("value", [])
                    
                    for subfolder in subfolders:
                        subfolder_name = subfolder["displayName"]
                        subfolder_id = subfolder["id"]
                        
                        if (subfolder_name not in all_folders and 
                            subfolder_name.lower() not in ['deleted items', 'trash', 'bin']):
                            all_folders[subfolder_name] = subfolder_id

                    
            
            logger.info(f"Found: {len(all_folders)} active folders: {list(all_folders.keys())}")
            return all_folders
            
        except Exception as e:
            logger.error(f"Error fetching mail folders: {e}")
            return {}
    
    async def create_folder(self, folder_name: str, parent_folder_id: Optional[str] = None) -> Optional[str]:
        """Create a new mail folder and return its ID"""
        try:
            folder_data = {"displayName": folder_name}
            
            if parent_folder_id:
                path = f"mailFolders/{parent_folder_id}/childFolders"
            else:
                path = "mailFolders"
            
            response = await self._post(path, folder_data)
            
            if 'id' in response:
                folder_id = response['id']
                logger.info(f" Created folder: {folder_name} with ID: {folder_id}")
                return folder_id
            else:
                logger.error(f" Failed to create folder {folder_name}: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating folder {folder_name}: {e}")
            return None
    
    async def get_message_rules(self) -> List[Dict[str, Any]]:
        """Get all existing message rules"""
        try:
            response = await self._get("mailFolders/inbox/messageRules")
            existing_rules = response.get("value", [])
            logger.info(f"Found {len(existing_rules)} existing rules")
            return existing_rules
        except Exception as e:
            logger.debug(f"Error fetching existing rules: {e}")
            return []
    
    async def create_message_rule(self, rule_data: Dict[str, Any]) -> Optional[str]:
        """Create a new message rule and return its ID"""
        try:
            response = await self._post("mailFolders/inbox/messageRules", rule_data)
            
            if 'id' in response:
                rule_id = response['id']
                rule_name = rule_data.get('displayName', 'Unknown')
                logger.info(f" Created rule: {rule_name} with ID: {rule_id}")
                return rule_id
            else:
                logger.error(f" Failed to create rule: {response}")
                return None
                
        except Exception as e:
            logger.error(f" Error creating rule: {e}")
            return None
    
    async def move_message(self, message_id: str, destination_folder_id: str) -> bool:
        """Move a message to a specific folder"""
        try:
            move_data = {"destinationId": destination_folder_id}
            await self._post(f"messages/{message_id}/move", move_data)
            logger.info(f" Moved message {message_id} to folder {destination_folder_id}")
            return True
        except Exception as e:
            logger.error(f"Error moving message {message_id}: {e}")
            return False
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
