# mypy: disable-error-code="syntax"
import os
from typing import Any, Optional

import httpx
from pydantic import Field
from pydantic_settings import BaseSettings


class UiPathCachedPathsSettings(BaseSettings):
    cached_completion_db: str = Field(
        default=os.path.join(
            os.path.dirname(__file__), "tests", "tests_uipath_cache.db"
        ),
        alias="CACHED_COMPLETION_DB",
    )
    cached_embeddings_dir: str = Field(
        default=os.path.join(os.path.dirname(__file__), "tests", "cached_embeddings"),
        alias="CACHED_EMBEDDINGS_DIR",
    )
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }


uipath_cached_paths_settings = UiPathCachedPathsSettings()
uipath_token_header: Optional[str] = None


class UiPathClientFactorySettings(BaseSettings):
    base_url: str = Field(default="", alias="UIPATH_BASE_URL")
    client_id: str = Field(default="", alias="UIPATH_CLIENT_ID")
    client_secret: str = Field(default="", alias="UIPATH_CLIENT_SECRET")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }


class UiPathClientSettings(BaseSettings):
    access_token: str = Field(default="", alias="UIPATH_ACCESS_TOKEN")
    base_url: str = Field(default="", alias="UIPATH_BASE_URL")
    org_id: str = Field(default="", alias="UIPATH_ORGANIZATION_ID")
    tenant_id: str = Field(default="", alias="UIPATH_TENANT_ID")
    requesting_product: str = Field(
        default="uipath-python-sdk", alias="UIPATH_REQUESTING_PRODUCT"
    )
    requesting_feature: str = Field(
        default="langgraph-agent", alias="UIPATH_REQUESTING_FEATURE"
    )
    timeout_seconds: str = Field(default="120", alias="UIPATH_TIMEOUT_SECONDS")
    action_name: str = Field(default="DefaultActionName", alias="UIPATH_ACTION_NAME")
    action_id: str = Field(default="DefaultActionId", alias="UIPATH_ACTION_ID")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }


def get_uipath_token_header(
    settings: Any = None,
) -> str:
    global uipath_token_header
    
    # First, try to get token from environment variable
    env_token = os.getenv("UIPATH_ACCESS_TOKEN") or os.getenv("UIPATH_SERVICE_TOKEN")
    if env_token:
        return env_token
    
    # If no environment token, try to get from client credentials
    if not uipath_token_header:
        settings = settings or UiPathClientFactorySettings()
        if settings.base_url and settings.client_id and settings.client_secret:
            try:
                url_get_token = f"{settings.base_url}/identity_/connect/token"
                token_credentials = dict(
                    client_id=settings.client_id,
                    client_secret=settings.client_secret,
                    grant_type="client_credentials",
                )
                with httpx.Client() as client:
                    res = client.post(url_get_token, data=token_credentials)
                    res_json = res.json()
                    uipath_token_header = res_json.get("access_token")
            except Exception:
                # If client credentials fail, return empty string
                pass

    return uipath_token_header or ""


async def get_token_header_async(
    settings: Any = None,
) -> str:
    global uipath_token_header
    
    # First, try to get token from environment variable
    env_token = os.getenv("UIPATH_ACCESS_TOKEN") or os.getenv("UIPATH_SERVICE_TOKEN")
    if env_token:
        return env_token
    
    # If no environment token, try to get from client credentials
    if not uipath_token_header:
        settings = settings or UiPathClientFactorySettings()
        if settings.base_url and settings.client_id and settings.client_secret:
            try:
                url_get_token = f"{settings.base_url}/identity_/connect/token"
                token_credentials = dict(
                    client_id=settings.client_id,
                    client_secret=settings.client_secret,
                    grant_type="client_credentials",
                )

                with httpx.Client() as client:
                    res_json = client.post(url_get_token, data=token_credentials).json()
                    uipath_token_header = res_json.get("access_token")
            except Exception:
                # If client credentials fail, return empty string
                pass

    return uipath_token_header or ""
