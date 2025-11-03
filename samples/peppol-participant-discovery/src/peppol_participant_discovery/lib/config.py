"""Configuration management with UiPath Asset support."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or UiPath Assets."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM Configuration (OpenAI-compatible)
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: Optional[str] = Field(
        default=None,
        alias="OPENAI_BASE_URL",
        description="Optional base URL for OpenAI-compatible endpoints (e.g., OpenRouter, Azure)",
    )
    model: str = Field(
        default="gpt-4o-mini-2024-07-18",
        description="Model identifier (OpenAI, Azure, or other compatible endpoint)",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: int = Field(default=30, gt=0)

    # Company Data Hub API Configuration
    api_endpoint: Optional[str] = Field(
        default="https://companydata-hub.cprimadotnet.workers.dev/unstructured/signature",
        alias="API_ENDPOINT",
    )
    companydatahub_api_key: Optional[str] = Field(default=None, alias="COMPANYDATAHUB_API_KEY")


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """
    Load settings from .env and environment variables.

    For UiPath deployment, use UiPath Assets to populate environment variables.
    For local development, use .env file.
    """
    load_dotenv(override=False)
    return Settings()


def load_settings_with_uipath(uipath) -> Settings:
    """
    Load settings from environment variables.

    Args:
        uipath: UiPath SDK instance (reserved for future use)

    Returns:
        Settings with values from .env or environment variables
    """
    load_dotenv(override=False)
    return Settings()
