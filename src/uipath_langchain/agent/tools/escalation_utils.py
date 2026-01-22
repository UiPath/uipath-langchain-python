"""Utility functions for escalation tool - separated to avoid circular imports."""

from uipath.agent.models.agent import (
    AgentEscalationRecipient,
    AssetRecipient,
    StandardRecipient,
)
from uipath.platform import UiPath


async def resolve_recipient_value(recipient: AgentEscalationRecipient) -> str | None:
    """Resolve recipient value based on recipient type."""
    if isinstance(recipient, AssetRecipient):
        return await resolve_asset(recipient.asset_name, recipient.folder_path)

    if isinstance(recipient, StandardRecipient):
        return recipient.value

    return None


async def resolve_asset(asset_name: str, folder_path: str) -> str | None:
    """Retrieve asset value."""
    try:
        client = UiPath()
        result = await client.assets.retrieve_async(
            name=asset_name, folder_path=folder_path
        )

        if not result or not result.value:
            raise ValueError(f"Asset '{asset_name}' has no value configured.")

        return result.value
    except Exception as e:
        raise ValueError(
            f"Failed to resolve asset '{asset_name}' in folder '{folder_path}': {str(e)}"
        ) from e
