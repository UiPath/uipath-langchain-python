"""Escalation recipient resolution.

Turns a channel's design-time recipient configuration — or an agent-inferred
("LLM inferred") value supplied at runtime — into a concrete ``TaskRecipient``
for Action Center assignment.
"""

import logging
import re
from typing import Any

from uipath.agent.models.agent import (
    AgentEscalationRecipient,
    AgentEscalationRecipientType,
    ArgumentEmailRecipient,
    ArgumentGroupNameRecipient,
    AssetRecipient,
    CustomAssigneesRecipient,
    RoundRobinRecipient,
    StandardRecipient,
    WorkloadRecipient,
)
from uipath.agent.utils.text_tokens import safe_get_nested
from uipath.platform import UiPath
from uipath.platform.action_center.tasks import TaskRecipient, TaskRecipientType
from uipath.platform.common import UiPathConfig

from uipath_langchain._utils import get_execution_folder_path

from .escalation_memory import _get_user_email

_logger = logging.getLogger(__name__)


# Reserved input-schema field that carries the agent-inferred ("Custom") recipient.
RESERVED_RECIPIENT_FIELD = "dynamic__escalationRecipient"

MAX_RECIPIENTS = 50
MAX_EMAIL_LENGTH = 254
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+(?:\.[^@\s.]+)+$")


async def _resolve_asset(asset_name: str, folder_path: str | None) -> str | None:
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


async def _filter_to_directory_users(emails: list[str]) -> list[str]:
    """Return only the emails that resolve to a real user in the tenant directory."""
    org_id = UiPathConfig.organization_id
    if not org_id:
        return []
    client = UiPath()
    valid: list[str] = []
    for email in emails:
        try:
            response = await client.api_client.request_async(
                "GET",
                f"/identity_/api/Directory/Search/{org_id}",
                scoped="org",
                params={
                    "startsWith": email,
                    "sourceFilter": ["directoryUsers", "localUsers"],
                },
            )
            if any(
                _get_user_email(entry) == email for entry in (response.json() or [])
            ):
                valid.append(email)
            else:
                _logger.warning(
                    "LLM recipient '%s' did not match a tenant directory user; dropping",
                    email,
                )
        except Exception:
            _logger.warning(
                "Directory lookup failed for LLM recipient '%s'; dropping",
                email,
                exc_info=True,
            )
    return valid


async def _build_llm_recipient(raw: Any) -> TaskRecipient | None:
    """Validate an LLM-supplied recipient value and shape it into a TaskRecipient."""
    if isinstance(raw, str):
        candidates = [s.strip() for s in raw.split(",")]
    elif isinstance(raw, list):
        candidates = [str(s).strip() for s in raw]
    else:
        return None

    emails = [
        c for c in candidates if c and len(c) <= MAX_EMAIL_LENGTH and _EMAIL_RE.match(c)
    ][:MAX_RECIPIENTS]
    if not emails:
        _logger.warning(
            "LLM recipient produced no valid email values; leaving task unassigned"
        )
        return None

    valid = await _filter_to_directory_users(emails)
    if not valid:
        _logger.warning(
            "LLM recipient values did not resolve to tenant users; "
            "leaving task unassigned"
        )
        return None

    return TaskRecipient(
        value=valid[0],
        values=valid,
        type=TaskRecipientType.WORKLOAD,
        displayName=", ".join(valid),
    )


async def resolve_recipient_value(
    recipient: AgentEscalationRecipient,
    input_args: dict[str, Any] | None = None,
) -> TaskRecipient | None:
    """Resolve recipient value based on recipient type."""
    if isinstance(recipient, AssetRecipient):
        value = await _resolve_asset(recipient.asset_name, get_execution_folder_path())
        type = None
        if recipient.type == AgentEscalationRecipientType.ASSET_USER_EMAIL:
            type = TaskRecipientType.EMAIL
        elif recipient.type == AgentEscalationRecipientType.ASSET_GROUP_NAME:
            type = TaskRecipientType.GROUP_NAME
        return TaskRecipient(value=value, type=type, displayName=value)

    if isinstance(recipient, ArgumentEmailRecipient):
        value = safe_get_nested(input_args or {}, recipient.argument_path)
        if value is None:
            raise ValueError(
                f"Argument '{recipient.argument_path}' has no value in agent input."
            )
        return TaskRecipient(
            value=value, type=TaskRecipientType.EMAIL, displayName=value
        )

    if isinstance(recipient, ArgumentGroupNameRecipient):
        value = safe_get_nested(input_args or {}, recipient.argument_path)
        if value is None:
            raise ValueError(
                f"Argument '{recipient.argument_path}' has no value in agent input."
            )
        return TaskRecipient(
            value=value, type=TaskRecipientType.GROUP_NAME, displayName=value
        )

    if isinstance(recipient, WorkloadRecipient):
        # Action Center expects the group NAME in assigneeNamesOrEmails;
        # `value` on the agent model is the group identifier, `display_name` is the name.
        return TaskRecipient(
            value=recipient.display_name,
            type=TaskRecipientType.WORKLOAD,
            displayName=recipient.display_name,
        )

    if isinstance(recipient, RoundRobinRecipient):
        return TaskRecipient(
            value=recipient.display_name,
            type=TaskRecipientType.ROUND_ROBIN,
            displayName=recipient.display_name,
        )

    if isinstance(recipient, CustomAssigneesRecipient):
        return None

    if isinstance(recipient, StandardRecipient):
        type = TaskRecipientType(recipient.type)
        if recipient.type == AgentEscalationRecipientType.USER_EMAIL:
            type = TaskRecipientType.EMAIL
        return TaskRecipient(
            value=recipient.value, type=type, displayName=recipient.value
        )

    return None
