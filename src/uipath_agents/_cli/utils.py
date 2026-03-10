import logging
import shutil
from pathlib import Path

from pydantic import ValidationError
from pydantic_core import ErrorDetails
from uipath.agent.models.agent import LowCodeAgentDefinition
from uipath.runtime.errors import UiPathErrorCategory
from uipath_langchain.agent.exceptions import AgentStartupError, AgentStartupErrorCode

from .constants import (
    AGENT_BUILDER_FILENAME,
    AGENT_ENTRYPOINT,
)

logger = logging.getLogger(__name__)


def _prepare_agent_execution_contract() -> None:
    """Copy all files from .agent-builder to root directory."""
    agent_builder_dir = Path(AGENT_BUILDER_FILENAME)

    if not agent_builder_dir.exists() or not agent_builder_dir.is_dir():
        logger.info(f"Agent builder directory not found at {agent_builder_dir}")
        return

    try:
        for file_path in agent_builder_dir.iterdir():
            if file_path.is_file():
                target_path = Path.cwd() / file_path.name
                shutil.copy2(file_path, target_path)
                logger.info(f"Copied {file_path.name} to {target_path}")
    except (OSError, PermissionError, shutil.Error) as e:
        logger.error(f"Failed to copy files from {agent_builder_dir}: {e}")
        raise


def load_agent_configuration(file_path: Path) -> LowCodeAgentDefinition:
    """Load and validate agent.json configuration.

    Raises:
        AgentStartupError: If file missing, has invalid structure, or fails schema validation
    """
    if not file_path.exists():
        raise AgentStartupError(
            AgentStartupErrorCode.FILE_NOT_FOUND,
            "Agent configuration not found",
            f"{AGENT_ENTRYPOINT} not found at {file_path}",
            UiPathErrorCategory.USER,
        )

    try:
        return LowCodeAgentDefinition.model_validate_json(
            file_path.read_text(encoding="utf-8")
        )
    except ValidationError as e:
        raise AgentStartupError(
            AgentStartupErrorCode.INVALID_AGENT_CONFIG,
            "Agent configuration invalid",
            f"{AGENT_ENTRYPOINT} failed schema validation: {e}\n\n{errorDetailsListToMessage(e.errors())}",
            UiPathErrorCategory.SYSTEM,
        ) from e
    except (ValueError, TypeError) as e:
        raise AgentStartupError(
            AgentStartupErrorCode.INVALID_AGENT_CONFIG,
            "Agent configuration invalid",
            f"Invalid {AGENT_ENTRYPOINT} structure: {e}",
            UiPathErrorCategory.SYSTEM,
        ) from e


def errorDetailsListToMessage(details: list[ErrorDetails]):
    return "\n  ".join(map(errorDetailsToMessage, details))


def errorDetailsToMessage(details: ErrorDetails):
    return f"(type: {details.get('type')}, loc: {details.get('loc')}, msg: {details.get('msg')})"
