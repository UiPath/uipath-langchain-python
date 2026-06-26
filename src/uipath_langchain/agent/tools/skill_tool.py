"""Skill tool: binds a vdbs Skill (versioned prompt) to the agent as a callable.

Patterned after LangChain's "skill loader" tool. Invoking the tool returns the
skill's published ``Content`` string verbatim; the parent agent's LLM treats
that string as additional context and continues its reasoning loop. There is
no sub-LLM call inside the tool — the parent owns execution, sees the prompt
as a tool result, and applies it.
"""

from logging import getLogger

from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from uipath.agent.models.agent import AgentSkillToolResourceConfig
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)

from .utils import sanitize_tool_name

logger = getLogger(__name__)


class _NoArgs(BaseModel):
    """Empty args schema for skill tools — they take no parameters."""


def create_skill_tool(resource: AgentSkillToolResourceConfig) -> StructuredTool:
    """Create a tool that returns a published vdbs Skill's prompt as a string.

    At tool-creation time the skill's published version Content is fetched
    from ``/ecs_/v2/Skills`` (pinned by ``version_id`` when set, otherwise the
    current published version). The Content is cached for the lifetime of the
    tool — invocations return it verbatim. The tool takes no arguments; the
    parent agent's LLM already has the task context in its own conversation
    and uses the returned prompt as guidance for its next step.

    Args:
        resource: The skill tool resource config from the agent definition.

    Returns:
        A StructuredTool whose result is the skill's prompt text.

    Raises:
        AgentRuntimeError: If the skill or its published version cannot be
            resolved at startup.
    """
    tool_name = sanitize_tool_name(resource.name)
    skill_id = resource.properties.skill_id
    version_id = resource.properties.version_id
    folder_path = resource.properties.folder_path
    folder_key = resource.properties.folder_key

    prompt = _resolve_skill_prompt(
        skill_id=skill_id,
        version_id=version_id,
        folder_path=folder_path,
        folder_key=folder_key,
        resource_name=resource.name,
    )

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=_NoArgs.model_json_schema(),
        output_schema={"type": "string"},
    )
    async def skill_tool_fn() -> str:
        return prompt

    return StructuredToolWithArgumentProperties(
        name=tool_name,
        description=resource.description,
        args_schema=_NoArgs,
        coroutine=skill_tool_fn,
        argument_properties={},
        metadata={
            "tool_type": resource.type.lower(),
            "display_name": tool_name,
            "skill_id": skill_id,
            "skill_version_id": version_id,
        },
    )


def _resolve_skill_prompt(
    *,
    skill_id: str,
    version_id: str,
    folder_path: str | None,
    folder_key: str | None,
    resource_name: str,
) -> str:
    """Fetch the prompt content for the bound skill version.

    The current BE doesn't route ``/Skills({key})/GetVersion(...)`` so we fetch
    the full skill with ``include_content=True`` and find the bound version
    in its ``versions`` array. Raises ``AgentRuntimeError`` on any resolution
    failure so the agent fails fast at startup.
    """
    sdk = UiPath()
    try:
        skill = sdk.skills.retrieve(
            key=skill_id,
            include_content=True,
            folder_path=folder_path,
            folder_key=folder_key,
        )
    except LookupError as exc:
        raise AgentRuntimeError(
            code=AgentRuntimeErrorCode.UNEXPECTED_ERROR,
            title=f"Skill '{resource_name}' not found",
            detail=str(exc),
            category=UiPathErrorCategory.DEPLOYMENT,
        ) from exc
    except Exception as exc:
        raise AgentRuntimeError(
            code=AgentRuntimeErrorCode.UNEXPECTED_ERROR,
            title=f"Failed to resolve skill '{resource_name}'",
            detail=f"Fetching skill content raised: {exc!r}",
            category=UiPathErrorCategory.SYSTEM,
        ) from exc

    target = next((v for v in skill.versions if v.id == version_id), None)
    if target is None:
        raise AgentRuntimeError(
            code=AgentRuntimeErrorCode.UNEXPECTED_ERROR,
            title=f"Skill '{resource_name}' has no version '{version_id}'",
            detail=(
                f"Skill {skill_id} does not have a version with id "
                f"{version_id}. Available versions: "
                + ", ".join(f"{v.version}({v.id})" for v in skill.versions)
            ),
            category=UiPathErrorCategory.DEPLOYMENT,
        )

    content = target.content or ""
    if not content.strip():
        logger.warning(
            "Skill '%s' (id=%s, version=%s) has empty content; the tool will "
            "return an empty string.",
            resource_name,
            skill_id,
            target.version,
        )
    return content
