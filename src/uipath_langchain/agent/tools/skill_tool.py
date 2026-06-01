"""Skill tool: binds a vdbs Skill (versioned prompt) to the agent as a callable."""

import json
from logging import getLogger
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from uipath.agent.models.agent import AgentSkillToolResourceConfig
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)
from uipath_langchain.chat.helpers import extract_text_content

from .utils import sanitize_tool_name

logger = getLogger(__name__)


def create_skill_tool(
    resource: AgentSkillToolResourceConfig, llm: BaseChatModel
) -> StructuredTool:
    """Create a structured tool that invokes a published vdbs Skill.

    At tool-creation time, the published version's Content is fetched from
    `/ecs_/v2/Skills` (pinned by version_id when set, otherwise the current
    published version on the skill). The Content becomes the SystemMessage of
    a sub-LLM call. The tool's input/output schema and argument properties
    come from the enclosing AgentSkillToolResourceConfig.

    Args:
        resource: The skill tool resource config from the agent definition.
        llm: The parent agent's chat model. Reused for the sub-LLM call so the
            skill inherits the same provider, credentials, and quota.

    Returns:
        A StructuredTool whose invocation runs the skill's prompt against the
        user-supplied arguments and returns output validated against the
        binding's output schema.

    Raises:
        AgentRuntimeError: If the skill or its published version cannot be
            resolved at startup, or if invocation fails.
    """
    tool_name = sanitize_tool_name(resource.name)
    skill_id = resource.properties.skill_id
    pinned_version_id = resource.properties.version_id
    folder_path = resource.properties.folder_path
    folder_key = resource.properties.folder_key

    prompt = _resolve_skill_prompt(
        skill_id=skill_id,
        version_id=pinned_version_id,
        folder_path=folder_path,
        folder_key=folder_key,
        resource_name=resource.name,
    )

    input_model: Any = create_model(resource.input_schema)
    output_model: Any = create_model(resource.output_schema)
    has_output_schema = bool(resource.output_schema.get("properties"))

    non_streaming_llm = llm.model_copy(update={"disable_streaming": True})

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=input_model.model_json_schema(),
        output_schema=output_model.model_json_schema(),
    )
    async def skill_tool_fn(**kwargs: Any) -> Any:
        args_model = input_model.model_validate(kwargs)
        user_message = args_model.model_dump_json()

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=user_message),
        ]

        if has_output_schema:
            structured_llm = non_streaming_llm.with_structured_output(output_model)
            try:
                result = await structured_llm.ainvoke(messages)
            except Exception as exc:
                raise AgentRuntimeError(
                    code=AgentRuntimeErrorCode.UNEXPECTED_ERROR,
                    title=f"Skill '{resource.name}' invocation failed",
                    detail=f"LLM call raised: {exc!r}",
                    category=UiPathErrorCategory.SYSTEM,
                ) from exc
            # with_structured_output returns the validated pydantic instance;
            # serialize to a plain dict for the tool result.
            if hasattr(result, "model_dump"):
                return result.model_dump()
            return result

        try:
            response = await non_streaming_llm.ainvoke(messages)
        except Exception as exc:
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.UNEXPECTED_ERROR,
                title=f"Skill '{resource.name}' invocation failed",
                detail=f"LLM call raised: {exc!r}",
                category=UiPathErrorCategory.SYSTEM,
            ) from exc

        text = extract_text_content(response)
        # If the skill response looks like JSON, return it parsed so the
        # downstream agent can consume structured output even without a schema.
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text

    return StructuredToolWithArgumentProperties(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=skill_tool_fn,
        output_type=output_model,
        argument_properties=resource.argument_properties,
        metadata={
            "tool_type": resource.type.lower(),
            "display_name": tool_name,
            "skill_id": skill_id,
            "skill_version_id": pinned_version_id,
            "args_schema": input_model,
            "output_schema": output_model,
        },
    )


def _resolve_skill_prompt(
    *,
    skill_id: str,
    version_id: str | None,
    folder_path: str | None,
    folder_key: str | None,
    resource_name: str,
) -> str:
    """Fetch the prompt content for the bound skill version.

    When ``version_id`` is provided we fetch that exact version; otherwise we
    resolve the skill's current published version. Raises ``AgentRuntimeError``
    on any resolution failure so the agent fails fast at startup rather than
    masking the misconfiguration at call time.
    """
    sdk = UiPath()
    try:
        if version_id is not None:
            version = sdk.skills.get_version(
                key=skill_id,
                version_id=version_id,
                folder_path=folder_path,
                folder_key=folder_key,
            )
        else:
            skill = sdk.skills.retrieve(
                key=skill_id,
                folder_path=folder_path,
                folder_key=folder_key,
            )
            published = skill.published_version
            if published is None:
                raise AgentRuntimeError(
                    code=AgentRuntimeErrorCode.UNEXPECTED_ERROR,
                    title=f"Skill '{resource_name}' has no published version",
                    detail=(
                        f"Skill {skill_id} is not published. Publish a version or "
                        "pin a specific versionId on the agent's skill binding."
                    ),
                    category=UiPathErrorCategory.DEPLOYMENT,
                )
            version = sdk.skills.get_version(
                key=skill_id,
                version_id=published.id,
                folder_path=folder_path,
                folder_key=folder_key,
            )
    except AgentRuntimeError:
        raise
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

    content = version.content or ""
    if not content.strip():
        logger.warning(
            "Skill '%s' (id=%s, version=%s) has empty content; the LLM will "
            "receive an empty system prompt.",
            resource_name,
            skill_id,
            version.version,
        )
    return content
