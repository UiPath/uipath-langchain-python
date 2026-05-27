"""Quick-form escalation (``escalationType=2``).

Quick-form escalations render a schema-first task in Action Center via
FormLib instead of dispatching to an Action Center app. The HITL schema
and its key live on the channel (``AgentEscalationChannel.schema`` /
``schema_id``) and are forwarded inline to Orchestrator's
``GenericTasks/CreateTask`` endpoint via
:meth:`uipath.platform.action_center.tasks.TasksService.create_quickform_async`.
"""

from typing import Any

from langchain_core.tools import StructuredTool
from uipath.agent.models.agent import (
    AgentEscalationChannel,
    AgentQuickFormEscalationResourceConfig,
    LowCodeAgentDefinition,
)
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.platform.action_center.tasks import Task
from uipath.platform.common import WaitEscalation

from uipath_langchain._utils.durable_interrupt import durable_interrupt
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)

from ..utils import sanitize_tool_name
from .common import (
    build_invocation_ctx,
    finalize_escalation_result,
    make_escalation_tool_output,
    make_escalation_wrapper,
)


def create_quick_form_escalation_tool(
    resource: AgentQuickFormEscalationResourceConfig,
    agent: LowCodeAgentDefinition | None = None,
) -> StructuredTool:
    """Create a structured tool that opens a quick-form HITL task.

    The returned tool suspends graph execution via ``durable_interrupt``
    until the form is completed, then resolves the configured outcome
    mapping into a continue/end action (mirroring
    :func:`create_escalation_tool`).

    Args:
        resource: The quick-form escalation resource (``escalationType=2``).
        agent: Optional parent agent definition; reserved for parity with
            :func:`create_escalation_tool` and future agent-scoped
            settings (e.g. escalation memory).

    Returns:
        A langchain ``StructuredTool`` representing the quick-form
        escalation.
    """
    del agent

    tool_name: str = f"escalate_{sanitize_tool_name(resource.name)}"
    channel: AgentEscalationChannel = resource.channels[0]

    # Orchestrator upserts the form schema by schemaId on every task creation,
    # so both schemaId and the inline schema are required for QuickForm.
    if not channel.schema_id or not channel.schema:
        raise ValueError(
            f"Quick-form escalation '{resource.name}' is missing 'schemaId' "
            "or 'schema' on its channel; both are required to create the "
            "QuickForm task."
        )

    task_schema_key: str = channel.schema_id
    task_schema_body: dict[str, Any] = channel.schema

    input_model: Any = create_model(channel.input_schema)
    output_model: Any = create_model(channel.output_schema)
    QuickFormEscalationToolOutput = make_escalation_tool_output(output_model)

    async def quick_form_escalation_tool_fn(**kwargs: Any) -> dict[str, Any]:
        ctx = await build_invocation_ctx(tool, channel, kwargs, input_model)

        @mockable(
            name=tool_name.lower(),
            description=resource.description,
            input_schema=input_model.model_json_schema(),
            output_schema=QuickFormEscalationToolOutput.model_json_schema(),
            example_calls=channel.properties.example_calls,
        )
        async def escalate(**_: Any):
            @durable_interrupt
            async def create_quick_form_task():
                client = UiPath()
                created_task = await client.tasks.create_quickform_async(
                    title=ctx.task_title,
                    task_schema_key=task_schema_key,
                    schema=task_schema_body,
                    data=ctx.serialized_data,
                    folder_path=ctx.folder_path,
                    recipient=ctx.recipient,
                    priority=channel.priority,
                    labels=channel.labels,
                    is_actionable_message_enabled=channel.properties.is_actionable_message_enabled,
                    actionable_message_metadata=channel.properties.actionable_message_meta_data,
                )

                return WaitEscalation(
                    action=created_task,
                    app_folder_path=ctx.folder_path,
                    app_name=channel.properties.app_name,
                    recipient=ctx.recipient,
                )

            return await create_quick_form_task()

        result = await escalate(**kwargs)
        if isinstance(result, dict):
            result = Task.model_validate(result)

        return finalize_escalation_result(
            result,
            input_model=input_model,
            output_model=output_model,
            outcome_mapping=channel.outcome_mapping,
        )

    tool = StructuredToolWithArgumentProperties(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        output_type=output_model,
        coroutine=quick_form_escalation_tool_fn,
        argument_properties=channel.argument_properties,
        metadata={
            "tool_type": "escalation",
            "escalation_subtype": "quick_form",
            "display_name": channel.properties.app_name,
            "channel_type": channel.type,
            "recipient": None,
            "args_schema": input_model,
            "output_schema": output_model,
            "schema_id": task_schema_key,
        },
    )
    tool.set_tool_wrappers(awrapper=make_escalation_wrapper(channel))

    return tool
