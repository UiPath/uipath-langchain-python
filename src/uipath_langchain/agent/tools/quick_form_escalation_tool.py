"""Quick-form escalation tool creation for schema-first HITL tasks.

Quick-form escalations (``escalationType=2``) render a schema-first task in
Action Center via FormLib instead of dispatching to an Action Center app.
The HITL schema and its key live on the channel
(``AgentEscalationChannel.schema`` / ``schema_id``) and are forwarded
inline to Orchestrator's ``GenericTasks/CreateTask`` endpoint via
:meth:`uipath.platform.action_center.tasks.TasksService.create_quickform_async`.
"""

from typing import Any, Literal

from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel
from uipath.agent.models.agent import (
    AgentEscalationChannel,
    AgentQuickFormEscalationResourceConfig,
    LowCodeAgentDefinition,
)
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.platform.action_center.tasks import Task, TaskRecipient
from uipath.platform.common import WaitEscalation
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain._utils import get_execution_folder_path
from uipath_langchain._utils.durable_interrupt import durable_interrupt
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)

from ..exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from ..react.types import AgentGraphState
from .escalation_tool import (
    EscalationAction,
    _parse_task_data,
    _resolve_escalation_action,
    resolve_recipient_value,
)
from .tool_node import ToolWrapperReturnType
from .utils import (
    resolve_task_title,
    sanitize_dict_for_serialization,
    sanitize_tool_name,
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

    class QuickFormEscalationToolOutput(BaseModel):
        action: Literal["approve", "reject"]
        data: output_model
        is_deleted: bool = False

    async def quick_form_escalation_tool_fn(**kwargs: Any) -> dict[str, Any]:
        agent_input: dict[str, Any] = (
            tool.metadata.get("agent_input") if tool.metadata else None
        ) or {}
        recipient: TaskRecipient | None = (
            await resolve_recipient_value(channel.recipients[0], input_args=agent_input)
            if channel.recipients
            else None
        )
        folder_path = get_execution_folder_path()

        task_title = "Escalation Task"
        if tool.metadata is not None:
            tool.metadata["recipient"] = recipient
            task_title = tool.metadata.get("task_title") or task_title

        serialized_data = input_model.model_validate(kwargs).model_dump(mode="json")

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
                    title=task_title,
                    task_schema_key=task_schema_key,
                    schema=task_schema_body,
                    data=serialized_data,
                    folder_path=folder_path,
                    recipient=recipient,
                    priority=channel.priority,
                    labels=channel.labels,
                    is_actionable_message_enabled=channel.properties.is_actionable_message_enabled,
                    actionable_message_metadata=channel.properties.actionable_message_meta_data,
                )

                return WaitEscalation(
                    action=created_task,
                    app_folder_path=folder_path,
                    app_name=channel.properties.app_name,
                    recipient=recipient,
                )

            return await create_quick_form_task()

        result = await escalate(**kwargs)
        if isinstance(result, dict):
            result = Task.model_validate(result)

        if result.is_deleted:
            return {
                "action": EscalationAction.END,
                "output": None,
                "outcome": "The escalation task was deleted",
            }

        raw_data = (
            result.data.model_dump()
            if isinstance(result.data, BaseModel)
            else (result.data or {})
        )
        escalation_output = _parse_task_data(
            raw_data,
            input_schema=input_model.model_json_schema(),
            output_schema=output_model.model_json_schema(),
        )
        escalation_action = _resolve_escalation_action(
            result.action,
            channel.outcome_mapping,
        )

        return {
            "action": escalation_action,
            "output": escalation_output,
            "outcome": result.action,
        }

    async def quick_form_escalation_wrapper(
        tool: BaseTool,
        call: ToolCall,
        state: AgentGraphState,
    ) -> ToolWrapperReturnType:
        if tool.metadata is None:
            raise RuntimeError("Tool metadata is required for task_title resolution")

        state_dict = sanitize_dict_for_serialization(dict(state))
        tool.metadata["task_title"] = resolve_task_title(
            channel.task_title,
            state_dict,
            default_title="Escalation Task",
        )
        internal_fields = set(AgentGraphState.model_fields.keys())
        tool.metadata["agent_input"] = {
            k: v for k, v in state_dict.items() if k not in internal_fields
        }

        tool.metadata["_call_id"] = call.get("id")
        tool.metadata["_call_args"] = dict(call.get("args", {}))

        result = await tool.ainvoke(call["args"])

        if result["action"] == EscalationAction.END:
            output_detail = f"Escalation output: {result['output']}"
            termination_title = (
                f"Agent run ended based on escalation outcome {result['action']} "
                f"with directive {result['outcome']}"
            )
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.TERMINATION_ESCALATION_REJECTED,
                title=termination_title,
                detail=output_detail,
                category=UiPathErrorCategory.USER,
            )

        return {
            "output": result["output"],
            "outcome": result["outcome"],
            "task_id": result.get("task_id"),
            "assigned_to": result.get("assigned_to"),
        }

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
    tool.set_tool_wrappers(awrapper=quick_form_escalation_wrapper)

    return tool
