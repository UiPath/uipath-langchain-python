import re
from typing import Any, Mapping

from langchain_core.messages import AnyMessage, ToolCall, ToolMessage
from langgraph.types import Overwrite
from pydantic import BaseModel
from uipath.platform.guardrails import BaseGuardrail, GuardrailScope
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.guardrails.masking import (
    Masker,
    PlaceholderAssigner,
    Restorer,
    decode_json_escapes,
    map_content,
    map_strings,
    map_tool_content,
)
from uipath_langchain.agent.guardrails.types import ExecutionStage
from uipath_langchain.agent.react.utils import (
    extract_current_tool_call_index,
    find_latest_ai_message,
)

from ...exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from ...messages.message_utils import replace_tool_calls
from ...react.types import AgentGuardrailsGraphState
from .base_action import GuardrailAction, GuardrailActionNode


class MaskAction(GuardrailAction):
    """Action that masks the PII a guardrail finds in a call's input.

    Before an agent or tool call, each value the guardrail detected in the call's
    input is replaced with a placeholder such as ``[Email_1]``. After the call the
    guardrail is not evaluated: the placeholders in the call's output are replaced
    with their values, after a tool call only those this guardrail wrote into that
    call. ``inner_state.masked_values`` keeps the run's placeholders.
    """

    @property
    def action_type(self) -> str:
        return "Mask"

    @property
    def text_only(self) -> bool:
        return True

    def action_node(
        self,
        *,
        guardrail: BaseGuardrail,
        scope: GuardrailScope,
        execution_stage: ExecutionStage,
        guarded_component_name: str,
    ) -> GuardrailActionNode:
        """Create the node that masks the detected values in the call's input."""
        metadata: dict[str, Any] = {
            "guardrail": guardrail,
            "scope": scope,
            "execution_stage": execution_stage,
            "masked_entities": {},
        }

        # Typed Any: LangGraph hands a node only the fields of its annotated state,
        # and the agent's input arguments are not among AgentGuardrailsGraphState's.
        async def _node(state: Any) -> dict[str, Any]:
            _ensure_maskable(guardrail, scope)
            error_type = state.inner_state.guardrail_validation_error_type
            if error_type is not None:
                raise _unmaskable(
                    guardrail,
                    f"could not be evaluated ({error_type}), so the input could "
                    "not be masked.",
                )
            values = _detected_values(
                state.inner_state.guardrail_detections,
                json_payload=scope == GuardrailScope.TOOL,
            )
            if not values:
                raise _unmaskable(
                    guardrail,
                    "reported a violation without saying where it is in the input, "
                    "so the input could not be masked.",
                )

            assigner = PlaceholderAssigner(state.inner_state.masked_values)
            masker = Masker(values, assigner)
            inner_update: dict[str, Any] = {}
            if scope == GuardrailScope.TOOL:
                update, call_id = _mask_tool_call(state, guarded_component_name, masker)
                if call_id is not None:
                    key = _tool_call_key(guardrail, call_id)
                    inner_update["masked_tool_calls"] = {key: masker.placeholders()}
            else:
                update = _mask_agent_input(state, masker)

            if missed := masker.missed_categories():
                raise _unmaskable(
                    guardrail,
                    f"found {', '.join(missed)} that could not be located in the "
                    "input, so it could not be masked.",
                )
            _replace_contents(metadata["masked_entities"], masker.counts())
            if assigner.added:
                inner_update["masked_values"] = assigner.added
            if inner_update:
                update["inner_state"] = inner_update
            return update

        _node.__metadata__ = metadata  # type: ignore[attr-defined]
        return _node_name(guardrail, scope, execution_stage, "mask"), _node

    def unconditional_node(
        self,
        *,
        guardrail: BaseGuardrail,
        scope: GuardrailScope,
        execution_stage: ExecutionStage,
        guarded_component_name: str,
    ) -> GuardrailActionNode | None:
        """After the call, restore the values in its output instead of evaluating."""
        if execution_stage != ExecutionStage.POST_EXECUTION:
            return None

        metadata: dict[str, Any] = {
            "guardrail": guardrail,
            "scope": scope,
            "execution_stage": execution_stage,
            "restored_entities": {},
        }

        async def _node(state: AgentGuardrailsGraphState) -> dict[str, Any]:
            _ensure_maskable(guardrail, scope)
            if scope == GuardrailScope.TOOL:
                restorer, update = _restore_tool_output(state, guardrail)
            else:
                restorer = Restorer(state.inner_state.masked_values)
                update = _restore_agent_result(state, restorer)
            _replace_contents(metadata["restored_entities"], restorer.counts())
            return update

        _node.__metadata__ = metadata  # type: ignore[attr-defined]
        return _node_name(guardrail, scope, execution_stage, "restore"), _node


def _node_name(
    guardrail: BaseGuardrail,
    scope: GuardrailScope,
    execution_stage: ExecutionStage,
    suffix: str,
) -> str:
    raw_node_name = f"{scope.name}_{execution_stage.name}_{guardrail.name}_{suffix}"
    return re.sub(r"\W+", "_", raw_node_name.lower()).strip("_")


def _tool_call_key(guardrail: BaseGuardrail, call_id: str) -> str:
    return f"{guardrail.id}:{call_id}"


def _ensure_maskable(guardrail: BaseGuardrail, scope: GuardrailScope) -> None:
    if scope not in (GuardrailScope.AGENT, GuardrailScope.TOOL):
        raise AgentRuntimeError(
            code=AgentRuntimeErrorCode.TERMINATION_GUARDRAIL_ERROR,
            title="Guardrail mask action not supported",
            detail=f"MaskAction of guardrail [{guardrail.name}] is not supported for "
            f"scope [{scope.name}].",
            category=UiPathErrorCategory.SYSTEM,
        )


def _unmaskable(guardrail: BaseGuardrail, reason: str) -> AgentRuntimeError:
    return AgentRuntimeError(
        code=AgentRuntimeErrorCode.TERMINATION_GUARDRAIL_ERROR,
        title="Guardrail could not mask the input",
        detail=f"Guardrail [{guardrail.name}] {reason}",
        category=UiPathErrorCategory.SYSTEM,
    )


def _replace_contents(target: dict[str, int], counts: Mapping[str, int]) -> None:
    # In place: the graph keeps a shallow copy of the node metadata sharing this dict.
    target.clear()
    target.update(counts)


def _detected_values(
    detections: list[dict[str, Any]] | None, *, json_payload: bool
) -> dict[str, str]:
    """Each value detected in the checked payload, mapped to its category.

    A detection without an offset comes from a file, not the payload, and is skipped.
    """
    values: dict[str, str] = {}
    for detection in detections or []:
        text = detection.get("text")
        if detection.get("offset") is None or not isinstance(text, str) or not text:
            continue
        if json_payload:
            text = decode_json_escapes(text)
        category = detection.get("category")
        values.setdefault(text, category if isinstance(category, str) else "")
    return values


def _mask_tool_call(
    state: AgentGuardrailsGraphState, tool_name: str, masker: Masker
) -> tuple[dict[str, Any], str | None]:
    """Mask the arguments of the ``tool_name`` call about to run; return its id too."""
    ai_message = find_latest_ai_message(state.messages)
    if ai_message is None or not ai_message.tool_calls:
        return {}, None
    index = extract_current_tool_call_index(state.messages, tool_name)
    if index is None:
        return {}, None

    tool_calls = list(ai_message.tool_calls)
    call_id = tool_calls[index].get("id") or ""
    args = tool_calls[index]["args"]
    masked_args = masker.mask(args)
    if masked_args == args:
        return {}, call_id
    masked_call: ToolCall = {**tool_calls[index], "args": masked_args}
    tool_calls[index] = masked_call
    return {"messages": [replace_tool_calls(ai_message, tool_calls)]}, call_id


def _mask_agent_input(
    state: AgentGuardrailsGraphState, masker: Masker
) -> dict[str, Any]:
    """Mask the agent's messages so far and its input arguments."""
    update: dict[str, Any] = {}
    messages = [_mask_message(message, masker) for message in state.messages]
    if any(new is not old for new, old in zip(messages, state.messages, strict=True)):
        update["messages"] = Overwrite(messages)

    agent_fields = AgentGuardrailsGraphState.model_fields
    for name in type(state).model_fields:
        if name in agent_fields:
            continue
        value = getattr(state, name)
        data = (
            value.model_dump(by_alias=True) if isinstance(value, BaseModel) else value
        )
        masked = masker.mask(data)
        if masked != data:
            update[name] = masked
    return update


def _mask_message(message: AnyMessage, masker: Masker) -> AnyMessage:
    content = map_content(message.content, masker.mask_text)
    if content == message.content:
        return message
    return message.model_copy(update={"content": content})


def _restore_tool_output(
    state: AgentGuardrailsGraphState, guardrail: BaseGuardrail
) -> tuple[Restorer, dict[str, Any]]:
    """Restore, in the result of the tool call that just ran, the values this
    guardrail masked in that call's arguments, then drop the call's record.

    Any other placeholder, such as one the agent's own mask wrote and the tool
    echoed, stays as it is.
    """
    if not state.messages or not isinstance(state.messages[-1], ToolMessage):
        return Restorer({}), {}
    result = state.messages[-1]
    key = _tool_call_key(guardrail, result.tool_call_id or "")
    placeholders = state.inner_state.masked_tool_calls.get(key)
    if placeholders is None:
        return Restorer({}), {}

    masked_values = state.inner_state.masked_values
    restorer = Restorer(
        {p: masked_values[p] for p in placeholders if p in masked_values}
    )
    update: dict[str, Any] = {"inner_state": {"masked_tool_calls": {key: None}}}
    content = map_tool_content(result.content, restorer.restore_text)
    if content != result.content:
        restored = result.model_copy(update={"content": content})
        update["messages"] = (
            [restored]
            if result.id is not None
            else Overwrite([*state.messages[:-1], restored])
        )
    return restorer, update


def _restore_agent_result(
    state: AgentGuardrailsGraphState, restorer: Restorer
) -> dict[str, Any]:
    """Restore the values in the agent's output."""
    result = state.inner_state.agent_result
    if result is None:
        return {}
    restored = map_strings(result, restorer.restore_text)
    if restored == result:
        return {}
    return {"inner_state": {"agent_result": restored}}
