from typing import Any, Dict, override
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field
import jsonschema

from .uipath_gym_types import StateBaseClass


META_DESCRIPTION_ESCALATION_TOOL: str = """This is an escalation. This tool must be called separately from other tool calls (i.e. not in parallel with other tool calls).
This is because you must wait for the response before deciding what tool to call next.
"""


class EndExecutionDefaultOutput(BaseModel):
    result: Dict[str, Any] = {}


class EndExecutionTool(StructuredTool):
    name: str = "end_execution"
    description: str = "Use this tool when you have gathered all required information and want to end execution. The input should match the expected output schema."
    output_schema: type[BaseModel] | None = None  # The final output schema to return

    @override
    def run(self, tool_input: StateBaseClass, *args: Any, **kwargs: Any):
        last_message = tool_input.messages[-1]
        # If the last message is a ToolMessage, this was the result of a validation step,
        # so we need to get the previous message.
        if isinstance(last_message, ToolMessage):
            last_message = tool_input.messages[-2]
        if not isinstance(last_message, AIMessage):
            raise ValueError(
                "The last message passed as input to the EndExecutionTool should have been an AIMessage"
            )
        arguments = last_message.tool_calls[0]["args"]
        schema = (
            self.args_schema
            if isinstance(self.args_schema, dict)
            else self.args_schema.model_json_schema()
        )
        jsonschema.validate(arguments, schema)
        tool_input.result = arguments

        # Return the output_schema if provided, otherwise return EndExecutionOutput
        if self.output_schema:
            # Convert the result dict to the output_schema type
            return self.output_schema.model_validate(tool_input.result)
        return EndExecutionDefaultOutput(result=tool_input.result)


class RaiseErrorInput(BaseModel):
    message: str = Field(
        description="The error message to display to the user. This should be a brief on line message."
    )
    details: str | None = Field(
        description="Optional additional details about the error. This can be a multiline text with more details. Only populate this if there are relevant details not already captured in the error message."
    )


class RaiseErrorTool(StructuredTool):
    name: str = "raise_error"
    description: str = "Raises an error and ends the execution of the agent"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["args_schema"] = RaiseErrorInput
        super().__init__(*args, **kwargs)

    @override
    def run(self, agent_state: StateBaseClass, *args: Any, **kwargs: Any):
        last_message = agent_state.messages[-1]
        # If the last message is a ToolMessage, this was the result of a validation step,
        # so we need to get the previous message.
        if isinstance(last_message, ToolMessage):
            last_message = agent_state.messages[-2]
        if not isinstance(last_message, AIMessage):
            raise ValueError(
                "The last message passed as input to the EndExecutionTool should have been an AIMessage"
            )
        arguments = last_message.tool_calls[0]["args"]
        arguments_model = RaiseErrorInput.model_validate(arguments)
        return arguments_model


class EscalationToolInput(BaseModel):
    query: str = Field(..., title="Query string")


class EscalationTool(BaseTool):
    name: str = "escalation_tool"
    description: str
    assign_to: str
    return_message: str | None = None

    def __init__(
        self,
        description: str = "",
        **kwargs: Any,
    ):
        super().__init__(
            description=f"{META_DESCRIPTION_ESCALATION_TOOL}{description}",
            args_schema=EscalationToolInput,
            **kwargs,
        )

    @override
    async def _arun(self, query: str) -> str:
        """Async run method required by BaseTool"""
        return self._run(query)

    @override
    def _run(self, query: str) -> str:
        """Synchronous run method required by BaseTool"""
        if self.return_message:
            return self.return_message
        return f'The escalation was successful. A task has been created for this query: "{query}", and assigned to {self.assign_to}. Call the end execution tool.'
