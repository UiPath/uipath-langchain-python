from typing import Annotated, Any, Dict, List, Literal, TypeAlias, override

import jsonschema
from langchain.tools import BaseTool, StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from functools import partial
from langchain_core.runnables import Runnable

Message: TypeAlias = BaseMessage | SystemMessage | ToolMessage | HumanMessage | AIMessage


class RaiseErrorInput(BaseModel):
    message: str = Field(
        description="The error message to display to the user. This should be a brief on line message."
    )
    details: str | None = Field(
        description="Optional additional details about the error. This can be a multiline text with more details. Only populate this if there are relevant details not already captured in the error message."
    )


class StateBaseClass(BaseModel):
    class Config:
        extra = "forbid"

    messages: Annotated[List[Message], add_messages] = []
    result: Dict[str, Any] = {}
    raised_error: RaiseErrorInput | None = None
    run_init_state: Dict[str, str] = {}


class EndExecutionOutput(BaseModel):
    result: Dict[str, Any] = {}


class EndExecutionTool(StructuredTool):
    name: str = "end_execution"
    description: str = "Use this tool when you have gathered all required information and want to end execution. The input should match the expected output schema."

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
        return EndExecutionOutput(result=tool_input.result)


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
        agent_state.raised_error = arguments_model
        return agent_state


class Datapoint(BaseModel):
    input: Dict[str, str | Dict[str, Any]]
    evaluation_criteria: Dict[str, Any]
    simulation_instructions: str


class AgentBaseClass(BaseModel):
    system_prompt: str
    user_prompt: str
    end_execution_tool: EndExecutionTool
    datapoints: List[Datapoint] = []
    raise_error_tool: RaiseErrorTool = RaiseErrorTool()
    tools: List[BaseTool] = []


async def chatbot_node(
    state: StateBaseClass,
    llm: Runnable,
    print_trace: bool = True,
    parallel_tool_calls: bool = False,
):
    if state.messages and isinstance(state.messages[-1], AIMessage) and "claude" in getattr(llm, "model_name", ""):
        state.messages.append(HumanMessage(content="Proceed with the next step"))
    ai_message = await llm.ainvoke(state.messages)  # get AI response

    if print_trace:
        print(f"\nAssistant: {ai_message.content}")
        if isinstance(ai_message, AIMessage) and ai_message.tool_calls:
            if parallel_tool_calls:
                for tool_call in ai_message.tool_calls:
                    tool_name = tool_call["name"]
                    print(f"Tool Call: {tool_name}\n")
            else:
                print(f"Tool Call: {ai_message.tool_calls[0]['name']}")
    state.messages.append(ai_message)  # append the new message to existing messages
    return state


def should_continue(state: StateBaseClass) -> Literal["tools", "raise_error", "end_execution"]:
    messages = state.messages
    last_message = messages[-1]
    tool_calls = last_message.tool_calls if isinstance(last_message, AIMessage) else []

    # Check if end_execution tool is called
    if any(tool_call.get("name") == "end_execution" for tool_call in tool_calls):
        return "end_execution"
    # If there are other tool calls, route to tools
    elif any(tool_call.get("name") == "raise_error" for tool_call in tool_calls):
        return "raise_error"
    elif tool_calls:
        return "tools"
    # If no tool calls, route to tools anyway (agent needs to make a decision)
    return "tools"


class BasicLoop:
    def __init__(
        self,
        scenario: AgentBaseClass,
        llm: BaseChatModel,
        print_trace: bool = False,
        parallel_tool_calls: bool = False,
        debug: bool = False,
    ):
        self.llm_with_tools = llm.bind_tools(
            list(scenario.tools) + [scenario.end_execution_tool, scenario.raise_error_tool],
            parallel_tool_calls=parallel_tool_calls,
            tool_choice="any",
        )
        self.scenario = scenario
        self.print_trace = print_trace
        self.parallel_tool_calls = parallel_tool_calls
        self.debug = debug


    def prepare_input_node(self, state: StateBaseClass, agent_input: Dict[str, Any]) -> StateBaseClass:
        """Node to handle input preparation and ensure proper initial messages."""
        if self.debug:
            print(f"[DEBUG] prepare_input_node - Received state with {len(state.messages)} messages")
            print(f"[DEBUG] prepare_input_node - agent_input: {agent_input}")

        # If no messages, use default scenario prompts (unattended mode)
        if not state.messages:
            # Check if agent_input contains datapoint data (like 'expression', 'message', etc.)
            assert agent_input, "Agent input is required to be non-empty when no messages are present"

            # This looks like unattended mode with datapoint - use scenario prompts
            try:
                state.messages = [
                        SystemMessage(content=self.scenario.system_prompt),
                        HumanMessage(content=self.scenario.user_prompt.format_map(agent_input))
                    ]
                if self.debug:
                    print("[DEBUG] Datapoint content detected - using scenario prompts (unattended mode)")
            except KeyError as e:
                # If format_map fails, fall back to basic setup
                state.messages = [
                    SystemMessage(content=self.scenario.system_prompt),
                    HumanMessage(content=f"Help me with: {agent_input}")
                ]
                if self.debug:
                    print(f"[DEBUG] Format error {e} - using fallback with agent_input")
        else:
            # Interactive mode: ensure system message is present
            has_system_message = any(isinstance(msg, SystemMessage) for msg in state.messages)
            if not has_system_message:
                state.messages.insert(0, SystemMessage(content=self.scenario.system_prompt))
                if self.debug:
                    print("[DEBUG] Added system message to user-provided messages (interactive mode)")

        if self.debug:
            print(f"[DEBUG] Final {len(state.messages)} messages")

        return state

    def build_graph(self, agent_input: Dict[str, Any] = {}) -> StateGraph:
        """Build the graph with proper input handling for both unattended and interactive operation."""
        # Create graph with input/output schema specification
        graph = StateGraph(StateBaseClass)

        # Add input preparation node as the first step
        graph.add_node("prepare_input", partial(self.prepare_input_node, agent_input=agent_input))

        graph.add_node(
            "chatbot",
            partial(
                chatbot_node,
                llm=self.llm_with_tools,
                print_trace=self.print_trace,
                parallel_tool_calls=self.parallel_tool_calls,
            ),
        )
        graph.add_node("tools", ToolNode(self.scenario.tools))
        graph.add_node("end_execution", self.scenario.end_execution_tool)
        graph.add_node("raise_error", self.scenario.raise_error_tool)

        # Route through prepare_input first to handle JSON input conversion
        graph.add_edge(START, "prepare_input")
        graph.add_edge("prepare_input", "chatbot")

        graph.add_conditional_edges("chatbot", should_continue)
        graph.add_edge("tools", "chatbot")
        graph.add_edge("end_execution", END)
        graph.add_edge("raise_error", END)
        return graph
