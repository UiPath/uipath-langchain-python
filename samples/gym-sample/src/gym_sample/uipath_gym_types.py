from typing import Annotated, Any, Dict, List, Literal, TypeAlias, override

import jsonschema
from langchain.tools import BaseTool, StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, create_model
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
        extra = "allow"  # Allow extra fields from input_schema (like 'expression')

    messages: Annotated[List[Message], add_messages] = []
    result: Dict[str, Any] = {}
    raised_error: RaiseErrorInput | None = None
    run_init_state: Dict[str, str] = {}


class EndExecutionOutput(BaseModel):
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
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
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


class BasicLoopInput(BaseModel):
    agent_input: Dict[str, Any]


class BasicLoopOutput(BaseModel):
    answer: str


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


    def prepare_input_node(self, agent_input: BaseModel, state: StateBaseClass | None = None) -> StateBaseClass:
        """Node to handle input preparation and ensure proper initial messages (evaluation mode)."""
        state = state or StateBaseClass()
        # If no messages, use default scenario prompts (unattended mode)
        if not state.messages:
            # Check if agent_input contains datapoint data (like 'expression', 'message', etc.)
            assert agent_input, "Agent input is required to be non-empty when no messages are present"

            # This looks like unattended mode with datapoint - use scenario prompts
            try:
                state.messages = [
                        SystemMessage(content=self.scenario.system_prompt),
                        HumanMessage(content=self.scenario.user_prompt.format_map(agent_input.model_dump()))
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

    def prepare_input_from_runtime(self, graph_input: BaseModel) -> StateBaseClass:
        """Node to handle runtime input from CLI invocation (following ticket-classification pattern).

        With input_schema=, LangGraph passes the input_schema type directly to the first node!
        This is like ticket-classification: prepare_input(graph_input: GraphInput) -> GraphState

        Args:
            graph_input: Typed input matching input_schema (e.g., CalculatorInput)

        Returns:
            StateBaseClass with messages prepared from the input
        """
        if self.debug:
            print(f"[DEBUG] prepare_input_from_runtime received: {type(graph_input).__name__}")
            print(f"[DEBUG] Input data: {graph_input.model_dump()}")

        # Create new state with messages
        state = StateBaseClass()

        # Create initial messages using scenario prompts
        try:
            state.messages = [
                SystemMessage(content=self.scenario.system_prompt),
                HumanMessage(content=self.scenario.user_prompt.format_map(graph_input.model_dump()))
            ]
            if self.debug:
                print(f"[DEBUG] Created messages from runtime input (CLI mode)")
                print(f"[DEBUG] Messages: {[m.content[:50] for m in state.messages]}")
        except KeyError as e:
            # If format_map fails, fall back to basic setup
            state.messages = [
                SystemMessage(content=self.scenario.system_prompt),
                HumanMessage(content=f"Help me with: {graph_input}")
            ]
            if self.debug:
                print(f"[DEBUG] Format error {e} - using fallback")

        if self.debug:
            print(f"[DEBUG] Final {len(state.messages)} messages from runtime input")

        return state

    # Create a wrapper for end_execution that returns the proper output schema
    def end_execution_node(self, state: StateBaseClass) -> BaseModel:
        """Wrapper node that calls end_execution tool and returns output_schema."""
        self.scenario.end_execution_tool.run(state)
        # Return the output_schema populated from state.result
        return self.scenario.output_schema.model_validate(state.result)

    def build_cli_graph(self) -> StateGraph:
        """Build graph for CLI mode - accepts input at runtime (like ticket-classification example).

        Returns a graph that:
        - Uses input= and output= parameters (separate from State)
        - Dynamically creates a State class that includes input fields
        - Has a prepare_input node that converts input → State
        - Accepts typed input at runtime via `graph.ainvoke(CalculatorInput(...))`
        """
        # Dynamically create a State class that includes input fields
        # This is necessary because LangGraph needs the State to have the same fields as the input
        # (like ticket-classification where GraphState includes message, ticket_id, assignee from GraphInput)
        input_fields = {
            field_name: (field_info.annotation, field_info.default if field_info.is_required() else None)
            for field_name, field_info in self.scenario.input_schema.model_fields.items()
        }

        # Create a new State class that inherits from StateBaseClass and adds input fields
        StateWithInput = create_model(  # pyright: ignore[reportCallIssue]
            'StateWith' + self.scenario.input_schema.__name__,
            __base__=StateBaseClass,
            **input_fields  # pyright: ignore[reportArgumentType]
        )

        # CLI mode: input and output are separate types from State
        graph = StateGraph(StateWithInput, input_schema=self.scenario.input_schema, output_schema=self.scenario.output_schema)

        # prepare_input node converts GraphInput → GraphState
        graph.add_node("prepare_input", self.prepare_input_from_runtime)  # pyright: ignore

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
        graph.add_node("end_execution", self.end_execution_node)
        graph.add_node("raise_error", self.scenario.raise_error_tool)

        # Route through prepare_input first
        graph.add_edge(START, "prepare_input")
        graph.add_edge("prepare_input", "chatbot")

        graph.add_conditional_edges("chatbot", should_continue)
        graph.add_edge("tools", "chatbot")
        graph.add_edge("end_execution", END)
        graph.add_edge("raise_error", END)
        return graph

    def build_evaluation_graph(self, agent_input: Dict[str, Any]) -> StateGraph:
        """Build graph for evaluation mode - pre-binds input at build time.

        Args:
            agent_input: The input data from a datapoint

        Returns a graph that:
        - Pre-binds the input at graph build time
        - No runtime input needed - just call graph.ainvoke({})
        """
        # Evaluation mode: use input_schema for validation but pre-bind the data
        graph = StateGraph(StateBaseClass, input_schema=self.scenario.input_schema, output_schema=self.scenario.output_schema)

        final_agent_input = self.scenario.input_schema.model_validate(agent_input)

        # Create a closure that captures the input (avoids partial conflict with positional args)
        def prepare_with_bound_input(state: StateBaseClass | None = None) -> StateBaseClass:
            # Ignore incoming state, use pre-bound input instead
            return self.prepare_input_node(agent_input=final_agent_input, state=state)

        # Pre-bind input at build time via closure
        graph.add_node("prepare_input", prepare_with_bound_input)  # pyright: ignore

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
        graph.add_node("end_execution", self.end_execution_node)
        graph.add_node("raise_error", self.scenario.raise_error_tool)

        # Route through prepare_input first
        graph.add_edge(START, "prepare_input")
        graph.add_edge("prepare_input", "chatbot")

        graph.add_conditional_edges("chatbot", should_continue)
        graph.add_edge("tools", "chatbot")
        graph.add_edge("end_execution", END)
        graph.add_edge("raise_error", END)
        return graph
