from typing import Any, Callable, Dict, List, Literal

from langchain.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, create_model
from functools import partial
from langchain_core.runnables import Runnable
from .tools import EndExecutionTool, RaiseErrorTool, StateBaseClass


class Datapoint(BaseModel):
    name: str = Field(description="The name of the datapoint")
    input: Dict[str, str | Dict[str, Any]]
    evaluation_criteria: Dict[str, Any]
    simulation_instructions: str = ""


class AgentBaseClass(BaseModel):
    system_prompt: str
    user_prompt: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    datapoints: List[Datapoint] = []
    raise_error_tool: RaiseErrorTool = RaiseErrorTool()
    tools: List[BaseTool] = []

    @property
    def end_execution_tool(self) -> EndExecutionTool:
        return EndExecutionTool(args_schema=self.output_schema, output_schema=self.output_schema)


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

    def prepare_input(self, state: BaseModel) -> StateBaseClass:
        """Node to handle input preparation - converts typed input to state with messages.

        Used by both CLI mode (direct binding) and evaluation mode (lambda binding).
        With input_schema=, LangGraph passes the typed input directly to this node.

        Args:
            state: The typed input matching input_schema (e.g., CalculatorInput).
                   Despite being named 'state', this receives the input object, not a state.

        Returns:
            StateBaseClass with messages prepared from the input
        """
        if self.debug:
            print(f"[DEBUG] prepare_input received: {type(state).__name__}")
            print(f"[DEBUG] Input data: {state.model_dump()}")

        # Create new state with messages from the input
        new_state = StateBaseClass()

        # Create initial messages using scenario prompts
        try:
            new_state.messages = [
                SystemMessage(content=self.scenario.system_prompt),
                HumanMessage(content=self.scenario.user_prompt.format_map(state.model_dump()))
            ]
            if self.debug:
                print(f"[DEBUG] Created messages from input")
                print(f"[DEBUG] Messages: {[m.content[:50] for m in new_state.messages]}")
        except KeyError as e:
            # If format_map fails, fall back to basic setup
            new_state.messages = [
                SystemMessage(content=self.scenario.system_prompt),
                HumanMessage(content=f"Help me with: {state}")
            ]
            if self.debug:
                print(f"[DEBUG] Format error {e} - using fallback")

        if self.debug:
            print(f"[DEBUG] Final {len(new_state.messages)} messages")

        return new_state

    def end_execution_node(self, state: StateBaseClass) -> BaseModel:
        """Wrapper node that calls end_execution tool and returns output_schema."""
        self.scenario.end_execution_tool.run(state)
        # Return the output_schema populated from state.result
        return self.scenario.output_schema.model_validate(state.result)

    def _create_state_with_input(self) -> type[BaseModel]:
        """Create a State class that includes input fields from input_schema.

        This is necessary for CLI mode where LangGraph needs the State to have
        the same fields as the input (like ticket-classification pattern).

        Returns:
            A new State class that inherits from StateBaseClass and includes input fields
        """
        # Extract fields from input_schema and convert to (type, Field) tuples
        input_fields: Dict[str, Any] = {}
        for field_name, field_info in self.scenario.input_schema.model_fields.items():
            # Get the annotation, defaulting to Any if not present
            field_type = field_info.annotation if field_info.annotation is not None else Any

            # Copy the field metadata if using Field(), otherwise create a simple tuple
            if field_info.is_required():
                # Required field: use (type, ...)
                input_fields[field_name] = (field_type, ...)
            else:
                # Optional field: use (type, default_value)
                input_fields[field_name] = (field_type, field_info.default)

        # Create a new State class that inherits from StateBaseClass and adds input fields
        StateWithInput = create_model(
            'StateWith' + self.scenario.input_schema.__name__,
            __base__=StateBaseClass,
            **input_fields
        )

        return StateWithInput

    def add_worker_graph(
        self,
        graph: StateGraph,
    ) -> None:
        """Add all worker nodes and edges to the graph.

        This method contains the common graph structure used by both CLI and evaluation modes.
        Override this method to customize the agent loop behavior.

        Args:
            graph: The StateGraph to add nodes and edges to
        """

        # Add worker nodes
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

        # Add edges
        graph.add_edge(START, "prepare_input")
        graph.add_edge("prepare_input", "chatbot")
        graph.add_conditional_edges("chatbot", should_continue)
        graph.add_edge("tools", "chatbot")
        graph.add_edge("end_execution", END)
        graph.add_edge("raise_error", END)

    def build_cli_graph(self) -> StateGraph:
        """Build graph for CLI mode - accepts input at runtime (like ticket-classification example).

        Returns a graph that:
        - Uses input= and output= parameters (separate from State)
        - Dynamically creates a State class that includes input fields
        - Has a prepare_input node that converts input → State
        - Accepts typed input at runtime via `graph.ainvoke(CalculatorInput(...))`
        """
        # Create State class with input fields for CLI mode
        StateWithInput = self._create_state_with_input()

        # CLI mode: input and output are separate types from State
        graph = StateGraph(
            StateWithInput,
            input_schema=self.scenario.input_schema,
            output_schema=self.scenario.output_schema
        )
        graph.add_node("prepare_input", self.prepare_input)

        # Add all worker nodes and edges
        self.add_worker_graph(graph)

        return graph

    def build_evaluation_graph(self, agent_input: Dict[str, Any]) -> StateGraph:
        """Build graph for evaluation mode - pre-binds input at build time.

        Args:
            agent_input: The input data from a datapoint

        Returns a graph that:
        - Pre-binds the input at graph build time
        - No runtime input needed - just call graph.ainvoke({})
        """
        # Validate input data against schema
        final_agent_input = self.scenario.input_schema.model_validate(agent_input)

        # Evaluation mode: use input_schema for validation but pre-bind the data
        graph = StateGraph(
            StateBaseClass,
            input_schema=self.scenario.input_schema,
            output_schema=self.scenario.output_schema
        )

        # Lambda captures the pre-bound input and ignores incoming state from LangGraph
        graph.add_node("prepare_input", lambda state=None: self.prepare_input(final_agent_input))

        # Add all worker nodes and edges
        self.add_worker_graph(graph)

        return graph
