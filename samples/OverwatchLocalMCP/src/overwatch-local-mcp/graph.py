import sys
import os
from contextlib import asynccontextmanager
from typing import Optional

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from system_prompts import OVERWATCH_SYSTEM_PROMPT

model = ChatAnthropic(model_name="claude-3-5-sonnet-latest")


class GraphInput(BaseModel):
    """Structured input for the Overwatch Agent - can be called from other processes."""
    instance_id: Optional[str] = Field(
        default=None,
        description="The UiPath process instance ID to analyze and manage (optional for task management operations)"
    )
    process_key: Optional[str] = Field(
        default=None,
        description="The UiPath process key to analyze and manage (optional for process-level operations)"
    )
    user_prompt: Optional[str] = Field(
        default=None,
        description="Custom user prompt to override the default analysis prompt"
    )
    user_ids: Optional[str] = Field(
        default=None,
        description="Comma-separated list of user IDs for task assignment operations (e.g., '123,456,789')"
    )


class GraphOutput(BaseModel):
    """Structured output from the Overwatch Agent."""
    result: str = Field(description="The result of the agent's analysis and actions")


class GraphState(MessagesState):
    """State for the Overwatch Agent graph."""
    instance_id: Optional[str]
    process_key: Optional[str]
    user_ids: Optional[str]
    result: Optional[str]


def prepare_input(state: GraphInput) -> GraphState:
    """Convert structured input into messages for the agent."""
    # Use the imported system prompt
    generic_system_prompt = OVERWATCH_SYSTEM_PROMPT

    # Use custom user prompt if provided, otherwise use default
    if state.user_prompt:
        if state.instance_id:
            user_message = f"Instance ID: {state.instance_id}\n\nUser Request: {state.user_prompt}"
        else:
            user_message = f"User Request: {state.user_prompt}"
    else:
        if state.instance_id:
            user_message = f"Analyze instance {state.instance_id} and take appropriate action based on these guidelines."
        else:
            user_message = "Please help me with UiPath operations. You can manage both process instances and tasks."
    
    # Add user_ids information if provided
    if state.process_key:
        user_message += f"\n\nProcess Key to analyze: {state.process_key}"
    
    if state.user_ids:
        user_message += f"\n\nAvailable user IDs for task assignment: {state.user_ids}"

    return GraphState(
        instance_id=state.instance_id,
        process_key=state.process_key,
        user_ids=state.user_ids,
        messages=[SystemMessage(content=generic_system_prompt), HumanMessage(content=user_message)],
        result=None
    )


async def execute_agent(state: GraphState) -> Command:
    """Execute the agent with the prepared state."""
    # Get environment variables for the MCP servers
    env_vars = {
        "UIPATH_ENVIRONMENT": os.getenv("UIPATH_ENVIRONMENT", "alpha"),
        "UIPATH_ORG_ID": os.getenv("UIPATH_ORGANIZATION_ID", ""),
        "UIPATH_TENANT": os.getenv("UIPATH_TENANT", "DefaultTenant"),
        "FOLDER_KEY": os.getenv("UIPATH_FOLDER_KEY", ""),
        "AUTH_TOKEN": os.getenv("AUTH_TOKEN", ""),
        "UIPATH_ACCESS_TOKEN": os.getenv("UIPATH_ACCESS_TOKEN", ""),
        "UIPATH_BASE_URL": os.getenv("UIPATH_BASE_URL", "alpha.uipath.com"),
    }
    
    client = MultiServerMCPClient({
        "instance-controller": {
            "command": sys.executable,
            "args": ["src/overwatch-local-mcp/instance_controller.py"],
            "transport": "stdio",
            "env": env_vars,
        },
        "task-controller": {
            "command": sys.executable,
            "args": ["src/overwatch-local-mcp/task_controller.py"],
            "transport": "stdio",
            "env": env_vars,
        },
     })
    agent = create_react_agent(model, await client.get_tools())
    result = await agent.ainvoke(state)
    
    return Command(
        update={
            "result": result["messages"][-1].content
        }
    )


def return_result(state: GraphState) -> GraphOutput:
    """Return the final result."""
    return GraphOutput(
        result=state["result"] or "No result available"
    )


def create_overwatch_agent():
    """Create the Overwatch Agent with structured input/output support.
    
    This function creates a StateGraph that accepts structured input with instance_id,
    process_key, user_prompt, and user_ids, and returns structured output with the agent's analysis and actions.
    
    Input parameters:
        - instance_id: The UiPath process instance ID to analyze and manage (optional for process-level operations)
        - process_key: The UiPath process key to analyze and manage (optional for process-level operations)
        - user_prompt: Optional custom prompt to override the default analysis prompt
        - user_ids: Optional comma-separated list of user IDs for task assignment operations
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Build the state graph
    builder = StateGraph(GraphState, input=GraphInput, output=GraphOutput)
    builder.add_node("prepare_input", prepare_input)
    builder.add_node("execute_agent", execute_agent)
    builder.add_node("return_result", return_result)

    builder.add_edge(START, "prepare_input")
    builder.add_edge("prepare_input", "execute_agent")
    builder.add_edge("execute_agent", "return_result")
    builder.add_edge("return_result", END)

    # Compile and return the graph
    return builder.compile()


# Create the graph instance
graph = create_overwatch_agent()


@asynccontextmanager
async def make_graph():
    """Create and yield the Overwatch Agent for use with langgraph.json."""
    # Get environment variables for the MCP servers
    env_vars = {
        "UIPATH_ENVIRONMENT": os.getenv("UIPATH_ENVIRONMENT", "alpha"),
        "UIPATH_ORG_ID": os.getenv("UIPATH_ORG_ID", ""),
        "UIPATH_TENANT": os.getenv("UIPATH_TENANT", ""),
        "FOLDER_KEY": os.getenv("FOLDER_KEY", ""),
        "AUTH_TOKEN": os.getenv("AUTH_TOKEN", ""),
        "UIPATH_ACCESS_TOKEN": os.getenv("UIPATH_ACCESS_TOKEN", ""),
        "UIPATH_BASE_URL": os.getenv("UIPATH_BASE_URL", "alpha.uipath.com"),
    }
    
    client = MultiServerMCPClient({
        "instance-controller": {
            "command": sys.executable,
            "args": ["src/overwatch-local-mcp/instance_controller.py"],
            "transport": "stdio",
            "env": env_vars,
        },
        "task-controller": {
            "command": sys.executable,
            "args": ["src/overwatch-local-mcp/task_controller.py"],
            "transport": "stdio",
            "env": env_vars,
        },
    })
    agent = create_react_agent(model, await client.get_tools())
    yield agent

# import dotenv
# import os
# import sys
# import traceback
# from contextlib import asynccontextmanager
# from typing import Optional
# from pydantic import BaseModel, Field
# from langgraph.prebuilt import create_react_agent
# from langchain_anthropic import ChatAnthropic
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_mcp_adapters.client import MultiServerMCPClient

# dotenv.load_dotenv()

# @asynccontextmanager
# async def make_graph():
#     try:
#         # Create MCP client with stdio transport for local server
#         client = MultiServerMCPClient({
#             "instance-controller": {
#                 "command": sys.executable,
#                 "args": ["src/overwatch-local-mcp/instance_controller.py"],
#                 "transport": "stdio",
#             },
#         })
        
#         # Load tools from the local MCP server
#         tools = await client.get_tools()
#         print(f"✅ Loaded {len(tools)} tools from local MCP server")
        
#         # Create the model
#         model = ChatAnthropic(model="claude-3-5-sonnet-latest")
        
#         # Create agent with tools
#         agent = create_react_agent(model, tools=tools)
        
#         print("✅ Agent created successfully with local MCP server")
#         yield agent
        
#     except Exception as e:
#         print("❌ An unexpected error occurred in make_graph.")
#         traceback.print_exc()
#         raise 