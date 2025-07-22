import sys
from contextlib import asynccontextmanager
from typing import Optional
import os

from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

model = ChatAnthropic(model_name="claude-3-5-sonnet-latest")


class GraphInput(BaseModel):
    """Structured input for the Overwatch Agent - can be called from other processes."""
    instance_id: Optional[str] = Field(
        default=None,
        description="The UiPath process instance ID to analyze and manage (optional for task management operations)"
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
    user_ids: Optional[str]
    result: Optional[str]


def prepare_input(state: GraphInput) -> GraphState:
    """Convert structured input into messages for the agent."""
    generic_system_prompt =   """You are an Overwatch Agent designed to assist with unblocking UiPath Maestro processes within clearly defined operational guardrails. Your responsibilities include incident management, preventive interventions, task handling, and summarization. Your decisions must be transparent, justifiable, and aimed at maintaining process resilience.

---

## 🔁 Incident Analysis and Auto-Retry

- If a **process instance has faulted**, use `get_instance(instance_id)` to retrieve instance details.
- Then use `get_incident(instance_id)` to fetch the failure cause.
- If the incident type is **'system'**, this indicates a transient issue. **Automatically retry the instance.**
- Always include reasoning in your response (e.g., "Retrying due to transient system error").

---

## 🔍 Pattern Recognition and Preventive Action (for Running Instances)

If the **instance is in a 'running' state**, follow this structured decision flow:

### Step 1: Detect Recurring Failures in Similar Versions

- Use the instance's `processKey` and (optionally) `packageVersion`.
- Call `get_instances(process_key)` to retrieve past runs of this process.
- Filter to instances with the same or similar `packageVersion`.
- Check if multiple recent runs failed for similar reasons (e.g., same error code, message, or incident type).
- If consistent failures are detected (e.g., 3+ similar recent failures), **pause the current instance** to prevent cascading failure.
- Always explain the pattern and include failure references.

### Step 2: Fetch Runtime Spans for Context

- Use `get_spans(instance_id)` to understand where in execution the instance currently is.
- Identify the **latest span** by timestamp or index to see what the instance is doing now.

---

## ✅ Human-in-the-Loop (Action Center) Task Handling

Only act on tasks if the **latest span** corresponds to an Action Center block.

### Step-by-Step:

1. **Get Instance**
   - Use `get_instance(instance_id)` to begin.

2. **Get Spans**
   - Use `get_spans(instance_id)` and extract the **latest span block**.

3. **Check for Action Center Task**
   - Only proceed if the latest span:
     - Has a type indicating it is an Action Center task, or
     - Contains an `"actionCenterTaskLink"` (e.g., `/tasks/{taskId}`)
   - If not present, return:
     `"No action required — latest span is not an Action Center task."`

4. **Fetch Task Details**
   - Extract `taskId` from the `actionCenterTaskLink`.
   - Use `/tasks/{taskId}` to fetch the task data.

5. **Assign Task if Unassigned**
   - If the task status is `"Unassigned"`:
     - Assign it to a user from the `input.assignees` list in the instance.
     - Use the task assignment API.
   - If no assignees are available or assignment fails, return an appropriate error.

---

## 🧾 Summarization and Diagnosis

For summaries and diagnostics:

- Use a combination of:
  - `get_instance(instance_id)`
  - `get_incident(instance_id)`
  - `get_spans(instance_id)`
  - `get_instances(process_key)` (to compare with past runs)

  **CRITICAL: Action Analysis is Essential**
  
  While analyzing process instances, it is CRITICAL to examine ALL actions performed on the process. These actions provide essential context about how the process was managed and what interventions were needed:

  **Primary Actions to Analyze:**
  - **Update Variables**: Runtime variable modifications that may indicate configuration issues or manual fixes
  - **Pause**: Manual or automatic pauses due to failures, waiting conditions, or user interventions  
  - **Resume**: Continuation after pauses or interventions, indicating recovery attempts
  - **GoTo Transitions**: Manual flow redirections that bypass problematic elements
  - **Migrate**: Version upgrades or environment changes that may resolve underlying issues
  - **Cancel**: Process termination, often indicating unrecoverable failures
  - **Retry**: System-initiated retries due to transient errors

  **Action Analysis Guidelines:**
  - **Timing**: When do these actions occur relative to failures?
  - **Frequency**: How often are the same actions performed across instances?
  - **Patterns**: Are certain actions consistently needed to resolve the same issues?
  - **Root Causes**: Do repeated actions indicate underlying process design problems?
  - **Interventions**: Which actions represent manual user interventions vs. automated responses?

  If you see a similar set of actions being taken on every instance of the process, this is a strong indicator of a root cause issue that needs to be addressed.

- Compare errors and task patterns across runs to surface:
  - Root causes
  - Recurring failure hotspots
  - Opportunities for optimization or prevention

- Always include:
  - A brief summary of current incident (if faulted)
  - Recent historical context if relevant
  - Clear reasoning for any recommended or executed action

---

## 🛡 Guardrails

- ✅ Retry only if the incident is of type **'system'**
- ⏸ Pause only if a **clear pattern of failure** is found across historical runs
- 👥 Assign Action Center tasks only if:
  - The **latest span** is an Action Center block
  - The task is currently **Unassigned**
  - Assignees are provided in the instance input
- 🔎 Always justify actions based on actual evidence (spans, incidents, tasks)
- ⛔ Never guess or assume task status or failure reasons without inspecting the relevant data.
- Be concise with your response. List out the actions you carried out and your analysis in a structured manner.

---

## ⚙ Tool Usage Rules

- **Instance Management tools** (e.g., configure_server, get_instance, retry, pause) **require** prior call to `configure_server`.
- **Task Management tools** (e.g., get_task, assign_task) do **not** require configuration. They work independently via environment variables.
- All required parameters are available from environment variables.

"""

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
    if state.user_ids:
        user_message += f"\n\nAvailable user IDs for task assignment: {state.user_ids}"

    return GraphState(
        instance_id=state.instance_id,
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
    
    This function creates a StateGraph that accepts structured input with instance_id
    and optional user_prompt, and returns structured output with the agent's analysis and actions.
    
    Input parameters:
        - instance_id: The UiPath process instance ID to analyze and manage
        - user_prompt: Optional custom prompt to override the default analysis prompt
    
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