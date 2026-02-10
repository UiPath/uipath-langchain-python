from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command, Send
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_openai import ChatOpenAI  # or any other chat model
from typing import Any, TypedDict, Annotated
from operator import add
from uipath.core.chat.interrupt import UiPathConversationToolCallConfirmationValue
from uipath.platform.common import InvokeProcess


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to someone."""
    return f"Email sent to {to} with subject '{subject}'"


@tool
def schedule_meeting(title: str, attendees: str, duration_minutes: int) -> str:
    """Schedule a meeting with attendees."""
    return f"Meeting '{title}' scheduled with {attendees} for {duration_minutes} minutes"


class State(TypedDict):
    messages: Annotated[list, add]


# Setup LLM with tools
llm = ChatOpenAI(model="gpt-5.2", temperature=0)
llm_with_tools = llm.bind_tools([send_email, schedule_meeting])


def call_llm(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


async def call_email_tool_with_confirmation(state: State):
    """Handle send_email with conversation-based confirmation interrupt."""
    last_message: AIMessage = state["messages"][-1]

    results = []
    for tool_call in last_message.tool_calls:
        if tool_call["name"] != "send_email":
            continue

        response = interrupt(
            UiPathConversationToolCallConfirmationValue(
                toolCallId=tool_call["id"],
                toolName=tool_call["name"],
                inputSchema=send_email.args_schema.model_json_schema(),
                inputValue=tool_call["args"]
            )
        )

        if response["value"].get("approved"):
            tool_call_args = response["value"].get("input")
            result = send_email.invoke(tool_call_args or tool_call["args"])
        else:
            result = "Cancelled by user"

        # Return ToolMessage with matching ID
        results.append(
            ToolMessage(content=result, tool_call_id=tool_call["id"])
        )

    # Use Command to explicitly route to join node after interrupt completes
    return Command(update={"messages": results}, goto="join")


async def call_meeting_tool_with_job_interrupt(state: State):
    """Handle schedule_meeting with JOB-based interrupt."""
    last_message: AIMessage = state["messages"][-1]

    results = []
    for tool_call in last_message.tool_calls:
        if tool_call["name"] != "schedule_meeting":
            continue

        # Invoke a UiPath process and wait for it to complete
        #job_result = interrupt(
        #    InvokeProcess(
        #        name="Wait_Idle",  # Replace with your actual process name
        #        input_arguments={
        #            #"title": tool_call["args"]["title"],
        #            #"attendees": tool_call["args"]["attendees"],
        #            #"duration_minutes": tool_call["args"]["duration_minutes"]
        #        },
        #        process_folder_path='Shared/Coded Conversational Interrupt',  # Optional: specify folder path
        #        #process_folder_key=None,   # Optional: specify folder key
        #        attachments=None
        #    )
        #)

        # The job_result contains the output from the UiPath process
        # For demonstration, we'll use the meeting tool directly
        #if job_result:
        #    result = f"Meeting scheduled via UiPath process. Result: {job_result}"
        #else:
        #    result = schedule_meeting.invoke(tool_call["args"])

        response = interrupt(
            UiPathConversationToolCallConfirmationValue(
                toolCallId=tool_call["id"],
                toolName=tool_call["name"],
                inputSchema=send_email.args_schema.model_json_schema(),
                inputValue=tool_call["args"]
            )
        )

        if response["value"].get("approved"):
            tool_call_args = response["value"].get("input")
            result = schedule_meeting.invoke(tool_call_args or tool_call["args"])
        else:
            result = "Cancelled by user"

        # Return ToolMessage with matching ID
        results.append(
            ToolMessage(content=result, tool_call_id=tool_call["id"])
        )

    # Use Command to explicitly route to join node after interrupt completes
    return Command(update={"messages": results}, goto="join")


def fan_out_tools(state: State):
    """Fan out to tool handlers in parallel to create multiple interrupts.

    Returns a list of Send objects to execute tool handlers in parallel.
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        tool_names = [call["name"] for call in last_message.tool_calls]

        # Return a list of Send objects to execute nodes in parallel
        sends = []
        if "send_email" in tool_names:
            sends.append(Send("email_tool", state))
        if "schedule_meeting" in tool_names:
            sends.append(Send("meeting_tool", state))

        return sends if sends else "end"
    return "end"


def join_results(state: State):
    """Join node that passes through - allows parallel branches to converge."""
    return state


# Build graph with parallel tool branches using Send API
builder = StateGraph(State)
builder.add_node("llm", call_llm)
builder.add_node("email_tool", call_email_tool_with_confirmation)
builder.add_node("meeting_tool", call_meeting_tool_with_job_interrupt)
builder.add_node("join", join_results)

# Start with LLM
builder.add_edge(START, "llm")

# Use Send API for parallel execution
# When fan_out_tools returns a list of Send objects, both tools execute in parallel
# Each tool uses Command(goto="join") to route to the join node after interrupt completes
builder.add_conditional_edges("llm", fan_out_tools)

# From join, return to LLM for next iteration
builder.add_edge("join", "llm")

graph = builder.compile(checkpointer=MemorySaver())