from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from uipath.models import CreateAction
from uipath_langchain.chat import UiPathChat
from langchain_core.messages import SystemMessage, HumanMessage
from uipath_langchain.retrievers import ContextGroundingRetriever
from uipath import UiPath
from typing import Dict, Any
from dotenv import load_dotenv
from datetime import datetime
from contextlib import asynccontextmanager
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
import json, os, logging, ast

load_dotenv()

logging.basicConfig(level=logging.INFO)

# Use UiPathChat for making LLM calls
llm = UiPathChat(model="gpt-4o-2024-08-06")

uipath_client = UiPath()

# ---------------- MCP Server Configuration ----------------
@asynccontextmanager
async def get_mcp_session():
    """MCP session management"""
    MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")
    
    if hasattr(uipath_client, 'api_client'):
                if hasattr(uipath_client.api_client, 'default_headers'):
                    auth_header = uipath_client.api_client.default_headers.get('Authorization', '')
                    if auth_header.startswith('Bearer '):
                        UIPATH_ACCESS_TOKEN = auth_header.replace('Bearer ', '')
                        logging.info("Retrieved token from UiPath API client")
    
    async with streamablehttp_client(
        url=MCP_SERVER_URL,
        headers={"Authorization": f"Bearer {UIPATH_ACCESS_TOKEN}"} if UIPATH_ACCESS_TOKEN else {},
        timeout=60,
    ) as (read, write, session_id_callback):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session

async def get_mcp_tools():
    """Load MCP tools for use with agents"""
    async with get_mcp_session() as session:
        tools = await load_mcp_tools(session)
        return tools

# Initialize Context Grounding for company policy
context_grounding = ContextGroundingRetriever(
    index_name="company-policy-index",
    folder_path="Shared",
    number_of_results=1
    )

# ---------------- State ----------------
class GraphState(BaseModel):
    """Enhanced state to track the complete leave request workflow"""
    leave_request: str | None = None
    employee_email: str | None = None
    employee_id: str | None = None
    employee_name: str | None = None
    leave_start: str | None = None
    leave_end: str | None = None
    leave_reason: str | None = None
    leave_category: str | None = None
    leave_days_requested: int | None = None
    available_leave_balance: Dict[str, int] | None = None
    policy_compliant: bool | None = None
    policy_violations: list | None = []
    applicable_policies: list | None = []
    hr_approval_required: bool = False
    hr_approved: bool | None = None
    hr_comments: str | None = None
    final_status: str | None = None  # "approved", "rejected"
    
    # Control flags
    hitl_required: bool = False
    validation_complete: bool = False


# ---------------- Helper Functions ----------------
async def check_company_policy_with_context(state: GraphState) -> Dict[str, Any]:
    """Check if leave request complies with company policy using Context Grounding"""
    
    # Default return value
    default_result = {
        "compliant": True,
        "violations": [],
        "applicable_policies": [],
        "requires_hr": state.leave_days_requested > 4
    }
    
    try:
        # Your existing policy check logic...
        policy_query = f"""What is the company policy on
        Leave type: {state.leave_category}
        Duration: {state.leave_days_requested} days
        Reason: {state.leave_reason}
        """
        
        # Try to get policy context
        try:
            policy_context = context_grounding.invoke(policy_query)
            logging.info(f"DEBUG: Retrieved {len(policy_context) if policy_context else 0} policy documents")
        except Exception as e:
            logging.warning(f"WARNING: Context grounding failed: {e}")
            return default_result
        
        if policy_context:
            # Process documents...
            applicable_policies = []
            for doc in policy_context:
                policy_text = doc.page_content
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page_number', '1')
                formatted_policy = f"Source: {source} (Page {page})\n{policy_text}"
                applicable_policies.append(formatted_policy)
            
            # LLM analysis...
            try:
                policy_check_prompt = f"""
                Based on the following company policies, check if this leave request is compliant:
                
                Leave Request:
                - Type: {state.leave_category}
                - Duration: {state.leave_days_requested} days
                - Dates: {state.leave_start} to {state.leave_end}
                - Reason: {state.leave_reason}
                
                Relevant Policies:
                {chr(10).join(applicable_policies)}
                
                Return output in this format but don't explicitly convert or return a json:
                {{
                    "compliant": true,
                    "violations": [],
                    "requires_hr": false
                }}
                """
                
                response = await llm.ainvoke([
                    SystemMessage("You are a policy compliance checker. Analyze the policies and return only JSON."),
                    HumanMessage(policy_check_prompt)
                ])
                
                result = json.loads(response.content)
                result["applicable_policies"] = applicable_policies
                return result
                
            except Exception as e:
                logging.warning(f"WARNING: LLM policy analysis failed: {e}")
                default_result["applicable_policies"] = applicable_policies
                return default_result
        else:
            logging.warning("WARNING: No policy documents found")
            return default_result
            
    except Exception as e:
        logging.error(f"ERROR: Policy check completely failed: {e}")
        return default_result

def calculate_leave_days(start: str, end: str) -> int:
    """Calculate number of leave days requested"""
    try:
        start_date = datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")
        return (end_date - start_date).days + 1
    except:
        return 1
    
def default_leave_balance():
    """Default leave balance fallback"""
    return {
        "Vacation": 20,
        "Sick Leave": 12,
        "Maternity/Paternity": 90,
        "Bereavement": 5,
        "Personal Leave": 3
    }

# ---------------- Tool Operations via MCP ----------------
async def get_employee_leave_balance_mcp(employee_email: str) -> Dict[str, int]:
    """Get leave balance using proper MCP tools"""
    
    try:
        async with get_mcp_session() as session:
            tools = await load_mcp_tools(session)
            
            # Find the database query tool
            GetLeaveBalance_tool = next((tool for tool in tools if "getleavebalance" in tool.name.lower()), None)
            if not GetLeaveBalance_tool:
                logging.warning("WARNING: Database query tool not found in MCP server")
                return default_leave_balance()
            
            try:
                result = await GetLeaveBalance_tool.ainvoke({
                    "employee_email": employee_email,
                })
                
                balance_dict = {}

                # Convert string to dict
                result = ast.literal_eval(result) if result else None

                if result and isinstance(result, dict) and "leave_balance" in result:
                    for key, value in result["leave_balance"].items():
                        balance_dict[key] = value

                    return balance_dict if balance_dict else default_leave_balance()
                else:
                    return default_leave_balance()
                    
            except Exception as e:
                logging.warning(f"WARNING: MCP database query failed: {e}")
                return default_leave_balance()
                
    except Exception as e:
        logging.warning(f"WARNING: MCP session creation failed: {e}")
        return default_leave_balance()

async def update_leave_balance_mcp(employee_email: str, leave_type: str, days: int):
    """Update leave balance using proper MCP tools"""
    async with get_mcp_session() as session:
        tools = await load_mcp_tools(session)
        
        # Find the database update tool
        update_tool = next((tool for tool in tools if "updateleavebalance" in tool.name.lower()), None)
        if not update_tool:
            logging.error("Database update tool not found in MCP server")
            raise Exception("Database update tool not available")
        
        try:
            await update_tool.ainvoke({
                "employee_email": employee_email,
                "leave_type": leave_type,
                "days": days
            })
            logging.info(f"Updated DB via MCP: Deducted {days} days of {leave_type} for {employee_email}")
            
        except Exception as e:
            logging.error(f"Error updating leave balance via MCP: {e}")
            raise

async def get_employee_details_mcp(employee_email: str):
    """Update employee details using proper MCP tools"""
    async with get_mcp_session() as session:
        tools = await load_mcp_tools(session)
        
        # Find the database update tool
        getemployeedetails_tool = next((tool for tool in tools if "getemployeedetails" in tool.name.lower()), None)
        if not getemployeedetails_tool:
            logging.error("Get Employee Details update tool not found in MCP server")
            raise Exception("Get Employee Details update tool not available")
        
        try:
            result = await getemployeedetails_tool.ainvoke({
                "employee_email": employee_email
            })
            logging.info(f"Retrieved employee details via MCP")

            employee_details_dict = {}

            # Convert string to dict (safe for Python dict-style strings)
            result = ast.literal_eval(result) if result else None

            if result and isinstance(result, dict) and "employee_details" in result:
                for key, value in result["employee_details"].items():
                    employee_details_dict[key] = value

                return employee_details_dict if employee_details_dict else None
            else:
                return None
            
        except Exception as e:
            logging.error(f"Error retrieving employee details via MCP: {e}")
            raise

async def send_email_mcp(to: str, subject: str, body: str):
    """Send email using proper MCP tools"""
    async with get_mcp_session() as session:
        tools = await load_mcp_tools(session)
        
        # Find the email tool
        email_tool = next((tool for tool in tools if "sendemail" in tool.name.lower()), None)
        if not email_tool:
            logging.error("Email tool not found in MCP server")
            raise Exception("Email tool not available")
        
        try:
            await email_tool.ainvoke({
                "Recipient": to,
                "Subject": subject,
                "Body": body
            })
            logging.info(f"Email sent via MCP to {to}")
            
        except Exception as e:
            logging.error(f"Error sending email via MCP: {e}")
            raise


# ---------------- Nodes ----------------
async def start_node(state: GraphState) -> GraphState:
    """Extract leave information from the request"""
    system_prompt = """You are a smart HR assistant tasked with extracting leave information from a user's message. 

    Your goal is to extract the following three fields:
    1. leave_start - the start date of the leave (try to parse informal dates like "next Monday", "Sep 25", etc. into YYYY-MM-DD if possible; otherwise return null)
    2. leave_end - the end date of the leave (same rules as leave_start)
    3. leave_reason - the reason for the leave

    Instructions:
    - Only return a JSON object with keys: leave_start, leave_end, leave_reason.
    - If a field cannot be determined, return null.
    - If dates are vague (like "next Monday"), try to infer the actual date relative to today, but if not possible, return null.
    - Only output the JSON. Do not include any explanations, commentary, or extra text.
    - Extract the most relevant reason from the user's message.

    Examples:

    User message: "I want to take leave from 2025-09-25 to 2025-09-27 for attending a wedding."
    Output:
    {
    "leave_start": "2025-09-25",
    "leave_end": "2025-09-27",
    "leave_reason": "attending a wedding"
    }

    User message: "I need a day off next Monday."
    Output:
    {
    "leave_start": "2025-09-30",
    "leave_end": "2025-09-30",
    "leave_reason": "a day off"
    }

    User message: "I want leave for a family function."
    Output:
    {
    "leave_start": null,
    "leave_end": null,
    "leave_reason": "family function"
    }
    """

    output = await llm.ainvoke(
        [SystemMessage(system_prompt),
         HumanMessage(state.leave_request)]
    )

    leave_data = json.loads(output.content)
    
    # Calculate leave days if dates are available
    leave_days = None
    if leave_data.get("leave_start") and leave_data.get("leave_end"):
        leave_days = calculate_leave_days(leave_data["leave_start"], leave_data["leave_end"])

    return state.model_copy(update={
        "leave_start": leave_data.get("leave_start"),
        "leave_end": leave_data.get("leave_end"),
        "leave_reason": leave_data.get("leave_reason"),
        "leave_days_requested": leave_days
    })


async def get_employee_details_node(state: GraphState) -> GraphState:
    """Get employee details via MCP integration"""

    employee_details = await get_employee_details_mcp(
        state.employee_email
    )
    
    return state.model_copy(update={
        "employee_id": employee_details['Employee ID'] or None,
        "employee_name": employee_details['Employee Name'] or None
    })


async def check_fields_node(state: GraphState) -> GraphState:
    """Check if all required fields are present"""
    hitl_required = not state.leave_start or not state.leave_end or not state.leave_reason
    return state.model_copy(update={"hitl_required": hitl_required})


async def employee_hitl_node(state: GraphState) -> Command:
    """Ask employee to fill missing details"""
    action_data = interrupt(
        CreateAction(
            app_name="LeaveRequestApp",
            title="Please fill missing leave details",
            data={
                "LeaveStart": state.leave_start or "",
                "LeaveEnd": state.leave_end or "",
                "LeaveReason": state.leave_reason or ""
            },
            app_version=1,
            app_folder_path="Shared"
        )
    )

    updates = {
        "leave_start": action_data.get("LeaveStart", state.leave_start),
        "leave_end": action_data.get("LeaveEnd", state.leave_end),
        "leave_reason": action_data.get("LeaveReason", state.leave_reason),
        "leave_days_requested": calculate_leave_days(
            action_data.get("LeaveStart", state.leave_start),
            action_data.get("LeaveEnd", state.leave_end)
        )
    }

    return Command(update=updates)


async def categorize_node(state: GraphState) -> GraphState:
    """Categorize the leave type based on the request reason"""
    system_prompt = """You are a leave categorization expert. Categorize the leave request into ONE of these exact categories based on the reason provided:

        1. Vacation - Personal time off for rest, travel, recreation, or personal activities
        2. Sick Leave - Personal illness, medical appointments, mental health, or caring for sick immediate family
        3. Maternity/Paternity - Pregnancy-related leave, childbirth, adoption, or bonding with new child
        4. Bereavement - Death of family member, friend, or attending funeral services
        5. Personal Leave - Personal matters not covered by other categories (legal issues, home emergencies, personal business)

        Guidelines for categorization:
        - Wedding attendance (own or family) = Vacation
        - Doctor appointments = Sick Leave  
        - Caring for sick child/parent = Sick Leave
        - Funeral attendance = Bereavement
        - Home repairs/emergencies = Personal Leave
        - Mental health days = Sick Leave

        Return ONLY the exact category name from the list above. Do not add explanations or additional text."""

    # Create the leave reason context for better categorization
    leave_context = f"Leave reason: {state.leave_reason}"
    if state.leave_start and state.leave_end:
        leave_context += f"\nDates: {state.leave_start} to {state.leave_end}"
    if state.leave_days_requested:
        leave_context += f"\nDuration: {state.leave_days_requested} days"

    output = await llm.ainvoke(
        [SystemMessage(system_prompt),
         HumanMessage(leave_context)]
    )

    # Clean the output and ensure it matches one of our categories
    category = output.content.strip()
    
    # Validate the category matches our expected types
    valid_categories = ["Vacation", "Sick Leave", "Maternity/Paternity", "Bereavement", "Personal Leave"]
    
    if category not in valid_categories:
        # Log the unexpected category and default to Personal Leave
        logging.warning(f"Unexpected category '{category}' returned, defaulting to 'Personal Leave'")
        category = "Personal Leave"

    return state.model_copy(update={"leave_category": category})


async def check_policy_node(state: GraphState) -> GraphState:
    """Check company policy using Context Grounding and leave availability via MCP"""
    
    # Initialize default values
    policy_check = {
        "compliant": True,
        "violations": [],
        "applicable_policies": [],
        "requires_hr": False
    }
    
    balance = default_leave_balance()  # Ensure we always have a balance
    
    try:
        # Check policy compliance using Context Grounding
        policy_result = await check_company_policy_with_context(state)
        if policy_result:
            policy_check = policy_result
        else:
            logging.warning("DEBUG: Policy check returned None, using defaults")
        
    except Exception as e:
        logging.error(f"ERROR: Policy check failed: {e}")
        # Use default policy_check values
    
    try:
        # Get available leave balance via MCP server
        balance_result = await get_employee_leave_balance_mcp(state.employee_email)
        if balance_result:
            balance = balance_result
        else:
            logging.warning("DEBUG: Balance check returned None, using defaults")
            
    except Exception as e:
        logging.error(f"ERROR: Balance check failed: {e}")
        # Use default balance
    
    # Ensure we have valid data
    if not isinstance(balance, dict):
        balance = default_leave_balance()
    
    if not isinstance(policy_check, dict):
        policy_check = {
            "compliant": True,
            "violations": [],
            "applicable_policies": [],
            "requires_hr": False
        }
    
    # Check if employee has sufficient leave balance
    if state.leave_category in balance:
        available = balance[state.leave_category]
        if state.leave_days_requested > available:
            policy_check["violations"].append(
                f"Insufficient {state.leave_category} balance. Available: {available} days"
            )
            policy_check["compliant"] = False
    else:
        # Add the category with default value
        balance[state.leave_category] = 0
        policy_check["violations"].append(f"No {state.leave_category} balance available")
        policy_check["compliant"] = False
    
    # Determine if HR approval is needed based on policy and other factors
    hr_required = (
        policy_check.get("requires_hr", False) or
        state.leave_days_requested > 5 or
        state.leave_category == "Maternity/Paternity" or
        not policy_check.get("compliant", True)
    )
    
    # Create the updated state
    updated_state = state.model_copy(update={
        "policy_compliant": policy_check.get("compliant", True),
        "policy_violations": policy_check.get("violations", []),
        "applicable_policies": policy_check.get("applicable_policies", []),
        "available_leave_balance": balance,
        "hr_approval_required": hr_required,
        "validation_complete": True
    })
    
    return updated_state


async def check_leave_availability_node(state: GraphState) -> GraphState:
    """Check if the detected leave type is available for the employee"""
    if not state.available_leave_balance:
        return state
    
    leave_available = False
    if state.leave_category in state.available_leave_balance:
        available_days = state.available_leave_balance[state.leave_category]
        leave_available = available_days >= state.leave_days_requested
    
    if not leave_available and "Insufficient" not in str(state.policy_violations):
        violations = state.policy_violations or []
        violations.append(f"Insufficient {state.leave_category} balance")
        return state.model_copy(update={
            "policy_violations": violations,
            "policy_compliant": False
        })
    
    return state


async def hr_approval_node(state: GraphState) -> Command:
    """Send to HR for approval"""
    violation_text = "\n".join(state.policy_violations) if state.policy_violations else "None"
    policy_text = "\n".join(state.applicable_policies[:2]) if state.applicable_policies else "Standard policies apply"
    
    action_data = interrupt(
        CreateAction(
            app_name="HRApprovalApp",
            title="Leave Request Needs HR Approval",
            data={
                "Employee": f"{state.employee_name} (ID: {state.employee_id})",
                "LeaveType": f"{state.leave_category} (Current Balance - {state.available_leave_balance.get(state.leave_category, 0)})",
                "Period": f"{state.leave_start} to {state.leave_end} ({state.leave_days_requested} days)",
                "Reason": state.leave_reason,
                "PolicyViolations": violation_text,
                "ApplicablePolicies": policy_text
            },
            app_version=2,
            app_folder_path="Shared"
        )
    )

    hr_approved = action_data.get("Approval").lower() == "approved"
    
    return Command(update={
        "hr_approved": hr_approved,
        "hr_comments": action_data.get("Comments", ""),
        "final_status": "approved" if hr_approved else "rejected"
    })


async def update_database_node(state: GraphState) -> GraphState:
    """Updated database node with proper MCP integration"""
    if state.final_status == "approved" or state.policy_compliant:
        await update_leave_balance_mcp(
            state.employee_email,
            state.leave_category,
            state.leave_days_requested
        )
    
    return state


async def send_approval_email_node(state: GraphState) -> GraphState:
    """Send approval email with proper MCP integration"""
    email_subject = f"Leave Request Approved - {state.leave_category}"
    email_body = f"""
        <p>Dear {state.employee_name},</p>

        <p>Your leave request has been <b>approved</b>.</p>

        <p><b>Details:</b><br>
        - Leave Type: {state.leave_category}<br>
        - Start Date: {state.leave_start}<br>
        - End Date: {state.leave_end}<br>
        - Duration: {state.leave_days_requested} days<br>
        - Reason: {state.leave_reason}
        </p>

        <p><b>HR Comments:</b> {state.hr_comments or 'N/A'}</p>

        <p>Your remaining balance for {state.leave_category}: {
            state.available_leave_balance.get(state.leave_category, 0) - state.leave_days_requested
        } days</p>

        <p>Best regards,<br>HR Team</p>
        """
    
    await send_email_mcp(
        to=state.employee_email,  # Fixed: was sender_email
        subject=email_subject,
        body=email_body
    )
    
    return state.model_copy(update={"final_status": "completed"})


async def send_rejection_email_node(state: GraphState) -> GraphState:
    """Send rejection email with proper MCP integration"""
    email_subject = f"Leave Request Rejected - {state.leave_category}"
    
    rejection_reasons = []
    if state.policy_violations:
        rejection_reasons.extend(state.policy_violations)
    if state.hr_comments:
        rejection_reasons.append(f"HR Comments: {state.hr_comments}")
    
    email_body = f"""
        <p>Dear {state.employee_name},</p>

        <p>Your leave request has been <b>rejected</b>.</p>

        <p><b>Details:</b><br>
        - Leave Type: {state.leave_category}<br>
        - Requested Dates: {state.leave_start} to {state.leave_end}<br>
        - Duration: {state.leave_days_requested} days
        </p>

        <p><b>Reason(s) for rejection:</b></p>
        <ul>
            {''.join(f'<li>{reason}</li>' for reason in rejection_reasons)}
        </ul>

        <p>Please contact HR for more information or to discuss alternative arrangements.</p>

        <p>Best regards,<br>HR Team</p>
        """
    
    await send_email_mcp(
        to=state.employee_email,  # Fixed: was sender_email
        subject=email_subject,
        body=email_body
    )
    
    return state.model_copy(update={"final_status": "completed"})


def end_node(state: GraphState) -> GraphState:
    """Final node to log the completion"""
    logging.info(f"Leave request processing completed. Status: {state.final_status}")
    return state


# ---------------- Condition Functions ----------------
def should_go_to_employee_hitl(state: GraphState):
    """Check if employee HITL is needed"""
    return "hitl_needed" if state.hitl_required else "hitl_not_needed"

def should_go_to_hr(state: GraphState):
    """Check if HR approval is needed"""
    if state.hr_approval_required:
        return "needs_hr_approval"
    elif state.policy_compliant:
        return "auto_approved"
    else:
        return "auto_rejected"

def hr_decision(state: GraphState):
    """Route based on HR decision"""
    return "approved" if state.hr_approved else "rejected"


# ---------------- Build Graph ----------------
graph = StateGraph(GraphState)

# Add all nodes
graph.add_node("start", start_node)
graph.add_node("get_employee_details", get_employee_details_node)
graph.add_node("check_fields", check_fields_node)
graph.add_node("employee_hitl", employee_hitl_node)
graph.add_node("categorize", categorize_node)
graph.add_node("check_policy", check_policy_node)
graph.add_node("check_availability", check_leave_availability_node)
graph.add_node("hr_approval", hr_approval_node)
graph.add_node("update_database", update_database_node)
graph.add_node("send_approval_email", send_approval_email_node)
graph.add_node("send_rejection_email", send_rejection_email_node)
graph.add_node("end", end_node)

# Set entry point
graph.set_entry_point("start")

# Add edges
graph.add_edge("start", "get_employee_details")
graph.add_edge("get_employee_details", "check_fields")

# Employee HITL routing
graph.add_conditional_edges(
    "check_fields",
    should_go_to_employee_hitl,
    {
        "hitl_needed": "employee_hitl",
        "hitl_not_needed": "categorize"
    }
)
graph.add_edge("employee_hitl", "check_fields")

# Continue flow
graph.add_edge("categorize", "check_policy")
graph.add_edge("check_policy", "check_availability")

# HR approval routing
graph.add_conditional_edges(
    "check_availability",
    should_go_to_hr,
    {
        "needs_hr_approval": "hr_approval",
        "auto_approved": "update_database",
        "auto_rejected": "send_rejection_email"
    }
)

# HR decision routing
graph.add_conditional_edges(
    "hr_approval",
    hr_decision,
    {
        "approved": "update_database",
        "rejected": "send_rejection_email"
    }
)

# Final steps
graph.add_edge("update_database", "send_approval_email")
graph.add_edge("send_approval_email", "end")
graph.add_edge("send_rejection_email", "end")
graph.add_edge("end", END)

# Compile the graph
agent = graph.compile()
