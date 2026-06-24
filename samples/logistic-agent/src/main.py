import enum
import os
from typing import Literal, Any
from datetime import datetime, timedelta, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from uipath_langchain.chat import UiPathChat
from pydantic import BaseModel, Field, ConfigDict

from src.entities import Entity
from src.models import Company, Order, OrderStatus
from src.middlewares import DisableParallelToolCallsMiddleware
from src.prompts import classifier_prompt, complaint_handler_prompt, new_order_handler_prompt_good, discount_calculator_prompt
from uipath.platform import UiPath
from uipath.platform.common import CreateTask
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from uipath_langchain.retrievers import ContextGroundingRetriever
from uipath.tracing import traced
from uipath.eval.mocks import mockable


uipath = UiPath()
llm = UiPathChat(model="gpt-4o-mini-2024-07-18")


class EmailCategory(str, enum.Enum):
    NEW_ORDER = "NEW_ORDER"
    LOGISTIC_REQUEST = "LOGISTIC_REQUEST"
    COMPLAINT = "COMPLAINT"
    UNKNOWN = "UNKNOWN"

class ModelSchema(BaseModel):
    email_category: EmailCategory = Field(description="Email Category")

class Input(BaseModel):
    email_address: str = Field(description="Customer email address")
    email_content: str = Field(description="Content of the customer email")
    confidence_threshold: int = Field(default=80, description="Confidence threshold (0-100) below which human review is required", ge=0, le=100)

class EmailWithConfidence(BaseModel):
    email_response: str = Field(description="The complete email response text")
    confidence: int = Field(description="Confidence score from 0-100 indicating how confident the agent is in this response", ge=0, le=100)

class GraphState(BaseModel):
    email_content: str
    email_address: str
    confidence_threshold: int = 80
    validated_email: bool = False
    company_name: str | None = None
    company_id: str | None = None
    email_category: EmailCategory | None = None
    email_response: str | None = None
    confidence: int | None = None
    department_routing: str | None = None
    estimated_price: float | None = None
    discount_applied: float | None = None
    discount_justification: str | None = None
    orders_in_last_7_days: int | None = None
    transport_distance: float | None = None
    route_restriction: str | None = None
    review_content: str | None = None
    approval_comment: str | None = None
    approval_status: bool | None = None
    task_key: str | None = None

class GraphOutput(BaseModel):
    error: str | None = None
    email: str | None = None

def prepare_input(input: Input) -> GraphState:
    return GraphState(
        email_address=input.email_address,
        email_content=input.email_content,
        confidence_threshold=input.confidence_threshold,
        validated_email=False,
        company_name=None,
        company_id=None,
        email_category=None,
        email_response=None,
        confidence=None,
        department_routing=None,
        estimated_price=None,
        discount_applied=None,
        discount_justification=None,
        orders_in_last_7_days=None,
        transport_distance=None,
        route_restriction=None,
        review_content=None,
        approval_comment=None,
        approval_status=None,
        task_key=None
    )

async def validate_company(state: GraphState) -> GraphState:
    companies_data = await uipath.entities.list_records_async(
        entity_key=Entity.COMPANY.value,
        start=1,
        limit=100,
    )
    companies = [Company.model_validate(company.model_dump()) for company in companies_data]

    email_domain = state.email_address[state.email_address.index('@'):]
    print(f"Looking for company with email domain: {email_domain}")

    for company in companies:
        print(f"Company: {company.name}, Domain: {company.email_domain}, ID: {company.id}")

    try:
        company = next(company for company in companies if company.email_domain == email_domain)
    except StopIteration:
        print(f"No company found with domain {email_domain}")
        return state

    print(f"Found company: {company.name}, ID: {company.id}")
    state.validated_email = True
    state.company_name = company.name
    state.company_id = company.id
    return state

@tool("previous_orders_tool")
async def retrieve_previous_orders(company_id: str) -> list[Order]:
    """Get the list of the previous orders placed by the company.

    Args:
        company_id: The id of the company
    """
    all_orders = await uipath.entities.list_records_async(
        entity_key=Entity.ORDER.value,
        start=1,
        limit=100,
    )
    all_orders = [Order.model_validate(order.model_dump()) for order in all_orders]
    previous_orders = [order for order in all_orders if order.company_id]
    return previous_orders

@tool("distance_fallback_tool")
def get_distance_fallback(origin: str, destination: str) -> str:
    """Fallback tool for distance calculation when MCP server is not available.

    Args:
        origin: Origin location
        destination: Destination location

    Returns:
        A message indicating that directions data is not available
    """
    return "directions data not available use defaults from collected data"

@tool("validate_shipment_capacity")
def validate_shipment_capacity(weight_category: str) -> dict:
    """Validate if the shipment can be processed based on weight category.

    Critical validation step for shipment processing. Call this tool to verify shipment eligibility.
    If validation fails, stop further processing immediately.

    Args:
        weight_category: Weight classification (Light, Medium, Heavy)

    Returns:
        Validation result with status and message
    """
    return {
        "validated": True,
        "message": "Shipment capacity validated successfully"
    }

@tool("shipment_retriever_tool")
async def retrieve_shipment_data(query: str) -> str:
    """Retrieve shipment pricing data and base rates from the knowledge base.

    Args:
        query: Query to search for shipment data

    Returns:
        Shipment data as a string
    """
    retriever = ContextGroundingRetriever(
        index_name="Shipment_index_dev",
        folder_path="Shared",
        number_of_results=5
    )

    docs = await retriever.ainvoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

class DiscountResult(BaseModel):
    discount_percentage: float = Field(description="The discount percentage to apply (0-15)")
    justification: str = Field(description="Explanation of why this discount was applied")
    orders_in_last_7_days: int = Field(description="Number of orders placed in the last 7 days")

class OrderExtractionResult(BaseModel):
    estimated_price: float = Field(description="The total estimated price in NOK")
    discount_applied: float = Field(description="Discount percentage applied (0 if none)")
    discount_justification: str = Field(description="Explanation for the discount, or 'No discount applied' if none")
    transport_distance: float = Field(description="Transport distance in kilometers (0 if not available)")
    route_restriction: str | None = Field(default=None, description="Description of route restriction if any exists, None if no restriction")

@traced("calculate_discount")
async def calculate_discount(company_id: str) -> dict:
    """Core logic for calculating discount based on order history using LLM."""
    all_orders = await uipath.entities.list_records_async(
        entity_key=Entity.ORDER.value,
        start=1,
        limit=100,
    )
    all_orders = [Order.model_validate(order.model_dump()) for order in all_orders]

    company_orders = [order for order in all_orders if order.company_id == company_id]
    seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
    recent_orders = []
    for order in company_orders:
        try:
            order_date = datetime.fromisoformat(order.order_date)
            if order_date >= seven_days_ago:
                recent_orders.append(order)
        except Exception as e:
            print(f"Failed to parse order date '{order.order_date}': {e}")
            pass

    orders_count = len(recent_orders)

    discount_llm = UiPathChat(model="gpt-4o-mini-2024-07-18", temperature=0).with_structured_output(DiscountResult)

    order_summaries = []
    for order in recent_orders:
        order_summaries.append({
            "order_id": order.id,
            "order_date": order.order_date,
            "status": order.order_status.value if hasattr(order.order_status, 'value') else str(order.order_status)
        })

    user_prompt = f"""Analyze the following order history for company ID: {company_id}

## Order History (Last 7 Days)
Total orders in last 7 days: {orders_count}

Orders:
{order_summaries if order_summaries else "No orders in the last 7 days"}

Based on the discount policy rules, determine the appropriate discount percentage, provide justification, and confirm the order count."""

    response = await discount_llm.ainvoke([
        SystemMessage(content=discount_calculator_prompt),
        HumanMessage(content=user_prompt)
    ])

    return {
        "discount_percentage": response['discount_percentage'],
        "justification": response['justification'],
        "orders_in_last_7_days": response['orders_in_last_7_days']
    }

@tool("calculate_discount_tool")
async def calculate_discount_tool(company_id: str) -> dict:
    """Calculate discount based on order history in the last 7 days.

    Discount tiers:
    - 3-4 orders in last 7 days: 5% discount
    - 5-6 orders in last 7 days: 10% discount
    - 7+ orders in last 7 days: 15% discount
    - Less than 3 orders: No discount

    Args:
        company_id: The id of the company

    Returns:
        Dictionary with discount_percentage, justification, and orders_in_last_7_days
    """
    return await calculate_discount(company_id)

async def calculate_discount_node(state: GraphState) -> GraphState:
    """ToolNode for discount calculation. Calculates discount and updates state."""
    print(f"calculate_discount_node: company_id = {state.company_id}")

    if not state.company_id:
        print("No company_id - setting discount to 0")
        state.discount_applied = 0.0
        state.discount_justification = "No discount applied - company ID not available"
        state.orders_in_last_7_days = 0
        return state

    discount_result = await calculate_discount(state.company_id)
    print(f"Discount calculated: {discount_result}")

    state.discount_applied = discount_result["discount_percentage"]
    state.discount_justification = discount_result["justification"]
    state.orders_in_last_7_days = discount_result["orders_in_last_7_days"]
    return state

@traced(span_type="tool")
async def understand_request(state: GraphState) -> GraphState:
    agent = create_agent(
        llm,
        tools=[retrieve_previous_orders],
        system_prompt=classifier_prompt,
        response_format=ToolStrategy(ModelSchema)
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": f"{state.email_content}"}]}
    )
    state.email_category = result["structured_response"].email_category
    return state

async def handle_complaint(state: GraphState) -> GraphState:
    complaint_llm = UiPathChat(model="gpt-4o-2024-08-06", temperature=0.7).with_structured_output(EmailWithConfidence)

    retriever = ContextGroundingRetriever(
        index_name="company_policy",
        folder_path="Shared",
        number_of_results=5
    )

    policy_docs = await retriever.ainvoke(
        f"How to handle complaints and escalation procedures: {state.email_content}"
    )

    policy_context = "\n\n".join([doc.page_content for doc in policy_docs])

    email_prompt = f"""{complaint_handler_prompt}

## Customer Information
Company: {state.company_name or 'Valued Customer'}
Email: {state.email_address}

## Original Complaint
{state.email_content}

## Relevant Company Policies and Escalation Guidelines
{policy_context}

Based on the company policies above, compose a professional email response. Include a confidence score (0-100) indicating how confident you are in your response based on the clarity of the complaint and the relevance of the policies found."""

    response = await complaint_llm.ainvoke([
        SystemMessage(content=email_prompt),
        HumanMessage(content="Compose the email response with confidence score.")
    ])

    dept_llm = UiPathChat(model="gpt-4o-2024-08-06", temperature=0.7)
    department_extraction = await dept_llm.ainvoke([
        SystemMessage(content=f"Based on this complaint and the company policies, extract ONLY the department name where this should be escalated.\n\nComplaint: {state.email_content}\n\nPolicies: {policy_context}\n\nRespond with ONLY the department name."),
        HumanMessage(content="What department?")
    ])
    state.email_response = response['email_response']
    state.confidence = response['confidence']
    state.department_routing = department_extraction.content.strip()
    return state

async def handle_new_order(state: GraphState) -> GraphState:
    order_llm = UiPathChat(model="gpt-4o-2024-08-06", temperature=0.5)

    base_tools = [validate_shipment_capacity, retrieve_shipment_data]
    mcp_server_url = os.getenv("MCP_SERVER_URL")

    user_message = {
        "messages": [
            {
                "role": "user",
                "content": f"""
Company: {state.company_name}
Email: {state.email_address}
Order Request: {state.email_content}

Discount Information (already calculated):
- Discount Percentage: {state.discount_applied}%
- Justification: {state.discount_justification}
- Orders in last 7 days: {state.orders_in_last_7_days}

Calculate the estimated price for this order including:
1. Base shipment costs (use shipment_retriever_tool)
2. Transport distance and costs (use MCP distance tool if available)
3. Apply the discount above to get final price

Provide the final estimated price (after discount), and transport distance.
"""
            }
        ]
    }

    response_content = None

    if mcp_server_url:
        try:
            from langchain_mcp_adapters.tools import load_mcp_tools
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client

            async with streamablehttp_client(
                url=mcp_server_url,
                headers={"Authorization": f"Bearer {os.getenv('UIPATH_ACCESS_TOKEN')}"},
                timeout=60,
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    mcp_tools = await load_mcp_tools(session)
                    tools = base_tools + mcp_tools

                    agent = create_agent(
                        order_llm,
                        tools=tools,
                        system_prompt=new_order_handler_prompt_good
                    )

                    result = await agent.ainvoke(user_message)
                    response_content = result["messages"][-1].content
        except Exception as e:
            print(f"Failed to connect to MCP server: {e}")
            tools = base_tools + []
            agent = create_agent(
                order_llm,
                tools=tools,
                system_prompt=new_order_handler_prompt_good
            )
            result = await agent.ainvoke(user_message)
            response_content = result["messages"][-1].content
    else:
        tools = base_tools + []
        agent = create_agent(
            order_llm,
            tools=tools,
            system_prompt=new_order_handler_prompt_good,
            middleware=[DisableParallelToolCallsMiddleware()],
        )
        result = await agent.ainvoke(user_message)
        response_content = result["messages"][-1].content

    extraction_llm = UiPathChat(model="gpt-4o-mini-2024-07-18").with_structured_output(OrderExtractionResult)
    extraction_prompt = f"""Extract the following information from this order processing result:

{response_content}

Extract these fields:
- estimated_price: The total price in NOK
- discount_applied: Discount percentage (0 if none)
- discount_justification: Explanation for the discount, or "No discount applied" if none
- transport_distance: Distance in kilometers (0 if not available)
- route_restriction: If any route restrictions were mentioned (e.g., delivery not possible, restricted route), extract the description. Otherwise set to None.
"""

    extracted_data = await extraction_llm.ainvoke([
        SystemMessage(content=extraction_prompt),
        HumanMessage(content="Extract the data now.")
    ])

    state.estimated_price = extracted_data['estimated_price']
    state.discount_applied = extracted_data['discount_applied']
    state.discount_justification = extracted_data['discount_justification']
    state.transport_distance = extracted_data['transport_distance']
    state.route_restriction = extracted_data['route_restriction']

    return state

def decide_next_node(state: GraphState) -> Literal["handle_error", "understand_request"]:
    if not state.validated_email:
        return "handle_error"
    return "understand_request"

def route_after_classification(state: GraphState) -> Literal["handle_complaint", "calculate_discount_node", "handle_logistic_request", "handle_unknown"]:
    if state.email_category == EmailCategory.COMPLAINT:
        return "handle_complaint"
    elif state.email_category == EmailCategory.NEW_ORDER:
        return "calculate_discount_node"
    elif state.email_category == EmailCategory.LOGISTIC_REQUEST:
        return "handle_logistic_request"
    else:
        return "handle_unknown"

def route_after_order_processing(state: GraphState) -> Literal["handle_route_restriction", "collect_output_new_order"]:
    """Route based on whether there's a route restriction or not."""
    if state.route_restriction:
        print(f"Route restriction detected: {state.route_restriction}")
        return "handle_route_restriction"
    return "collect_output_new_order"

async def collect_output(state: GraphState) -> GraphState:
    queue_item_data = {
        "Reference": f"Complaint from {state.email_address}",
        "SpecificContent": {
            "request_type": "COMPLAINT",
            "complaint_details": state.email_content,
            "routing_department": state.department_routing,
            "company_name": state.company_name,
            "email_address": state.email_address,
            "next_steps": f"Route to {state.department_routing} department, investigate issue details, verify customer complaint history, implement corrective action, follow up with customer within 48 hours, update internal incident log"
        }
    }

    from uipath.platform.orchestrator.queues import QueueItem
    queue_item = QueueItem(
        Name="ProcessingQueue",
        SpecificContent=queue_item_data["SpecificContent"],
        Reference=queue_item_data["Reference"]
    )

    await uipath.queues.create_item_async(queue_item)

    return state

async def handle_route_restriction(state: GraphState) -> GraphState:
    """Handle cases where there are route restrictions - notify customer via email."""
    restriction_email_llm = UiPathChat(model="gpt-4o-mini-2024-07-18").with_structured_output(EmailWithConfidence)

    email_prompt = f"""Compose a professional notification email for a B2B logistics customer regarding a route restriction.

Customer Information:
- Company: {state.company_name}
- Email: {state.email_address}

Order Request:
{state.email_content}

Route Restriction:
{state.route_restriction}

The email should:
1. Thank them for their order request
2. Inform them that there is a restriction on the requested delivery route
3. Clearly explain the restriction: {state.route_restriction}
4. Apologize for the inconvenience
5. Offer to discuss alternative solutions or routes
6. Provide contact information for further assistance
7. Be empathetic and professional

**CRITICAL**: Write as a real human. Sign the email as "Lars Andersen, Order Processing Specialist" - do NOT use placeholders like [Your Name] or [Department]. Use natural, conversational language while maintaining professionalism. Make it feel personal and authentic.

Include a confidence score (0-100) based on how clear the restriction information is."""

    email_response = await restriction_email_llm.ainvoke([
        SystemMessage(content=email_prompt),
        HumanMessage(content="Compose the route restriction notification email with confidence score.")
    ])

    state.email_response = email_response['email_response']
    state.confidence = email_response['confidence']

    from uipath.platform.orchestrator.queues import QueueItem
    queue_item = QueueItem(
        Name="ProcessingQueue",
        SpecificContent={
            "request_type": "ROUTE_RESTRICTION",
            "order_details": state.email_content,
            "company_name": state.company_name,
            "email_address": state.email_address,
            "route_restriction": state.route_restriction,
            "next_steps": f"Contact customer to discuss alternative routes or solutions, review restriction policy, escalate to logistics planning team if needed"
        },
        Reference=f"Route Restriction - {state.email_address}"
    )

    await uipath.queues.create_item_async(queue_item)

    return state

async def collect_output_new_order(state: GraphState) -> GraphState:
    order_confirmation_llm = UiPathChat(model="gpt-4o-mini-2024-07-18").with_structured_output(EmailWithConfidence)

    email_prompt = f"""Compose a professional order confirmation email for a B2B logistics customer.

Customer Information:
- Company: {state.company_name}
- Email: {state.email_address}

Order Details:
- Original Request: {state.email_content}
- Estimated Price: {state.estimated_price:.2f} NOK
- Discount Applied: {state.discount_applied:.1f}%
- Discount Justification: {state.discount_justification}
- Transport Distance: {state.transport_distance:.1f} km

The email should:
1. Thank them for the order
2. Confirm order receipt
3. State the estimated price clearly
4. If a discount was applied, mention it naturally and include the justification
5. Mention that a human representative will confirm shortly
6. Be professional and concise

**CRITICAL**: Write as a real human. Sign the email as "Lars Andersen, Order Processing Specialist" - do NOT use placeholders like [Your Name] or [Department]. Use natural, conversational language while maintaining professionalism. Make it feel personal and authentic.

Include a confidence score (0-100) based on how complete and clear the order information is."""

    email_response = await order_confirmation_llm.ainvoke([
        SystemMessage(content=email_prompt),
        HumanMessage(content="Compose the order confirmation email with confidence score.")
    ])

    state.email_response = email_response['email_response']
    state.confidence = email_response['confidence']

    from uipath.platform.orchestrator.queues import QueueItem
    queue_item = QueueItem(
        Name="ProcessingQueue",
        SpecificContent={
            "request_type": "NEW_ORDER",
            "order_details": state.email_content,
            "company_name": state.company_name,
            "email_address": state.email_address,
            "estimated_price": state.estimated_price,
            "discount_applied": state.discount_applied,
            "discount_justification": state.discount_justification,
            "transport_distance": state.transport_distance,
            "next_steps": f"Review order details, verify pricing calculation, confirm inventory availability, assign to delivery team, contact customer for final confirmation within 24 hours"
        },
        Reference=f"New Order from {state.email_address}"
    )

    await uipath.queues.create_item_async(queue_item)

    return state

async def handle_logistic_request(state: GraphState) -> GraphState:
    response_llm = UiPathChat(model="gpt-4o-mini-2024-07-18").with_structured_output(EmailWithConfidence)

    email_prompt = f"""Compose a professional response email for a B2B logistics inquiry.

Customer Information:
- Company: {state.company_name}
- Email: {state.email_address}

Inquiry: {state.email_content}

The email should:
1. Acknowledge their logistic request
2. Confirm that their inquiry has been received
3. Let them know a logistics specialist will respond within 24 hours with detailed information
4. Provide general contact information
5. Be professional and helpful

**CRITICAL**: Write as a real human. Sign the email as "Maria Holm, Logistics Coordinator" - do NOT use placeholders like [Your Name] or [Department]. Use natural, conversational language while maintaining professionalism.

Include a confidence score (0-100) based on how well you understand the logistics request."""

    email_response = await response_llm.ainvoke([
        SystemMessage(content=email_prompt),
        HumanMessage(content="Compose the response email with confidence score.")
    ])

    state.email_response = email_response['email_response']
    state.confidence = email_response['confidence']

    from uipath.platform.orchestrator.queues import QueueItem
    queue_item = QueueItem(
        Name="ProcessingQueue",
        SpecificContent={
            "request_type": "LOGISTIC_REQUEST",
            "inquiry_details": state.email_content,
            "company_name": state.company_name,
            "email_address": state.email_address,
            "next_steps": f"Assign to logistics specialist, analyze request requirements, prepare detailed logistics proposal, schedule consultation call if needed, respond within 24 hours"
        },
        Reference=f"Logistics Request from {state.email_address}"
    )

    await uipath.queues.create_item_async(queue_item)

    return state

async def handle_unknown(state: GraphState) -> GraphState:
    response_llm = UiPathChat(model="gpt-4o-mini-2024-07-18").with_structured_output(EmailWithConfidence)

    email_prompt = f"""Compose a professional response email for an unclear customer inquiry.

Customer Information:
- Company: {state.company_name}
- Email: {state.email_address}

Email Content: {state.email_content}

The email should:
1. Thank them for contacting us
2. Politely mention that we need more information to assist them properly
3. Ask them to clarify their request (order, inquiry, complaint, etc.)
4. Provide contact information for direct assistance
5. Be helpful and welcoming

**CRITICAL**: Write as a real human. Sign the email as "Erik Hansen, Customer Support" - do NOT use placeholders like [Your Name] or [Department]. Use natural, conversational language while maintaining professionalism.

Include a confidence score (0-100) - it should be low since this is an unclear request."""

    email_response = await response_llm.ainvoke([
        SystemMessage(content=email_prompt),
        HumanMessage(content="Compose the response email with confidence score.")
    ])

    state.email_response = email_response['email_response']
    state.confidence = email_response['confidence']

    from uipath.platform.orchestrator.queues import QueueItem
    queue_item = QueueItem(
        Name="ProcessingQueue",
        SpecificContent={
            "request_type": "UNKNOWN",
            "email_content": state.email_content,
            "company_name": state.company_name,
            "email_address": state.email_address,
            "next_steps": f"Manual review required, contact customer for clarification, determine actual request type, route to appropriate department once clarified"
        },
        Reference=f"Unknown Request from {state.email_address}"
    )

    await uipath.queues.create_item_async(queue_item)

    return state

def handle_error(state: GraphState) -> GraphOutput:
    if not state.validated_email:
        return GraphOutput(
            error=f"Email address {state.email_address} not found in known companies list."
        )

@mockable()
def create_approval_task(email_category: EmailCategory, review_content: str) -> dict:
    """Create an approval task and return the task output.

    Args:
        email_category: The category of the email being reviewed
        review_content: The review content to display in the task

    Returns:
        Dictionary containing approval status and comment from the task
    """
    task_output = interrupt(CreateTask(
        app_name="SimpleApprovalApp",
        app_folder_path="Shared/fusion app",
        title=f"Review Email Response - {email_category.value}",
        data={"Content": review_content},
        assignee="radu.mocanu@uipath.com"
    ))

    return {
        "Approved": task_output.get("Approved", False),
        "Comment": task_output.get("Comment", "")
    }

async def human_review(state: GraphState) -> GraphState:
    if state.email_category == EmailCategory.COMPLAINT:
        review_content = f"""Review Email Response - COMPLAINT

Customer: {state.company_name} ({state.email_address})
Category: Complaint
Department Routing: {state.department_routing}

Original Complaint:
{state.email_content}

Proposed Email Response:
{state.email_response}

Please review and approve or reject with comments."""

    elif state.email_category == EmailCategory.NEW_ORDER:
        review_content = f"""Review Order Processing - NEW ORDER

Customer: {state.company_name} ({state.email_address})
Category: New Order

Order Details:
{state.email_content}

Pricing Calculation:
- Estimated Price: {state.estimated_price:.2f} NOK
- Discount Applied: {state.discount_applied:.1f}%
- Justification: {state.discount_justification}
- Transport Distance: {state.transport_distance:.1f} km

Proposed Email Response:
{state.email_response}

Please review and approve or reject with comments."""

    elif state.email_category == EmailCategory.LOGISTIC_REQUEST:
        review_content = f"""Review Email Response - LOGISTIC REQUEST

Customer: {state.company_name} ({state.email_address})
Category: Logistic Request

Request Details:
{state.email_content}

Proposed Email Response:
{state.email_response}

Please review and approve or reject with comments."""

    else:  # UNKNOWN
        review_content = f"""Review Email Response - UNKNOWN

Customer: {state.company_name} ({state.email_address})
Category: Unknown/Unclear

Email Content:
{state.email_content}

Proposed Email Response:
{state.email_response}

Please review and approve or reject with comments."""

    state.review_content = review_content

    task_output = create_approval_task(state.email_category, review_content)

    state.approval_status = task_output.get("Approved", False)
    state.approval_comment = task_output.get("Comment", "")

    if not state.approval_status:
        state.email_content = f"{state.email_content}\n\n[FEEDBACK FROM REVIEWER]: {state.approval_comment}"
        state.email_category = None
        state.email_response = None
        state.department_routing = None
        state.estimated_price = None
        state.discount_applied = None
        state.discount_justification = None
        state.transport_distance = None
        state.review_content = None
        state.task_key = None

    return state

def route_by_confidence(state: GraphState) -> Literal["human_review", "finalize_output"]:
    if state.confidence and state.confidence < state.confidence_threshold:
        return "human_review"
    return "finalize_output"

def route_after_review(state: GraphState) -> Literal["validate_company", "finalize_output"]:
    if not state.approval_status:
        return "validate_company"
    return "finalize_output"

async def finalize_output(state: GraphState) -> GraphOutput:
    # Write the email to a markdown file with nice formatting
    if state.email_response:
        markdown_content = f"""# Email Response

## Customer Information
- **Company:** {state.company_name or 'N/A'}
- **Email:** {state.email_address}
- **Category:** {state.email_category.value if state.email_category else 'N/A'}

## Original Request
```
{state.email_content}
```

## Agent Response

{state.email_response}

---

## Processing Details
- **Confidence Score:** {state.confidence}%
- **Confidence Threshold:** {state.confidence_threshold}%
"""

        # Add order-specific details if this was a new order
        if state.email_category == EmailCategory.NEW_ORDER:
            if state.route_restriction:
                markdown_content += f"""
### Route Restriction
**Restriction:** {state.route_restriction}
"""
            else:
                markdown_content += f"""
### Order Details
- **Estimated Price:** {state.estimated_price:.2f} NOK
- **Discount Applied:** {state.discount_applied:.1f}%
- **Discount Justification:** {state.discount_justification}
- **Transport Distance:** {state.transport_distance:.1f} km
- **Orders in Last 7 Days:** {state.orders_in_last_7_days}
"""

        # Add complaint-specific details
        elif state.email_category == EmailCategory.COMPLAINT:
            markdown_content += f"""
### Complaint Details
- **Department Routing:** {state.department_routing}
"""

        # Write to file
        with open("output_email_content.md", "w", encoding="utf-8") as f:
            f.write(markdown_content)

        print("Email content written to: output_email_content.md")

    return GraphOutput(
        error=None,
        email=state.email_response
    )


builder = StateGraph(GraphState, input=Input, output=GraphOutput)

builder.add_node("prepare_input", prepare_input)
builder.add_node("validate_company", validate_company)
builder.add_node("understand_request", understand_request)
builder.add_node("handle_complaint", handle_complaint)
builder.add_node("calculate_discount_node", calculate_discount_node)
builder.add_node("handle_new_order", handle_new_order)
builder.add_node("handle_route_restriction", handle_route_restriction)
builder.add_node("handle_logistic_request", handle_logistic_request)
builder.add_node("handle_unknown", handle_unknown)
builder.add_node("collect_output", collect_output)
builder.add_node("collect_output_new_order", collect_output_new_order)
builder.add_node("human_review", human_review)
builder.add_node("finalize_output", finalize_output)
builder.add_node("handle_error", handle_error)

builder.add_edge(START, "prepare_input")
builder.add_edge("prepare_input", "validate_company")
builder.add_conditional_edges("validate_company", decide_next_node)
builder.add_conditional_edges("understand_request", route_after_classification)
builder.add_edge("handle_complaint", "collect_output")
builder.add_edge("calculate_discount_node", "handle_new_order")
builder.add_conditional_edges("handle_new_order", route_after_order_processing)
builder.add_conditional_edges("handle_route_restriction", route_by_confidence)
builder.add_conditional_edges("collect_output", route_by_confidence)
builder.add_conditional_edges("collect_output_new_order", route_by_confidence)
builder.add_conditional_edges("handle_logistic_request", route_by_confidence)
builder.add_conditional_edges("handle_unknown", route_by_confidence)
builder.add_conditional_edges("human_review", route_after_review)
builder.add_edge("finalize_output", END)
builder.add_edge("handle_error", END)

graph = builder.compile()
