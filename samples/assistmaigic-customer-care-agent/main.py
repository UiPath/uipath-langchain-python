import asyncio
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
    MCP_SERVER_URL = os.getenv("UIPATH_MCP_SERVER_URL")
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
    logging.info(f"Loading MCP tools...")
    async with get_mcp_session() as session:
        tools = await load_mcp_tools(session)
        return tools
    
# Initialize Context Grounding for company policy
context_grounding = ContextGroundingRetriever(
    index_name="AssistmAIgic_Index",
    folder_path="Shared",
    number_of_results=1
    )

# ---------------- State ----------------
class GraphState(BaseModel):
    """Enhanced state to track the complete assistmAIgic workflow"""
    message_id: str | None = None
    agent_language: str | None = "English"
    email_body: str | None = None
    email_subject: str | None = None
    translated_email_body: str | None = None
    translated_email_subject: str | None = None
    mail_communication_language: str | None = None
    order_id: str | None = None
    order_details: Dict[str, Any] | None = None
    email_sentiment: Dict[str, Any] | None = None
    isorder_valid: bool | None = None
    email_category: str | None = None
    email_response: str | None = None
    hitl_response: str | None = None
    final_status: str | None = None  


## Translate email body and subject language if required
async def translate_email_language_mcp(email_subject: str, email_body: str, agent_language: str):
    """Translate email body and subject language if required by MCP tools"""
    async with get_mcp_session() as session:
        tools = await load_mcp_tools(session)
        
        # Translate Email details tool
        translategetemaildetails_tool = next((tool for tool in tools if "translateemailsubjectandbodylanguage" in tool.name.lower()), None)
        if not translategetemaildetails_tool:
            logging.error("Translate Email Subject And BodybLanguage tool not found in MCP server")
            raise Exception("translateEmailSubjectAndBodyLanguage tool not available")
        
        logging.info(f"Invoking Translate Email Subject And Body Language MCP tool...")
        try:
            result = await translategetemaildetails_tool.ainvoke({
                "in_OriginalEmailSubject": email_subject,
                "in_OriginalEmailBody": email_body,
                "in_Language": agent_language
            })
            logging.info(f"Email language translation done via MCP")        
            # Assuming 'result' is your string variable
            email_details_dict = ast.literal_eval(result) if result else None  
            return email_details_dict if email_details_dict else None   
            
        except Exception as e:
            logging.error(f"Error translating email details via MCP: {e}")
            raise

## Get Order Details by order_id via MCP
async def get_order_details_mcp(order_id: str):
    """Get Order Details by order_id with MCP tool"""
    async with get_mcp_session() as session:
        tools = await load_mcp_tools(session)
        
        # Get Order Details tool
        order_details_tool = next((tool for tool in tools if "getOrderDetailsByOrderId".lower() in tool.name.lower()), None)
        if not order_details_tool:
            logging.error("getOrderDetailsByOrderId tool not found in MCP server")
            raise Exception("getOrderDetailsByOrderId tool not available")
        
        logging.info(f"Invoking getOrderDetailsByOrderId MCP tool...")
        try:
            result = await order_details_tool.ainvoke({
                "in_OrderNumber": order_id
            })
            logging.info(f"Fetched order details via MCP tool.")        
            # Assuming 'result' is your string variable
            order_details_dict = ast.literal_eval(result) if result else None  
            return order_details_dict if order_details_dict else None   
            
        except Exception as e:
            logging.error(f"Error Fetching order details via MCP: {e}")
            raise

## Get Email Categorization done via MCP
async def categorize_email_mcp(content_to_categorize: str):
    """Categorize email body and subject with MCP tool"""
    async with get_mcp_session() as session:
        tools = await load_mcp_tools(session)
        
        # Get email categorized using Categorize Email tool
        categorize_email_tool = next((tool for tool in tools if "categorizeEmail".lower() in tool.name.lower()), None)
        if not categorize_email_tool:
            logging.error("categorizeEmail tool not found in MCP server")
            raise Exception("Categorize Email tool not available")
        
        logging.info(f"Invoking Categorize Email tool...")
        try:
            result = await categorize_email_tool.ainvoke({
                "in_Content_To_Categorize": content_to_categorize
            })
            logging.info(f"Categorized email via MCP tool.")        
            # Assuming 'result' is your string variable
            email_category_dict = ast.literal_eval(result) if result else None  
            return email_category_dict if email_category_dict else None   
            
        except Exception as e:
            logging.error(f"Error Categorizing email via MCP: {e}")
            raise

## Email Sentiment Analysis via MCP
async def analyze_email_sentiment_mcp(content_to_sentiment_analysis: str):
    """Email Sentiment Analysis email body and subject with MCP tool"""
    async with get_mcp_session() as session:
        tools = await load_mcp_tools(session)
        
        # Email Sentiment Analysis tool
        email_sentiment_analysis_tool = next((tool for tool in tools if "emailSentimentAnalysis".lower() in tool.name.lower()), None)
        if not email_sentiment_analysis_tool:
            logging.error("emailSentimentAnalysis tool not found in MCP server")
            raise Exception("Email Sentiment Analysis tool not available")
        
        logging.info(f"Invoking emailSentimentAnalysis tool...")
        try:
            result = await email_sentiment_analysis_tool.ainvoke({
                "in_Content_To_Analysis": content_to_sentiment_analysis
            })
            logging.info(f"Email Sentiment Analysis via MCP tool.")        
            # Assuming 'result' is your string variable
            sentiment_analysis_dict = ast.literal_eval(result) if result else None  
            return sentiment_analysis_dict if sentiment_analysis_dict else None   
            
        except Exception as e:
            logging.error(f"Error Email Sentiment Analysis via MCP: {e}")
            raise

# ---------------- Graph Nodes ----------------
class GraphOutput(BaseModel):
    report: str

# ---------------- Nodes ----------------
# Translate email body and subject language if required via MCP integration
async def translate_email_language_node(state: GraphState) -> GraphOutput:
    """Translate email body and subject language if required by MCP integration"""
    email_details = await translate_email_language_mcp(
        state.email_subject,
        state.email_body,
        state.agent_language
    )
    
    return state.model_copy(update={
        "mail_communication_language": email_details['out_CommunicationLanguage'] or None,
        "translated_email_body": email_details['out_TranslatedMailBody'] or None,
        "translated_email_subject": email_details['out_TranslatedMailSubject'] or None
    })

# ---------------- Nodes ----------------
# Extract order id from email body & subject   
async def extract_order_id_node(state: GraphState) -> GraphState:
    """Extract order id information from the request"""
    system_prompt = """You are a data extraction expert tasked with extracting order id information from input text. 

    Your goal is to extract the following fields:
    1. order id - the order id will be 8 digit number

    Instructions:
    - Only return a JSON object with keys: order_id
    - If a field cannot be determined, return null.
    - If multiple order ids are present, extract the first one only.
    - Only output the JSON. Do not include any explanations, commentary, or extra text.

    Examples:

    User message: "My order number is 12345678 and I need help."
    Output:
    {
    "order_id": "12345678"
    }
    User message: "My order number is #12345678 and I need help."
    Output:
    {
    "order_id": "12345678"
    }
    User message: "My order no is #12345678 and I need help."
    Output:
    {
    "order_id": "12345678"
    }
    User message: "My order #12345678 and I need help."
    Output:
    {
    "order_id": "12345678"
    }
    User message: "I have purchased a fan with order id 87654321 last week."
    Output:
    {
    "order_id": "87654321",
    }

    User message: "My TV isn't working properly."
    Output:
    {
    "order_id": null
    }
    """

    output = await llm.ainvoke(
        [SystemMessage(system_prompt),
         HumanMessage("Extract the order ID from this message: " + state.email_body + "\n" + state.email_subject)]
    )
    try:
        payload = json.loads(output.content)
    except Exception:
        payload = None

    raw = (output.content or "").lower()
    null_patterns = ['"order_id": null', "'order_id' is null", "order_id is null", "order_id: null"]

    if any(p in raw for p in null_patterns):
        
        extracted_order_id = None
    else:
        extracted_order_id = None
        if isinstance(payload, dict):
            extracted_order_id = payload.get("order_id")

    logging.info(f"Extracted Order ID: {extracted_order_id}")
    return state.model_copy(update={
        "order_id": extracted_order_id or None
    })

# ---------------- Nodes ----------------
# Get Order Details by order_id via MCP integration
async def get_order_details_node(state: GraphState) -> GraphOutput:
    """Get Order Details by order_id with MCP tool"""
    order_details_obj = await get_order_details_mcp(
        state.order_id
    )
    
    return state.model_copy(update={
        "order_details": order_details_obj["out_OrderDetails"] or None
    })

def end_node(state: GraphState) -> GraphState:
    """Final node to log the agent run completion"""
    logging.info(f"Email processing completed. Status: {state.final_status}")
    return state

# ---------------- Nodes ----------------
# Auto-reject email via MCP integration
async def auto_reject_node(state: GraphState) -> GraphState:
    """Send auto-rejection email via MCP integration"""
    logging.info(f"Auto-rejecting email due to missing order id.")
    await reply_email_mcp(
        message_id=state.message_id,
        llmprompt_to_prepare_reply="We regret to inform you that your order Id is missing from your email. Please provide a valid order Id for us to assist you further.",
        reply_language=state.mail_communication_language
    )
    
    return state.model_copy(update={"final_status": "completed"})

# ---------------- Nodes ----------------
# Reply to email via MCP integration
async def reply_to_email_node(state: GraphState) -> GraphState:
    """Send email reply via MCP integration"""
    logging.info(f"Replying to email via MCP integration.")
    await reply_email_mcp(
        message_id=state.message_id,
        llmprompt_to_prepare_reply = state.hitl_response if state.hitl_response else state.email_response,
        reply_language=state.mail_communication_language
    )
    
    return state.model_copy(update={"final_status": "completed"})

# ---------------- MCP Email Reply Function ----------------
async def reply_email_mcp(message_id: str, llmprompt_to_prepare_reply: str, reply_language: str):
    """Send email reply via MCP integration"""
    async with get_mcp_session() as session:
        tools = await load_mcp_tools(session)
        
        # Find the email tool
        reply_to_email_tool = next((tool for tool in tools if "replyToEmail".lower() in tool.name.lower()), None)
        if not reply_to_email_tool:
            logging.error("Email reply tool not found in MCP server.")
            raise Exception("Email reply tool not available.")
        
        try:
            await reply_to_email_tool.ainvoke({
                "in_Message_Id": message_id,
                "in_llmprompt_to_prepare_reply": llmprompt_to_prepare_reply,
                "in_Reply_Language": reply_language
            })
            logging.info(f"Replied to email via MCP tool.")
            
        except Exception as e:
            logging.error(f"Error replying to email via MCP: {e}")
            raise

# ---------------- Nodes ----------------
# Categorize email via MCP integration
async def categorize_email_node(state: GraphState) -> GraphOutput:
    """Categorize email by body and subject"""
    categorize_email_obj = await categorize_email_mcp(
        state.translated_email_body + "\n" + state.translated_email_subject
    )
    
    return state.model_copy(update={
        "email_category": categorize_email_obj["out_Category"] or None
    })

# ---------------- Helper Functions ----------------
async def get_answer_with_context(state: GraphState) -> str:
    """Get answer to user query with context grounding and LLM analysis"""
    
    # Default return value
    email_response = None
    logging.info(f"Generating email response with context grounding and LLM analysis...")
    try:
        # Your existing policy check logic...
        context_query = f"""Get details for the product {state.order_details.get("ProductName", "")}. Provide detailed information about the product features, specifications, warranty policy, and any other relevant details that can help in addressing customer inquiries or issues related to this product. If required, also include information about common troubleshooting steps or usage guidelines for this product and customer care contact details."""
        
        # Try to get products context
        try:
            products_context = context_grounding.invoke(context_query)
            logging.debug(f"Retrieved {len(products_context) if products_context else 0} documents")
        except Exception as e:
            logging.error(f"Context grounding failed: {e}")
            return email_response
        
        if products_context:
            # Process documents...
            llm_context = []
            for doc in products_context:
                product_specification = doc.page_content
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page_number', '1')
                product_specification_docs = f"Source: {source} (Page {page})\n{product_specification}"
                llm_context.append(product_specification_docs)
            
            # # LLM analysis...
            try:
                system_prompt = f"""
                You are an AI product assistant designed to help customers with their product-related queries. Your primary responsibilities are:
                1. Understand and interpret customer queries about products.
                2. Use the provided context grounding index to find relevant information about the products.
                3. Formulate clear, concise, and accurate responses to customer queries based on the information from the context index.
                4. If the exact information is not available, provide the most relevant information you can find and be transparent about any limitations.
                5. Always maintain a professional, helpful, and friendly tone in your responses.
                6. If you cannot find an answer to a query, politely inform the customer and suggest alternative ways they might get the information they need.
                
                Use this context to answer the customer query:
                {llm_context}

                Remember, your goal is to provide the best possible assistance to customers regarding their product inquiries.
                
                Return the output in plain string. No JSON format is required.
                """
                user_prompt = f"""
                Please process the following customer query about our product:
                Query: {state.translated_email_subject} + "\n" +{state.translated_email_body}
                
                To answer this query:
                1. Analyze the query to understand what information the customer is seeking.
                2. Search the context grounding index for relevant product information.
                3. Formulate a clear and concise answer based on the information found.
                4. If additional product details are available and relevant, include them in your response.
                5. Ensure your response is helpful and directly addresses the customer's query.
                
                Provide your response in the following format:
                - Your response to the customer's query

                """

                response = await llm.ainvoke([
                    SystemMessage(system_prompt),
                    HumanMessage(user_prompt)
                ])
                
                result = response.content
                return result
                
            except Exception as e:
                logging.error(f"LLM get_answer_with_context failed: {e}")
                return email_response or None
        else:
            logging.error("No context found")
            return email_response or None
            
    except Exception as e:
        logging.error(f"Get_answer_with_context completely failed: {e}")
        return email_response or None
    
async def get_answer_to_query_node(state: GraphState) -> GraphState:
    """Get answer to user query with context grounding and LLM analysis"""
    answer_of_query = await get_answer_with_context(state)
    
    return state.model_copy(update={
        "email_response": answer_of_query or None
    })

# ---------------- Nodes ----------------
# Analyze email sentiment via MCP integration
async def analyze_email_sentiment_node(state: GraphState) -> GraphOutput:
    """Analyze email sentiment by body and subject"""
    logging.info(f"Analyzing email sentiment via MCP integration...")
    email_sentiment_obj = await analyze_email_sentiment_mcp(
        state.translated_email_body + "\n" + state.translated_email_subject
    )
    # Normalize email sentiment to a dict (MCP may return a stringified dict)
    
    sentiment_raw = None
    if isinstance(email_sentiment_obj, dict):
        sentiment_raw = email_sentiment_obj.get("out_Email_Sentiment")
    else:
        sentiment_raw = email_sentiment_obj

    sentiment_dict = None
    if isinstance(sentiment_raw, dict):
        sentiment_dict = sentiment_raw
    elif isinstance(sentiment_raw, str):
        try:
            sentiment_dict = json.loads(sentiment_raw)
        except Exception:
            try:
                sentiment_dict = ast.literal_eval(sentiment_raw)
            except Exception:
                sentiment_dict = {"raw": sentiment_raw}
    else:
        sentiment_dict = None

    return state.model_copy(update={
        "email_sentiment": sentiment_dict or None
    })

# ---------------- Nodes ----------------
# Human-in-the-loop review in case of negative sentiment node
async def human_review_node(state: GraphState) -> Command:
    """Send to human review"""
    logging.info(f"Sending email response to HITL review due to very negative sentiment.")
    action_data = interrupt(
        CreateAction(
            app_name="HITL_Review",
            title="Very Negative sentiment email response review",
            data={
                "EmailSubject": state.translated_email_subject,
                "EmailBody": state.translated_email_body,
                "InitialResponse": state.email_response
            },
            app_version=1,
            app_folder_path="Shared/ReviewEmailResponseSolution"
        )
    )

    email_response = action_data.get("InitialResponse")
    
    return Command(update={
        "hitl_response": email_response
    })

# ---------------- Condition Functions ----------------
def should_go_to_order_id_auto_reject(state: GraphState):
    """Check if order_id is missing and autorejection is required"""
    return "order_id_missing" if state.order_id is None else "order_id_available"

# ---------------- Condition Functions ----------------
def should_go_to_HITL_review(state: GraphState):
    """Check if email sentiment is very negative and HITL review is required"""
    return "very_negative_sentiment" if state.email_sentiment.get("label") == "Very Negative" else "agent_can_proceed_sentiment"

# ---------------- Build Graph ----------------
graph = StateGraph(GraphState)

# Add all nodes
graph.add_node("translate_email_language", translate_email_language_node) #first node
graph.add_node("extract_order_id", extract_order_id_node)
graph.add_node("get_order_details", get_order_details_node)
graph.add_node("auto_reject", auto_reject_node)
graph.add_node("categorize_email", categorize_email_node)
graph.add_node("analyze_email_sentiment", analyze_email_sentiment_node)
graph.add_node("get_answer_to_query", get_answer_to_query_node)
graph.add_node("human_review", human_review_node)
graph.add_node("reply_to_email", reply_to_email_node)
graph.add_node("step_end", end_node)

# Set entry point
graph.set_entry_point("translate_email_language")
# Add edges
graph.add_edge("translate_email_language", "extract_order_id")
#graph.add_edge("extract_order_id", "get_order_details")
graph.add_conditional_edges(
    "extract_order_id", 
    should_go_to_order_id_auto_reject,
    {
        "order_id_missing": "auto_reject",
        "order_id_available": "get_order_details"
    }
 )
graph.add_edge("auto_reject", "step_end")
graph.add_edge("get_order_details", "categorize_email")
graph.add_edge("categorize_email", "analyze_email_sentiment")
graph.add_edge("analyze_email_sentiment", "get_answer_to_query")
graph.add_conditional_edges(
    "get_answer_to_query", 
    should_go_to_HITL_review,
    {
        "very_negative_sentiment": "human_review",
        "agent_can_proceed_sentiment": "reply_to_email"
    }
 )
graph.add_edge("human_review", "reply_to_email")
graph.add_edge("reply_to_email", "step_end")
graph.add_edge("step_end", END)

# Compile the graph
agent = graph.compile()